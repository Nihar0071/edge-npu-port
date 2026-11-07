"""A flat, per-layer view of an ONNX graph: shapes, weight sizes, MACs, liveness.

`inspect` and `estimate` both build on this. Everything is counted in *elements*;
callers multiply by bytes-per-element for the target dtype (1 for int8, 4 for fp32).
"""

from dataclasses import dataclass, field
from math import prod
from pathlib import Path

import onnx
from onnx import TensorProto, numpy_helper, shape_inference

FLOAT_TYPES = {TensorProto.FLOAT, TensorProto.FLOAT16, TensorProto.BFLOAT16, TensorProto.DOUBLE}


@dataclass
class Layer:
    index: int
    name: str
    op: str
    inputs: list[str]  # activation inputs only (weights excluded)
    outputs: list[str]
    in_shapes: list[tuple[int, ...]]
    out_shapes: list[tuple[int, ...]]
    weight_elems: int = 0  # float weights + biases this layer reads
    macs: int = 0  # multiply-accumulates (Conv / MatMul / Gemm)
    attrs: dict = field(default_factory=dict)

    @property
    def act_in_elems(self) -> int:
        return sum(prod(s) for s in self.in_shapes)

    @property
    def act_out_elems(self) -> int:
        return sum(prod(s) for s in self.out_shapes)

    def working_set(self, bpe: int = 1) -> int:
        """Bytes that must be reachable at once to run this layer in one go."""
        return (self.act_in_elems + self.act_out_elems + self.weight_elems) * bpe


@dataclass
class Graph:
    path: Path
    opset: int
    layers: list[Layer]
    shapes: dict[str, tuple[int, ...]]
    inputs: list[str]
    outputs: list[str]
    weight_elems: int

    @property
    def total_macs(self) -> int:
        return sum(l.macs for l in self.layers)

    def peak_live_elems(self) -> tuple[int, int]:
        """Largest total size of activations alive at the same moment, and where.

        A tensor is alive from the layer that produces it until its last consumer.
        Skip connections (YOLO's Concat of early and late features) keep early,
        large tensors alive for a long time -- that's what pushes peak memory up.
        Assumes ONNX's required topological node order and no in-place reuse.
        """
        last_use: dict[str, int] = {}
        for l in self.layers:
            for t in l.inputs:
                last_use[t] = l.index
        for t in self.outputs:
            last_use[t] = len(self.layers)

        live: dict[str, int] = {t: prod(self.shapes[t]) for t in self.inputs}
        peak, at = sum(live.values()), -1
        for l in self.layers:
            for t, s in zip(l.outputs, l.out_shapes):
                live[t] = prod(s)
            total = sum(live.values())
            if total > peak:
                peak, at = total, l.index
            for t in list(live):
                if last_use.get(t, -1) <= l.index:
                    del live[t]
        return peak, at


def fuse_silu(g: Graph) -> Graph:
    """Fold `Conv -> Sigmoid -> Mul(conv_out, sigmoid_out)` into one Conv with a SiLU epilogue.

    This is what an NPU compiler does: the activation is applied to each output
    value while it's still in registers, so the conv output, the sigmoid output
    and the product never exist as three separate tensors in memory. Fewer
    tensors in memory means lower peak memory and less traffic.
    Only fuses when the conv output feeds nothing else.
    """
    consumers: dict[str, list[Layer]] = {}
    for l in g.layers:
        for t in l.inputs:
            consumers.setdefault(t, []).append(l)

    drop: set[int] = set()
    fused: list[Layer] = []
    for l in g.layers:
        if l.index in drop:
            continue
        if l.op == "Conv" and len(l.outputs) == 1 and l.outputs[0] not in g.outputs:
            x = l.outputs[0]
            users = consumers.get(x, [])
            sig = next((u for u in users if u.op == "Sigmoid"), None)
            mul = next((u for u in users if u.op == "Mul" and sig and sig.outputs[0] in u.inputs), None)
            if sig and mul and len(users) == 2 and len(consumers.get(sig.outputs[0], [])) == 1:
                l = Layer(**{**l.__dict__, "outputs": mul.outputs, "attrs": {**l.attrs, "fused": "SiLU"}})
                drop |= {sig.index, mul.index}
        fused.append(l)

    for i, l in enumerate(fused):
        l.index = i
    return Graph(path=g.path, opset=g.opset, layers=fused, shapes=g.shapes,
                 inputs=g.inputs, outputs=g.outputs, weight_elems=g.weight_elems)


def _shape(vi: onnx.ValueInfoProto) -> tuple[int, ...] | None:
    dims = vi.type.tensor_type.shape.dim
    if not dims or any(not d.HasField("dim_value") for d in dims):
        return None
    return tuple(d.dim_value for d in dims)


def _macs(op: str, ins: list[tuple], outs: list[tuple], wshape: tuple | None, attrs: dict) -> int:
    if op in ("Conv", "ConvTranspose") and wshape and outs:
        # weight: [C_out, C_in/groups, kH, kW]; every output element needs C_in/groups*kH*kW MACs
        return prod(outs[0]) * prod(wshape[1:])
    if op == "MatMul" and outs:
        a = ins[0] if ins else wshape
        return prod(outs[0]) * a[-1]
    if op == "Gemm" and outs:
        a = ins[0] if ins else wshape
        k = a[0] if attrs.get("transA") else a[-1]
        return prod(outs[0]) * k
    return 0


def load(path: str | Path) -> Graph:
    model = onnx.load(str(path))
    model = shape_inference.infer_shapes(model, strict_mode=True)
    g = model.graph

    inits = {t.name: t for t in g.initializer}
    shapes: dict[str, tuple[int, ...]] = {}
    for vi in [*g.input, *g.value_info, *g.output]:
        if (s := _shape(vi)) is not None:
            shapes[vi.name] = s
    for name, t in inits.items():
        shapes[name] = tuple(t.dims)

    # Constant nodes hold small tensors (reshape targets, anchor grids). Treat them like weights.
    const_outs = {n.output[0]: numpy_helper.to_array(n.attribute[0].t) for n in g.node if n.op_type == "Constant"}
    for name, arr in const_outs.items():
        shapes[name] = tuple(arr.shape)

    layers, total_w = [], 0
    for n in g.node:
        if n.op_type == "Constant":
            continue
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in n.attribute}
        act_in = [i for i in n.input if i and i not in inits and i not in const_outs]
        w_names = [i for i in n.input if i in inits and inits[i].data_type in FLOAT_TYPES]
        w_elems = sum(prod(inits[i].dims) for i in w_names)
        wshape = tuple(inits[w_names[0]].dims) if w_names else None
        missing = [t for t in [*act_in, *n.output] if t not in shapes]
        if missing:
            raise ValueError(f"no static shape for {missing} at node {n.name}; export with static shapes")
        ins = [shapes[t] for t in act_in]
        outs = [shapes[t] for t in n.output]
        layers.append(Layer(index=len(layers), name=n.name or f"{n.op_type}_{len(layers)}", op=n.op_type,
                            inputs=act_in, outputs=list(n.output), in_shapes=ins, out_shapes=outs,
                            weight_elems=w_elems, macs=_macs(n.op_type, ins, outs, wshape, attrs), attrs=attrs))
        total_w += w_elems

    return Graph(path=Path(path), opset=model.opset_import[0].version, layers=layers, shapes=shapes,
                 inputs=[i.name for i in g.input if i.name not in inits],
                 outputs=[o.name for o in g.output], weight_elems=total_w)
