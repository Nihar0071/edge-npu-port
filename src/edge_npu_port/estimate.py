"""Step 3: "How fast will it be, and where does the time go?"

A per-layer roofline model. For every layer:

    compute_s = MACs / peak MAC rate                     (MAC array)
    vector_s  = element ops / vector rate                (ALUs: activations, adds, pools)
    memory_s  = bytes that must cross DDR / DDR bandwidth
    layer_s   = max(compute_s + vector_s, memory_s) + fixed per-layer overhead

`max` because a well-built kernel overlaps data movement with math (double
buffering), so the slower of the two sets the pace. That's the roofline idea:
a layer is either compute-bound or memory-bound, and arithmetic intensity
(MACs per DDR byte) tells you which, compared against the chip's ridge point.

Which bytes cross DDR (simple, explicit rules):
  * weights: re-read from DDR every inference if the model's weights don't fit on-chip
  * activations: if a layer's inputs+outputs+weights don't fit on-chip, its inputs
    are read from and its outputs written to DDR (it must be tiled through DDR)
  * the graph input is read from DDR; graph outputs are written to DDR
"""

from dataclasses import dataclass
from math import prod

from .graph import Graph, Layer, fuse_silu
from .hardware import Profile

# Element-wise work per output element, in "vector ops". Overridable per profile
# (e.g. an NPU with a hardware sigmoid lookup table sets Sigmoid: 1).
DEFAULT_VECTOR_COST = {
    "Add": 1, "Sub": 1, "Mul": 1, "Div": 1, "Relu": 1, "Clip": 1, "Neg": 1, "Abs": 1,
    "Min": 1, "Max": 1, "Sigmoid": 4, "Tanh": 4, "Exp": 4, "Softmax": 6, "Erf": 4,
    "HardSigmoid": 2, "HardSwish": 3, "LeakyRelu": 1, "Pow": 4, "Sqrt": 4, "Resize": 2,
    "ReduceMean": 1, "ReduceMax": 1, "GlobalAveragePool": 1,
}
FREE_OPS = {"Reshape", "Flatten", "Squeeze", "Unsqueeze", "Identity", "Shape"}  # metadata only
SILU_EPILOGUE_COST = 2  # sigmoid (lookup) + multiply, applied to each conv output in registers


@dataclass
class LayerEstimate:
    index: int
    name: str
    op: str
    macs: int
    ddr_bytes: int
    compute_s: float
    vector_s: float
    memory_s: float
    total_s: float

    @property
    def bound(self) -> str:
        work = self.compute_s + self.vector_s
        if max(work, self.memory_s) < 1e-12:
            return "overhead"
        if self.memory_s > work:
            return "memory"
        return "compute" if self.compute_s >= self.vector_s else "vector"

    @property
    def intensity(self) -> float:
        """MACs per DDR byte (inf if no DDR traffic)."""
        return self.macs / self.ddr_bytes if self.ddr_bytes else float("inf")


@dataclass
class Estimate:
    model: str
    profile: str
    fused: bool
    layers: list[LayerEstimate]
    total_macs: int
    peak_macs_per_s: float
    ridge_point: float

    @property
    def total_s(self) -> float:
        return sum(l.total_s for l in self.layers)

    @property
    def fps(self) -> float:
        return 1 / self.total_s

    @property
    def mac_utilization(self) -> float:
        """Fraction of the MAC array's peak actually used over the whole inference."""
        return self.total_macs / self.peak_macs_per_s / self.total_s

    def time_by_bound(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for l in self.layers:
            out[l.bound] = out.get(l.bound, 0.0) + l.total_s
        return out

    def by_module(self) -> dict[str, float]:
        """Sum per YOLO-style module (`/model.N/...`), the unit humans reason about."""
        out: dict[str, float] = {}
        for l in self.layers:
            out[module_of(l.name)] = out.get(module_of(l.name), 0.0) + l.total_s
        return out


def module_of(name: str) -> str:
    parts = name.strip("/").split("/")
    return parts[1] if len(parts) > 1 and parts[0] == "model" else parts[0].split("_")[0]


def _vector_ops(l: Layer, cost: dict[str, float]) -> float:
    out = sum(prod(s) for s in l.out_shapes)
    if l.op in ("MaxPool", "AveragePool"):
        return out * prod(l.attrs.get("kernel_shape", [1]))
    if l.op in ("Conv", "ConvTranspose", "MatMul", "Gemm"):
        return out * SILU_EPILOGUE_COST if l.attrs.get("fused") == "SiLU" else 0
    return out * cost.get(l.op, 0)


def estimate(raw: Graph, hw: Profile) -> Estimate:
    g = fuse_silu(raw) if hw.fuse_silu else raw
    bpe, cost = hw.bytes_per_elem, {**DEFAULT_VECTOR_COST, **hw.vector_op_cost}
    weights_resident = g.weight_elems * bpe <= hw.onchip_bytes
    vector_rate = hw.vector_ops_per_cycle * hw.clock_hz

    est = []
    for l in g.layers:
        spills = l.working_set(bpe) > hw.onchip_bytes
        ddr = 0 if weights_resident else l.weight_elems * bpe
        if spills:
            ddr += (l.act_in_elems + l.act_out_elems) * bpe
        else:
            ddr += sum(prod(s) for t, s in zip(l.inputs, l.in_shapes) if t in g.inputs) * bpe
            ddr += sum(prod(s) for t, s in zip(l.outputs, l.out_shapes) if t in g.outputs) * bpe

        compute_s = l.macs / hw.macs_per_s(l.op)
        vector_s = _vector_ops(l, cost) / vector_rate
        if l.op in FREE_OPS:
            moved_s = 0.0
        else:  # even without math, every byte touched passes through on-chip memory
            moved_s = (l.act_in_elems + l.act_out_elems) * bpe / hw.onchip_bytes_per_s
        vector_s = max(vector_s, moved_s) if l.macs == 0 else vector_s
        memory_s = ddr / hw.ddr_bytes_per_s
        overhead = 0.0 if l.op in FREE_OPS else hw.layer_overhead_s
        total = max(compute_s + vector_s, memory_s) + overhead
        est.append(LayerEstimate(l.index, l.name, l.op, l.macs, ddr, compute_s, vector_s, memory_s, total))

    return Estimate(model=g.path.name, profile=hw.name, fused=hw.fuse_silu, layers=est,
                    total_macs=g.total_macs, peak_macs_per_s=hw.peak_macs_per_s, ridge_point=hw.ridge_point)


def render(e: Estimate, top: int = 10, budget_ms: float | None = None) -> str:
    ms = lambda s: f"{s * 1e3:.3f}"
    shares = e.time_by_bound()
    out = [
        f"{e.model} on {e.profile}{' (Conv+SiLU fused)' if e.fused else ''}",
        f"  estimated latency  {e.total_s * 1e3:.2f} ms  ({e.fps:.0f} FPS)",
        f"  ideal (MACs/peak)  {e.total_macs / e.peak_macs_per_s * 1e3:.2f} ms   "
        f"-> MAC utilization {e.mac_utilization:.0%}",
        f"  ridge point        {e.ridge_point:.0f} MACs per DDR byte",
        "  time by bottleneck " + ", ".join(f"{k} {v / e.total_s:.0%}" for k, v in
                                           sorted(shares.items(), key=lambda kv: -kv[1])),
    ]
    if budget_ms:
        verdict = "MEETS" if e.total_s * 1e3 <= budget_ms else "MISSES"
        out.append(f"  budget {budget_ms:.2f} ms     -> {verdict} ({e.total_s * 1e3 / budget_ms:.0%} of budget)")
    out += ["", f"  top {top} layers by time:",
            f"  {'#':>4} {'op':<9}{'ms':>8}{'share':>7}  {'bound':<8}{'MAC/B':>8}  name"]
    for l in sorted(e.layers, key=lambda l: -l.total_s)[:top]:
        ai = "-" if l.intensity == float("inf") else f"{l.intensity:.0f}"
        out.append(f"  {l.index:>4} {l.op:<9}{ms(l.total_s):>8}{l.total_s / e.total_s:>7.1%}  {l.bound:<8}{ai:>8}  {l.name}")
    return "\n".join(out)
