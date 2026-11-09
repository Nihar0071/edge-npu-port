"""Step 4: "Why did int8 hurt my accuracy, and what does fixing it cost on my chip?"

1. Stage-by-stage accuracy: PyTorch float -> ONNX float -> ONNX int8, same evaluator.
   If ONNX float already lost accuracy, it isn't quantization's fault.
2. Quantization configs: per-tensor vs per-channel weights, MinMax vs Percentile calibration.
3. Sensitivity: SQNR of every activation tensor, comparing the int8 model with the float one
   (ONNX Runtime's qdq_loss_debug). Low SQNR = quantizing that tensor destroys information.
4. Fix ladder: keep the k worst layers in float (`nodes_to_exclude`), re-measure mAP,
   and price each fix in milliseconds on the target chip with the Step 3 estimator.
   That last part -- accuracy recovered *per millisecond* -- is what the vendor tools don't print.

SQNR (signal-to-quantization-noise ratio), in dB:  10 * log10( sum(x^2) / sum((x - x_q)^2) )
Rule of thumb: ~6 dB per bit, so ideal int8 on a well-scaled tensor is ~40 dB; < 20 dB is trouble.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import onnx
from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization import qdq_loss_debug as qd
from onnxruntime.quantization.shape_inference import quant_pre_process

from . import coco_eval, estimate, graph, hardware, preprocess

DATA = coco_eval.DATA


@dataclass
class QuantConfig:
    name: str
    per_channel: bool
    method: CalibrationMethod
    extra: dict = field(default_factory=dict)


CONFIGS = [
    QuantConfig("per-tensor / minmax", False, CalibrationMethod.MinMax),
    QuantConfig("per-channel / minmax", True, CalibrationMethod.MinMax),
    QuantConfig("per-channel / percentile 99.999", True, CalibrationMethod.Percentile, {"CalibPercentile": 99.999}),
]


class ImageReader(CalibrationDataReader):
    def __init__(self, images: list[Path], input_name: str = "images", imgsz: int = 640):
        self.images, self.input_name, self.imgsz, self.i = images, input_name, imgsz, 0

    def get_next(self):
        if self.i >= len(self.images):
            return None
        x, _ = preprocess.load(self.images[self.i], self.imgsz)
        self.i += 1
        return {self.input_name: x}

    def rewind(self):
        self.i = 0


def calib_images(n: int) -> list[Path]:
    return sorted((DATA / "coco128" / "images" / "train2017").glob("*.jpg"))[:n]


def prepare(onnx_path: Path) -> Path:
    """ORT's recommended pre-step: shape inference + basic graph cleanup before quantizing."""
    out = onnx_path.with_suffix(".pre.onnx")
    if not out.exists():
        quant_pre_process(str(onnx_path), str(out))
    return out


def quantize(pre: Path, out: Path, cfg: QuantConfig, n_calib: int, exclude: list[str] = ()) -> Path:
    quantize_static(
        str(pre), str(out), ImageReader(calib_images(n_calib)),
        quant_format=QuantFormat.QDQ, per_channel=cfg.per_channel,
        activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
        calibrate_method=cfg.method, nodes_to_exclude=list(exclude),
        extra_options={**cfg.extra, "ActivationSymmetric": False, "WeightSymmetric": True},
        # cache calibration ranges per config: the fix ladder re-quantizes without re-calibrating
        calibration_cache_path=str(out.parent / f"calib-{cfg.name.replace(' ', '').replace('/', '_')}.json"),
    )
    return out


@dataclass
class TensorErr:
    tensor: str
    node: str
    op: str
    sqnr_db: float  # error introduced by quantizing this tensor alone
    xmodel_db: float  # accumulated error vs the float model at this point


@dataclass
class RangeFlag:
    """A Concat whose inputs have very different ranges but must share one int8 scale."""

    node: str
    output: str
    input_bits: list[float]  # effective int8 bits each input keeps under the shared scale

    @property
    def worst_bits(self) -> float:
        return min(self.input_bits)


def range_mixing(pre: Path, acts_f: dict[str, list[np.ndarray]], min_bits: float = 4.0) -> list[RangeFlag]:
    """Find Concats that crush a small-range input.

    Per-tensor SQNR is blind to this: the tensor's error is dominated by its large
    values, so SQNR looks fine while the small-range slice (e.g. YOLO's 0..1 class
    scores concatenated with 0..640 box coordinates) rounds to zero.
    bits_i = log2(span_i / scale_out), where scale_out = span_out / 255.
    """
    flags = []
    for n in onnx.load(str(pre)).graph.node:
        if n.op_type != "Concat" or not all(i in acts_f for i in n.input):
            continue
        spans = []
        for i in n.input:
            a = np.concatenate([x.ravel() for x in acts_f[i]])
            spans.append(float(a.max() - a.min()))
        scale = max(spans) / 255 if max(spans) > 0 else 1.0
        bits = [float(np.log2(max(s / scale, 1e-9))) if s > 0 else 8.0 for s in spans]
        if min(bits) < min_bits:
            flags.append(RangeFlag(n.name, n.output[0], [round(b, 2) for b in bits]))
    return sorted(flags, key=lambda f: f.worst_bits)


def sensitivity(pre: Path, qdq: Path, n_images: int, work: Path) -> tuple[list[TensorErr], list[RangeFlag]]:
    aug_f, aug_q = work / "aug_float.onnx", work / "aug_qdq.onnx"
    qd.modify_model_output_intermediate_tensors(str(pre), str(aug_f))
    qd.modify_model_output_intermediate_tensors(str(qdq), str(aug_q))
    imgs = sorted((DATA / "val2017").glob("*.jpg"))[:n_images]
    acts_q = qd.collect_activations(str(aug_q), ImageReader(imgs))
    acts_f = qd.collect_activations(str(aug_f), ImageReader(imgs))
    errs = qd.compute_activation_error(qd.create_activation_matching(acts_q, acts_f))

    producer = {o: n for n in onnx.load(str(pre)).graph.node for o in n.output}
    out = []
    for t, e in errs.items():
        if t in producer and np.isfinite(e.get("qdq_err", np.nan)):
            n = producer[t]
            out.append(TensorErr(t, n.name, n.op_type, float(e["qdq_err"]), float(e.get("xmodel_err", np.nan))))
    return sorted(out, key=lambda r: r.sqnr_db), range_mixing(pre, acts_f)


def decode_tail(pre: Path, heavy_macs: float = 1e6) -> list[str]:
    """Nodes after the last heavy (MAC-dominated) layer on every path to the output.

    In detectors this is the post-processing baked into the graph: DFL softmax,
    anchor/stride math, sigmoid, concat. It's cheap in MACs but numerically
    fussy (mixed ranges, exact pixel coordinates), so a common vendor recipe is to
    keep it in float or run it on the host.
    """
    g = graph.load(pre)
    consumers: dict[str, list] = {}
    for l in g.layers:
        for t in l.inputs:
            consumers.setdefault(t, []).append(l)
    has_heavy_below: dict[int, bool] = {}
    for l in reversed(g.layers):  # reverse topological order: descendants first
        below = [c for t in l.outputs for c in consumers.get(t, [])]
        has_heavy_below[l.index] = any(c.macs > heavy_macs or has_heavy_below[c.index] for c in below)
    return [l.name for l in g.layers if l.macs <= heavy_macs and not has_heavy_below[l.index]
            and any(u.macs > heavy_macs for u in g.layers[: l.index])]


def fp_cost_ms(onnx_path: Path, hw: hardware.Profile, nodes: list[str], fp_mac_ratio: float = 0.25) -> float:
    """Extra latency on `hw` if `nodes` stay in fp16 instead of int8.

    Assumption (printed in the report): fp16 MACs run at `fp_mac_ratio` of the int8 rate and
    move 2x the bytes; plus a quantize/dequantize pass at each boundary.
    """
    g = graph.load(onnx_path)
    est8 = estimate.estimate(g, hw)
    hw16 = hardware.Profile(**{**hw.__dict__, "bytes_per_elem": 2,
                               "macs_per_cycle": int(hw.macs_per_cycle * fp_mac_ratio)})
    est16 = estimate.estimate(g, hw16)
    owner = {}
    for l in (graph.fuse_silu(g) if hw.fuse_silu else g).layers:
        owner[l.name] = l.index
        for a in l.attrs.get("absorbed", []):
            owner[a] = l.index
    hit = {owner[n] for n in nodes if n in owner}
    vec_rate = hw.vector_ops_per_cycle * hw.clock_hz
    extra = 0.0
    for i in hit:
        a, b = est8.layers[i], est16.layers[i]
        extra += b.total_s - a.total_s
        L = (graph.fuse_silu(g) if hw.fuse_silu else g).layers[i]
        extra += (L.act_in_elems + L.act_out_elems) / vec_rate  # Q/DQ at the boundary
    return extra * 1e3


@dataclass
class Row:
    label: str
    map50_95: float
    map50: float
    added_ms: float = 0.0
    excluded: list[str] = field(default_factory=list)


def run(onnx_path: Path, weights: Path, hw_name: str, n_eval: int, n_calib: int, ladder: list[int],
        out_dir: Path, sens_images: int = 8) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    hw = hardware.load(hw_name)
    rows: list[Row] = []

    def score(label, infer, **kw):
        m = coco_eval.evaluate(infer, n_eval)
        rows.append(Row(label, m.map50_95, m.map50, **kw))
        print(f"  {label:<44} mAP50-95 {m.map50_95:.4f}  mAP50 {m.map50:.4f}", flush=True)

    print(f"stage-by-stage accuracy on {n_eval} COCO val2017 images:")
    score("pytorch float32", coco_eval.torch_infer(weights))
    score("onnx float32", coco_eval.onnx_infer(onnx_path))

    pre = prepare(onnx_path)

    # 1. naive int8, then diagnose it
    naive_cfg = CONFIGS[1]
    naive = quantize(pre, out_dir / "int8-naive.onnx", naive_cfg, n_calib)
    score(f"int8 {naive_cfg.name} (naive, nothing excluded)", coco_eval.onnx_infer(naive))
    print(f"\nsensitivity (SQNR per tensor), naive '{naive_cfg.name}':")
    sens, flags = sensitivity(pre, naive, sens_images, out_dir)
    for s in sens[:10]:
        print(f"  {s.sqnr_db:>6.1f} dB  (acc. {s.xmodel_db:>5.1f} dB)  {s.op:<8} {s.node}")
    print("\nrange mixing (Concat inputs crushed by a shared int8 scale):")
    for f in flags:
        print(f"  {f.node}: effective bits per input {f.input_bits}")
    if not flags:
        print("  none")
    blockers = [f.node for f in flags]

    # 2. compare configs with the blockers fixed, so the comparison means something
    print("\nconfigs, with range-mixing blockers kept in float:")
    for i, cfg in enumerate(CONFIGS):
        q = quantize(pre, out_dir / f"int8-cfg{i}.onnx", cfg, n_calib, blockers)
        score(f"int8 {cfg.name} + blockers float", coco_eval.onnx_infer(q),
              added_ms=fp_cost_ms(onnx_path, hw, blockers), excluded=blockers)
    cfg_rows = rows[-len(CONFIGS):]
    best_cfg = CONFIGS[max(range(len(CONFIGS)), key=lambda i: cfg_rows[i].map50_95)]

    # 3. fix ladder on the best config
    tail = decode_tail(pre)
    heavy_sqnr = [s.node for s in sens if s.node not in tail and s.node not in blockers]
    print(f"\nfix ladder on '{best_cfg.name}' (decode tail = {len(tail)} nodes after the last heavy conv):")
    steps = [("blockers + decode tail", blockers + tail)]
    steps += [(f"blockers + tail + {k} lowest-SQNR layers", blockers + tail + heavy_sqnr[:k]) for k in ladder]
    for i, (label, excl) in enumerate(steps):
        excl = list(dict.fromkeys(excl))
        q = quantize(pre, out_dir / f"int8-fix{i}.onnx", best_cfg, n_calib, excl)
        score(f"int8 {best_cfg.name} + {label}", coco_eval.onnx_infer(q),
              added_ms=fp_cost_ms(onnx_path, hw, excl), excluded=excl)

    float_map = rows[1].map50_95
    report = {
        "model": onnx_path.name, "target": hw.name, "eval_images": n_eval, "calib_images": n_calib,
        "float_onnx_map50_95": float_map,
        "rows": [asdict(r) | {"drop_pts": (float_map - r.map50_95) * 100} for r in rows],
        "sensitivity_top": [asdict(s) for s in sens[:20]],
        "range_mixing": [asdict(f) for f in flags],
        "assumptions": ["layers kept in float run as fp16 on the target at 1/4 the int8 MAC rate, 2x bytes"],
    }
    (out_dir / "quant_doctor.json").write_text(json.dumps(report, indent=2))
    return report


def plot(report: dict, path: Path, budget_drop_pts: float = 1.0) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fmap = report["float_onnx_map50_95"] * 100
    rows = [r for r in report["rows"] if r["label"].startswith("int8")]
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
    ax.axhline(fmap, color="#555", lw=1, ls="--", label=f"float ONNX ({fmap:.1f})")
    ax.axhline(fmap - budget_drop_pts, color="#c0392b", lw=1, ls=":", label=f"spec: ≤ {budget_drop_pts:g} pt drop")
    ladder = [r for r in rows if "tail" in r["label"]]
    base = [r for r in rows if "tail" not in r["label"]]
    for r in base:
        ax.scatter(r["added_ms"], r["map50_95"] * 100, s=30, zorder=3, color="#888")
        ax.annotate(r["label"].replace("int8 ", "").replace(" + blockers float", ""),
                    (r["added_ms"], r["map50_95"] * 100), xytext=(6, 0),
                    textcoords="offset points", fontsize=6, va="center")
    if ladder:
        xs = [r["added_ms"] for r in ladder]
        ys = [r["map50_95"] * 100 for r in ladder]
        ax.plot(xs, ys, "o-", color="#1f6feb", zorder=3, label="fix ladder (layers kept in fp16)")
        for r, x, y in zip(ladder, xs, ys):
            ax.annotate(f"{len(r['excluded'])} nodes", (x, y), xytext=(4, -10), textcoords="offset points", fontsize=6)
    ax.set_xlabel(f"added latency on {report['target']} (ms, estimated)")
    ax.set_ylabel(f"COCO mAP50-95 ({report['eval_images']} val images)")
    ax.set_title(f"{report['model']}: accuracy recovered vs latency cost", fontsize=10)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    return path
