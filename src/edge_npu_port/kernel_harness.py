"""Step 5 harness: feed the C++ simulator a *real* layer from the int8 model.

Takes one QDQ Conv (+ SiLU) node from the quantized model and extracts:
  * its actual int8 input on a real COCO image (as ONNX Runtime computes it)
  * int8 weights, per-channel weight scales, int32 bias
  * the conv-output and SiLU-output quantization parameters
Then converts the float scales into what integer hardware uses -- a fixed-point
multiplier (m0, shift) per output channel -- and builds the 256-entry SiLU table.

Checks, in order (every kernel has a CPU reference):
  1. numpy integer reference == C++ scalar reference     (bit-exact)
  2. C++ tiled kernel on the simulator == C++ reference  (bit-exact, every mode)
  3. simulator output vs ONNX Runtime's int8 output      (within +-1 LSB, and why)
"""

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper

from . import preprocess

ROOT = Path(__file__).resolve().parents[2]
SIM = ROOT / "sim" / "build" / "npusim"


@dataclass
class QParams:
    scale: np.ndarray
    zero: np.ndarray


def _producers_consumers(model):
    prod = {o: n for n in model.graph.node for o in n.output}
    cons: dict[str, list] = {}
    for n in model.graph.node:
        for i in n.input:
            cons.setdefault(i, []).append(n)
    return prod, cons


def _init(model, name):
    for t in model.graph.initializer:
        if t.name == name:
            return numpy_helper.to_array(t)
    raise KeyError(name)


def _q(model, node) -> QParams:
    return QParams(_init(model, node.input[1]).astype(np.float64), _init(model, node.input[2]).astype(np.int64))


def quantize_multiplier(m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Real multiplier M in (0, 1) -> (m0 in [2^30, 2^31), right shift) with M ~= m0 * 2^-31 * 2^-shift."""
    frac, exp = np.frexp(m)  # m = frac * 2^exp, frac in [0.5, 1)
    m0 = np.round(frac * (1 << 31)).astype(np.int64)
    fix = m0 == (1 << 31)
    m0[fix] //= 2
    exp[fix] += 1
    shift = -exp
    assert (shift >= 0).all(), "multiplier >= 1 needs a left shift; not supported in this kernel"
    return m0.astype(np.int32), shift.astype(np.int32)


def _trunc_div(a: np.ndarray, d: int) -> np.ndarray:
    """Integer division rounding toward zero, like C++ '/' (numpy '//' floors)."""
    return np.where(a >= 0, a // d, -((-a) // d))


def requantize_np(acc: np.ndarray, m0: np.ndarray, shift: np.ndarray, zero: int) -> np.ndarray:
    """numpy twin of npusim::requantize (gemmlowp SaturatingRoundingDoublingHighMul + RoundingDivideByPOT)."""
    ab = acc.astype(np.int64) * m0.astype(np.int64)
    nudge = np.where(ab >= 0, 1 << 30, 1 - (1 << 30))
    hi = _trunc_div(ab + nudge, 1 << 31)
    mask = (1 << shift.astype(np.int64)) - 1
    rem = hi & mask
    thr = (mask >> 1) + (hi < 0)
    out = (hi >> shift) + (rem > thr) + zero
    return np.clip(out, -128, 127)


def extract(qdq_path: Path, conv_name: str, image: Path, out_dir: Path) -> dict:
    model = onnx.load(str(qdq_path))
    prod, cons = _producers_consumers(model)
    conv = next(n for n in model.graph.node if n.name == conv_name)
    dq_x, dq_w, dq_b = (prod[i] for i in conv.input[:3])
    x_q_name = dq_x.input[0]
    qx = _q(model, dq_x)
    w = _init(model, dq_w.input[0]).astype(np.int8)
    sw = _init(model, dq_w.input[1]).astype(np.float64)
    bias = _init(model, dq_b.input[0]).astype(np.int32)

    q_conv = next(n for n in cons[conv.output[0]] if n.op_type == "QuantizeLinear")
    qc = _q(model, q_conv)
    # follow DQ -> (Sigmoid, Mul) -> Q to the SiLU output's quantization
    dq_c = cons[q_conv.output[0]][0]
    mul = next(n for n in cons[dq_c.output[0]] if n.op_type == "Mul")
    q_y = next(n for n in cons[mul.output[0]] if n.op_type == "QuantizeLinear")
    qy = _q(model, q_y)

    # run ORT on a real image, exposing the int8 input and int8 output of this layer
    for name in (x_q_name, q_y.output[0]):
        model.graph.output.append(helper.make_empty_tensor_value_info(name))
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    x, _ = preprocess.load(image)
    x_q, y_ort = sess.run([x_q_name, q_y.output[0]], {sess.get_inputs()[0].name: x})

    _, cin, h, wd = x_q.shape
    cout = w.shape[0]
    x_hwc = x_q[0].reshape(cin, h * wd).T.copy()  # HWC: each pixel's channels contiguous
    y_ort_hwc = y_ort[0].reshape(cout, h * wd).T.copy()
    w2 = w.reshape(cout, cin)

    m = qx.scale * sw / qc.scale  # per output channel
    m0, shift = quantize_multiplier(np.atleast_1d(m))
    q = np.arange(-128, 128)
    v = (q - qc.zero) * qc.scale
    silu = v / (1 + np.exp(-v))
    lut = np.clip(np.rint(silu / qy.scale) + qy.zero, -128, 127).astype(np.int8)

    out_dir.mkdir(parents=True, exist_ok=True)
    x_hwc.astype(np.int8).tofile(out_dir / "x.bin")
    w2.tofile(out_dir / "w.bin")
    bias.tofile(out_dir / "bias.bin")
    m0.tofile(out_dir / "m0.bin")
    shift.tofile(out_dir / "shift.bin")
    lut.tofile(out_dir / "lut.bin")
    (out_dir / "meta.txt").write_text(
        f"pixels {h * wd}\ncin {cin}\ncout {cout}\nx_zero {int(qx.zero)}\npre_zero {int(qc.zero)}\n")

    # numpy integer reference
    acc = (x_hwc.astype(np.int64) - int(qx.zero)) @ w2.astype(np.int64).T + bias
    pre = requantize_np(acc, m0[None, :], shift[None, :], int(qc.zero))
    y_np = lut[(pre + 128).astype(np.int64)]
    np.save(out_dir / "y_numpy.npy", y_np)
    np.save(out_dir / "y_ort.npy", y_ort_hwc)
    return {"layer": conv_name, "pixels": h * wd, "cin": cin, "cout": cout, "hw": [h, wd],
            "s_x": float(qx.scale), "z_x": int(qx.zero), "s_conv": float(qc.scale), "z_conv": int(qc.zero),
            "s_y": float(qy.scale), "z_y": int(qy.zero),
            "m_range": [float(m.min()), float(m.max())], "shift_range": [int(shift.min()), int(shift.max())]}


def simulate(data_dir: Path, mode: str, sweep: bool = False, pixels_per_pe: str = "auto") -> dict | list:
    args = [str(SIM), str(data_dir), "--mode", mode, "--pixels-per-pe", pixels_per_pe]
    if sweep:
        args.append("--sweep")
    p = subprocess.run(args, capture_output=True, text=True)
    if p.returncode == 2:
        raise RuntimeError(p.stderr)
    return json.loads(p.stdout)


def compare(data_dir: Path) -> dict:
    y_sim = np.fromfile(data_dir / "y_sim.bin", dtype=np.int8)
    y_np = np.load(data_dir / "y_numpy.npy").ravel()
    y_ort = np.load(data_dir / "y_ort.npy").ravel()
    d = y_sim.astype(np.int16) - y_ort.astype(np.int16)
    return {"sim_vs_numpy_bit_exact": bool((y_sim == y_np).all()),
            "sim_vs_ort_exact_pct": float((d == 0).mean() * 100),
            "sim_vs_ort_within_1lsb_pct": float((np.abs(d) <= 1).mean() * 100),
            "sim_vs_ort_max_abs_lsb": int(np.abs(d).max())}


def plot_timeline(results: dict[str, dict], path: Path, max_tiles: int = 6) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"load": "#1f6feb", "store": "#8957e5", "compute": "#2da44e"}
    fig, axes = plt.subplots(len(results), 1, figsize=(9, 1.3 * len(results) + 0.6), dpi=150, sharex=True)
    for ax, (mode, r) in zip(np.atleast_1d(axes), results.items()):
        lanes = {"dma": 1, "compute": 0}
        horizon = 0
        for res, kind, tile, s, e in r["timeline"]:
            if tile >= max_tiles:
                continue
            ax.barh(lanes[res], (e - s) / 1e3, left=s / 1e3, color=colors[kind], edgecolor="white", height=0.6)
            horizon = max(horizon, e)
        ax.set_yticks([0, 1], ["MAC array", "DMA"])
        ax.set_title(f"{mode}: {r['cycles'] / 1e3:.0f}k cycles total, MAC utilization {r['mac_utilization']:.0%} "
                     f"(first {max_tiles} of {r['tiles']} tiles, {r['pixels_per_pe']} px/PE)", fontsize=8, loc="left")
        ax.grid(axis="x", alpha=0.3)
    np.atleast_1d(axes)[-1].set_xlabel("thousand cycles")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()]
    fig.legend(handles, colors.keys(), loc="upper right", fontsize=7, ncol=3)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    return path


def plot_sweep(sweep: list[dict], path: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 3.8), dpi=150)
    for mode in ("v0", "v1", "v2"):
        pts = [r for r in sweep if r["mode"] == mode]
        ax.plot([r["pixels_per_pe"] for r in pts], [r["cycles"] / 1e3 for r in pts], "o-", label=mode, ms=4)
    ideal = sweep[0]["ideal_mac_cycles"] / 1e3
    ax.axhline(ideal, color="#555", ls="--", lw=1, label=f"MAC roofline ({ideal:.0f}k)")
    ax.set_xlabel("pixels per PE per tile (bigger tile = fewer DMA start-ups, more local memory)")
    ax.set_ylabel("thousand cycles")
    ax.set_title("tile-size sweep (each line stops where the tile no longer fits in 4 KB)", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    return path
