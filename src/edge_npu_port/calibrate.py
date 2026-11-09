"""Measure a local CPU's roofline numbers with ONNX Runtime microbenchmarks.

Spec sheets give *theoretical* peaks; a model's runtime sees the *achievable*
ones, through the same runtime, threads and kernels. So for the local CPU we
measure instead of guessing (an "empirical roofline"):

  peak MAC rate    one big MatMul (2048^3)        -> compute ceiling
  DDR bandwidth    Add of two 256 MB tensors      -> streams far beyond any cache
  on-chip bw       Add of two 2 MB tensors        -> stays in L2
  vector rate      Sigmoid on an in-cache tensor  -> element-wise throughput
  layer overhead   Add of two 1-element tensors   -> pure per-op dispatch cost

Each is timed with the ORT profiler (kernel time, not Python call overhead),
median of many runs.
"""

import json
import platform
import subprocess
import tempfile
from pathlib import Path
from statistics import median

import numpy as np
import onnx
import onnxruntime as ort
import yaml
from onnx import TensorProto, helper


def _model(op: str, shapes: list[list[int]], out_shape: list[int], **attrs) -> bytes:
    ins = [helper.make_tensor_value_info(f"x{i}", TensorProto.FLOAT, s) for i, s in enumerate(shapes)]
    node = helper.make_node(op, [f"x{i}" for i in range(len(shapes))], ["y"], **attrs)
    g = helper.make_graph([node], op, ins, [helper.make_tensor_value_info("y", TensorProto.FLOAT, out_shape)])
    # ir_version 8 matches opset 17; the onnx package's newest default can be ahead of onnxruntime
    return helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)], ir_version=8).SerializeToString()


def session_options(threads: int, level=ort.GraphOptimizationLevel.ORT_ENABLE_BASIC, profile_dir: str | None = None):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = level
    if profile_dir:
        so.enable_profiling = True
        so.profile_file_prefix = str(Path(profile_dir) / "ort")
    return so


def kernel_times(sess: ort.InferenceSession, feeds: dict, runs: int, warmup: int = 3) -> dict[str, list[float]]:
    """Run, then read per-node kernel durations (seconds) from the ORT profile."""
    # ORT can stop profiling only once per session, so record everything and drop
    # each node's first `warmup` samples (cold caches, lazy allocations).
    for _ in range(warmup + runs):
        sess.run(None, feeds)
    events = json.loads(Path(sess.end_profiling()).read_text())
    out: dict[str, list[float]] = {}
    for e in events:
        if e.get("cat") == "Node" and e["name"].endswith("_kernel_time"):
            out.setdefault(e["name"][: -len("_kernel_time")], []).append(e["dur"] * 1e-6)
    return {k: v[warmup:] or v for k, v in out.items()}


def _bench(op: str, shapes, out_shape, threads: int, runs: int, **attrs) -> float:
    with tempfile.TemporaryDirectory() as d:
        sess = ort.InferenceSession(_model(op, shapes, out_shape, **attrs), session_options(threads, profile_dir=d),
                                    providers=["CPUExecutionProvider"])
        feeds = {f"x{i}": np.random.rand(*s).astype(np.float32) for i, s in enumerate(shapes)}
        times = kernel_times(sess, feeds, runs)
        return median(next(iter(times.values())))


def _cpu_name() -> str:
    try:
        return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except Exception:
        return platform.processor() or "cpu"


def calibrate(threads: int, onchip_bytes: int, name: str) -> dict:
    n = 2048
    t_mm = _bench("MatMul", [[n, n], [n, n]], [n, n], threads, 10)
    # A big GEMM is the best case. Real CNN convs are smaller and oddly shaped, and may
    # take a different kernel path, so measure a representative one too: 3x3, 64->64 ch, 80x80.
    c, hw_ = 64, 80
    t_conv = _bench("Conv", [[1, c, hw_, hw_], [c, c, 3, 3]], [1, c, hw_, hw_], threads, 20, pads=[1, 1, 1, 1])
    conv_macs = c * hw_ * hw_ * c * 9
    big = 64 * 2**20  # 64 M floats = 256 MB per tensor
    t_ddr = _bench("Add", [[big], [big]], [big], threads, 5)
    small = 512 * 1024  # 2 MB per tensor, 6 MB total: in L2
    t_l2 = _bench("Add", [[small], [small]], [small], threads, 50)
    t_sig = _bench("Sigmoid", [[small]], [small], threads, 50)
    t_tiny = _bench("Add", [[1], [1]], [1], threads, 200)

    macs_per_s = n**3 / t_mm
    vector_ops_per_s = small * 4 / t_sig  # Sigmoid counted as 4 vector ops (see estimate.DEFAULT_VECTOR_COST)
    return {
        "name": name,
        "kind": "cpu",
        "description": f"{_cpu_name()}, ONNX Runtime CPU EP, {threads} threads, fp32. "
                       f"Calibrated by microbenchmarks (edgeport calibrate), not from spec sheets.",
        # 1 GHz "nominal clock" so per-cycle fields read as per-nanosecond rates
        "compute": {"macs_per_cycle": round(macs_per_s / 1e9),
                    "conv_macs_per_cycle": round(conv_macs / t_conv / 1e9),
                    "vector_ops_per_cycle": round(vector_ops_per_s / 1e9),
                    "clock_hz": 1.0e9, "layer_overhead_s": float(f"{t_tiny:.3g}")},
        "memory": {"onchip_bytes": onchip_bytes, "onchip_bytes_per_s": float(f"{3 * small * 4 / t_l2:.3g}"),
                   "ddr_bytes_per_s": float(f"{3 * big * 4 / t_ddr:.3g}")},
        "bytes_per_elem": 4,
        # validated at ORT_ENABLE_BASIC: no Conv+SiLU fusion, so the estimate must not assume it
        "compiler": {"fuse_silu": False},
        "op_support": {"default": "vector", "matrix": ["Conv", "ConvTranspose", "MatMul", "Gemm"],
                       "layout": ["Concat", "Split", "Reshape", "Transpose", "Slice", "Flatten", "Squeeze",
                                  "Unsqueeze", "Identity", "Gather", "Pad", "Expand", "Cast", "Shape"]},
        "measured": {"matmul_2048_s": t_mm, "conv3x3_64ch_80px_s": t_conv, "add_256MB_s": t_ddr, "add_2MB_s": t_l2,
                     "sigmoid_2MB_s": t_sig, "add_1elem_s": t_tiny, "threads": threads,
                     "onnxruntime": ort.__version__},
        "assumptions": [{"field": "compute.conv_macs_per_cycle", "value": "one 3x3 64ch 80x80 conv",
                         "note": "real conv rate varies ~40x with shape; one representative conv is a simplification"}],
    }


def write(profile: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(profile, sort_keys=False))
    return path
