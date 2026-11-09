"""Step 6: measure on a real phone NPU via Qualcomm AI Hub.

Two jobs, both compiled for the Hexagon NPU (QNN DLC) and profiled on a
real device:
  * the deployable int8 model from Step 4 (per-layer time + which unit ran each layer)
  * a microbenchmark model (one 3x3 conv, one 1x1 conv, one big Add) used to
    *calibrate* a hardware profile for this NPU -- same empirical-roofline method
    as `edgeport calibrate` on the local CPU, so the estimate isn't built on
    marketing TOPS.

Needs `uv sync --extra aihub` and a configured token (`qai-hub configure --api_token ...`).
"""

import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

DEFAULT_DEVICE = "Samsung Galaxy S25 (Family)"


def strict_ir(src: Path, dst: Path) -> Path:
    """Drop value_info entries that duplicate graph inputs/outputs.

    The ONNX IR spec forbids it; ONNX Runtime tolerates it, AI Hub's compiler rejects it
    ("Tensors {'output0'} occur in value_info but also in model IO"). ORT's quantizer emits it.
    """
    m = onnx.load(str(src))
    io = {v.name for v in [*m.graph.input, *m.graph.output]}
    keep = [v for v in m.graph.value_info if v.name not in io]
    del m.graph.value_info[:]
    m.graph.value_info.extend(keep)
    onnx.checker.check_model(m)
    onnx.save(m, str(dst))
    return dst


def compile_and_profile(onnx_path: Path, device: str, out_dir: Path, name: str, input_specs: dict) -> dict:
    import qai_hub as hub

    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = strict_ir(onnx_path, out_dir / f"{name}.strict.onnx")
    dev = hub.Device(device)
    cj = hub.submit_compile_job(model=str(onnx_path), device=dev, name=f"{name}-compile",
                                input_specs=input_specs, options="--target_runtime qnn_dlc")
    target = cj.get_target_model()
    if target is None:
        raise RuntimeError(f"compile failed: {cj.get_status().message} ({cj.url})")
    pj = hub.submit_profile_job(model=target, device=dev, name=f"{name}-profile")
    prof = pj.download_profile()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.profile.json").write_text(json.dumps(prof, indent=2, default=str))
    (out_dir / f"{name}.jobs.json").write_text(json.dumps({"compile": cj.url, "profile": pj.url, "device": device}))
    return prof


def microbench_model(path: Path, calib_dir: Path) -> Path:
    """Float graph with three independent ops, then int8 QDQ via ONNX Runtime."""
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static

    rng = np.random.default_rng(0)
    w3 = numpy_helper.from_array((rng.standard_normal((64, 64, 3, 3)) * 0.05).astype(np.float32), "w3")
    w1 = numpy_helper.from_array((rng.standard_normal((64, 256, 1, 1)) * 0.05).astype(np.float32), "w1")
    nodes = [
        helper.make_node("Conv", ["a", "w3"], ["conv3x3"], name="bench_conv3x3", pads=[1, 1, 1, 1]),
        helper.make_node("Conv", ["b", "w1"], ["conv1x1"], name="bench_conv1x1"),
        helper.make_node("Add", ["c", "d"], ["add"], name="bench_add"),
    ]
    big = [1, 64, 320, 320]  # 6.5 M elements: far larger than any on-chip memory at int8
    ins = [helper.make_tensor_value_info("a", TensorProto.FLOAT, [1, 64, 80, 80]),
           helper.make_tensor_value_info("b", TensorProto.FLOAT, [1, 256, 80, 80]),
           helper.make_tensor_value_info("c", TensorProto.FLOAT, big),
           helper.make_tensor_value_info("d", TensorProto.FLOAT, big)]
    outs = [helper.make_tensor_value_info(n, TensorProto.FLOAT, None) for n in ("conv3x3", "conv1x1", "add")]
    g = helper.make_graph(nodes, "microbench", ins, outs, initializer=[w3, w1])
    float_path = calib_dir / "microbench.float.onnx"
    calib_dir.mkdir(parents=True, exist_ok=True)
    onnx.save(helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)], ir_version=8), float_path)

    shapes = {i.name: [d.dim_value for d in i.type.tensor_type.shape.dim] for i in ins}

    class R(CalibrationDataReader):
        def __init__(self):
            self.n = 0

        def get_next(self):
            if self.n >= 4:
                return None
            self.n += 1
            return {k: rng.standard_normal(s).astype(np.float32) for k, s in shapes.items()}

    quantize_static(str(float_path), str(path), R(), quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QInt8, weight_type=QuantType.QInt8)
    return path


MICROBENCH_SPECS = {"a": (1, 64, 80, 80), "b": (1, 256, 80, 80), "c": (1, 64, 320, 320), "d": (1, 64, 320, 320)}


# ---- validation against the device profile ------------------------------------------
# QNN's per-op numbers are HTP *cycles*, and they don't add up to wall time (ops run on
# parallel hardware threads). So we stay in one unit: calibrate the profile in QNN cycles
# from the microbenchmark, estimate YOLO in the same unit, compare with YOLO's QNN cycles.
# End-to-end wall time is reported separately (AI Hub's estimated_inference_time).

import re  # noqa: E402

MICROBENCH_MACS = {"bench_conv3x3": 64 * 80 * 80 * 64 * 9, "bench_conv1x1": 64 * 80 * 80 * 256}
MICROBENCH_ADD_BYTES = 3 * 64 * 320 * 320  # two int8 inputs read + one written


def _norm(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


def _cycles(prof: dict) -> dict[str, int]:
    return {r["name"]: int(r["execution_cycles"]) for r in prof["execution_detail"]}


def npu_profile(micro: dict, device: str) -> dict:
    """A hardware-profile dict for this NPU, in QNN-cycle units (clock_hz = 1 'cycle' per second)."""
    c = _cycles(micro)
    conv3 = MICROBENCH_MACS["bench_conv3x3"] / c["bench_conv3x3"]
    conv1 = MICROBENCH_MACS["bench_conv1x1"] / c["bench_conv1x1"]
    ddr = MICROBENCH_ADD_BYTES / c["bench_add"]
    return {
        "name": "snapdragon-8-elite-htp",
        "kind": "fixed-function-npu",
        "description": f"Hexagon v79 NPU ({device}) via QNN DLC, int8. Rates calibrated from an AI Hub "
                       f"microbenchmark, in QNN cycles (not seconds).",
        "compute": {"macs_per_cycle": round(conv3), "conv_macs_per_cycle": round(conv3),
                    "vector_ops_per_cycle": 512, "clock_hz": 1.0, "layer_overhead_s": 0},
        "memory": {"onchip_bytes": 8 * 2**20, "onchip_bytes_per_s": 256, "ddr_bytes_per_s": round(ddr, 2)},
        "bytes_per_elem": 1,
        "compiler": {"fuse_silu": True},
        "op_support": {"default": "fallback",
                       "matrix": ["Conv", "MatMul", "Gemm", "ConvTranspose"],
                       "vector": ["Add", "Sub", "Mul", "Div", "Sigmoid", "Softmax", "MaxPool", "Resize", "Relu"],
                       "layout": ["Concat", "Split", "Reshape", "Transpose", "Slice"]},
        "measured": {"conv3x3_macs_per_cycle": conv3, "conv1x1_macs_per_cycle": conv1, "add_bytes_per_cycle": ddr},
        "assumptions": [
            {"field": "units", "value": "QNN cycles", "note": "per-op QNN cycles don't sum to wall time; compare like with like"},
            {"field": "compute.vector_ops_per_cycle, memory.onchip_*", "value": "512, 8 MB, 256 B/cycle",
             "note": "not measured (the Add was DDR-bound); VTCM size for v79 not confirmed"},
        ],
    }


def compare(yolo: dict, est, module_of) -> dict:
    """Per-module measured QNN cycles vs estimate (same unit)."""
    meas: dict[str, int] = {}
    unmatched = 0
    known = {_norm(l.name): l.name for l in est.layers}
    for name, cyc in _cycles(yolo).items():
        n = _norm(name)
        m = re.search(r"model_(\d+)_", n + "_")
        if m:
            meas[f"model.{m.group(1)}"] = meas.get(f"model.{m.group(1)}", 0) + cyc
        elif n in known:
            meas[module_of(known[n])] = meas.get(module_of(known[n]), 0) + cyc
        else:
            unmatched += cyc
    est_mod = est.by_module()
    rows = [{"module": k, "estimated": est_mod.get(k, 0.0), "measured": v} for k, v in meas.items()]
    rows.sort(key=lambda r: -r["measured"])
    tot_m = sum(r["measured"] for r in rows)
    tot_e = sum(r["estimated"] for r in rows)
    wmape = sum(abs(r["estimated"] - r["measured"]) for r in rows) / tot_m if tot_m else float("nan")
    units = {}
    for r in yolo["execution_detail"]:
        units[r["compute_unit"]] = units.get(r["compute_unit"], 0) + 1
    return {"rows": rows, "total_estimated": tot_e, "total_measured": tot_m, "error": (tot_e - tot_m) / tot_m,
            "per_module_wmape": wmape, "unmatched_cycles": unmatched, "compute_units": units,
            "wall_us": yolo["execution_summary"]["estimated_inference_time"]}
