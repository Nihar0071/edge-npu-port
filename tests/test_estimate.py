"""Roofline estimator: hand-checkable cases on a made-up chip."""

from pathlib import Path

import pytest
import yaml

from edge_npu_port import estimate, graph, hardware

from .test_inspect import _conv_silu_model

ROOT = Path(__file__).resolve().parents[1]

TOY = {
    "name": "toy", "kind": "programmable-npu", "description": "toy",
    "compute": {"macs_per_cycle": 100, "vector_ops_per_cycle": 10, "clock_hz": 1.0e6, "layer_overhead_s": 0.0},
    "memory": {"onchip_bytes": 10**9, "onchip_bytes_per_s": 1e12, "ddr_bytes_per_s": 1e6},
    "bytes_per_elem": 1, "compiler": {"fuse_silu": True},
    "op_support": {"default": "custom", "matrix": ["Conv"]},
}


def _toy(tmp_path, **mem):
    d = {**TOY, "memory": {**TOY["memory"], **mem}}
    p = tmp_path / "toy.yaml"
    p.write_text(yaml.safe_dump(d))
    return hardware.load(str(p))


def test_compute_bound_when_everything_fits(tmp_path):
    g = graph.load(_conv_silu_model(tmp_path))
    # fast DDR: 6 KB of input/output at 1 GB/s = 6 us, far below ~3 ms of compute
    e = estimate.estimate(g, _toy(tmp_path, ddr_bytes_per_s=1e9))
    conv = e.layers[0]
    macs = 16 * 16 * 16 * 8 * 9
    assert conv.compute_s == pytest.approx(macs / (100 * 1e6))
    # fused SiLU epilogue: 2 vector ops per output element
    assert conv.vector_s == pytest.approx(16 * 16 * 16 * 2 / (10 * 1e6))
    # only the graph input is read from DDR and the graph output written: x + y bytes
    assert conv.ddr_bytes == 8 * 16 * 16 + 16 * 16 * 16 + 16 * 8 * 9 * 0  # weights resident (fit)
    assert conv.bound == "compute"


def test_spilling_layer_becomes_memory_bound(tmp_path):
    g = graph.load(_conv_silu_model(tmp_path))
    e = estimate.estimate(g, _toy(tmp_path, onchip_bytes=1000))  # tiny on-chip memory: nothing fits
    conv = e.layers[0]
    # weights (1168 B) no longer resident + inputs/outputs spill
    assert conv.ddr_bytes == (8 * 16 * 16 + 16 * 16 * 16) + (16 * 8 * 9 + 16)
    assert conv.total_s == pytest.approx(conv.memory_s)
    assert conv.bound == "memory"


@pytest.mark.skipif(not (ROOT / "models" / "yolo11n.onnx").exists(), reason="run `edgeport twin` first")
def test_nano_estimate_sane():
    e = estimate.estimate(graph.load(ROOT / "models" / "yolo11n.onnx"), hardware.load("nano-gpnpu"))
    ideal = e.total_macs / e.peak_macs_per_s
    assert ideal < e.total_s < 3 * ideal  # never faster than the MAC roofline
