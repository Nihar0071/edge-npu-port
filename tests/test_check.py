"""The CI gate must pass a model within budget and fail one outside it."""

from pathlib import Path

import pytest

from edge_npu_port.checks import run_checks

ONNX = Path(__file__).resolve().parents[1] / "models" / "yolo11n.onnx"
pytestmark = pytest.mark.skipif(not ONNX.exists(), reason="run `edgeport twin` first")


def test_passes_within_budget():
    assert run_checks(str(ONNX), "nano-gpnpu", max_ms=16.7, max_peak_mb=4)["pass"]


def test_fails_over_budget():
    r = run_checks(str(ONNX), "nano-gpnpu", max_ms=5)
    assert not r["pass"]
    assert next(c for c in r["checks"] if c["name"] == "latency")["pass"] is False
