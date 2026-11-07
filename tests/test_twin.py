"""The twin check must pass on a faithful export and FAIL on a subtly broken one.

A check that has never been seen to fail proves nothing. This "negative control"
scales a single conv weight tensor by 1.01 -- the kind of silent corruption a
bad BatchNorm fold or wrong scale would cause -- and expects a FAIL.
"""

from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import numpy_helper

from edge_npu_port import twin

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "models" / "yolo11n.pt"
ONNX = ROOT / "models" / "yolo11n.onnx"

pytestmark = pytest.mark.skipif(not ONNX.exists(), reason="run `edgeport twin` first to export the model")


def _images():
    import ultralytics
    return sorted((Path(ultralytics.__file__).parent / "assets").glob("*.jpg"))


def test_faithful_export_passes():
    reports = twin.twin_check(str(WEIGHTS), ONNX, _images())
    assert all(twin.passes(r) for r in reports)


def test_one_percent_weight_error_fails(tmp_path):
    model = onnx.load(ONNX)
    # pick a mid-network conv weight (4-D initializer), not the first or last layer
    convs = [t for t in model.graph.initializer if len(t.dims) == 4]
    victim = convs[len(convs) // 2]
    w = numpy_helper.to_array(victim) * np.float32(1.01)
    victim.CopyFrom(numpy_helper.from_array(w, victim.name))
    broken = tmp_path / "broken.onnx"
    onnx.save(model, broken)

    reports = twin.twin_check(str(WEIGHTS), broken, _images())
    assert not any(twin.passes(r) for r in reports), "twin check missed a 1% weight corruption"
