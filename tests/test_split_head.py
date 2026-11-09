"""Moving the box decode to the host must not change the float model's output."""

from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest

from edge_npu_port import preprocess, split_head as sh

ROOT = Path(__file__).resolve().parents[1]
ONNX = ROOT / "models" / "yolo11n.onnx"
pytestmark = pytest.mark.skipif(not ONNX.exists(), reason="run `edgeport twin` first")


def _image():
    import ultralytics
    return sorted((Path(ultralytics.__file__).parent / "assets").glob("*.jpg"))[0]


@pytest.mark.parametrize("variant", [{}, {"logits": True}, {"dfl_on_host": True}])
def test_host_decode_is_a_twin(tmp_path, variant):
    anchors, strides = sh.decode_constants(ONNX)
    cut = sh.cut(ONNX, tmp_path / "cut.onnx", **variant)
    x, _ = preprocess.load(_image())
    ref = ort.InferenceSession(str(ONNX)).run(None, {"images": x})[0][0]
    got = sh.infer(cut, anchors, strides, bins=sh.dfl_weights(ONNX))(x)
    for rows in (slice(0, 4), slice(4, None)):  # boxes and scores separately (Step 1)
        rel = np.abs(ref[rows] - got[rows]).max() / np.abs(ref[rows]).max()
        assert rel < 1e-4


def test_logit_threshold_matches_probability_threshold():
    rng = np.random.default_rng(0)
    logits = rng.normal(-4, 3, (80, 500)).astype(np.float32)
    keep, cls, score = sh.candidates_from_logits(logits, conf=0.25)
    probs = 1 / (1 + np.exp(-logits))
    expected = np.nonzero(probs.max(0) > 0.25)[0]
    assert (keep == expected).all()
    assert np.allclose(score, probs.max(0)[keep])
