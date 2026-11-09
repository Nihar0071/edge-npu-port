"""Fix for the int8 detection head: cut the graph before the box decode.

Problem (Steps 4 and 6): YOLO's graph ends with box *pixel* coordinates (0..640) and
class scores (0..1). In int8:
  * sharing one scale zeroes the scores (fixed by keeping the Concat in float),
  * even with their own scale, pixel coordinates get ~2.5 px steps -> mAP50-95 drops,
  * keeping the whole decode tail in float fixes accuracy but runs as fp16 on the
    Snapdragon NPU and costs 57% of its cycles.

Fix: end the NPU graph at the DFL output -- box *distances* in grid units (~0..16,
so an int8 step is ~0.06 grid cells = 0.5..2 px depending on stride) -- plus the
class scores. The anchor/stride arithmetic (a few ops per box, 8400 boxes) runs on
the host. On a GPNPU that's plain C++ on the same core.

The anchors and strides are read from the model's own constants, so the host decode
is the graph's decode, not a reimplementation that could drift.
"""

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper

DIST = "/model.23/dfl/Reshape_1_output_0"  # [1, 4, 8400] lt, rb distances in grid units
SCORES = "/model.23/Sigmoid_output_0"  # [1, 80, 8400] probabilities 0..1
LOGITS = "/model.23/Concat_1_output_0"  # [1, 80, 8400] the same, before the sigmoid
# Why logits: an int8 step on probabilities is 1/255 ~ 0.004, so every score < 0.004 rounds
# to zero and the low end of the precision-recall curve disappears. Logits are spread out
# evenly, so int8 keeps small probabilities apart. Sigmoid preserves order, so the host can
# threshold in logit space and apply the sigmoid only to the candidates that survive.


def _const(model, name: str) -> np.ndarray:
    for n in model.graph.node:
        if n.op_type == "Constant" and n.output[0] == name:
            return numpy_helper.to_array(n.attribute[0].t)
    for t in model.graph.initializer:
        if t.name == name:
            return numpy_helper.to_array(t)
    raise KeyError(name)


def decode_constants(onnx_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Anchor points [2, 8400] and strides [1, 8400] exactly as baked into the graph."""
    m = onnx.load(str(onnx_path))
    by_name = {n.name: n for n in m.graph.node}
    anchors = _const(m, by_name["/model.23/Sub"].input[0]).reshape(2, -1)
    strides = _const(m, by_name["/model.23/Mul_2"].input[1]).reshape(1, -1)
    return anchors.astype(np.float32), strides.astype(np.float32)


BOX_LOGITS = "/model.23/Concat_output_0"  # [1, 64, 8400] = 4 sides x 16 DFL bins, before softmax


def cut(onnx_path: Path, out_path: Path, logits: bool = False, dfl_on_host: bool = False) -> Path:
    """Extract images -> box output + class output.

    dfl_on_host=True ends the graph at the last convolutions ("convs only"): box DFL
    logits and class logits. Everything after is host work.
    """
    box = BOX_LOGITS if dfl_on_host else DIST
    cls = LOGITS if (logits or dfl_on_host) else SCORES
    onnx.utils.extract_model(str(onnx_path), str(out_path), ["images"], [box, cls])
    return out_path


def dfl_weights(onnx_path: Path) -> np.ndarray:
    """The DFL conv's fixed weights (0, 1, ..., 15): expected value over the bins."""
    m = onnx.load(str(onnx_path))
    conv = next(n for n in m.graph.node if n.name == "/model.23/dfl/conv/Conv")
    w = next(t for t in m.graph.initializer if t.name == conv.input[1])
    return numpy_helper.to_array(w).reshape(-1).astype(np.float32)


def dfl_decode(box_logits: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """(64, N) logits -> (4, N) distances: softmax over each side's 16 bins, then expectation."""
    l = box_logits.reshape(4, len(bins), -1)
    e = np.exp(l - l.max(1, keepdims=True))
    return (e / e.sum(1, keepdims=True) * bins[None, :, None]).sum(1)


def host_decode(dist: np.ndarray, scores: np.ndarray, anchors: np.ndarray, strides: np.ndarray) -> np.ndarray:
    """(4, N) distances + (80, N) scores -> (84, N) [cx, cy, w, h, scores...] like the original output."""
    lt, rb = dist[:2], dist[2:]
    x1y1, x2y2 = anchors - lt, anchors + rb
    cxcy = (x1y1 + x2y2) / 2
    wh = x2y2 - x1y1
    return np.concatenate([np.concatenate([cxcy, wh]) * strides, scores])


def infer(model_path: Path, anchors: np.ndarray, strides: np.ndarray, threads: int = 6,
          bins: np.ndarray | None = None):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    sess = ort.InferenceSession(str(model_path), so, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    outs = [o.name for o in sess.get_outputs()]
    logits = LOGITS in outs
    dfl_host = BOX_LOGITS in outs
    i_d = outs.index(BOX_LOGITS if dfl_host else DIST)
    i_s = outs.index(LOGITS if logits else SCORES)

    def run(x: np.ndarray) -> np.ndarray:
        r = sess.run(None, {name: x})
        s = r[i_s][0]
        if logits:  # the evaluator wants every score, so apply sigmoid to all here
            s = 1 / (1 + np.exp(-s))
        d = dfl_decode(r[i_d][0], bins) if dfl_host else r[i_d][0]
        return host_decode(d, s, anchors, strides)
    return run


def candidates_from_logits(logits: np.ndarray, conf: float = 0.25) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deployment-style host post-processing: threshold in logit space, sigmoid only survivors."""
    t = np.log(conf / (1 - conf))
    best = logits.max(0)
    keep = np.nonzero(best > t)[0]
    cls = logits[:, keep].argmax(0)
    return keep, cls, 1 / (1 + np.exp(-best[keep]))
