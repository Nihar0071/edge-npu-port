"""Step 1: export a PyTorch model to ONNX and prove the two are twins.

Porting rule #1: before quantizing or writing kernels, prove the
model you are about to port computes the same thing as the original. Every
later error (quantization loss, kernel bugs) is measured against this baseline,
so if the baseline is already wrong you can't tell which step broke things.

What can make an ONNX export differ from PyTorch even in float32:
  * graph rewrites at export: Conv+BatchNorm folding, constant folding, op fusion
  * different kernels: ONNX Runtime and PyTorch sum in different orders, and
    float addition isn't associative, so tiny (~1e-5 relative) differences are normal
  * real bugs: wrong opset semantics, a dynamic shape baked in, a skipped layer

So we don't ask "are they equal?", we ask "is the difference at the size of
float rounding, or bigger?" -- and we check it on the model's *outputs that
matter* (box coordinates vs class scores separately, plus final detections).
"""

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from . import preprocess

# YOLO11 detect output: (batch, 4 + num_classes, num_anchors)
#   rows 0..3  = box (cx, cy, w, h) in input pixels, range ~0..640
#   rows 4..83 = per-class scores after sigmoid, range 0..1
# The two groups live on very different scales, so we compare them separately:
# a 0.01 error is noise on a box coordinate but huge on a class score.
NUM_BOX_ROWS = 4


@dataclass
class GroupDiff:
    max_abs: float
    mean_abs: float
    max_rel: float  # max |a-b| / max |a| over the group: scale-free
    cosine: float


@dataclass
class TwinReport:
    image: str
    boxes: GroupDiff
    scores: GroupDiff
    detections_pt: int
    detections_onnx: int
    detections_matched: int  # same class, IoU > 0.99


def export_onnx(weights: str, out_dir: Path, imgsz: int = 640, opset: int = 17) -> Path:
    """Export with Ultralytics' exporter.

    opset 17 is pinned on purpose: newer opsets add ops that some edge
    toolchains (QNN / Qualcomm AI Hub, vendor compilers) don't accept yet.
    Static shapes (dynamic=False) because NPU compilers plan memory ahead of
    time and want every tensor size known at compile time.
    """
    from ultralytics import YOLO

    out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(weights)
    path = model.export(format="onnx", imgsz=imgsz, opset=opset,
                        dynamic=False, simplify=True, half=False)
    dst = out_dir / Path(path).name
    if Path(path).resolve() != dst.resolve():
        Path(path).replace(dst)
    return dst


def _diff(a: np.ndarray, b: np.ndarray) -> GroupDiff:
    d = np.abs(a - b)
    a64, b64 = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    cos = float(a64 @ b64 / (np.linalg.norm(a64) * np.linalg.norm(b64) + 1e-12))
    return GroupDiff(max_abs=float(d.max()), mean_abs=float(d.mean()),
                     max_rel=float(d.max() / (np.abs(a).max() + 1e-12)), cosine=cos)


def decode(pred: np.ndarray, conf: float = 0.25, iou: float = 0.7) -> np.ndarray:
    """(4+C, N) raw output -> (K, 6) [x1, y1, x2, y2, score, class] after NMS.

    NMS (non-max suppression) runs outside the ONNX graph, on the host CPU.
    On a programmable NPU it can be ordinary C++ on the same core.
    """
    import torchvision

    boxes, scores = pred[:NUM_BOX_ROWS].T, pred[NUM_BOX_ROWS:].T
    cls = scores.argmax(1)
    best = scores[np.arange(len(cls)), cls]
    keep = best > conf
    boxes, best, cls = boxes[keep], best[keep], cls[keep]
    xyxy = np.concatenate([boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2], 1)
    # class-aware NMS: offset boxes per class so different classes never suppress each other
    offset = (cls[:, None] * 4096.0).astype(np.float32)
    idx = torchvision.ops.nms(torch.from_numpy(xyxy + offset), torch.from_numpy(best.astype(np.float32)), iou).numpy()
    return np.concatenate([xyxy[idx], best[idx, None], cls[idx, None]], 1)


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:4], b[None, :, 2:4])
    inter = np.prod(np.clip(br - tl, 0, None), 2)
    area = lambda x: np.prod(x[:, 2:4] - x[:, :2], 1)
    return inter / (area(a)[:, None] + area(b)[None] - inter + 1e-9)


def _match(dp: np.ndarray, do: np.ndarray) -> int:
    if len(dp) == 0 or len(do) == 0:
        return 0
    ious = _iou(dp, do)
    same_cls = dp[:, None, 5] == do[None, :, 5]
    return int(((ious > 0.99) & same_cls).any(1).sum())


def twin_check(weights: str, onnx_path: Path, images: list[Path], imgsz: int = 640) -> list[TwinReport]:
    from ultralytics import YOLO

    # The *original* PyTorch module: un-fused BatchNorm, PyTorch kernels.
    # Comparing against this (not the exporter's fused copy) also verifies the
    # Conv+BN folding the exporter did.
    torch_model = YOLO(weights).model.float().eval()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    reports = []
    for img in images:
        x, _ = preprocess.load(img, imgsz)
        with torch.inference_mode():
            out = torch_model(torch.from_numpy(x))
        pt = (out[0] if isinstance(out, (list, tuple)) else out).numpy()[0]
        ox = sess.run(None, {input_name: x})[0][0]
        assert pt.shape == ox.shape, f"shape mismatch {pt.shape} vs {ox.shape}"

        dp, do = decode(pt), decode(ox)
        reports.append(TwinReport(
            image=img.name,
            boxes=_diff(pt[:NUM_BOX_ROWS], ox[:NUM_BOX_ROWS]),
            scores=_diff(pt[NUM_BOX_ROWS:], ox[NUM_BOX_ROWS:]),
            detections_pt=len(dp), detections_onnx=len(do),
            detections_matched=_match(dp, do),
        ))
    return reports


def passes(r: TwinReport, rel_tol: float = 1e-4) -> bool:
    """Float32 twins should agree to ~1e-5 relative; 1e-4 leaves headroom.

    Plus a functional check: every detection must match one-to-one.
    """
    return (r.boxes.max_rel < rel_tol and r.scores.max_rel < rel_tol
            and r.detections_pt == r.detections_onnx == r.detections_matched)


def as_dict(reports: list[TwinReport]) -> list[dict]:
    return [asdict(r) for r in reports]
