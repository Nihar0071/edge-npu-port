"""COCO mAP for any `x -> raw YOLO output` function, so PyTorch, float ONNX and int8
ONNX are all scored by the *same* preprocessing, decoding and evaluator.
That's what makes the stage-by-stage loss (PyTorch -> ONNX -> int8) attributable.
"""

import contextlib
import io
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort

from . import preprocess
from .twin import decode

DATA = Path(__file__).resolve().parents[2] / "data"


@dataclass
class MAP:
    map50_95: float
    map50: float
    images: int


def coco80_to_91() -> list[int]:
    from ultralytics.data.converter import coco80_to_coco91_class
    return coco80_to_coco91_class()


def image_ids(n: int, ann: Path = DATA / "annotations" / "instances_val2017.json"):
    from pycocotools.coco import COCO
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(str(ann))
    ids = sorted(coco.getImgIds())[:n]  # deterministic subset: first n ids
    return coco, ids


def evaluate(infer: Callable[[np.ndarray], np.ndarray], n: int = 500, imgsz: int = 640) -> MAP:
    """Ultralytics-style eval settings: conf 0.001, IoU 0.7, max 300 detections, single label per box."""
    from pycocotools.cocoeval import COCOeval

    coco, ids = image_ids(n)
    cat = coco80_to_91()
    dets = []
    for img_id in ids:
        info = coco.loadImgs(img_id)[0]
        x, lb = preprocess.load(DATA / "val2017" / info["file_name"], imgsz)
        d = decode(infer(x), conf=0.001, iou=0.7)[:300]
        if not len(d):
            continue
        # undo the letterbox: model pixels -> original image pixels
        b = d[:, :4].copy()
        b[:, [0, 2]] = ((b[:, [0, 2]] - lb.pad_x) / lb.scale).clip(0, info["width"])
        b[:, [1, 3]] = ((b[:, [1, 3]] - lb.pad_y) / lb.scale).clip(0, info["height"])
        for (x1, y1, x2, y2), s, c in zip(b, d[:, 4], d[:, 5]):
            dets.append({"image_id": img_id, "category_id": cat[int(c)],
                         "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)], "score": float(s)})

    if not dets:  # e.g. a broken int8 model whose scores all quantized to zero
        return MAP(0.0, 0.0, len(ids))
    with contextlib.redirect_stdout(io.StringIO()):
        ev = COCOeval(coco, coco.loadRes(dets), "bbox")
        ev.params.imgIds = ids
        ev.evaluate(), ev.accumulate(), ev.summarize()
    return MAP(map50_95=float(ev.stats[0]), map50=float(ev.stats[1]), images=len(ids))


def onnx_infer(path: str | Path, threads: int = 6) -> Callable[[np.ndarray], np.ndarray]:
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    return lambda x: sess.run(None, {name: x})[0][0]


def torch_infer(weights: str | Path) -> Callable[[np.ndarray], np.ndarray]:
    import torch
    from ultralytics import YOLO

    m = YOLO(str(weights)).model.float().eval()

    def run(x):
        with torch.inference_mode():
            out = m(torch.from_numpy(x))
        return (out[0] if isinstance(out, (list, tuple)) else out).numpy()[0]
    return run
