"""Image preprocessing shared by every step.

Every step (twin check, quantization calibration, accuracy eval, hardware runs)
must feed the model exactly the same tensor for the same image. If calibration
used one resize and evaluation used another, accuracy numbers would drift for
reasons that have nothing to do with quantization.

The recipe matches Ultralytics' default "letterbox":
  1. resize so the long side is `size`, keeping the aspect ratio
  2. pad the short side with gray (114) to a square `size` x `size`
  3. BGR -> RGB, HWC -> CHW, uint8 -> float32 in [0, 1], add a batch dim
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class Letterbox:
    """How an original image maps into the model input. Needed to map boxes back."""

    scale: float  # model pixels per original pixel
    pad_x: float  # left padding, in model pixels
    pad_y: float  # top padding, in model pixels


def letterbox(img_bgr: np.ndarray, size: int = 640) -> tuple[np.ndarray, Letterbox]:
    h, w = img_bgr.shape[:2]
    scale = min(size / h, size / w)
    new_w, new_h = round(w * scale), round(h * scale)
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_x, pad_y = (size - new_w) / 2, (size - new_h) / 2
    top, bottom = round(pad_y - 0.1), round(pad_y + 0.1)
    left, right = round(pad_x - 0.1), round(pad_x + 0.1)
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return padded, Letterbox(scale=scale, pad_x=left, pad_y=top)


def to_tensor(img_bgr: np.ndarray) -> np.ndarray:
    """uint8 HWC BGR -> float32 NCHW RGB in [0, 1]."""
    rgb = img_bgr[:, :, ::-1]
    chw = np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32) / 255.0
    return chw[None]


def load(path: str | Path, size: int = 640) -> tuple[np.ndarray, Letterbox]:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    boxed, lb = letterbox(img, size)
    return to_tensor(boxed), lb
