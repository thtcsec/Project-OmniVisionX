from typing import Optional

import cv2
import numpy as np


def should_drop_oversized_bbox(
    frame_shape: tuple[int, int, int] | tuple[int, int],
    bbox: tuple[int, int, int, int],
    confidence: float,
    oversized_ratio: float,
    oversized_low_conf: float,
) -> tuple[bool, float]:
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = bbox
    det_area = max(0, x2 - x1) * max(0, y2 - y1)
    frame_area = max(1, h * w)
    area_ratio = det_area / frame_area
    should_drop = area_ratio > oversized_ratio and confidence < oversized_low_conf
    return should_drop, area_ratio


def compute_vehicle_crop_bbox(
    frame_shape: tuple[int, int, int] | tuple[int, int],
    bbox: tuple[int, int, int, int],
    class_name: str,
) -> tuple[int, int, int, int]:
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = bbox
    v_w = max(0, x2 - x1)
    v_h = max(0, y2 - y1)

    if class_name == "motorcycle":
        pad_x = min(120, max(28, int(max(v_w, v_h) * 0.16)))
        pad_top = min(90, max(20, int(v_h * 0.12)))
        pad_bottom = min(130, max(34, int(v_h * 0.28)))
    else:
        pad = min(80, max(20, int(min(v_w, v_h) * 0.08)))
        pad_x = pad
        pad_top = pad
        pad_bottom = pad

    crop_x1 = max(0, x1 - pad_x)
    crop_y1 = max(0, y1 - pad_top)
    crop_x2 = min(w, x2 + pad_x)
    crop_y2 = min(h, y2 + pad_bottom)
    return crop_x1, crop_y1, crop_x2, crop_y2


def should_try_legacy_merge(plates: list[dict], strict_conf: float) -> bool:
    if not plates:
        return False
    gate = max(0.18, strict_conf - 0.22)
    best_conf = max(float(p.get("confidence", 0.0)) for p in plates)
    return best_conf < gate


def get_relaxed_ocr_thresholds(strict_conf: float, strict_len: int, class_name: str) -> tuple[float, int]:
    relaxed_conf = max(0.15, strict_conf - 0.27)
    relaxed_len = max(4, strict_len - 2)
    if class_name == "motorcycle":
        relaxed_conf = max(0.12, relaxed_conf - 0.05)
        relaxed_len = max(3, relaxed_len - 1)
    return relaxed_conf, relaxed_len


def should_accept_raw_plate_candidate(
    plate_text: str,
    plate_confidence: float,
    class_name: str,
    strict_conf: float,
    strict_len: int,
) -> tuple[bool, float]:
    if not plate_text or plate_text.startswith("UNKNOWN-"):
        return False, 0.0
    # Reject obvious garbage (all digits > 10 chars, random symbols)
    alnum = "".join(c for c in plate_text if c.isalnum())
    if len(alnum) >= 12 and alnum.isdigit():
        return False, 0.0  # Likely phone number or noise
    alnum_len = sum(1 for c in plate_text if c.isalnum())
    relaxed_conf, relaxed_len = get_relaxed_ocr_thresholds(strict_conf, strict_len, class_name)
    return not (plate_confidence < relaxed_conf or alnum_len < relaxed_len), relaxed_conf


def compute_plate_sharpness(plate_crop: Optional[np.ndarray]) -> float:
    if plate_crop is None or not hasattr(plate_crop, "shape") or plate_crop.size <= 0:
        return 0.0
    try:
        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY) if len(plate_crop.shape) == 3 else plate_crop
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
    except Exception:
        return 0.0
