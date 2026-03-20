from __future__ import annotations

from typing import Tuple
from app.services.plate.plate_utils import normalize_vn_plate_confusions


def expand_bbox(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    img_w: int,
    img_h: int,
    scale: float = 0.2,
) -> Tuple[int, int, int, int]:
    """Expand tight bbox to avoid cutting plate borders."""
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return x1, y1, x2, y2

    dx = int(w * scale)
    dy = int(h * scale)

    nx1 = max(0, x1 - dx)
    ny1 = max(0, y1 - dy)
    nx2 = min(img_w, x2 + dx)
    ny2 = min(img_h, y2 + dy)

    return nx1, ny1, nx2, ny2


def normalize_vn_plate(text: str) -> str:
    return normalize_vn_plate_confusions(text)
