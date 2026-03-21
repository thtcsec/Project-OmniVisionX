import time

import cv2
import numpy as np

from app.services.plate.vehicle_types import normalize_vehicle_type


def estimate_vehicle_color(
    vehicle_crop: np.ndarray | None,
    crop_bbox: tuple | None = None,
    plate_bbox: tuple | list | None = None,
) -> str:
    if vehicle_crop is None or vehicle_crop.size == 0:
        return "unknown"

    try:
        hsv = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2HSV)
        hue = hsv[:, :, 0]
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        mean_val = float(np.mean(val))
        mean_sat = float(np.mean(sat))

        if mean_val < 28:
            return "unknown"
        if mean_sat < 18 and mean_val < 75:
            return "unknown"

        height, width = hue.shape
        mask = np.zeros((height, width), dtype=np.uint8)
        margin_x = max(1, int(width * 0.08))
        margin_y = max(1, int(height * 0.08))
        mask[margin_y:height - margin_y, margin_x:width - margin_x] = 1

        if crop_bbox and plate_bbox and len(crop_bbox) == 4 and len(plate_bbox) == 4:
            px1 = max(0, int(plate_bbox[0]) - int(crop_bbox[0]))
            py1 = max(0, int(plate_bbox[1]) - int(crop_bbox[1]))
            px2 = min(width, int(plate_bbox[2]) - int(crop_bbox[0]))
            py2 = min(height, int(plate_bbox[3]) - int(crop_bbox[1]))
            if px2 > px1 and py2 > py1:
                pad_x = max(2, int((px2 - px1) * 0.12))
                pad_y = max(2, int((py2 - py1) * 0.2))
                mask[max(0, py1 - pad_y):min(height, py2 + pad_y), max(0, px1 - pad_x):min(width, px2 + pad_x)] = 0

        valid = mask.astype(bool)
        if not np.any(valid):
            valid = np.ones((height, width), dtype=bool)

        neutral = valid & (sat < 42)
        neutral_ratio = float(np.count_nonzero(neutral)) / max(float(np.count_nonzero(valid)), 1.0)
        if neutral_ratio >= 0.58:
            mean_v = float(np.mean(val[neutral])) if np.any(neutral) else float(np.mean(val[valid]))
            if mean_v < 55 and mean_sat < 28:
                return "unknown"
            if mean_v < 65:
                return "black"
            if mean_v > 205:
                return "white"
            if mean_v > 155:
                return "silver"
            return "gray"

        chromatic = valid & (sat >= 42) & (val >= 35)
        if not np.any(chromatic):
            mean_v = float(np.mean(val[valid]))
            if mean_v < 65:
                return "black"
            if mean_v > 205:
                return "white"
            return "gray"

        weights = (sat[chromatic].astype(np.float32) / 255.0) * (0.35 + val[chromatic].astype(np.float32) / 255.0)
        hues = hue[chromatic]

        color_scores = {
            "red": weights[(hues < 8) | (hues >= 172)].sum(),
            "orange": weights[(hues >= 8) & (hues < 22)].sum(),
            "yellow": weights[(hues >= 22) & (hues < 38)].sum(),
            "green": weights[(hues >= 38) & (hues < 82)].sum(),
            "blue": weights[(hues >= 82) & (hues < 128)].sum(),
            "purple": weights[(hues >= 128) & (hues < 172)].sum(),
        }
        best_color = max(color_scores, key=color_scores.get)
        if best_color == "orange":
            return "yellow"

        min_score = 0.02 * float(np.count_nonzero(valid))
        return best_color if color_scores[best_color] > min_score else "unknown"
    except Exception:
        return "unknown"


def refine_vehicle_type(
    detected_type: str,
    plate_text: str | None,
    vehicle_bbox: tuple[int, int, int, int] | None,
    plate_bbox: tuple | list | None,
) -> str:
    normalized = normalize_vehicle_type(detected_type, plate_text)

    if normalized not in {"car", "unknown"}:
        return normalized

    if not vehicle_bbox or not plate_bbox or len(plate_bbox) != 4:
        return normalized

    try:
        vx1, vy1, vx2, vy2 = [int(v) for v in vehicle_bbox]
        px1, py1, px2, py2 = [int(v) for v in plate_bbox]
        vehicle_w = max(1, vx2 - vx1)
        vehicle_h = max(1, vy2 - vy1)
        plate_w = max(1, px2 - px1)
        plate_h = max(1, py2 - py1)

        vehicle_aspect = vehicle_w / vehicle_h
        plate_aspect = plate_w / plate_h
        plate_coverage = (plate_w * plate_h) / float(vehicle_w * vehicle_h)
        vehicle_area = vehicle_w * vehicle_h

        if plate_aspect <= 2.55 and vehicle_aspect <= 1.7 and plate_coverage >= 0.015:
            if vehicle_area > 15000:
                return normalized
            return "motorcycle"
    except Exception:
        return normalized

    return normalized


class VehicleBboxSmoother:
    def __init__(self, alpha: float = 0.25, ttl_s: float = 2.5):
        self._cache: dict[str, tuple[float, tuple[int, int, int, int]]] = {}
        self._alpha = max(0.05, min(0.95, float(alpha)))
        self._ttl_s = max(0.5, float(ttl_s))

    @property
    def ttl_s(self) -> float:
        return self._ttl_s

    def smooth(
        self,
        camera_id: str,
        track_id: int,
        bbox: tuple[int, int, int, int],
    ) -> tuple[int, int, int, int]:
        if track_id < 0:
            return bbox

        key = f"{camera_id}:{track_id}"
        now = time.monotonic()
        prev = self._cache.get(key)
        if prev and (now - float(prev[0])) <= self._ttl_s:
            prev_bbox = prev[1]
            smoothed = [
                int(round(self._alpha * float(new_v) + (1.0 - self._alpha) * float(prev_v)))
                for new_v, prev_v in zip(bbox, prev_bbox)
            ]
        else:
            smoothed = [int(v) for v in bbox]

        if smoothed[2] <= smoothed[0]:
            smoothed[2] = smoothed[0] + 1
        if smoothed[3] <= smoothed[1]:
            smoothed[3] = smoothed[1] + 1

        result = (smoothed[0], smoothed[1], smoothed[2], smoothed[3])
        self._cache[key] = (now, result)
        return result

    def compute_sharpness(self, img_bgr: np.ndarray, bbox: tuple[int, int, int, int]) -> float:
        try:
            h, w = img_bgr.shape[:2]
            x1, y1, x2, y2 = bbox
            x1 = max(0, min(w - 1, int(x1)))
            y1 = max(0, min(h - 1, int(y1)))
            x2 = max(1, min(w, int(x2)))
            y2 = max(1, min(h, int(y2)))
            if x2 <= x1 or y2 <= y1:
                return 0.0
            roi = img_bgr[y1:y2, x1:x2]
            if roi.size == 0:
                return 0.0
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
            return float(cv2.Laplacian(gray, cv2.CV_64F).var())
        except Exception:
            return 0.0

    def evict_stale(self, now: float | None = None) -> int:
        now = time.monotonic() if now is None else now
        stale_keys = [
            key for key, (updated_at, _) in self._cache.items()
            if now - float(updated_at) > self._ttl_s
        ]
        for key in stale_keys:
            del self._cache[key]
        return len(stale_keys)
