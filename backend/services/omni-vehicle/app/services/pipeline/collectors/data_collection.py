import json
import os
import time
import random
import re
import threading
from datetime import datetime
from typing import Optional, Tuple
import cv2


def _safe_name(value: str) -> str:
    if not value:
        return ""
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value)[:48]


class DataCollector:
    """Lightweight image collector for active learning."""

    def __init__(
        self,
        enabled: bool,
        base_dir: str,
        sample_rate: float = 1.0,
        collect_vehicles: bool = True,
        collect_plates: bool = True,
        min_conf: float = 0.0,
        max_conf: float = 1.0,
        low_conf_only: bool = False,
        quality_filter: bool = True,
        min_sharpness: float = 40.0,
        min_brightness: float = 35.0,
        max_brightness: float = 230.0,
        min_vehicle_area: int = 2500,
        min_plate_area: int = 600,
    ) -> None:
        self.enabled = enabled
        self.base_dir = base_dir
        self.sample_rate = max(0.0, min(1.0, sample_rate))
        self.collect_vehicles = collect_vehicles
        self.collect_plates = collect_plates
        self.min_conf = max(0.0, min(1.0, min_conf))
        self.max_conf = max(0.0, min(1.0, max_conf))
        self.low_conf_only = low_conf_only
        self.quality_filter = quality_filter
        self.min_sharpness = min_sharpness
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.min_vehicle_area = min_vehicle_area
        self.min_plate_area = min_plate_area

        self.vehicle_dir = os.path.join(self.base_dir, "vehicles")
        self.plate_dir = os.path.join(self.base_dir, "plates")
        self.stats_path = os.path.join(self.base_dir, "quality_stats.json")
        self._stats_lock = threading.Lock()
        self._stats = self._load_stats()
        self._last_save_time = time.time()

        if self.enabled:
            os.makedirs(self.vehicle_dir, exist_ok=True)
            os.makedirs(self.plate_dir, exist_ok=True)

    def _should_collect(self, conf: Optional[float]) -> bool:
        if not self.enabled:
            return False
        if self.sample_rate < 1.0 and random.random() > self.sample_rate:
            return False
        if conf is None:
            return True
        if self.low_conf_only and conf >= self.max_conf:
            return False
        return self.min_conf <= conf <= self.max_conf

    def _check_quality(self, image, kind: str) -> Tuple[bool, Optional[str]]:
        if not self.quality_filter:
            return True, None
        if image is None or image.size == 0:
            return False, "invalid_image"
        h, w = image.shape[:2]
        area = int(h * w)
        min_area = self.min_vehicle_area if kind == "vehicle" else self.min_plate_area
        if min_area and area < min_area:
            return False, "too_small"
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        brightness = float(gray.mean())
        if self.min_brightness and brightness < self.min_brightness:
            return False, "too_dark"
        if self.max_brightness and brightness > self.max_brightness:
            return False, "too_bright"
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if self.min_sharpness and sharpness < self.min_sharpness:
            return False, "blurry"
        return True, None

    def _bump_stat(self, kind: str, field: str, reason: Optional[str] = None) -> None:
        with self._stats_lock:
            bucket = self._stats.setdefault(kind, {"attempted": 0, "saved": 0, "rejected": 0, "reasons": {}})
            bucket[field] = int(bucket.get(field, 0)) + 1
            if reason:
                reasons = bucket.setdefault("reasons", {})
                reasons[reason] = int(reasons.get(reason, 0)) + 1
            self._stats["updated_at"] = datetime.utcnow().isoformat() + "Z"
            
            now = time.time()
            if now - self._last_save_time > 10.0:
                self._save_stats()
                self._last_save_time = now

    def _load_stats(self) -> dict:
        try:
            if os.path.exists(self.stats_path):
                with open(self.stats_path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            pass
        return {"vehicle": {"attempted": 0, "saved": 0, "rejected": 0, "reasons": {}},
                "plate": {"attempted": 0, "saved": 0, "rejected": 0, "reasons": {}}}

    def _save_stats(self) -> None:
        try:
            os.makedirs(self.base_dir, exist_ok=True)
            with open(self.stats_path, "w", encoding="utf-8") as f:
                json.dump(self._stats, f, indent=2)
        except Exception:
            pass

    def save_vehicle(self, image, frame_id: int, conf: Optional[float] = None) -> None:
        if not self.collect_vehicles or not self._should_collect(conf):
            return
        if image is None or image.size == 0:
            self._bump_stat("vehicle", "rejected", "invalid_image")
            return
        self._bump_stat("vehicle", "attempted")
        ok, reason = self._check_quality(image, "vehicle")
        if not ok:
            self._bump_stat("vehicle", "rejected", reason)
            return
        ts = int(time.time() * 1000)
        conf_tag = f"{conf:.2f}" if conf is not None else "na"
        filename = f"veh_{frame_id}_{ts}_{conf_tag}.jpg"
        path = os.path.join(self.vehicle_dir, filename)
        cv2.imwrite(path, image)
        self._bump_stat("vehicle", "saved")

    def save_plate(
        self,
        image,
        frame_id: int,
        conf: Optional[float] = None,
        plate_text: Optional[str] = None,
    ) -> None:
        if not self.collect_plates or not self._should_collect(conf):
            return
        if image is None or image.size == 0:
            self._bump_stat("plate", "rejected", "invalid_image")
            return
        self._bump_stat("plate", "attempted")
        ok, reason = self._check_quality(image, "plate")
        if not ok:
            self._bump_stat("plate", "rejected", reason)
            return
        ts = int(time.time() * 1000)
        conf_tag = f"{conf:.2f}" if conf is not None else "na"
        text_tag = _safe_name(plate_text or "")
        suffix = f"_{text_tag}" if text_tag else ""
        filename = f"plate_{frame_id}_{ts}_{conf_tag}{suffix}.jpg"
        path = os.path.join(self.plate_dir, filename)
        cv2.imwrite(path, image)
        self._bump_stat("plate", "saved")
