import os
import time
from dataclasses import dataclass
from typing import Optional

from app.services.plate.vehicle_types import is_probable_vehicle_label, normalize_vehicle_type

VEHICLE_CLASSES = {"car", "truck", "motorcycle", "bus"}


@dataclass(slots=True)
class ParsedDetectionEvent:
    class_name: str
    camera_id: str
    track_id: int
    bbox_str: str
    confidence: float
    timestamp: float
    frame_stream_id: Optional[str]


def _decode(raw: dict, key: bytes, default: str = "") -> str:
    value = raw.get(key, default.encode() if isinstance(default, str) else default)
    if isinstance(value, bytes):
        return value.decode(errors="ignore")
    return str(value)


def parse_vehicle_detection_event(data: dict) -> Optional[ParsedDetectionEvent]:
    raw_class_name = _decode(data, b"class_name", "")
    class_name = normalize_vehicle_type(raw_class_name)
    if class_name not in VEHICLE_CLASSES:
        if class_name == "unknown" and is_probable_vehicle_label(raw_class_name):
            class_name = "car"
        else:
            return None

    camera_id = _decode(data, b"camera_id", "")
    bbox_str = _decode(data, b"bbox", "0,0,0,0")
    frame_stream_id = _decode(data, b"frame_stream_id", "") or None
    try:
        track_id = int(float(_decode(data, b"global_track_id", "-1")))
    except ValueError:
        track_id = -1
    try:
        confidence = float(_decode(data, b"confidence", "0"))
    except ValueError:
        confidence = 0.0
    try:
        timestamp = float(_decode(data, b"timestamp", "0"))
    except ValueError:
        timestamp = 0.0

    return ParsedDetectionEvent(
        class_name=class_name,
        camera_id=camera_id,
        track_id=track_id,
        bbox_str=bbox_str,
        confidence=confidence,
        timestamp=timestamp,
        frame_stream_id=frame_stream_id,
    )


def compute_effective_drop_ratio(global_drop_ratio: float, camera_drop_ratio: float, class_name: str) -> float:
    ratio = max(global_drop_ratio, camera_drop_ratio)
    if class_name == "motorcycle":
        ratio *= 0.5
    return ratio


def apply_track_keep_budget(
    effective_drop_ratio: float,
    track_keep_budget: dict[str, tuple[float, int]],
    camera_id: str,
    track_id: int,
) -> float:
    if track_id < 0:
        return effective_drop_ratio

    track_key = f"{camera_id}:{track_id}"
    keep_first = int(os.environ.get("LPR_STREAM_KEEP_FIRST_EVENTS_PER_TRACK", "2"))
    keep_first = max(0, min(5, keep_first))
    if keep_first <= 0:
        return effective_drop_ratio

    ttl_s = float(os.environ.get("LPR_STREAM_TRACK_KEEP_TTL", "30"))
    ttl_s = max(5.0, min(180.0, ttl_s))
    last_ts, budget = track_keep_budget.get(track_key, (0.0, keep_first))
    now_ts = time.monotonic()
    if now_ts - last_ts > ttl_s:
        budget = keep_first

    if budget > 0:
        track_keep_budget[track_key] = (now_ts, budget - 1)
        return 0.0

    track_keep_budget[track_key] = (now_ts, budget)
    return effective_drop_ratio


def event_age_exceeded(settings, timestamp: float) -> tuple[bool, float, float]:
    max_event_age = float(
        getattr(
            settings,
            "lpr_stream_max_event_age_s",
            os.environ.get("LPR_STREAM_MAX_EVENT_AGE", "20.0"),
        )
    )
    age = time.time() - timestamp
    return age > max_event_age, age, max_event_age


def parse_bbox(bbox_str: str) -> Optional[tuple[int, int, int, int]]:
    parts = bbox_str.split(",")
    if len(parts) != 4:
        return None
    try:
        x1, y1, x2, y2 = [int(float(p)) for p in parts]
        return x1, y1, x2, y2
    except ValueError:
        return None
