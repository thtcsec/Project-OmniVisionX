"""
OmniVision omni-human — configuration

SINGLE SOURCE OF TRUTH for face/human detection, InsightFace recognition,
stream consumer tuning, and DB search parameters.

Every tunable field carries:
  - description  : explains what it does (shown in UI tooltip)
  - ge / le      : validation range  (Pydantic will reject out-of-range)
  - json_schema_extra.tier : "live" | "restart" | "infra"
        live    → hot-reloadable without restart
        restart → needs worker restart (not container)
        infra   → needs container restart
"""
import json
import os
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings
from typing import Any

SETTINGS_FILE = "/app/config/settings.json"

_default_data_root = "/data" if os.path.exists("/data") else str(Path.home() / ".omnivision" / "data")

# ── Tier constants (for UI grouping) ──────────────────────────
_LIVE = {"tier": "live"}
_RESTART = {"tier": "restart"}
_INFRA = {"tier": "infra"}


class Settings(BaseSettings):
    # ── Infrastructure (tier: infra) ──────────────────────────
    data_root: str = Field(
        default=_default_data_root,
        description="Thư mục gốc chứa weights, thumbnails, dataset",
        json_schema_extra=_INFRA,
    )
    thumbnail_path: str = Field(
        default="/app/thumbnails",
        description="Thư mục lưu ảnh crop khuôn mặt đã nhận diện",
        json_schema_extra=_INFRA,
    )
    redis_streams_url: str = Field(
        default="redis://omni-bus:6379",
        description="URL Redis Streams dùng làm event bus",
        json_schema_extra=_INFRA,
    )
    stream_prefix: str = Field(
        default="omni",
        description="Redis stream prefix (shared with omni-object / omni-vehicle)",
        json_schema_extra=_INFRA,
    )
    redis_consumer_group: str = Field(
        default="omni-human-group",
        description="Consumer group on detections stream",
        json_schema_extra=_INFRA,
    )

    # ── Face Detection (tier: live) ───────────────────────────
    face_confidence_threshold: float = Field(
        default=0.5, ge=0.1, le=0.95,
        description="Ngưỡng tin cậy YOLOv8-face — khuôn mặt dưới giá trị này bị bỏ qua. "
                    "Giảm xuống 0.3 nếu camera xa/chất lượng thấp.",
        json_schema_extra=_LIVE,
    )
    human_confidence_threshold: float = Field(
        default=0.25, ge=0.1, le=0.95,
        description="Ngưỡng tin cậy YOLO phát hiện người — person detection từ omni-object "
                    "dưới giá trị này sẽ bị bỏ qua. omni-object publish ở ≥0.3 nên giá trị "
                    "này cần ≤0.3 để tránh dead zone. Giảm nếu bỏ sót người ở xa.",
        json_schema_extra=_LIVE,
    )

    # ── InsightFace Recognition (tier: live) ──────────────────
    insightface_det_score: float = Field(
        default=0.3, ge=0.05, le=0.9,
        description="Ngưỡng điểm phát hiện InsightFace — khuôn mặt dưới giá trị này "
                    "bị bỏ qua khi trích embedding. Giảm nếu bị miss mặt nghiêng.",
        json_schema_extra=_LIVE,
    )
    face_db_match_distance: float = Field(
        default=0.45, ge=0.2, le=0.8,
        description="Ngưỡng khoảng cách cosine pgvector để xác định là cùng một người. "
                    "Giảm = chặt hơn (ít false positive), tăng = dễ match hơn (ít miss).",
        json_schema_extra=_LIVE,
    )

    # ── Stream Consumer Tuning (tier: live) ───────────────────
    track_dedup_cooldown: float = Field(
        default=3.0, ge=0.5, le=30.0,
        description="Thời gian (giây) cùng track_id phải chờ trước khi xử lý lại. "
                    "Tăng = ít tải GPU hơn, giảm = cập nhật khuôn mặt nhanh hơn.",
        json_schema_extra=_LIVE,
    )
    stale_event_threshold: float = Field(
        default=10.0, ge=2.0, le=60.0,
        description="Bỏ qua detection event cũ hơn N giây (chống xử lý event backlog).",
        json_schema_extra=_LIVE,
    )
    min_person_crop_px: int = Field(
        default=40, ge=10, le=200,
        description="Kích thước tối thiểu (pixel) person crop để chạy face detection. "
                    "Giảm nếu camera đặt xa, tăng nếu muốn bỏ người quá nhỏ.",
        json_schema_extra=_LIVE,
    )
    min_face_crop_px: int = Field(
        default=20, ge=5, le=100,
        description="Kích thước tối thiểu (pixel) face crop để lưu thumbnail.",
        json_schema_extra=_LIVE,
    )
    person_publish_conf: float = Field(
        default=0.30, ge=0.1, le=0.95,
        description="Ngưỡng confidence tối thiểu để publish person event khi không "
                    "phát hiện được khuôn mặt (event chỉ chứa person bbox). "
                    "Nên ≤ human_confidence_threshold để không tạo dead zone drop event.",
        json_schema_extra=_LIVE,
    )

    class Config:
        env_file = ".env"
        extra = "ignore"

    def get_field_descriptions(self) -> dict[str, dict[str, Any]]:
        """Return field metadata for UI rendering: {field_name: {description, tier, type, min, max, value}}"""
        result: dict[str, dict[str, Any]] = {}
        for name, field_info in self.model_fields.items():
            meta: dict[str, Any] = {
                "description": field_info.description or "",
                "tier": (field_info.json_schema_extra or {}).get("tier", "infra"),
                "type": field_info.annotation.__name__ if hasattr(field_info.annotation, '__name__') else str(field_info.annotation),
                "value": getattr(self, name),
                "default": field_info.default,
            }
            # Extract validation constraints
            for m in field_info.metadata:
                if hasattr(m, 'ge'):
                    meta["min"] = m.ge
                if hasattr(m, 'le'):
                    meta["max"] = m.le
            result[name] = meta
        return result

    def save_to_file(self):
        """Save mutable settings to JSON file"""
        data = {
            "face_confidence_threshold": self.face_confidence_threshold,
            "human_confidence_threshold": self.human_confidence_threshold,
            "insightface_det_score": self.insightface_det_score,
            "face_db_match_distance": self.face_db_match_distance,
            "track_dedup_cooldown": self.track_dedup_cooldown,
            "stale_event_threshold": self.stale_event_threshold,
            "min_person_crop_px": self.min_person_crop_px,
            "min_face_crop_px": self.min_face_crop_px,
            "person_publish_conf": self.person_publish_conf,
        }
        os.makedirs(os.path.dirname(SETTINGS_FILE), exist_ok=True)
        with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    def load_from_file(self):
        """Load settings from JSON file if exists"""
        import logging
        _logger = logging.getLogger("omni-human.config")
        if os.path.exists(SETTINGS_FILE):
            try:
                with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                for key, value in data.items():
                    if hasattr(self, key):
                        setattr(self, key, value)
                _logger.info("Loaded settings from %s (%d keys)", SETTINGS_FILE, len(data))
            except json.JSONDecodeError as e:
                _logger.warning("Corrupt settings file %s: %s — using defaults", SETTINGS_FILE, e)
            except Exception as e:
                _logger.warning("Failed to load settings from %s: %s — using defaults", SETTINGS_FILE, e)


_settings_instance = None


def _resolve_thumbnail_path(path: str, data_root: str) -> str:
    candidates: list[str] = []
    if path:
        candidates.append(path)
    if path == "/app/thumbnails":
        if data_root:
            candidates.insert(0, str(Path(data_root) / "thumbnails"))
        candidates.append("/app/thumbnails")
    else:
        if data_root:
            data_thumb = str(Path(data_root) / "thumbnails")
            if data_thumb not in candidates:
                candidates.append(data_thumb)
    for candidate in candidates:
        try:
            os.makedirs(candidate, exist_ok=True)
            return candidate
        except Exception:
            continue
    return candidates[0] if candidates else "/app/thumbnails"

def get_settings():
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings()
        _settings_instance.load_from_file()
        _settings_instance.thumbnail_path = _resolve_thumbnail_path(
            _settings_instance.thumbnail_path,
            _settings_instance.data_root
        )
    return _settings_instance
