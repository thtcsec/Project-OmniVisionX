import logging
import uuid
from typing import Optional

import redis.asyncio as aioredis

from app.services.plate.vehicle_types import normalize_vehicle_type

logger = logging.getLogger("omni-vehicle.stream_event_publisher")


def normalize_bbox_list(bbox: tuple | list | None) -> list[int] | None:
    if not bbox or len(bbox) != 4:
        return None
    return [int(v) for v in bbox]


def bbox_to_csv(bbox: tuple | list | None) -> str:
    normalized = normalize_bbox_list(bbox)
    if not normalized:
        return "0,0,0,0"
    return ",".join(str(v) for v in normalized)


class LprEventPublisher:
    def __init__(self, settings):
        self.settings = settings

    @staticmethod
    def _encode(value) -> bytes:
        return str(value).encode(errors="ignore")

    async def publish_plate_event(
        self,
        redis_client: Optional[aioredis.Redis],
        redis_connected: bool,
        camera_id: str,
        track_id: int,
        class_name: str,
        bbox: tuple,
        plate_text: str,
        plate_confidence: float,
        timestamp: float,
        plate_crop_path: str = "",
        full_frame_path: str = "",
        cache_entry: dict = None,
        plate_bbox: tuple | list | None = None,
        vehicle_color: str | None = None,
    ) -> None:
        normalized_vehicle_type = normalize_vehicle_type(class_name, plate_text)
        vehicle_bbox = normalize_bbox_list(bbox)
        normalized_plate_bbox = normalize_bbox_list(plate_bbox)
        metadata = {
            "vehicle_bbox": vehicle_bbox,
            "plate_bbox": normalized_plate_bbox,
            "vehicle_color": vehicle_color,
            "vehicle_type": normalized_vehicle_type,
        }
        metadata = {key: value for key, value in metadata.items() if value not in (None, "", [])}
        bbox_csv = bbox_to_csv(bbox)

        try:
            from app.services.pipeline.repositories.event_repository import PlateEventRepository

            repo = PlateEventRepository.get_instance()
            is_update = False
            evt_id = uuid.uuid4()
            plate_id = uuid.uuid4()
            if cache_entry and cache_entry.get("event_id") and cache_entry["event_id"] not in ["PENDING", "DEFERRED"]:
                evt_id = cache_entry["event_id"]
                plate_id = cache_entry.get("plate_id", plate_id)
                is_update = True

            await repo.save_plate_event(
                event_id=evt_id,
                plate_id=plate_id,
                camera_id=camera_id,
                plate_number=plate_text,
                vehicle_type=normalized_vehicle_type,
                confidence=plate_confidence,
                thumbnail_filename=plate_crop_path,
                full_frame_filename=full_frame_path,
                bbox=vehicle_bbox,
                tracking_id=str(track_id),
                is_update=is_update,
                metadata=metadata or None,
                color=vehicle_color,
            )

            if cache_entry and not is_update:
                cache_entry["event_id"] = evt_id
                cache_entry["plate_id"] = plate_id
        except Exception as e:
            logger.warning("DB persist failed (stream path): %s", e)

        if not redis_connected or redis_client is None:
            return
        try:
            entry = {
                b"camera_id": self._encode(camera_id),
                b"global_track_id": self._encode(track_id),
                b"class_name": self._encode(normalized_vehicle_type),
                b"bbox": self._encode(bbox_csv),
                b"confidence": self._encode(f"{plate_confidence:.3f}"),
                b"timestamp": self._encode(f"{timestamp:.3f}"),
                b"plate_text": self._encode(plate_text),
                b"plate_confidence": self._encode(f"{plate_confidence:.3f}"),
                b"plate_crop_path": self._encode(plate_crop_path),
                b"full_frame_path": self._encode(full_frame_path),
            }
            if normalized_plate_bbox:
                entry[b"plate_bbox"] = self._encode(",".join(str(v) for v in normalized_plate_bbox))
            if vehicle_color:
                entry[b"vehicle_color"] = self._encode(vehicle_color)
            maxlen = int(getattr(self.settings, "lpr_event_stream_maxlen", 1000))
            approximate = bool(getattr(self.settings, "lpr_event_stream_approximate_trim", True))
            pfx = getattr(self.settings, "stream_prefix", "omni")
            await redis_client.xadd(f"{pfx}:vehicles", entry, maxlen=maxlen, approximate=approximate)
        except Exception as e:
            logger.warning("Failed to publish lpr_event: %s", e)
