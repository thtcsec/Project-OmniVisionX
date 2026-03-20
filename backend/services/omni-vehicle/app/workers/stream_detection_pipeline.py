import logging
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

import numpy as np

from app.workers.stream_event_utils import (
    apply_track_keep_budget,
    compute_effective_drop_ratio,
    event_age_exceeded,
    parse_bbox,
    parse_vehicle_detection_event,
)
from app.workers.stream_metrics_utils import update_camera_processing_stats
from app.workers.stream_pipeline_utils import (
    compute_vehicle_crop_bbox,
    should_drop_oversized_bbox,
    should_try_legacy_merge,
)
from app.workers.stream_vehicle_vision import VehicleBboxSmoother

logger = logging.getLogger("omni-vehicle.stream_detection_pipeline")


@dataclass
class DetectionPorts:
    get_camera_stat_entry: Callable[[str], dict[str, float]]
    update_camera_drop_ratio: Callable[[str], float]
    should_drop_event: Callable[..., bool]
    record_drop: Callable[[str, Optional[dict[str, float]]], None]
    get_frame_bgr: Callable[..., Awaitable[Optional[np.ndarray]]]
    detect_plates_on_vehicle: Callable[..., Awaitable[list[dict]]]
    legacy_detect_plates: Callable[..., Awaitable[list[dict]]]
    merge_plate_candidates: Callable[[list[dict], list[dict]], list[dict]]
    handle_plate_candidate: Callable[..., Awaitable[None]]
    get_backlog_len: Callable[[], int]


class LprDetectionPipeline:
    def __init__(
        self,
        settings,
        ambient,
        process_frame_fn,
        stats: dict,
        track_keep_budget: dict[str, tuple[float, int]],
        bbox_smoother: VehicleBboxSmoother,
        frame_blur_min_var: float,
        ports: DetectionPorts,
    ):
        self.settings = settings
        self._ambient = ambient
        self._process_frame = process_frame_fn
        self._stats = stats
        self._track_keep_budget = track_keep_budget
        self._bbox_smoother = bbox_smoother
        self._frame_blur_min_var = frame_blur_min_var
        self._ports = ports

    async def process_detection(self, msg_id, data: dict, global_drop_ratio: float) -> None:
        started_at = time.monotonic()

        if isinstance(msg_id, bytes):
            msg_id = msg_id.decode()

        event = parse_vehicle_detection_event(data)
        if event is None:
            return

        class_name = event.class_name
        camera_id = event.camera_id
        track_id = event.track_id
        bbox_str = event.bbox_str
        confidence = event.confidence
        timestamp = event.timestamp
        frame_stream_id = event.frame_stream_id

        cam_stat = self._ports.get_camera_stat_entry(camera_id)
        cam_stat["received"] = float(cam_stat.get("received", 0.0)) + 1.0
        cam_stat["last_update"] = time.monotonic()

        camera_drop_ratio = self._ports.update_camera_drop_ratio(camera_id)
        effective_drop_ratio = compute_effective_drop_ratio(global_drop_ratio, camera_drop_ratio, class_name)
        effective_drop_ratio = apply_track_keep_budget(
            effective_drop_ratio,
            self._track_keep_budget,
            camera_id,
            track_id,
        )

        if effective_drop_ratio > 0 and self._ports.should_drop_event(
            camera_id,
            msg_id,
            effective_drop_ratio,
            track_id=track_id,
            timestamp=timestamp,
        ):
            self._ports.record_drop("backpressure", cam_stat)
            logger.debug(
                "📉 DROP[backpressure] cam=%s track=%d ratio=%.2f (backlog=%d, cam_ratio=%.2f)",
                camera_id[:8], track_id, effective_drop_ratio,
                self._ports.get_backlog_len(), camera_drop_ratio,
            )
            return

        too_old, age, max_event_age = event_age_exceeded(self.settings, timestamp)
        if too_old:
            self._ports.record_drop("age", cam_stat)
            logger.debug(
                "📉 DROP[age] cam=%s track=%d age=%.1fs > max %.1fs",
                camera_id[:8], track_id, age, max_event_age,
            )
            return

        img_bgr = await self._ports.get_frame_bgr(
            camera_id,
            detection_ts=timestamp,
            frame_stream_id=frame_stream_id,
        )
        if img_bgr is None:
            self._ports.record_drop("no_frame", cam_stat)
            logger.debug(
                "📉 DROP[no_frame] cam=%s track=%d ts=%.3f",
                camera_id[:8], track_id, timestamp,
            )
            return

        parsed_bbox = parse_bbox(bbox_str)
        if parsed_bbox is None:
            self._ports.record_drop("invalid_bbox", cam_stat)
            return

        x1, y1, x2, y2 = self._bbox_smoother.smooth(camera_id, track_id, parsed_bbox)

        oversized_ratio = float(getattr(self.settings, "lpr_vehicle_bbox_max_ratio", 0.97))
        oversized_low_conf = float(getattr(self.settings, "lpr_vehicle_bbox_oversized_low_conf", 0.45))
        should_drop_oversized, area_ratio = should_drop_oversized_bbox(
            img_bgr.shape,
            (x1, y1, x2, y2),
            confidence,
            oversized_ratio,
            oversized_low_conf,
        )
        if should_drop_oversized:
            self._ports.record_drop("oversized_bbox", cam_stat)
            logger.debug(
                "DROP[oversized_bbox] cam=%s track=%d area_ratio=%.2f conf=%.2f",
                camera_id[:8], track_id, area_ratio, confidence,
            )
            return

        frame_sharpness = self._bbox_smoother.compute_sharpness(img_bgr, (x1, y1, x2, y2))
        if frame_sharpness < self._frame_blur_min_var:
            self._ports.record_drop("blur_frame", cam_stat)
            logger.debug(
                "DROP[blur_frame] cam=%s track=%d sharpness=%.1f<th=%.1f",
                camera_id[:8], track_id, frame_sharpness, self._frame_blur_min_var,
            )
            return

        self._ambient.update_brightness(camera_id, img_bgr, plate_bbox=(x1, y1, x2, y2))

        crop_x1, crop_y1, crop_x2, crop_y2 = compute_vehicle_crop_bbox(
            img_bgr.shape,
            (x1, y1, x2, y2),
            class_name,
        )
        vehicle_crop = img_bgr[crop_y1:crop_y2, crop_x1:crop_x2]

        if vehicle_crop.size == 0:
            self._ports.record_drop("empty_vehicle_crop", cam_stat)
            return

        try:
            plates = await self._ports.detect_plates_on_vehicle(
                vehicle_crop=vehicle_crop,
                vehicle_bbox=(crop_x1, crop_y1, crop_x2, crop_y2),
                camera_id=camera_id,
            )

            strict_conf = float(self.settings.ocr_confidence_threshold)
            strict_len = int(self.settings.ocr_min_text_length)

            if plates and self._process_frame is not None and should_try_legacy_merge(plates, strict_conf):
                legacy_plates = await self._ports.legacy_detect_plates(
                    img_bgr=img_bgr,
                    camera_id=camera_id,
                    vehicle_crop_bgr=vehicle_crop,
                    crop_bbox=(crop_x1, crop_y1, crop_x2, crop_y2),
                )
                plates = self._ports.merge_plate_candidates(plates, legacy_plates)

            if not plates and self._process_frame is not None:
                plates = await self._ports.legacy_detect_plates(
                    img_bgr=img_bgr,
                    camera_id=camera_id,
                    vehicle_crop_bgr=vehicle_crop,
                    crop_bbox=(crop_x1, crop_y1, crop_x2, crop_y2),
                )

            for plate in plates:
                self._stats["plates_detected"] += 1
                await self._ports.handle_plate_candidate(
                    plate=plate,
                    class_name=class_name,
                    strict_conf=strict_conf,
                    strict_len=strict_len,
                    camera_id=camera_id,
                    track_id=track_id,
                    timestamp=timestamp,
                    vehicle_crop=vehicle_crop,
                    crop_bbox=(crop_x1, crop_y1, crop_x2, crop_y2),
                    vehicle_bbox=(x1, y1, x2, y2),
                    img_bgr=img_bgr,
                )

            processed_ms = (time.monotonic() - started_at) * 1000.0
            e2e_lag_ms = max(0.0, (time.time() - timestamp) * 1000.0) if timestamp > 0 else 0.0
            self._stats["processed_frames"] += 1
            self._stats["processing_time_ms_total"] += processed_ms
            self._stats["e2e_lag_ms_total"] += e2e_lag_ms
            update_camera_processing_stats(cam_stat, processed_ms, e2e_lag_ms)
        except Exception as exc:
            logger.warning("Plate detection failed: %s", exc)
