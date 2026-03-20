import asyncio
import logging
from typing import Any, Callable

import cv2

import numpy as np
from app.services.plate.plate_utils import normalize_vn_plate_confusions, is_valid_vn_plate_format

from app.workers.stream_pipeline_utils import (
    compute_plate_sharpness,
    should_accept_raw_plate_candidate,
)

try:
    from app.services.plate.vn_plate_validator import (
        get_adjusted_confidence,
        validate_and_correct_plate,
    )
    VN_PLATE_VALIDATOR_AVAILABLE = True
except Exception:
    VN_PLATE_VALIDATOR_AVAILABLE = False

logger = logging.getLogger("omni-vehicle.stream_plate_workflow")


class LprPlateWorkflow:
    def __init__(
        self,
        consensus,
        drop_reasons: dict[str, int],
        save_thumbnails_fn: Callable[..., tuple],
        publish_plate_event_fn: Callable[..., Any],
        refine_vehicle_type_fn: Callable[..., str],
        estimate_vehicle_color_fn: Callable[..., str],
    ):
        self._consensus = consensus
        self._drop_reasons = drop_reasons
        self._save_thumbnails = save_thumbnails_fn
        self._publish_plate_event = publish_plate_event_fn
        self._refine_vehicle_type = refine_vehicle_type_fn
        self._estimate_vehicle_color = estimate_vehicle_color_fn

    def _runtime_sr_params(self) -> tuple[bool, int, int, float]:
        settings = getattr(self._consensus, "settings", None)
        enabled = bool(getattr(settings, "lpr_sr_enable", True))
        min_h = max(16, int(getattr(settings, "lpr_sr_min_height", 40)))
        min_w = max(32, int(getattr(settings, "lpr_sr_min_width", 120)))
        conf_th = max(0.0, min(1.0, float(getattr(settings, "lpr_sr_conf_threshold", 0.82))))
        return enabled, min_h, min_w, conf_th

    async def handle_candidate(
        self,
        plate: dict,
        class_name: str,
        strict_conf: float,
        strict_len: int,
        camera_id: str,
        track_id: int,
        timestamp: float,
        vehicle_crop: np.ndarray,
        crop_bbox: tuple[int, int, int, int],
        vehicle_bbox: tuple[int, int, int, int],
        img_bgr: np.ndarray,
    ) -> None:
        plate_text = plate.get("plate_text", "")
        plate_confidence = plate.get("confidence", 0.0)
        plate_bbox = plate.get("bbox")
        plate_crop = plate.get("plate_crop")
        plate_sharpness = compute_plate_sharpness(plate_crop)
        accepted_raw, relaxed_conf = should_accept_raw_plate_candidate(
            plate_text=plate_text,
            plate_confidence=float(plate_confidence),
            class_name=class_name,
            strict_conf=strict_conf,
            strict_len=strict_len,
        )
        if not accepted_raw:
            return

        validation = None
        if VN_PLATE_VALIDATOR_AVAILABLE:
            try:
                raw_plate_text = plate_text
                validation = validate_and_correct_plate(plate_text, float(plate_confidence))
                adjusted_confidence = float(get_adjusted_confidence(float(plate_confidence), validation))
                if validation.corrected:
                    plate_text = validation.corrected
                plate_confidence = adjusted_confidence
                if (not validation.is_valid) and plate_confidence < max(0.28, relaxed_conf):
                    return
                # Log serious OCR edge cases for debugging (province invalid, too short, etc.)
                if validation.errors and plate_confidence >= 0.5:
                    logger.warning(
                        "OCR edge case cam=%s track=%d raw=%r corrected=%r conf=%.2f errors=%s",
                        camera_id[:8], track_id, raw_plate_text, plate_text, plate_confidence, validation.errors,
                    )
            except Exception as validator_error:
                logger.debug("VN plate validator failed, fallback raw OCR: %s", validator_error)

        rescue_enabled = bool(getattr(self._consensus.settings, "lpr_ocr_rescue_enabled", True))
        rescue_conf_threshold = float(getattr(self._consensus.settings, "lpr_ocr_rescue_conf_threshold", 0.78))
        should_rescue = (
            rescue_enabled
            and plate_crop is not None
            and hasattr(plate_crop, "size")
            and plate_crop.size != 0
            and (float(plate_confidence) <= rescue_conf_threshold or (validation is not None and not validation.is_valid))
        )
        if should_rescue:
            try:
                from PIL import Image as PILImage
                from app.services.pipeline.orchestration.vn_plate_reader import read_plate_hybrid

                rescue_crop = plate_crop
                crop_h, crop_w = int(plate_crop.shape[0]), int(plate_crop.shape[1])
                sr_enabled, sr_min_h, sr_min_w, sr_conf_threshold = self._runtime_sr_params()
                should_try_sr = (
                    sr_enabled
                    and float(plate_confidence) <= sr_conf_threshold
                    and (crop_h < sr_min_h or crop_w < sr_min_w)
                )

                if should_try_sr:
                    try:
                        from app.services.ocr.super_resolution import get_super_resolution_service

                        sr_service = get_super_resolution_service()
                        if sr_service.is_available():
                            plate_rgb_for_sr = (
                                cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB)
                                if len(plate_crop.shape) == 3
                                else cv2.cvtColor(plate_crop, cv2.COLOR_GRAY2RGB)
                            )
                            vehicle_width = max(1, int(vehicle_bbox[2] - vehicle_bbox[0])) if vehicle_bbox else None
                            sr_pil, was_upscaled, sr_reason = await sr_service.conditional_upscale(
                                PILImage.fromarray(plate_rgb_for_sr),
                                vehicle_width=vehicle_width,
                                force=False,
                            )
                            if was_upscaled:
                                rescue_crop = cv2.cvtColor(np.array(sr_pil), cv2.COLOR_RGB2BGR)
                                logger.debug(
                                    "OCR rescue SR applied cam=%s track=%s crop=%dx%d reason=%s",
                                    camera_id[:8],
                                    track_id,
                                    crop_w,
                                    crop_h,
                                    sr_reason,
                                )
                    except Exception as sr_error:
                        logger.debug("SR pre-rescue skipped: %s", sr_error)

                plate_rgb = cv2.cvtColor(rescue_crop, cv2.COLOR_BGR2RGB) if len(rescue_crop.shape) == 3 else rescue_crop
                rescue_text, rescue_conf, _rescue_crop, _ = await asyncio.to_thread(
                    read_plate_hybrid,
                    PILImage.fromarray(plate_rgb),
                )
                if rescue_text and float(rescue_conf) > float(plate_confidence) + 0.03:
                    logger.info("OCR rescue override: '%s'(%.2f) -> '%s'(%.2f)", plate_text, plate_confidence, rescue_text, rescue_conf)
                    plate_text = rescue_text
                    plate_confidence = float(rescue_conf)
            except Exception as rescue_error:
                logger.debug("OCR rescue failed: %s", rescue_error)

        plate_text = normalize_vn_plate_confusions(plate_text)
        if not plate_text:
            return

        if not is_valid_vn_plate_format(plate_text) and float(plate_confidence) < max(0.36, relaxed_conf + 0.05):
            return

        normalized_vehicle_type = self._refine_vehicle_type(
            class_name,
            plate_text,
            vehicle_bbox,
            plate_bbox,
        )
        vehicle_color = self._estimate_vehicle_color(
            vehicle_crop,
            crop_bbox=crop_bbox,
            plate_bbox=plate_bbox,
        )

        plate_crop_path, full_frame_path = await asyncio.to_thread(
            self._save_thumbnails,
            img_bgr,
            plate_crop,
            camera_id,
            plate_text,
            plate_bbox,
            vehicle_bbox,
            normalized_vehicle_type,
            vehicle_color,
        )

        should_process, cache_entry = await self._consensus.should_process_plate(
            camera_id=camera_id,
            plate_text=plate_text,
            confidence=plate_confidence,
            thumbnail_path=plate_crop_path,
            full_frame_path=full_frame_path,
            bbox=list(plate_bbox) if plate_bbox else None,
            track_id=track_id,
            sharpness=plate_sharpness,
        )
        if not should_process:
            self._drop_reasons["consensus_skip"] = int(self._drop_reasons.get("consensus_skip", 0)) + 1
            return

        if cache_entry:
            plate_text = cache_entry.get("plate", plate_text)
            plate_confidence = cache_entry.get("confidence", plate_confidence)
            plate_crop_path = cache_entry.get("thumb_path", plate_crop_path)
            full_frame_path = cache_entry.get("full_path", full_frame_path)

        await self._publish_plate_event(
            camera_id=camera_id,
            track_id=track_id,
            class_name=normalized_vehicle_type,
            bbox=vehicle_bbox,
            plate_text=plate_text,
            plate_confidence=plate_confidence,
            timestamp=timestamp,
            plate_crop_path=plate_crop_path,
            full_frame_path=full_frame_path,
            cache_entry=cache_entry,
            plate_bbox=plate_bbox,
            vehicle_color=vehicle_color,
        )
        await self._consensus.cleanup_plate_cache()
