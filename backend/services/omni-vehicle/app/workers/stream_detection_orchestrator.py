import asyncio
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger("omni-vehicle.stream_detection")

# Timeout for GPU inference — tránh treo khi model bị kẹt
INFERENCE_TIMEOUT_S = 5.0


class LprDetectionOrchestrator:
    def __init__(self, settings, process_frame_fn=None):
        self.settings = settings
        self.process_frame_fn = process_frame_fn
        max_workers = max(4, int(getattr(settings, "lpr_max_concurrent_tasks", 10)))
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="lpr-fortress")
        self._telemetry: dict[str, int] = {
            "fortress_attempts": 0,
            "fortress_success": 0,
            "fortress_empty": 0,
            "fortress_errors": 0,
            "legacy_attempts": 0,
            "legacy_success": 0,
            "legacy_rejected": 0,
            "legacy_errors": 0,
        }
        self._telemetry_lock = threading.Lock()

    def _bump(self, key: str, inc: int = 1) -> None:
        with self._telemetry_lock:
            self._telemetry[key] = int(self._telemetry.get(key, 0)) + int(inc)

    def get_telemetry_snapshot(self) -> dict[str, int | float]:
        with self._telemetry_lock:
            base = {k: int(v) for k, v in self._telemetry.items()}
        fa = base.get("fortress_attempts", 0)
        la = base.get("legacy_attempts", 0)
        if fa > 0:
            base["fortress_success_rate"] = round(base.get("fortress_success", 0) / fa, 3)
        if la > 0:
            base["legacy_fallback_rate"] = round(base.get("legacy_success", 0) / la, 3)
        return base

    @staticmethod
    def bbox_iou(box_a: tuple, box_b: tuple) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        denom = area_a + area_b - inter_area
        if denom <= 0:
            return 0.0
        return inter_area / denom

    def filter_overlapping_plates(self, plates: list[dict]) -> list[dict]:
        if not plates:
            return []
        threshold = float(getattr(self.settings, "nms_threshold", 0.45))
        sorted_plates = sorted(plates, key=lambda p: p.get("confidence", 0.0), reverse=True)
        kept: list[dict] = []
        for plate in sorted_plates:
            bbox = plate.get("bbox")
            if not bbox:
                continue
            should_keep = True
            for kept_plate in kept:
                kept_bbox = kept_plate.get("bbox")
                if kept_bbox and self.bbox_iou(bbox, kept_bbox) > threshold:
                    should_keep = False
                    break
            if should_keep:
                kept.append(plate)
        return kept

    def merge_plate_candidates(self, primary: list[dict], secondary: list[dict]) -> list[dict]:
        if not secondary:
            return primary
        if not primary:
            return secondary

        merged = list(primary)
        for candidate in secondary:
            bbox = candidate.get("bbox")
            if not bbox:
                merged.append(candidate)
                continue
            duplicate = False
            merge_iou = float(getattr(self.settings, "plate_merge_iou_threshold", 0.55))
            for existing in merged:
                existing_bbox = existing.get("bbox")
                if existing_bbox and self.bbox_iou(existing_bbox, bbox) > merge_iou:
                    duplicate = True
                    if float(candidate.get("confidence", 0.0)) > float(existing.get("confidence", 0.0)):
                        for k, v in candidate.items():
                            if v is not None:
                                existing[k] = v
                    break
            if not duplicate:
                merged.append(candidate)
        return self.filter_overlapping_plates(merged)

    async def detect_plates_on_vehicle(
        self,
        vehicle_crop: np.ndarray,
        vehicle_bbox: tuple,
        camera_id: str = "unknown",
    ) -> list[dict]:
        enabled_env = os.getenv("ENABLE_FORTRESS_LPR", "").strip().lower()
        if enabled_env in {"0", "false", "no"}:
            return []
        if not bool(getattr(self.settings, "enable_fortress_lpr", True)):
            return []
        self._bump("fortress_attempts")
        try:
            from app.services.core.ambient_adapter import AmbientAdapter
            from app.services.pipeline.orchestration.fortress_lpr import get_fortress_lpr

            lpr = get_fortress_lpr()

            def _detect_with_context(crop, bbox):
                AmbientAdapter.set_active_camera(camera_id)
                with lpr._gpu_semaphore:
                    return lpr.plate_detector.detect(crop, bbox)

            def _recognize_with_context(batch):
                AmbientAdapter.set_active_camera(camera_id)
                with lpr._gpu_semaphore:
                    return lpr.recognizer.recognize_batch(batch)

            loop = asyncio.get_running_loop()
            detected = await asyncio.wait_for(
                loop.run_in_executor(self._executor, _detect_with_context, vehicle_crop, vehicle_bbox),
                timeout=INFERENCE_TIMEOUT_S,
            )
            if not detected:
                self._bump("fortress_empty")
                h, w = vehicle_crop.shape[:2]
                logger.debug(
                    "Fortress: Không tìm thấy biển số — crop %dx%d bbox=%s (có thể ảnh mờ, không có biển, hoặc góc chụp)",
                    w, h, vehicle_bbox,
                )
                return []
            detected = self.filter_overlapping_plates(detected)

            batch_inputs = [(p.get("crop"), p.get("plate_type")) for p in detected]
            rec_results = await asyncio.wait_for(
                loop.run_in_executor(self._executor, _recognize_with_context, batch_inputs),
                timeout=INFERENCE_TIMEOUT_S,
            )

            if len(detected) != len(rec_results):
                logger.error(
                    "Mismatch detect vs recognize: detected=%d rec_results=%d — bỏ qua batch",
                    len(detected), len(rec_results),
                )
                self._bump("fortress_errors")
                return []

            plates: list[dict] = []
            for plate_info, (plate_text, conf_score) in zip(detected, rec_results):
                plate_text, conf_score = lpr._validate_plate(plate_text, conf_score, plate_info.get("plate_type"))
                if not plate_text:
                    continue
                plates.append({
                    "plate_text": plate_text,
                    "confidence": float(conf_score),
                    "bbox": plate_info.get("bbox"),
                    "plate_crop": plate_info.get("crop"),
                })
            if plates:
                self._bump("fortress_success")
            else:
                self._bump("fortress_empty")
            return plates
        except asyncio.TimeoutError:
            self._bump("fortress_errors")
            logger.warning(
                "Fortress inference timeout (%.1fs) — crop %s",
                INFERENCE_TIMEOUT_S,
                vehicle_crop.shape if vehicle_crop is not None else None,
            )
            return []
        except Exception as e:
            self._bump("fortress_errors")
            logger.warning(
                "Fortress vehicle-crop detect failed: %s: %s — crop shape=%s",
                type(e).__name__, e,
                vehicle_crop.shape if vehicle_crop is not None else None,
            )
            return []

    async def legacy_detect_plates(
        self,
        img_bgr: np.ndarray,
        camera_id: str,
        vehicle_crop_bgr: np.ndarray,
        crop_bbox: Optional[tuple] = None,
    ) -> list[dict]:
        self._bump("legacy_attempts")
        try:
            from PIL import Image as PILImage
            from app.services.pipeline.orchestration.vn_plate_reader import VNPlateReader

            reader = VNPlateReader.get_instance()
            if not reader.is_available():
                self._bump("legacy_rejected")
                return []

            rgb_crop = cv2.cvtColor(vehicle_crop_bgr, cv2.COLOR_BGR2RGB)
            pil_crop = PILImage.fromarray(rgb_crop)

            plate_text, conf_score, plate_crop_pil, plate_bbox = await reader.predict(pil_crop)
            strict_conf = float(getattr(self.settings, "ocr_confidence_threshold", 0.40))
            offset = float(getattr(self.settings, "lpr_legacy_accept_conf_offset", 0.25))
            floor = float(getattr(self.settings, "lpr_legacy_accept_conf_floor", 0.12))
            legacy_accept_conf = max(floor, strict_conf - offset)
            if not plate_text or conf_score < legacy_accept_conf:
                self._bump("legacy_rejected")
                logger.debug(
                    "Legacy: Bỏ qua — text=%r conf=%.2f < %.2f (ảnh mờ hoặc không có biển hợp lệ)",
                    plate_text, conf_score, legacy_accept_conf,
                )
                return []

            plate_crop_np = None
            if plate_crop_pil is not None:
                plate_crop_np = cv2.cvtColor(np.array(plate_crop_pil), cv2.COLOR_RGB2BGR)

            if plate_bbox and crop_bbox and len(plate_bbox) == 4 and len(crop_bbox) == 4:
                plate_bbox = [
                    int(plate_bbox[0]) + int(crop_bbox[0]),
                    int(plate_bbox[1]) + int(crop_bbox[1]),
                    int(plate_bbox[2]) + int(crop_bbox[0]),
                    int(plate_bbox[3]) + int(crop_bbox[1]),
                ]

            logger.info("🔑 Legacy fallback detected: '%s' conf=%.2f", plate_text, conf_score)
            self._bump("legacy_success")
            return [{
                "plate_text": plate_text,
                "confidence": float(conf_score),
                "bbox": plate_bbox,
                "plate_crop": plate_crop_np,
            }]
        except Exception as e:
            self._bump("legacy_errors")
            logger.warning(
                "Legacy plate detect fallback failed: %s: %s",
                type(e).__name__, e,
            )
            return []
