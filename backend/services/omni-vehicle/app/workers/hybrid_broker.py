"""
Hybrid broker for external detections (SDK → consensus → persistence).
Migrated from root ai-engine/app/workers/hybrid_broker.py to omni-vehicle.
"""
import logging
import os
import uuid
from datetime import datetime
from typing import Optional, Tuple

from PIL import Image

logger = logging.getLogger(__name__)

class HybridBroker:
    def __init__(self, settings, consensus, event_repository):
        self.settings = settings
        self.consensus = consensus
        self.event_repository = event_repository

    async def inject_external_detection(
        self,
        camera_id: str,
        plate_text: str,
        confidence: float,
        plate_image: Optional[Image.Image] = None,
        global_image: Optional[Image.Image] = None,
        track_id: int = -1,
        source: str = "dahua_sdk",
        plate_color: str = "",
        vehicle_type: str = "car",
        timestamp: Optional[datetime] = None,
        needs_verification: bool = False
    ) -> Tuple[bool, Optional[str]]:
        now = datetime.utcnow()
        ts = timestamp or now

        clean_plate = plate_text.replace("-", "").replace(".", "").replace(" ", "").upper()
        if len(clean_plate) < self.settings.ocr_min_text_length:
            logger.info("🌉 [%s] Reject short plate: %s (len=%d, min=%d)", source, plate_text, len(clean_plate), self.settings.ocr_min_text_length)
            return False, None

        is_sensitive = self._is_sensitive_plate(plate_color, confidence)
        high_conf = self.settings.event_instant_confidence
        should_verify = needs_verification or is_sensitive or confidence < high_conf

        final_plate = plate_text
        final_conf = confidence

        if should_verify and plate_image is not None:
            verified_plate, verified_conf = await self._verify_with_fortress(plate_image)

            if verified_plate:
                if verified_conf > confidence or verified_plate != plate_text:
                    logger.info("🔧 [%s] Fortress correction: %s → %s (conf: %.2f → %.2f)", source, plate_text, verified_plate, confidence, verified_conf)
                    final_plate = verified_plate
                    final_conf = verified_conf
                else:
                    logger.info("✅ [%s] Fortress confirmed: %s", source, plate_text)
            else:
                logger.warning("⚠️ [%s] Fortress verification failed, using SDK result", source)
        else:
            logger.info("⚡ [%s] Fast-track: %s (conf=%.2f)", source, plate_text, confidence)

        timestamp_str = ts.strftime("%Y%m%d_%H%M%S")
        safe_plate = "".join([c for c in final_plate if c.isalnum()])

        plate_filename = f"{timestamp_str}_{camera_id[:8]}_{safe_plate}_plate.jpg" if plate_image else None
        full_frame_filename = f"{timestamp_str}_{camera_id[:8]}_{safe_plate}_full.jpg" if global_image else None

        effective_track_id = track_id if track_id != -1 else abs(hash(f"{camera_id}_{final_plate}_{ts.timestamp():.0f}"))

        should_save, cache_entry = await self.consensus.should_process_plate(
            camera_id=camera_id,
            plate_text=final_plate,
            confidence=final_conf,
            thumbnail_path=plate_filename or "",
            full_frame_path=full_frame_filename or "",
            bbox=None,  # SDK detections don't have frame-space coordinates
            track_id=effective_track_id
        )

        if not should_save:
            logger.info("⏭️ [%s] Skip (Duplicate/Frozen): %s", source, final_plate)
            return False, final_plate

        os.makedirs(self.settings.thumbnail_path, exist_ok=True)

        try:
            if plate_image and plate_filename:
                plate_path = os.path.join(self.settings.thumbnail_path, plate_filename)
                plate_image.save(plate_path, "JPEG", quality=95)

            if global_image and full_frame_filename:
                full_path = os.path.join(self.settings.thumbnail_path, full_frame_filename)
                global_image.save(full_path, "JPEG", quality=90)

            is_update = False
            if cache_entry and cache_entry.get("event_id") and cache_entry.get("event_id") not in ["PENDING", "DEFERRED"]:
                evt_id = cache_entry["event_id"]
                plate_id = cache_entry["plate_id"]
                is_update = True
            else:
                evt_id = uuid.uuid4()
                plate_id = uuid.uuid4()

            await self.event_repository.save_plate_event(
                event_id=evt_id,
                plate_id=plate_id,
                camera_id=camera_id,
                plate_number=final_plate,
                vehicle_type=vehicle_type,
                confidence=final_conf,
                thumbnail_filename=plate_filename,
                full_frame_filename=full_frame_filename,
                bbox=None,
                tracking_id=str(effective_track_id),
                is_update=is_update
            )

            # Only stamp cache AFTER DB commit succeeds to prevent
            # stale cache entries when the DB write fails.
            if cache_entry and not is_update:
                cache_entry["event_id"] = evt_id
                cache_entry["plate_id"] = plate_id

        except Exception:
            logger.exception("💥 [%s] Failed to persist plate event %s — "
                             "cache entry left uncommitted so next detection retries",
                             source, final_plate)
            return False, final_plate

        logger.info("💾 [%s] Saved: %s (conf=%.2f, update=%s)", source, final_plate, final_conf, is_update)
        return True, final_plate

    def _is_sensitive_plate(self, plate_color: str, confidence: float) -> bool:
        high_conf = self.settings.event_instant_confidence
        sensitive_colors = ["red", "blue", "green", "đỏ", "xanh"]

        if plate_color.lower() in sensitive_colors:
            return True

        if 0.70 <= confidence < high_conf:
            return True

        return False

    async def _verify_with_fortress(self, plate_image: Image.Image) -> Tuple[Optional[str], float]:
        try:
            from app.services.pipeline.orchestration.fortress_lpr import get_fortress_lpr, PlateType, LONG_PLATE_AR
            import numpy as np
            import cv2
            import asyncio

            img_array = np.array(plate_image)
            if img_array.size == 0:
                return None, 0.0

            if len(img_array.shape) == 2:
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
            else:
                if img_array.shape[2] == 4:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

            h, w = img_bgr.shape[:2]
            lpr = get_fortress_lpr()

            if max(h, w) >= 720:
                # Wrap sync GPU inference in thread to avoid blocking event loop
                result = await asyncio.to_thread(lpr.process_frame, img_bgr)
                if not result.plates:
                    return None, 0.0
                best = max(result.plates, key=lambda p: p.confidence)
                if best.plate_text and len(best.plate_text) >= self.settings.ocr_min_text_length:
                    return best.plate_text, float(best.confidence)
                return None, 0.0

            aspect = w / h if h > 0 else 1.0
            plate_type = PlateType.ONE_LINE if aspect >= LONG_PLATE_AR else PlateType.TWO_LINE
            plate_text, conf_score = await asyncio.to_thread(lpr.recognizer.recognize, img_bgr, plate_type)
            if plate_text and len(plate_text) >= self.settings.ocr_min_text_length:
                return plate_text, float(conf_score)
            return None, 0.0
        except Exception:
            logger.exception("Fortress verification error")
            return None, 0.0
