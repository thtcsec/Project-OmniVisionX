"""
Legacy core LPR pipeline ported from ai-engine/app/workers/plate_pipeline.py.
Vehicle-first pipeline with quality filters + consensus + image saving.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import threading

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app.config import get_settings
from app.services.pipeline.collectors.data_collection import DataCollector
from app.services.pipeline.repositories.event_repository import PlateEventRepository
from app.services.pipeline.orchestration.fortress_lpr import get_fortress_lpr
from app.services.core.enhancer import ImageEnhancer
from app.services.pipeline.application.lpr_utils import expand_bbox
from app.services.ocr.super_resolution import get_super_resolution_service
from app.services.pipeline.orchestration.vn_plate_reader import VNPlateReader

logger = logging.getLogger("omni-vehicle.legacy")


def _verbose_enabled() -> bool:
    return os.environ.get("LEGACY_LPR_VERBOSE", "0").strip().lower() in {"1", "true", "yes"}


def _vlog(message: str, *args) -> None:
    if _verbose_enabled():
        logger.info(message, *args)


def _safe_plate(text: str) -> str:
    if not text:
        return "UNKNOWN"
    return "".join(ch for ch in text if ch.isalnum()) or "UNKNOWN"


def _point_in_polygon(x: float, y: float, polygon: List[Tuple[int, int]]) -> bool:
    inside = False
    n = len(polygon)
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        intersect = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / (yj - yi + 1e-9) + xi
        )
        if intersect:
            inside = not inside
        j = i
    return inside


def _draw_detection_box(
    image: Image.Image,
    bbox: Tuple[int, int, int, int],
    label: Optional[str] = None,
    color: str = "red",
    line_width: int = 4,
) -> Image.Image:
    if image is None:
        return image
    draw = ImageDraw.Draw(image)
    x1, y1, x2, y2 = bbox
    for i in range(max(1, line_width)):
        draw.rectangle((x1 - i, y1 - i, x2 + i, y2 + i), outline=color)
    if label:
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        bbox_text = draw.textbbox((0, 0), label, font=font)
        text_w = bbox_text[2] - bbox_text[0]
        text_h = bbox_text[3] - bbox_text[1]
        draw.rectangle((x1, y1 - text_h - 6, x1 + text_w + 6, y1), fill=color)
        draw.text((x1 + 3, y1 - text_h - 3), label, fill="white", font=font)
    return image


def _estimate_vehicle_color(vehicle_crop: Image.Image) -> str:
    if vehicle_crop is None:
        return "unknown"
    try:
        np_img = np.array(vehicle_crop)
        if np_img.size == 0:
            return "unknown"
        hsv = cv2.cvtColor(np_img, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        mean_h = float(np.mean(h))
        mean_s = float(np.mean(s))
        mean_v = float(np.mean(v))
        if mean_v < 50:
            return "black"
        if mean_v > 200 and mean_s < 40:
            return "white"
        if mean_s < 40:
            return "gray"
        if 0 <= mean_h < 15 or mean_h >= 165:
            return "red"
        if 15 <= mean_h < 35:
            return "yellow"
        if 35 <= mean_h < 85:
            return "green"
        if 85 <= mean_h < 130:
            return "blue"
        if 130 <= mean_h < 165:
            return "purple"
        return "unknown"
    except Exception:
        return "unknown"


@dataclass
class _ConsensusEntry:
    last_emit_ts: float = 0.0
    sightings: List[float] = field(default_factory=list)
    event_id: Optional[uuid.UUID] = None
    plate_id: Optional[uuid.UUID] = None
    old_thumb: Optional[str] = None
    old_full: Optional[str] = None


class PlateConsensus:
    _MAX_ENTRIES = 5000  # Hard cap to prevent unbounded growth

    def __init__(self) -> None:
        self._entries: Dict[str, _ConsensusEntry] = {}
        self._last_eviction: float = 0.0

    def _evict_stale(self, ttl: float) -> None:
        """Remove entries older than ttl. Called periodically."""
        now = time.time()
        if now - self._last_eviction < ttl * 0.5:
            return  # Don't evict too frequently
        self._last_eviction = now
        stale_keys = [
            k for k, v in self._entries.items()
            if now - (v.last_emit_ts or (v.sightings[-1] if v.sightings else now)) > ttl * 2
        ]
        for k in stale_keys:
            del self._entries[k]
        # Hard cap: if still too large, evict oldest entries
        if len(self._entries) > self._MAX_ENTRIES:
            sorted_keys = sorted(
                self._entries.keys(),
                key=lambda k: self._entries[k].last_emit_ts or (self._entries[k].sightings[-1] if self._entries[k].sightings else 0.0),
            )
            for k in sorted_keys[:len(self._entries) - self._MAX_ENTRIES]:
                del self._entries[k]

    def _key(self, camera_id: str, plate_text: str, track_id: Optional[int]) -> str:
        tid = track_id if track_id is not None else "na"
        return f"{camera_id}:{tid}:{plate_text}"

    def should_process_plate(
        self,
        camera_id: str,
        plate_text: str,
        confidence: float,
        plate_filename: str,
        full_frame_filename: str,
        bbox: List[int],
        track_id: Optional[int],
    ) -> Tuple[bool, Optional[dict]]:
        settings = get_settings()
        now = time.time()
        ttl = float(settings.event_dedup_ttl)

        # Periodic eviction of stale entries to prevent memory leak
        self._evict_stale(ttl)

        # AmbientAdapter: interpolate event thresholds by ambient brightness
        try:
            from app.services.core.ambient_adapter import AmbientAdapter
            _amb = AmbientAdapter.get_instance()
            instant = _amb.get_threshold(camera_id, "event_instant_confidence",
                                         float(settings.event_instant_confidence))
            min_votes = _amb.get_int_threshold(camera_id, "event_min_vote_count",
                                               int(settings.event_min_vote_count))
        except Exception:
            instant = float(settings.event_instant_confidence)
            min_votes = int(settings.event_min_vote_count)

        key = self._key(camera_id, plate_text, track_id)
        entry = self._entries.get(key)
        if entry is None:
            entry = _ConsensusEntry(last_emit_ts=0.0, sightings=[])
            self._entries[key] = entry

        # Cleanup old sightings
        entry.sightings = [t for t in (entry.sightings or []) if now - t <= ttl]
        entry.sightings.append(now)

        # Dedup emit
        if entry.last_emit_ts and now - entry.last_emit_ts < ttl:
            return False, None

        # Consensus decision
        should_emit = confidence >= instant or len(entry.sightings) >= min_votes
        if not should_emit:
            return False, None

        entry.last_emit_ts = now
        cache_entry = {
            "event_id": entry.event_id,
            "plate_id": entry.plate_id,
            "old_thumb": entry.old_thumb,
            "old_full": entry.old_full,
        }
        # Update cache for potential overwrite
        entry.old_thumb = plate_filename
        entry.old_full = full_frame_filename
        return True, cache_entry


class LegacyPlatePipeline:
    _instance: "LegacyPlatePipeline | None" = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self.settings = get_settings()
        self.event_repository = PlateEventRepository.get_instance()
        self.ocr_service = VNPlateReader.get_instance()
        self.sr_service = get_super_resolution_service()
        self.consensus = PlateConsensus()

        self._min_component_count = 6
        self._min_quality_score = 0.45
        self._min_edge_density = 0.02
        self._adaptive_conf_base = float(self.settings.adaptive_confidence_base)
        self._adaptive_conf_alpha = float(self.settings.adaptive_confidence_alpha)
        self._adaptive_conf_min = float(self.settings.adaptive_confidence_min)
        self._adaptive_conf_max = float(self.settings.adaptive_confidence_max)

    @classmethod
    def get_instance(cls) -> "LegacyPlatePipeline":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = LegacyPlatePipeline()
        return cls._instance

    def _pil_from_bgr(self, img_bgr: np.ndarray) -> Image.Image:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    def _score_plate_quality(self, plate_crop: Image.Image) -> dict:
        try:
            img = np.array(plate_crop)
            if img.size == 0:
                return {"score": 0.0, "components": 0, "sharpness": 0.0, "is_incomplete": True}

            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                gray = img

            sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            blur_score = sharpness
            edges = cv2.Canny(gray, 50, 150)
            edge_density = float(np.mean(edges > 0))
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            num_labels, labels = cv2.connectedComponents(255 - binary)

            components = 0
            for label in range(1, num_labels):
                area = int(np.sum(labels == label))
                if area >= 10:
                    components += 1

            edge_score = min(edge_density / 0.12, 1.0)
            comp_score = 0.0 if components < self._min_component_count else min((components - 4) / 6.0, 1.0)
            sharp_score = min(sharpness / 200.0, 1.0)

            score = edge_score * 0.5 + comp_score * 0.3 + sharp_score * 0.2

            h, w = gray.shape[:2]
            area = float(h * w)
            area_score = min(area / 15000.0, 1.0)
            contrast = float(np.std(gray))
            contrast_score = min(contrast / 64.0, 1.0)
            if len(img.shape) == 3:
                hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
                brightness_std = float(np.std(hsv[:, :, 2]))
            else:
                brightness_std = float(np.std(gray))
            brightness_score = min(brightness_std / 64.0, 1.0)
            blur_norm = min(blur_score / 200.0, 1.0)
            quality_score = (
                area_score * 0.30 +
                blur_norm * 0.30 +
                contrast_score * 0.20 +
                brightness_score * 0.20
            )

            is_incomplete = (
                components < self._min_component_count or
                edge_density < self._min_edge_density or
                score < self._min_quality_score
            )

            return {
                "score": score,
                "components": components,
                "sharpness": sharpness,
                "is_incomplete": is_incomplete,
                "area": area,
                "blur": blur_score,
                "contrast": contrast,
                "brightness_std": brightness_std,
                "quality_score": quality_score,
            }
        except Exception:
            logger.exception("Plate quality scoring error")
            return {"score": 0.0, "components": 0, "sharpness": 0.0, "is_incomplete": True}

    def _adaptive_confidence_threshold(self, quality: Optional[dict],
                                        camera_id: str = "unknown") -> float:
        quality_score = 0.0
        if quality:
            quality_score = float(quality.get("quality_score", 0.0))

        # AmbientAdapter: interpolate base/alpha/min/max by ambient brightness
        try:
            from app.services.core.ambient_adapter import AmbientAdapter
            _amb = AmbientAdapter.get_instance()
            base = _amb.get_threshold(camera_id, "adaptive_confidence_base",
                                      self._adaptive_conf_base)
            alpha = _amb.get_threshold(camera_id, "adaptive_confidence_alpha",
                                       self._adaptive_conf_alpha)
            mn = _amb.get_threshold(camera_id, "adaptive_confidence_min",
                                    self._adaptive_conf_min)
        except Exception:
            base = self._adaptive_conf_base
            alpha = self._adaptive_conf_alpha
            mn = self._adaptive_conf_min

        threshold = base - alpha * quality_score
        return float(min(self._adaptive_conf_max, max(mn, threshold)))

    async def process_frame(
        self,
        img_bgr: np.ndarray,
        camera_id: str,
        roi_points: Optional[List[Tuple[int, int]]] = None,
    ) -> List[Dict]:
        _vlog("LEGACY pipeline start cam=%s", camera_id)
        if not self.settings.enable_plate_ocr:
            return []

        image = self._pil_from_bgr(img_bgr)
        is_night = False
        try:
            is_night = ImageEnhancer.is_night_time(image)
        except Exception:
            is_night = False

        if is_night and not self.settings.enable_night_lpr:
            return []

        if not self.ocr_service.is_available():
            return []

        data_collector = DataCollector(
            enabled=self.settings.enable_lpr_data_collection,
            base_dir=self.settings.lpr_collection_dir,
            sample_rate=self.settings.lpr_collect_sample_rate,
            collect_vehicles=self.settings.lpr_collect_vehicle,
            collect_plates=self.settings.lpr_collect_plate,
            min_conf=self.settings.lpr_collect_min_conf,
            max_conf=self.settings.lpr_collect_max_conf,
            low_conf_only=self.settings.lpr_collect_low_conf_only,
            quality_filter=self.settings.lpr_collect_quality_filter,
            min_sharpness=self.settings.lpr_collect_min_sharpness,
            min_brightness=self.settings.lpr_collect_min_brightness,
            max_brightness=self.settings.lpr_collect_max_brightness,
            min_vehicle_area=self.settings.lpr_collect_min_vehicle_area,
            min_plate_area=self.settings.lpr_collect_min_plate_area,
        )

        fortress = get_fortress_lpr()
        vehicles = fortress.vehicle_detector.detect(img_bgr)
        results: List[Dict] = []

        _vlog("LEGACY vehicles=%s", len(vehicles))

        for vehicle in vehicles:
            try:
                bbox = vehicle["bbox"]
                orig_w = bbox[2] - bbox[0]
                orig_h = bbox[3] - bbox[1]

                if orig_w < 60:
                    _vlog("Skip small vehicle width=%.0f", orig_w)
                    continue

                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2
                img_w, img_h = image.size

                if roi_points and not _point_in_polygon(cx, cy, roi_points):
                    _vlog("Skip vehicle outside ROI")
                    continue

                is_large_vehicle = vehicle["class"] in ["truck", "bus"] or orig_w > 300
                edge_margin = 0.04 if is_large_vehicle else 0.08
                vehicle_conf = float(vehicle.get("confidence") or 0.0)

                if not (edge_margin * img_w < cx < (1 - edge_margin) * img_w and
                        edge_margin * img_h < cy < (1 - edge_margin) * img_h):
                    if vehicle_conf < 0.6:
                        _vlog("Skip edge vehicle class=%s", vehicle.get("class"))
                        continue

                expansion_scale = 0.40 if is_night else 0.30
                ex1, ey1, ex2, ey2 = expand_bbox(
                    bbox[0], bbox[1], bbox[2], bbox[3],
                    img_w, img_h,
                    scale=expansion_scale
                )

                vehicle_crop = image.crop((ex1, ey1, ex2, ey2))
                crop_width = ex2 - ex1
                scale_factor = 1.0
                vehicle_crop_upscaled, was_upscaled, _sr_reason = await self.sr_service.conditional_upscale(
                    vehicle_crop,
                    vehicle_width=int(crop_width)
                )
                if was_upscaled:
                    _vlog("SR applied to vehicle crop")
                    vehicle_crop = vehicle_crop_upscaled
                    scale_factor = 2.0

                if data_collector.enabled:
                    frame_id = int(time.time() * 1000)
                    vehicle_cv = cv2.cvtColor(np.array(vehicle_crop), cv2.COLOR_RGB2BGR)
                    data_collector.save_vehicle(vehicle_cv, frame_id, vehicle_conf)

                plate_text, conf_score, plate_crop, plate_bbox_in_vehicle = await self.ocr_service.predict(vehicle_crop)
                _vlog("OCR result text='%s' conf=%.2f", plate_text, conf_score)

                quality = None
                if plate_crop is not None:
                    quality = self._score_plate_quality(plate_crop)

                if plate_crop is not None and data_collector.enabled:
                    frame_id = int(time.time() * 1000)
                    plate_cv = cv2.cvtColor(np.array(plate_crop), cv2.COLOR_RGB2BGR)
                    data_collector.save_plate(plate_cv, frame_id, conf_score, plate_text)

                if not plate_text or len(plate_text) < 6:
                    _vlog("Reject short plate '%s'", plate_text)
                    continue

                if plate_bbox_in_vehicle:
                    px1, py1, px2, py2 = plate_bbox_in_vehicle
                    pcx_crop = (px1 + px2) / 2
                    pcy_crop = (py1 + py2) / 2

                    pcx_crop_orig = pcx_crop / scale_factor
                    pcy_crop_orig = pcy_crop / scale_factor

                    plate_center_x = ex1 + pcx_crop_orig
                    plate_center_y = ey1 + pcy_crop_orig

                    vx1, vy1, vx2, vy2 = bbox
                    if not (vx1 <= plate_center_x <= vx2 and vy1 <= plate_center_y <= vy2):
                        continue

                clean_plate = plate_text.replace("-", "").replace(".", "").replace(" ", "").upper()
                if len(clean_plate) < 6:
                    _vlog("Reject invalid plate length '%s'", plate_text)
                    continue
                if len(clean_plate) >= 2 and (not clean_plate[0].isdigit() or not clean_plate[1].isdigit()):
                    _vlog("Reject invalid province '%s'", plate_text)
                    continue
                try:
                    province = int(clean_plate[:2])
                    if province < 11 or province > 99:
                        _vlog("Reject province out of range '%s'", plate_text)
                        continue
                except Exception:
                    _vlog("Reject province parse '%s'", plate_text)
                    continue
                if len(clean_plate) >= 3 and not clean_plate[2].isalpha():
                    _vlog("Reject missing series '%s'", plate_text)
                    continue

                if len(clean_plate) < 8 or (quality and quality["is_incomplete"]):
                    conf_score = min(conf_score, 0.84)
                dynamic_threshold = self._adaptive_confidence_threshold(quality, camera_id=camera_id)
                if conf_score < dynamic_threshold:
                    _vlog("Reject low conf %.2f plate='%s'", conf_score, plate_text)
                    continue

                if vehicle.get("class") == "motorcycle" and conf_score < 0.95:
                    conf_score = min(conf_score, 0.84)

                timestamp_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                safe_plate = _safe_plate(plate_text)
                base_filename = f"{timestamp_str}_{camera_id[:8]}_{safe_plate}"

                full_frame_filename = f"{base_filename}_full.jpg"
                plate_filename = f"{base_filename}_plate.jpg"
                vehicle_filename = f"{base_filename}_vehicle.jpg"

                should_save, cache_entry = self.consensus.should_process_plate(
                    camera_id, plate_text, conf_score, plate_filename, full_frame_filename, list(bbox), None
                )

                if not should_save:
                    _vlog("Consensus skip plate='%s'", plate_text)
                    continue

                os.makedirs(self.settings.thumbnail_path, exist_ok=True)

                full_frame_path = os.path.join(self.settings.thumbnail_path, full_frame_filename)
                annotated_full = image.copy()
                annotated_full = _draw_detection_box(
                    annotated_full,
                    bbox,
                    label=(vehicle.get("class") or "vehicle"),
                    color="green",
                    line_width=6,
                )
                if plate_bbox_in_vehicle:
                    try:
                        vx1, vy1, _, _ = [int(v) for v in bbox]
                        p1x, p1y, p2x, p2y = [int(v) for v in plate_bbox_in_vehicle]
                        plate_bbox_frame = (vx1 + p1x, vy1 + p1y, vx1 + p2x, vy1 + p2y)
                        annotated_full = _draw_detection_box(
                            annotated_full,
                            plate_bbox_frame,
                            label=plate_text,
                            color="red",
                            line_width=6,
                        )
                    except Exception:
                        pass
                # Save images in thread to avoid blocking event loop
                await asyncio.to_thread(annotated_full.save, full_frame_path, "JPEG", quality=90)

                vehicle_path = os.path.join(self.settings.thumbnail_path, vehicle_filename)
                annotated_vehicle = vehicle_crop
                if plate_bbox_in_vehicle:
                    annotated_vehicle = _draw_detection_box(
                        annotated_vehicle,
                        plate_bbox_in_vehicle,
                        label=plate_text,
                        color="red",
                        line_width=8,
                    )
                await asyncio.to_thread(annotated_vehicle.save, vehicle_path, "JPEG", quality=90)

                plate_path = os.path.join(self.settings.thumbnail_path, plate_filename)
                if plate_bbox_in_vehicle:
                    px1, py1, px2, py2 = [int(v) for v in plate_bbox_in_vehicle]
                    img_w2, img_h2 = vehicle_crop.size
                    ex1p, ey1p, ex2p, ey2p = expand_bbox(px1, py1, px2, py2, img_w2, img_h2, scale=0.15)
                    manual_plate_crop = vehicle_crop.crop((ex1p, ey1p, ex2p, ey2p))
                    await asyncio.to_thread(manual_plate_crop.save, plate_path, "JPEG", quality=95)
                elif plate_crop:
                    await asyncio.to_thread(plate_crop.save, plate_path, "JPEG", quality=95)
                else:
                    continue

                is_update = False
                evt_id = uuid.uuid4()
                plate_id = uuid.uuid4()
                if cache_entry and cache_entry.get("event_id") and cache_entry.get("event_id") not in ["PENDING", "DEFERRED"]:
                    evt_id = cache_entry["event_id"]
                    plate_id = cache_entry["plate_id"]
                    is_update = True

                vehicle_color = _estimate_vehicle_color(vehicle_crop)

                await self.event_repository.save_plate_event(
                    event_id=evt_id,
                    plate_id=plate_id,
                    camera_id=camera_id,
                    plate_number=plate_text,
                    vehicle_type=vehicle["class"],
                    confidence=conf_score,
                    thumbnail_filename=plate_filename,
                    full_frame_filename=full_frame_filename,
                    bbox=list(bbox),
                    tracking_id=None,
                    is_update=is_update,
                    metadata={"vehicle_color": vehicle_color},
                )

                # Write back IDs so next re-detection can UPDATE instead of INSERT
                consensus_key = self.consensus._key(camera_id, plate_text, None)
                consensus_entry = self.consensus._entries.get(consensus_key)
                if consensus_entry is not None:
                    consensus_entry.event_id = evt_id
                    consensus_entry.plate_id = plate_id

                _vlog("Saved plate='%s' conf=%.2f", plate_text, conf_score)

                results.append({
                    "plate_text": plate_text,
                    "confidence": float(conf_score),
                    "bbox": [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
                    "track_id": None,
                })

            except Exception:
                logger.exception("Legacy LPR error")

        return results
