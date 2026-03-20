# omni-vehicle: License plate recognition service
# FastAPI microservice for Vietnamese license plate detection and OCR

import asyncio
import logging
import os
import time
import uuid
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np
from PIL import Image
import io
import cv2
import redis.asyncio as aioredis

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("omni-vehicle")

from app.services.core.model_loader import ModelLoader
from app.services.pipeline.application.lpr_utils import expand_bbox
from app.services.pipeline.application.lpr_service import OCRService
from app.services.core.simple_tracker import TrackerManager
from app.services.pipeline.repositories.event_repository import PlateEventRepository
from app.services.plate.plate_utils import normalize_plate_basic, plate_edit_distance
from app.services.plate.vehicle_types import normalize_vehicle_type
from app.services.pipeline.orchestration.legacy_plate_pipeline import LegacyPlatePipeline
from app.config import get_settings
from app.workers.lpr_scheduler import LprScheduler, scheduler_enabled

# === MIGRATED FROM ROOT ai-engine ===
from app.services.integration.camera.dahua_sdk_manager import get_dahua_sdk_manager
from app.workers.hybrid_broker import HybridBroker
from app.workers.tracking_consensus import TrackingConsensus
from app.workers.training_worker import LprTrainingWorker

# === EVENT-DRIVEN (Master Plan V8) ===
from app.workers.stream_consumer import LprStreamConsumer


class PlateResult(BaseModel):
    plate_text: str
    confidence: float
    bbox: List[int]  # [x1, y1, x2, y2]
    track_id: int | None = None
    

class DetectionResponse(BaseModel):
    success: bool
    plates: List[PlateResult]
    processing_time_ms: float


import threading

_tracker_manager = TrackerManager()
_plate_cache: Dict[str, Dict[str, float]] = {}
_plate_cache_lock = threading.Lock()
_plate_cache_last_full_cleanup: float = 0.0
_PLATE_CACHE_MAX_ENTRIES = 5000  # Hard cap to prevent unbounded memory growth
_event_repo = PlateEventRepository.get_instance()
_scheduler = LprScheduler()

# === Global instances for migrated components ===
_dahua_manager = None
_hybrid_broker = None
_training_worker = None
_stream_consumer = None
_redis_dedup: Optional[aioredis.Redis] = None

_MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB hard limit for /detect uploads


def _plate_cache_key(camera_id: str, track_id: int | None, plate_text: str) -> str:
    return f"{camera_id}:{track_id if track_id is not None else 'na'}:{plate_text}"


# _plate_edit_distance → moved to plate_utils.plate_edit_distance


def _should_emit_plate_memory(camera_id: str, track_id: int | None, plate_text: str, ttl_sec: float) -> bool:
    """
    In-memory dedup gate (single-worker fallback).
    
    Uses FUZZY matching (edit distance ≤ 1) to catch minor OCR jitter
    while preventing different plates from merging (e.g., 65A03977 vs 65B03977).
    Also requires province+series prefix match to avoid cross-plate merges.
    """
    global _plate_cache_last_full_cleanup
    now = time.time()
    
    with _plate_cache_lock:
        # Periodic full cleanup across ALL cameras (every 30s) to prevent unbounded growth
        if now - _plate_cache_last_full_cleanup > 30.0:
            _plate_cache_last_full_cleanup = now
            total_entries = 0
            empty_cams = []
            for cam_id, cam_cache in _plate_cache.items():
                expired = [k for k, ts in cam_cache.items() if now - ts > ttl_sec]
                for k in expired:
                    del cam_cache[k]
                total_entries += len(cam_cache)
                if not cam_cache:
                    empty_cams.append(cam_id)
            for cam_id in empty_cams:
                del _plate_cache[cam_id]
            # Hard cap: if still too many entries, evict oldest across all cameras
            if total_entries > _PLATE_CACHE_MAX_ENTRIES:
                all_entries = []
                for cam_id, cam_cache in _plate_cache.items():
                    for k, ts in cam_cache.items():
                        all_entries.append((ts, cam_id, k))
                all_entries.sort()  # oldest first
                evict_count = total_entries - _PLATE_CACHE_MAX_ENTRIES
                for _, cam_id, k in all_entries[:evict_count]:
                    _plate_cache.get(cam_id, {}).pop(k, None)

        cache = _plate_cache.setdefault(camera_id, {})
        
        # Cleanup expired entries
        for key, ts in list(cache.items()):
            if now - ts > ttl_sec:
                cache.pop(key, None)
        
        # Exact match (fast path)
        key = _plate_cache_key(camera_id, track_id, plate_text)
        last = cache.get(key)
        if last is not None and now - last < ttl_sec:
            return False
        
        # Fuzzy match: check ALL recent plates for this camera
        # This catches OCR jitter where same plate reads differently across frames
        # Uses edit_distance ≤ 1 + province prefix match to prevent cross-plate merges
        clean_text = plate_text.replace("-", "").replace(".", "").replace(" ", "").upper()
        if len(clean_text) >= 6:
            # Extract province+series prefix (e.g., "65A" from "65A03977")
            prefix = clean_text[:3] if len(clean_text) >= 3 else clean_text
            for cached_key, cached_ts in list(cache.items()):
                if now - cached_ts >= ttl_sec:
                    continue
                # Extract plate text from cache key (format: camera_id:track:plate)
                parts = cached_key.split(":", 2)
                if len(parts) >= 3:
                    cached_plate = parts[2].replace("-", "").replace(".", "").replace(" ", "").upper()
                    if len(cached_plate) >= 6:
                        # Province+series must match exactly to prevent cross-plate merges
                        cached_prefix = cached_plate[:3] if len(cached_plate) >= 3 else cached_plate
                        if prefix != cached_prefix:
                            continue
                        dist = plate_edit_distance(clean_text, cached_plate)
                        if dist <= 1:  # max 1 char difference = same plate with OCR jitter
                            # Update timestamp on the original entry to extend TTL
                            cache[cached_key] = now
                            return False
        
        cache[key] = now
        return True


async def _should_emit_plate_redis(camera_id: str, plate_text: str, ttl_sec: float) -> bool:
    """
    Redis-backed dedup gate.
    Uses a sorted-set per camera with plate text as member, timestamp as score.
    Works correctly across multiple uvicorn workers.
    """
    key = f"plate_dedup:{camera_id}"
    now = time.time()
    cutoff = now - ttl_sec

    pipe = _redis_dedup.pipeline()
    pipe.zremrangebyscore(key, "-inf", cutoff)
    pipe.zrangebyscore(key, cutoff, "+inf")
    results = await pipe.execute()
    recent_plates: list = results[1]

    clean = plate_text.replace("-", "").replace(".", "").replace(" ", "").upper()
    prefix = clean[:3] if len(clean) >= 3 else ""

    if len(clean) >= 6 and prefix:
        for cached_raw in recent_plates:
            cached = (cached_raw.decode() if isinstance(cached_raw, bytes) else cached_raw)
            cached_clean = cached.replace("-", "").replace(".", "").replace(" ", "").upper()
            if len(cached_clean) < 6:
                continue
            cached_prefix = cached_clean[:3]
            if prefix != cached_prefix:
                continue
            dist = plate_edit_distance(clean, cached_clean)
            if dist <= 1:
                await _redis_dedup.zadd(key, {cached: now})
                return False

    await _redis_dedup.zadd(key, {plate_text: now})
    await _redis_dedup.expire(key, int(ttl_sec * 2))
    return True


async def _should_emit_plate(camera_id: str, track_id: int | None, plate_text: str, ttl_sec: float) -> bool:
    """Dedup gate: prefer Redis (cross-worker safe), fallback to in-memory."""
    if _redis_dedup:
        try:
            return await _should_emit_plate_redis(camera_id, plate_text, ttl_sec)
        except Exception:
            pass
    return _should_emit_plate_memory(camera_id, track_id, plate_text, ttl_sec)


def _parse_roi(roi_json: str | None) -> List[Tuple[int, int]] | None:
    if not roi_json:
        return None
    import json

    try:
        points = json.loads(roi_json)
    except Exception:
        return None
    if not isinstance(points, list) or len(points) < 3:
        return None
    parsed: List[Tuple[int, int]] = []
    for p in points:
        if not isinstance(p, (list, tuple)) or len(p) != 2:
            return None
        parsed.append((int(p[0]), int(p[1])))
    return parsed


def _point_in_polygon(x: float, y: float, polygon: List[Tuple[int, int]]) -> bool:
    inside = False
    n = len(polygon)
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        dy = yj - yi
        if abs(dy) < 1e-12:
            j = i
            continue
        intersect = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / dy + xi
        )
        if intersect:
            inside = not inside
        j = i
    return inside


def _filter_by_roi(detections: List[List[float]], roi: List[Tuple[int, int]] | None) -> List[List[float]]:
    if not roi:
        return detections
    filtered: List[List[float]] = []
    for det in detections:
        x1, y1, x2, y2 = det[:4]
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        if _point_in_polygon(cx, cy, roi):
            filtered.append(det)
    return filtered


def _safe_plate_name(text: str) -> str:
    cleaned = normalize_plate_basic(text).upper()
    if not cleaned:
        return "UNKNOWN"
    return "".join(ch for ch in cleaned if ch.isalnum()) or "UNKNOWN"


def _save_plate_images(
    img_bgr: np.ndarray,
    plate_crop_bgr: np.ndarray,
    bbox: Tuple[int, int, int, int],
    camera_id: str,
    plate_text: str,
) -> Tuple[Optional[str], Optional[str]]:
    settings = get_settings()
    os.makedirs(settings.thumbnail_path, exist_ok=True)

    timestamp_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    safe_plate = _safe_plate_name(plate_text)
    short_cam = camera_id[:8] if camera_id else "cam"
    suffix = uuid.uuid4().hex[:6]
    base = f"{timestamp_str}_{short_cam}_{safe_plate}_{suffix}"

    plate_filename = f"{base}_plate.jpg"
    full_filename = f"{base}_full.jpg"

    plate_path = os.path.join(settings.thumbnail_path, plate_filename)
    full_path = os.path.join(settings.thumbnail_path, full_filename)

    try:
        import cv2

        if plate_crop_bgr is not None and plate_crop_bgr.size > 0:
            if not cv2.imwrite(plate_path, plate_crop_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92]):
                logger.warning("cv2.imwrite failed for plate crop: %s", plate_path)
                plate_filename = None

        # Resize full frame if wider than 1440px to save storage
        h, w = img_bgr.shape[:2]
        max_w = 1440
        if w > max_w:
            scale = max_w / w
            img_bgr = cv2.resize(img_bgr, (max_w, int(h * scale)), interpolation=cv2.INTER_AREA)

        if not cv2.imwrite(full_path, img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 88]):
            logger.warning("cv2.imwrite failed for full image: %s", full_path)
            full_filename = None
        return plate_filename, full_filename
    except Exception as exc:
        logger.warning("Failed to save plate images: %s", exc)
        return None, None


async def _persist_plate_event(
    camera_id: str,
    plate_text: str,
    confidence: float,
    bbox: Tuple[int, int, int, int],
    plate_crop_bgr: np.ndarray,
    img_bgr: np.ndarray,
    track_id: int | None,
    metadata: Optional[dict] = None,
) -> None:
    if not camera_id:
        return

    try:
        uuid.UUID(camera_id)
    except Exception:
        logger.warning("Skip save: invalid camera_id '%s'", camera_id)
        return

    save_enabled = os.environ.get("SAVE_PLATE_EVENTS", "1").strip().lower() in {"1", "true", "yes"}
    if not save_enabled:
        return

    plate_filename, full_filename = await asyncio.to_thread(
        _save_plate_images,
        img_bgr, plate_crop_bgr, bbox, camera_id, plate_text,
    )

    try:
        await _event_repo.save_plate_event(
            event_id=uuid.uuid4(),
            plate_id=uuid.uuid4(),
            camera_id=camera_id,
            plate_number=plate_text,
            vehicle_type=normalize_vehicle_type(metadata.get("vehicle_type") if metadata else None, plate_text),
            confidence=confidence,
            thumbnail_filename=plate_filename,
            full_frame_filename=full_filename,
            bbox=list(bbox),
            tracking_id=str(track_id) if track_id is not None else None,
            metadata=metadata,
            is_update=False,
        )
    except Exception as exc:
        logger.warning("DB persist failed (main path): %s", exc)
    

async def _process_frame(
    img_bgr: np.ndarray,
    camera_id: str,
    roi_points: List[Tuple[int, int]] | None,
    is_night: bool,
    use_tracking: bool,
    use_fortress: bool,
    dedup: bool,
    allow_legacy: bool = True,
) -> List[PlateResult]:
    legacy_enabled = allow_legacy and os.environ.get("LEGACY_CORE_PIPELINE", "1").strip().lower() in {"1", "true", "yes"}
    if legacy_enabled:
        try:
            legacy = LegacyPlatePipeline.get_instance()
            legacy_results = await legacy.process_frame(
                img_bgr=img_bgr,
                camera_id=camera_id,
                roi_points=roi_points,
            )
            return [
                PlateResult(
                    plate_text=r["plate_text"],
                    confidence=float(r["confidence"]),
                    bbox=[int(v) for v in r["bbox"]],
                    track_id=r.get("track_id"),
                )
                for r in legacy_results
            ]
        except Exception as exc:
            logger.exception("Legacy core pipeline failed, fallback to default: %s", exc)

    loader = ModelLoader.get_instance()
    detector, paddle_ocr = loader.get_models()

    if detector is None:
        logger.warning("LP_detector not loaded — cannot process frame")
        return []  # Return empty instead of HTTPException (called from non-HTTP contexts too)
    if paddle_ocr is None:
        logger.warning("PaddleOCR not loaded — cannot process frame")
        return []  # Return empty instead of HTTPException (called from non-HTTP contexts too)

    plates: List[PlateResult] = []
    dedup_ttl = float(os.environ.get("PLATE_DEDUP_TTL", "30.0"))

    if use_fortress:
        try:
            from app.services.pipeline.orchestration.fortress_lpr import get_fortress_lpr

            fortress = get_fortress_lpr()
            # Run blocking GPU inference in a thread to avoid blocking the event loop
            frame_result = await asyncio.to_thread(fortress.process_frame, img_bgr, True)
            track_assignments: Dict[Tuple[int, int, int, int], int] = {}
            if use_tracking and camera_id:
                track_iou = float(os.environ.get("TRACK_IOU", "0.3"))
                track_max_age = float(os.environ.get("TRACK_MAX_AGE_SEC", "5.0"))
                track_min_hits = int(os.environ.get("TRACK_MIN_HITS", "2"))
                tracker = _tracker_manager.get_tracker(
                    camera_id,
                    iou_threshold=track_iou,
                    max_age_seconds=track_max_age,
                    min_hits=track_min_hits,
                )
                dets = [
                    ((int(p.bbox[0]), int(p.bbox[1]), int(p.bbox[2]), int(p.bbox[3])), float(p.confidence))
                    for p in frame_result.plates
                ]
                track_results = tracker.update(dets, include_unconfirmed=True)
                track_assignments = {bbox: tid for bbox, tid in track_results}

            for plate in frame_result.plates:
                x1, y1, x2, y2 = map(int, plate.bbox)
                if roi_points and not _point_in_polygon((x1 + x2) / 2.0, (y1 + y2) / 2.0, roi_points):
                    continue
                track_id = track_assignments.get((x1, y1, x2, y2))
                if dedup and not await _should_emit_plate(camera_id, track_id, plate.plate_text, dedup_ttl):
                    continue
                # NOTE: DB persist removed from HTTP path — stream consumer
                # handles persistence with TrackingConsensus gate.
                # HTTP /detect-advanced is detection-only.
                plates.append(PlateResult(
                    plate_text=plate.plate_text,
                    confidence=float(plate.confidence),
                    bbox=[x1, y1, x2, y2],
                    track_id=track_id,
                ))
        except Exception as exc:
            logger.exception("Fortress pipeline unavailable, fallback to plate detector: %s", exc)
            use_fortress = False

    if not use_fortress:
        results = detector(img_bgr, size=640)
        raw_dets = results.pandas().xyxy[0].values.tolist()
        detections = _filter_by_roi(raw_dets, roi_points)
        img_h, img_w = img_bgr.shape[:2]
        expand_day = float(os.environ.get("PLATE_BBOX_EXPAND", "0.2"))
        expand_night = float(os.environ.get("PLATE_BBOX_EXPAND_NIGHT", "0.28"))
        ocr_service = OCRService.get_instance()

        track_assignments: Dict[Tuple[int, int, int, int], int] = {}
        if use_tracking and camera_id:
            track_iou = float(os.environ.get("TRACK_IOU", "0.3"))
            track_max_age = float(os.environ.get("TRACK_MAX_AGE_SEC", "5.0"))
            track_min_hits = int(os.environ.get("TRACK_MIN_HITS", "2"))
            tracker = _tracker_manager.get_tracker(
                camera_id,
                iou_threshold=track_iou,
                max_age_seconds=track_max_age,
                min_hits=track_min_hits,
            )
            track_results = tracker.update([
                ((int(d[0]), int(d[1]), int(d[2]), int(d[3])), float(d[4])) for d in detections
            ], include_unconfirmed=True)
            track_assignments = {bbox: tid for bbox, tid in track_results}

        for det in detections:
            x1, y1, x2, y2, conf, _cls_id, _cls_name = det[:7]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            scale = expand_night if is_night else expand_day
            x1, y1, x2, y2 = expand_bbox(x1, y1, x2, y2, img_w, img_h, scale=scale)

            plate_crop = img_bgr[y1:y2, x1:x2]
            if plate_crop.size == 0:
                continue

            plate_pil = Image.fromarray(cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB))
            bbox_area = max((x2 - x1) * (y2 - y1), 1)
            track_id = track_assignments.get((int(det[0]), int(det[1]), int(det[2]), int(det[3])))

            ocr_result = await ocr_service.predict_plate(
                plate_image=plate_pil,
                track_id=track_id,
                camera_id=camera_id,
                bbox_area=bbox_area,
                is_night=is_night,
            )
            if ocr_result:
                text, ocr_conf, _box = ocr_result
                if dedup and not await _should_emit_plate(camera_id, track_id, text, dedup_ttl):
                    continue
                # NOTE: DB persist removed — stream consumer handles with consensus.
                # Use OCR confidence directly — detector conf already gates detection,
                # multiplying them deflates scores (0.7×0.8=0.56 → unfairly low).
                plates.append(PlateResult(
                    plate_text=text,
                    confidence=float(ocr_conf),
                    bbox=[x1, y1, x2, y2],
                    track_id=track_id,
                ))

    return plates


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup + start migrated services + Redis Stream consumer"""
    global _dahua_manager, _hybrid_broker, _training_worker, _stream_consumer, _redis_dedup

    logger.info("🚀 omni-vehicle service starting...")

    # --- Connect Redis for plate dedup (cross-worker safe) ---
    settings = get_settings()
    try:
        _redis_dedup = aioredis.from_url(
            settings.redis_streams_url,
            decode_responses=False,
        )
        await _redis_dedup.ping()
        logger.info("   ✅ Redis plate dedup cache: connected")
    except Exception as e:
        logger.warning("   ⚠️  Redis dedup unavailable: %s (in-memory fallback)", e)
        _redis_dedup = None

    # --- Load AI models in thread to avoid blocking the event loop ---
    # loader.load() calls torch.hub.load() and initializes PaddleOCR,
    # both are synchronous heavy GPU operations (15-60+ seconds on cold start).
    loader = ModelLoader.get_instance()
    try:
        detector, paddle_ocr = await asyncio.to_thread(loader.load)
        await asyncio.to_thread(loader.warmup)
    except Exception as e:
        logger.error("❌ Model loading failed: %s — LPR detection disabled", e)
        detector, paddle_ocr = None, None

    ocr_service = OCRService.get_instance()
    ocr_service.set_external_model(paddle_ocr)

    logger.info("🎯 omni-vehicle models loaded!")

    # --- Start HybridBroker (Dahua SDK → Consensus → DB) ---
    settings = get_settings()
    try:
        consensus = TrackingConsensus(settings)
        _hybrid_broker = HybridBroker(
            settings=settings,
            consensus=consensus,
            event_repository=_event_repo,
        )
        logger.info("   ✅ HybridBroker: Ready")
    except Exception as e:
        logger.warning("   ⚠️  HybridBroker init failed: %s", e)
        _hybrid_broker = None

    # --- Start Dahua SDK Manager ---
    try:
        _dahua_manager = get_dahua_sdk_manager(hybrid_broker=_hybrid_broker)
        await _dahua_manager.start()
        logger.info("   ✅ Dahua SDK Manager: Started (%d cameras)", len(_dahua_manager._bridges))
    except Exception as e:
        logger.warning("   ⚠️  Dahua SDK Manager failed: %s", e)
        logger.info("   💡 Dahua LPR will use RTSP fallback mode")
        _dahua_manager = None

    # --- Start LPR Training Worker ---
    try:
        _training_worker = LprTrainingWorker()
        await _training_worker.start()
        logger.info("   ✅ LPR Training Worker: Started")
    except Exception as e:
        logger.warning("   ⚠️  LPR Training Worker failed: %s", e)
        _training_worker = None

    # --- Start Redis stream consumer (event-driven from omni-object) ---
    try:
        _stream_consumer = LprStreamConsumer(settings, process_frame_fn=_process_frame)
        await _stream_consumer.connect()
        await _stream_consumer.ensure_group()
        await _stream_consumer.start()
        logger.info("   ✅ Redis Stream Consumer: Started (event-driven mode)")
    except Exception as e:
        logger.warning("   ⚠️  Redis Stream Consumer failed: %s (pull-mode only)", e)
        _stream_consumer = None

    # --- Start LPR Scheduler (pull-based polling fallback) ---
    if scheduler_enabled():
        await _scheduler.start(_process_frame)
        logger.info("   ✅ LPR Scheduler: Started (pull-based fallback)")

    logger.info("🎯 omni-vehicle service ready! (Full LPR stack + event-driven)")

    yield

    # --- Shutdown ---
    logger.info("👋 omni-vehicle service shutting down...")
    if _stream_consumer:
        await _stream_consumer.disconnect()
    await _scheduler.stop()
    if _dahua_manager:
        await _dahua_manager.stop()
    if _training_worker:
        await _training_worker.stop()
    # Dispose the DB connection pool to prevent leaked connections
    try:
        await _event_repo.dispose()
    except Exception as exc:
        logger.warning("Failed to dispose event repo: %s", exc)
    if _redis_dedup:
        try:
            await _redis_dedup.aclose()
        except Exception:
            pass
    logger.info("👋 omni-vehicle service stopped.")


app = FastAPI(
    title="omni-vehicle Service",
    description="License plate recognition microservice for OmniVision",
    version="2.0.0",
    lifespan=lifespan
)

# Static files mount — use check_dir=False so missing dir doesn't crash at startup
try:
    _thumbnail_dir = get_settings().thumbnail_path
    app.mount("/thumbnails", StaticFiles(directory=_thumbnail_dir, check_dir=False), name="thumbnails")
except Exception as _e:
    logger.warning("Failed to mount /thumbnails static files: %s", _e)


@app.middleware("http")
async def limit_upload_size(request: Request, call_next):
    """Reject uploads larger than _MAX_UPLOAD_BYTES to prevent RAM spikes."""
    cl = request.headers.get("content-length")
    if cl and int(cl) > _MAX_UPLOAD_BYTES:
        return JSONResponse(status_code=413, content={"detail": "Payload too large"})
    return await call_next(request)

# Mount routers defensively — if one fails to import (missing dep, syntax error),
# the others still register. This prevents honeypot.py failure from killing
# the settings and detection routes (was causing 404 on /vehicle/settings).
try:
    from app.api.honeypot import router as honeypot_router
    app.include_router(honeypot_router)
except Exception as _e:
    logger.warning("⚠️ Honeypot router failed to load: %s", _e)

try:
    from app.api.settings import router as settings_router
    app.include_router(settings_router)
except Exception as _e:
    import traceback
    logger.error("❌ Settings router failed to load: %s — /vehicle/settings will be unavailable!", _e)
    logger.error("   Full traceback:\n%s", traceback.format_exc())

try:
    from app.api.detection import router as detection_router
    app.include_router(detection_router)
except Exception as _e:
    logger.error("❌ Detection router failed to load: %s", _e)


@app.get("/vehicle/health")
async def health_check():
    """Health check endpoint — reports accurate status based on model/component state."""
    loader = ModelLoader.get_instance()
    detector, paddle_ocr = loader.get_models()
    stream_ok = _stream_consumer is not None and _stream_consumer._connected
    detector_ok = detector is not None
    ocr_ok = paddle_ocr is not None

    fortress_status = {}
    try:
        from app.services.pipeline.orchestration.fortress_lpr import _fortress_lpr
        if _fortress_lpr is not None:
            f = _fortress_lpr
            fortress_status = {
                "initialized": True,
                "vehicle_detector": f.vehicle_detector.model is not None if f.vehicle_detector else False,
                "plate_detector_obb": not f.plate_detector.use_cv_fallback if f.plate_detector else False,
                "stn_lprnet": f.recognizer.model is not None if f.recognizer else False,
                "ocr_fallback": "paddle" if (f.recognizer and f.recognizer.model is None) else "lprnet",
            }
        else:
            fortress_status = {"initialized": False}
    except Exception:
        fortress_status = {"initialized": False, "error": "import failed"}

    if detector_ok and ocr_ok and stream_ok:
        status = "healthy"
    elif detector_ok and ocr_ok:
        status = "degraded"
    else:
        status = "unhealthy"

    return {
        "status": status,
        "service": "omni-vehicle",
        "version": "2.0.0",
        "detector_loaded": detector_ok,
        "ocr_loaded": ocr_ok,
        "fortress": fortress_status,
        "dahua_sdk": _dahua_manager is not None,
        "hybrid_broker": _hybrid_broker is not None,
        "training_worker": _training_worker is not None,
        "stream_consumer": stream_ok,
        "stream_metrics": _stream_consumer.get_metrics_snapshot() if _stream_consumer else None,
    }


@app.get("/vehicle/stats/dahua")
async def dahua_stats():
    """Get Dahua SDK Manager statistics"""
    if _dahua_manager is None:
        return {"status": "not_running", "cameras": {}}
    return _dahua_manager.get_stats()


@app.get("/vehicle/stats/training")
async def training_stats():
    """Get LPR Training Worker status"""
    if _training_worker is None:
        return {"status": "not_running"}
    return {
        "status": "running" if _training_worker._running else "stopped",
        "has_active_task": _training_worker._train_task is not None and not _training_worker._train_task.done() if _training_worker._train_task else False,
    }


@app.post("/vehicle/detect-advanced", response_model=DetectionResponse)
async def detect_plates_advanced(
    file: UploadFile = File(...),
    camera_id: str = Form(...),
    roi: str | None = Form(None),
    is_night: bool = Form(False),
    use_tracking: bool = Form(True),
    use_fortress: bool = Form(True),
    dedup: bool = Form(True),
):
    """
    Detect plates with optional ROI filtering and tracking.

    - ROI expects JSON list of [x, y] points (polygon).
    - Tracking assigns stable IDs across frames (per camera).
    """
    import time
    start_time = time.time()

    try:
        contents = await file.read()
        if len(contents) > _MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Image too large")
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        img_np = np.array(image)

        import cv2

        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        roi_points = _parse_roi(roi)
        plates = await _process_frame(
            img_bgr=img_bgr,
            camera_id=camera_id,
            roi_points=roi_points,
            is_night=is_night,
            use_tracking=use_tracking,
            use_fortress=use_fortress,
            dedup=dedup,
        )

        processing_time = (time.time() - start_time) * 1000

        return DetectionResponse(
            success=True,
            plates=plates,
            processing_time_ms=round(processing_time, 2),
        )
    except Exception as e:
        logger.error(f"Advanced detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
