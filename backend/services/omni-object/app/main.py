"""
omni-object: RTSP ingest, YOLO detection, ByteTrack, Redis Streams.

1. Persistent RTSP connections (CapturePool)
2. YOLO + ByteTrack global track IDs
3. Publish detections / frames to Redis Streams (prefix ``omni`` by default)
4. Serve /snap/{cam_id} for on-demand frames

Downstream: omni-vehicle, omni-human, omni-fusion (consumer groups).
"""
import asyncio
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Dict, Optional, List, Tuple
from urllib.parse import quote, urlsplit, urlunsplit

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
import httpx

from app.config import get_settings
from app.capture import CapturePool
from app.tracker import Detection, TrackerPool
from app.redis_publisher import RedisStreamPublisher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("omni-object")

# === Globals ===
_capture_pool: Optional[CapturePool] = None
_tracker_pool: Optional[TrackerPool] = None
_publisher: Optional[RedisStreamPublisher] = None
_yolo_model = None
_yolo_device: str = "cuda:0"  # actual device after load (cuda:0 or cpu)
_db_engine = None
_detection_task: Optional[asyncio.Task] = None
_cpu_executor: Optional[ThreadPoolExecutor] = None
_mediamtx_lock: Optional[asyncio.Lock] = None
_latest_detections: Dict[str, dict] = {}
_latest_detections_lock: Optional[asyncio.Lock] = None

# Per-track publish state for stationary dedup: {(cam_id, track_id): {center, last_publish_ts, still_frames}}
_track_publish_state: Dict[tuple, dict] = {}

# COCO classes that are vehicles or persons
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
PERSON_CLASS = {0: "person"}
TARGET_CLASSES = {**VEHICLE_CLASSES, **PERSON_CLASS}


def _parse_bbox_csv(value: bytes | str | None) -> Optional[list[int]]:
    if value is None:
        return None
    if isinstance(value, bytes):
        try:
            value = value.decode(errors="ignore")
        except Exception:
            return None
    parts = [part.strip() for part in str(value).split(",")]
    if len(parts) != 4:
        return None
    try:
        x1, y1, x2, y2 = [int(float(part)) for part in parts]
    except Exception:
        return None
    return [x1, y1, x2, y2]


async def _get_latest_plate_overlays(requested_ids: Optional[set[str]]) -> Dict[str, list[dict]]:
    """Read recent LPR events and extract latest plate bbox overlays per camera."""
    redis_client = getattr(_publisher, "_redis", None) if _publisher is not None else None
    if redis_client is None:
        return {}

    try:
        scan_count = max(20, int(os.getenv("LIVE_OVERLAY_LPR_SCAN_COUNT", "240")))
        max_boxes_per_camera = max(2, int(os.getenv("LIVE_OVERLAY_LPR_MAX_BOXES", "10")))
    except Exception:
        scan_count = 240
        max_boxes_per_camera = 10

    try:
        sp = get_settings().stream_prefix
        rows = await redis_client.xrevrange(f"{sp}:vehicles", count=scan_count)
    except Exception:
        return {}

    per_camera: Dict[str, list[dict]] = {}
    track_seen: Dict[str, set[int]] = {}

    for msg_id, payload in rows:
        if not isinstance(payload, dict):
            continue

        camera_raw = payload.get(b"camera_id") or payload.get("camera_id")
        if not camera_raw:
            continue
        if isinstance(camera_raw, bytes):
            camera_id = camera_raw.decode(errors="ignore")
        else:
            camera_id = str(camera_raw)

        if not camera_id:
            continue
        if requested_ids is not None and camera_id not in requested_ids:
            continue

        plate_bbox = _parse_bbox_csv(payload.get(b"plate_bbox") or payload.get("plate_bbox"))
        if not plate_bbox:
            continue

        track_raw = payload.get(b"global_track_id") or payload.get("global_track_id")
        try:
            if isinstance(track_raw, bytes):
                track_id = int(track_raw.decode(errors="ignore"))
            else:
                track_id = int(track_raw) if track_raw is not None else -1
        except Exception:
            track_id = -1

        confidence_raw = payload.get(b"plate_confidence") or payload.get("plate_confidence") or payload.get(b"confidence") or payload.get("confidence")
        try:
            if isinstance(confidence_raw, bytes):
                confidence = float(confidence_raw.decode(errors="ignore"))
            else:
                confidence = float(confidence_raw) if confidence_raw is not None else 0.0
        except Exception:
            confidence = 0.0

        timestamp_raw = payload.get(b"timestamp") or payload.get("timestamp")
        try:
            if isinstance(timestamp_raw, bytes):
                timestamp = float(timestamp_raw.decode(errors="ignore"))
            else:
                timestamp = float(timestamp_raw) if timestamp_raw is not None else 0.0
        except Exception:
            timestamp = 0.0

        per_camera.setdefault(camera_id, [])
        track_seen.setdefault(camera_id, set())
        if track_id >= 0 and track_id in track_seen[camera_id]:
            continue

        per_camera[camera_id].append({
            "track_id": track_id,
            "class_name": "plate",
            "class": "plate",
            "confidence": confidence,
            "bbox": plate_bbox,
            "timestamp": timestamp,
            "source": "omni-vehicle",
        })
        if track_id >= 0:
            track_seen[camera_id].add(track_id)

        if len(per_camera[camera_id]) >= max_boxes_per_camera:
            continue

    return per_camera


def _build_authenticated_rtsp_url(stream_url: str, username: Optional[str], password: Optional[str]) -> str:
    """Inject credentials into an RTSP URL when they are stored separately in DB."""
    raw_url = (stream_url or "").strip()
    if not raw_url or not username or not password:
        return raw_url

    try:
        parsed = urlsplit(raw_url)
        if not parsed.scheme:
            return raw_url

        host = parsed.hostname or ""
        if not host:
            return raw_url

        port = f":{parsed.port}" if parsed.port else ""
        encoded_user = quote(username, safe="")
        encoded_pass = quote(password, safe="")
        netloc = f"{encoded_user}:{encoded_pass}@{host}{port}"
        return urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))
    except Exception:
        return raw_url


def _try_load_on_device(model_path, device):
    """Load YOLO model on given device. Fallback to CPU only when CUDA fails (hạn chế dùng CPU kẻo quá tải)."""
    from ultralytics import YOLO
    model = YOLO(model_path)
    try:
        model.to(device)
        return model, device
    except Exception as dev_err:
        if device != 'cpu':
            logger.warning("⚠️ Device '%s' failed (%s), falling back to CPU (load will be capped)", device, dev_err)
            model = YOLO(model_path)
            model.to('cpu')
            return model, 'cpu'
        raise


def _load_yolo(settings):
    """Load YOLOv11m model via Ultralytics. Prefer GPU; persist to volume on first download."""
    from ultralytics import YOLO
    import shutil

    model_path = settings.yolo_model
    device = settings.device
    actual_device = device
    logger.info("📦 Loading YOLO model: %s (device=%s)", model_path, device)

    try:
        if os.path.exists(model_path):
            model, actual_device = _try_load_on_device(model_path, device)
            if actual_device != device:
                logger.warning("⚠️ Running YOLO on CPU — detection will be slower; load will be capped")
        else:
            logger.warning("Model not found at %s — auto-downloading yolo11m.pt...", model_path)
            model = YOLO("yolo11m.pt")
            ckpt_path = getattr(model, 'ckpt_path', None)
            if ckpt_path and os.path.isfile(str(ckpt_path)) and str(ckpt_path) != model_path:
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                shutil.copy2(str(ckpt_path), model_path)
                logger.info("✅ Copied model to %s for persistence", model_path)
                model, actual_device = _try_load_on_device(model_path, device)
            else:
                if not ckpt_path:
                    logger.warning("Could not determine ckpt_path; model won't persist")
                try:
                    model.to(device)
                    actual_device = device
                except Exception:
                    model.to('cpu')
                    actual_device = 'cpu'
    except Exception as primary_error:
        logger.warning("Primary YOLO load failed (%s). Trying v11 fallback weights only...", primary_error)
        # Chỉ fallback trong họ v11 (yolo11m / yolo11n), không dùng v8 để ép YOLO v11m + GPU
        fallback_models = [
            "/app/weights/yolov11m.pt",
            "/app/weights/yolo11n.pt",
            "/app/weights/yolov11n.pt",
        ]
        model = None
        for fallback_path in fallback_models:
            if not os.path.exists(fallback_path):
                continue
            try:
                model, actual_device = _try_load_on_device(fallback_path, device)
                logger.warning("⚠️ Using fallback YOLO model: %s", fallback_path)
                break
            except Exception as fallback_error:
                logger.warning("Fallback model failed (%s): %s", fallback_path, fallback_error)
        if model is None:
            raise

    # Warm up (prefer common 720p shape to reduce first real-frame latency)
    try:
        if hasattr(model, "warmup"):
            model.warmup(imgsz=(1, 3, 720, 1280))
        else:
            raise AttributeError("warmup not available")
    except Exception:
        dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
        model(dummy, verbose=False)
    logger.info("✅ YOLO model loaded and warmed up (device=%s)", actual_device)
    return model, actual_device


async def _run_cpu(fn, *args):
    loop = asyncio.get_running_loop()
    if _cpu_executor is not None:
        return await loop.run_in_executor(_cpu_executor, lambda: fn(*args))
    return await asyncio.to_thread(fn, *args)


async def _get_active_cameras(engine) -> Optional[Dict[str, str]]:
    """
    Query DB for active cameras with RTSP URLs.
    Returns: {camera_id: rtsp_url}

    When use_mediamtx_relay=True, registers each camera as a source path
    with MediaMTX and returns the relay URL instead of the raw camera URL.
    This implements Single Ingress: only MediaMTX connects to the camera,
    omni-object and HLS/WebRTC clients all consume through MediaMTX.
    """
    settings = get_settings()
    query = text("""
                                SELECT "Id"::text, "StreamUrl", "Username", "Password"
                FROM "Cameras"
                WHERE "Status" = 'Online'
                  AND "StreamUrl" IS NOT NULL
                  AND "StreamUrl" != ''
                  AND (
                    "EnableObjectDetection" = TRUE
                    OR "EnablePlateOcr" = TRUE
                    OR "EnableFaceRecognition" = TRUE
                  )
            """)

    max_attempts = max(0, int(settings.camera_query_retries))
    for attempt in range(max_attempts + 1):
        raw_cameras: Dict[str, str] = {}
        try:
            async with engine.connect() as conn:
                result = await asyncio.wait_for(
                    conn.execute(query),
                    timeout=float(settings.camera_query_timeout_s),
                )
                for row in result:
                    cam_id, rtsp_url, username, password = row
                    raw_cameras[cam_id] = _build_authenticated_rtsp_url(rtsp_url, username, password)

            if not settings.use_mediamtx_relay or not raw_cameras:
                return raw_cameras

            relay_cameras = {}
            for cam_id, raw_url in raw_cameras.items():
                # Path name MUST match web UI: http://host:8888/{cameraId}/index.m3u8
                path_name = cam_id
                relay_url = f"{settings.mediamtx_rtsp_url}/{path_name}"
                await _ensure_mediamtx_path(settings, path_name, raw_url)
                relay_cameras[cam_id] = relay_url
            return relay_cameras
        except Exception as e:
            if attempt < max_attempts:
                logger.warning(
                    "Camera query failed (attempt %d/%d): %s",
                    attempt + 1,
                    max_attempts + 1,
                    e,
                )
                await asyncio.sleep(float(settings.camera_query_retry_delay_s))
                continue
            logger.error("Failed to query cameras after %d attempts: %s", max_attempts + 1, e)
            return None


_mediamtx_registered_paths = {}

async def _ensure_mediamtx_path(settings, path_name: str, source_url: str):
    """
    Register or update a camera path in MediaMTX via its API.
    MediaMTX will pull from the source RTSP URL on demand.
    """
    global _mediamtx_lock
    if _mediamtx_lock is None:
        _mediamtx_lock = asyncio.Lock()

    async with _mediamtx_lock:
        if _mediamtx_registered_paths.get(path_name) == source_url:
            return

        api_url = f"{settings.mediamtx_api_url}/v3/config/paths/replace/{path_name}"
        payload = {
            "source": source_url,
            "sourceOnDemand": settings.mediamtx_source_on_demand,
            "sourceOnDemandStartTimeout": settings.mediamtx_source_on_demand_start_timeout,
            "sourceOnDemandCloseAfter": settings.mediamtx_source_on_demand_close_after,
        }
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(api_url, json=payload, timeout=5.0)
                if resp.status_code in (200, 204):
                    _mediamtx_registered_paths[path_name] = source_url
                    logger.debug("MediaMTX path %s registered → %s", path_name, source_url[:40])
                else:
                    logger.warning("MediaMTX path %s register failed: %s %s",
                                   path_name, resp.status_code, resp.text[:100])
        except Exception as e:
            logger.warning("MediaMTX API error for %s: %s", path_name, e)


def _run_detection(model, frame_bgr: np.ndarray, settings) -> list:
    """
    Run YOLO inference on a frame.
    Returns list of Detection objects for target classes only.
    """
    results = model(
        frame_bgr,
        conf=settings.confidence_threshold,
        iou=settings.nms_threshold,
        verbose=False,
        classes=list(TARGET_CLASSES.keys()),
    )

    frame_h, frame_w = frame_bgr.shape[:2]
    frame_area = frame_h * frame_w

    detections = []
    for r in results:
        boxes = r.boxes
        if boxes is None:
            continue
        for box in boxes:
            cls_id = int(box.cls[0])
            if cls_id not in TARGET_CLASSES:
                continue
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            conf = float(box.conf[0])
            # Skip detections that cover >85% of frame (false positive / noise)
            det_area = max(0, x2 - x1) * max(0, y2 - y1)
            if frame_area > 0 and det_area / frame_area > 0.85:
                continue
            detections.append(Detection(
                bbox=(x1, y1, x2, y2),
                confidence=conf,
                class_id=cls_id,
                class_name=TARGET_CLASSES[cls_id],
            ))
    return detections


def _run_detection_batch(model, batch_frames: List[Tuple[str, np.ndarray]], settings) -> Dict[str, List[Detection]]:
    """Run batched YOLO inference for multiple cameras in one GPU call."""
    if not batch_frames:
        return {}

    frames = [frame for _, frame in batch_frames]
    results = model(
        frames,
        conf=settings.confidence_threshold,
        iou=settings.nms_threshold,
        verbose=False,
        classes=list(TARGET_CLASSES.keys()),
    )

    detections_by_camera: Dict[str, List[Detection]] = {}
    for index, result in enumerate(results):
        cam_id = batch_frames[index][0]
        frame_bgr = batch_frames[index][1]
        frame_h, frame_w = frame_bgr.shape[:2]
        frame_area = frame_h * frame_w
        cam_detections: List[Detection] = []
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                cls_id = int(box.cls[0])
                if cls_id not in TARGET_CLASSES:
                    continue
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                conf = float(box.conf[0])
                # Skip detections that cover >85% of frame (false positive)
                det_area = max(0, x2 - x1) * max(0, y2 - y1)
                if frame_area > 0 and det_area / frame_area > 0.85:
                    continue
                cam_detections.append(Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=conf,
                    class_id=cls_id,
                    class_name=TARGET_CLASSES[cls_id],
                ))
        detections_by_camera[cam_id] = cam_detections
    return detections_by_camera


def _update_tracker_with_dt(tracker, detections: List[Detection], settings, last_ts: float, now: float):
    """Update a per-camera tracker using the actual elapsed time between frames."""
    dt = now - last_ts if last_ts > 0 else (1.0 / settings.frame_rate)
    dt = max(dt, 1e-3)
    effective_fps = 1.0 / dt

    if hasattr(tracker, '_tracker') and tracker._tracker is not None:
        orig_fr = getattr(tracker._tracker, 'frame_rate', settings.frame_rate)
        tracker._tracker.frame_rate = max(1, int(round(effective_fps)))
        tracked = tracker.update(detections)
        tracker._tracker.frame_rate = orig_fr
        return tracked

    return tracker.update(detections)


def _frame_to_jpeg(frame_bgr: np.ndarray, quality: int = 85) -> bytes:
    """Encode BGR frame to JPEG bytes. Quality 85 balances size vs OCR accuracy."""
    success, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not success or buf is None:
        raise ValueError("JPEG encode failed")
    return buf.tobytes()


async def _detection_loop():
    """
    Main detection loop:
    1. For each camera, get best frame
    2. Run YOLO detection
    3. Track with ByteTrack
    4. Publish to Redis Streams
    """
    global _capture_pool, _tracker_pool, _publisher, _yolo_model, _db_engine
    settings = get_settings()
    interval = 1.0 / settings.capture_fps  # process at capture FPS

    logger.info("🔄 Detection loop started (%.1f FPS)", settings.capture_fps)

    camera_poll_timer = 0.0
    
    batch_size = max(1, int(getattr(settings, "batch_inference_size", getattr(settings, "max_concurrent_inference", 4))))

    while True:
        try:
            loop_start = time.time()

            # Periodically sync cameras from DB
            if time.time() - camera_poll_timer > settings.camera_poll_interval:
                cameras = await _get_active_cameras(_db_engine)
                if cameras is None:
                    logger.warning("Camera sync skipped due to DB error")
                else:  # {} = all cameras removed
                    # Detect cameras about to be removed
                    current_cam_ids = set(_capture_pool.get_stats().keys())
                    desired_cam_ids = set(cameras.keys())
                    for removed_id in current_cam_ids - desired_cam_ids:
                        _tracker_pool.remove_tracker(removed_id)
                        if hasattr(_detection_loop, '_cam_last_ts'):
                            _detection_loop._cam_last_ts.pop(removed_id, None)
                        _latest_detections.pop(removed_id, None)
                        # Prune stationary dedup state for removed camera
                        stale_keys = [k for k in _track_publish_state if k[0] == removed_id]
                        for k in stale_keys:
                            del _track_publish_state[k]

                    _capture_pool.sync_cameras(cameras)
                    _publisher._shm_ring.prune(desired_cam_ids)
                    _publisher.prune_idle_timestamps(desired_cam_ids)

                    # Prune stale track publish state (tracks dropped by ByteTrack)
                    prune_cutoff = time.time() - max(30.0, settings.stationary_publish_interval_s * 6)
                    stale = [k for k, v in _track_publish_state.items() if v["last_publish_ts"] < prune_cutoff]
                    for k in stale:
                        del _track_publish_state[k]

                    logger.info("📹 Synced %d cameras", len(cameras))
                camera_poll_timer = time.time()

            def _collect_frames_sync(camera_ids: List[str]) -> List[Tuple[str, np.ndarray]]:
                collected: List[Tuple[str, np.ndarray]] = []
                for cam_id in camera_ids:
                    frame_bgr = _capture_pool.get_best_frame(cam_id)
                    if frame_bgr is not None:
                        collected.append((cam_id, frame_bgr))
                return collected

            # Per-camera last-timestamp for dt compensation
            if not hasattr(_detection_loop, '_cam_last_ts'):
                _detection_loop._cam_last_ts = {}

            stats = _capture_pool.get_stats()
            healthy_cam_ids = [cam_id for cam_id, cam_stats in stats.items() if cam_stats["healthy"]]

            # Defensive cleanup to avoid stale per-camera timestamps during partial failures
            _detection_loop._cam_last_ts = {
                cam_id: ts
                for cam_id, ts in _detection_loop._cam_last_ts.items()
                if cam_id in healthy_cam_ids
            }

            if not healthy_cam_ids:
                await asyncio.sleep(interval)
                continue

            latest_frames: Dict[str, np.ndarray] = {}
            collect_deadline = time.time() + (max(0, settings.batch_collect_window_ms) / 1000.0)
            while True:
                frames_pass = await _run_cpu(_collect_frames_sync, healthy_cam_ids)
                for cam_id, frame_bgr in frames_pass:
                    latest_frames[cam_id] = frame_bgr
                if time.time() >= collect_deadline:
                    break
                await asyncio.sleep(0.002)

            collected_frames = list(latest_frames.items())

            for batch_start in range(0, len(collected_frames), batch_size):
                batch = collected_frames[batch_start:batch_start + batch_size]
                if not batch:
                    continue

                detections_by_camera = await _run_cpu(
                    _run_detection_batch,
                    _yolo_model,
                    batch,
                    settings,
                )

                for cam_id, frame_bgr in batch:
                    tracker = _tracker_pool.get_tracker(cam_id)
                    last_ts = _detection_loop._cam_last_ts.get(cam_id, 0.0)
                    ts_now = time.time()
                    detections = detections_by_camera.get(cam_id, [])
                    tracked = _update_tracker_with_dt(tracker, detections, settings, last_ts, ts_now)
                    _detection_loop._cam_last_ts[cam_id] = ts_now

                    frame_h, frame_w = frame_bgr.shape[:2]

                    tracked_payload = [
                        {
                            "track_id": int(t.track_id),
                            "class_name": t.class_name,
                            "class_id": int(t.class_id),
                            "confidence": float(t.confidence),
                            "bbox": [int(t.bbox[0]), int(t.bbox[1]), int(t.bbox[2]), int(t.bbox[3])],
                        }
                        for t in tracked
                    ]

                    if bool(getattr(settings, "enable_live_bbox_overlay", True)):
                        if _latest_detections_lock is not None:
                            async with _latest_detections_lock:
                                _latest_detections[cam_id] = {
                                    "timestamp": ts_now,
                                    "frame_width": int(frame_w),
                                    "frame_height": int(frame_h),
                                    "detections": tracked_payload,
                                }

                    frame_jpeg = None
                    if not _publisher.can_use_shm_only():
                        frame_jpeg = await _run_cpu(_frame_to_jpeg, frame_bgr)

                    if not tracked:
                        if _publisher.should_publish_idle_frame(cam_id, ts_now):
                            await _publisher.publish_frame_only(
                                camera_id=cam_id,
                                frame_jpeg=frame_jpeg,
                                timestamp=ts_now,
                                frame_bgr=frame_bgr,
                            )
                        continue

                    frame_stream_id = await _publisher.publish_frame_only(
                        camera_id=cam_id,
                        frame_jpeg=frame_jpeg,
                        timestamp=ts_now,
                        frame_bgr=frame_bgr,
                    )

                    # Stationary vehicle dedup: only publish moving/new tracks to Redis
                    move_thresh = settings.stationary_movement_threshold
                    still_limit = settings.stationary_frame_count
                    repub_interval = settings.stationary_publish_interval_s

                    publish_dicts = []
                    for t in tracked_payload:
                        tid = t["track_id"]
                        bbox = t["bbox"]
                        cx = (bbox[0] + bbox[2]) / 2.0
                        cy = (bbox[1] + bbox[3]) / 2.0
                        key = (cam_id, tid)
                        state = _track_publish_state.get(key)

                        if state is None:
                            _track_publish_state[key] = {
                                "center": (cx, cy),
                                "last_publish_ts": ts_now,
                                "still_frames": 0,
                            }
                            publish_dicts.append(t)
                            continue

                        dx = cx - state["center"][0]
                        dy = cy - state["center"][1]
                        dist = (dx * dx + dy * dy) ** 0.5

                        if dist > move_thresh:
                            state["center"] = (cx, cy)
                            state["still_frames"] = 0
                            state["last_publish_ts"] = ts_now
                            publish_dicts.append(t)
                        else:
                            state["center"] = (cx, cy)
                            state["still_frames"] += 1
                            if state["still_frames"] < still_limit:
                                publish_dicts.append(t)
                                state["last_publish_ts"] = ts_now
                            elif (ts_now - state["last_publish_ts"]) >= repub_interval:
                                state["last_publish_ts"] = ts_now
                                publish_dicts.append(t)

                    if publish_dicts:
                        await _publisher.publish_detections(
                            camera_id=cam_id,
                            detections=publish_dicts,
                            frame_jpeg=frame_jpeg,
                            timestamp=ts_now,
                            frame_bgr=frame_bgr,
                            frame_stream_id=frame_stream_id,
                        )

            # Pace the loop
            elapsed = time.time() - loop_start
            if elapsed > interval:
                logger.warning("Detection lagging: %.3fs > %.3fs (check GPU/Threadpool load)", elapsed, interval)
            sleep_time = max(0, interval - elapsed)
            await asyncio.sleep(sleep_time)

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.exception("Detection loop error: %s", e)
            await asyncio.sleep(2)

    logger.info("Detection loop stopped")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _capture_pool, _tracker_pool, _publisher, _yolo_model, _yolo_device, _db_engine, _detection_task, _cpu_executor, _mediamtx_lock, _latest_detections_lock
    settings = get_settings()
    _cpu_executor = ThreadPoolExecutor(max_workers=max(4, int(settings.cpu_worker_threads)))
    _mediamtx_lock = asyncio.Lock()
    _latest_detections_lock = asyncio.Lock()


    logger.info("=" * 60)
    logger.info("🚀 omni-object starting (RTSP + detection)...")
    logger.info("   Model: %s", settings.yolo_model)
    logger.info("   Device: %s", settings.device)
    logger.info("   Capture FPS: %d", settings.capture_fps)
    logger.info("   Redis: %s", settings.redis_streams_url)
    logger.info("=" * 60)

    # 0. Verify model integrity (crash if wrong model deployed)
    try:
        from shared.model_integrity import verify_model
        if os.path.exists(settings.yolo_model):
            await _run_cpu(
                verify_model, settings.yolo_model, strict=True
            )
    except SystemExit:
        raise
    except Exception as e:
        logger.warning("⚠️ Model integrity check skipped: %s", e)

    # 1. Database engine
    _db_engine = create_async_engine(settings.database_url)

    # 2. Load YOLO model — run in thread to avoid blocking the event loop
    #    (model loading can take 30-60s on cold start, blocking async lifespan)
    try:
        _yolo_model, _yolo_device = await _run_cpu(_load_yolo, settings)
        # Hạn chế fallback CPU: khi chạy trên CPU thì giới hạn FPS và concurrency để tránh quá tải
        if _yolo_device == 'cpu':
            old_fps, old_conc = settings.capture_fps, settings.max_concurrent_inference
            settings.capture_fps = min(settings.capture_fps, 4)
            settings.max_concurrent_inference = min(settings.max_concurrent_inference, 2)
            logger.warning(
                "⚠️ CPU mode: capping capture_fps %d→%d, max_concurrent_inference %d→%d to avoid overload",
                old_fps, settings.capture_fps, old_conc, settings.max_concurrent_inference,
            )
    except Exception as e:
        logger.error("❌ YOLO model load failed: %s", e)
        _yolo_model = None
        _yolo_device = "none"

    # 3. Initialize capture pool
    _capture_pool = CapturePool(settings)

    # 4. Initialize tracker pool
    _tracker_pool = TrackerPool(settings)

    # 5. Connect Redis publisher (SHM capacity auto-checked at init)
    _publisher = RedisStreamPublisher(settings)
    await _publisher.connect()
    await _publisher.ensure_consumer_groups()
    if settings.shm_enabled:
        shm_ok = _publisher._shm_ring._space_ok
        logger.info(
            "   SHM: %s (dir=%s, slots=%d/cam)",
            "✅ enabled" if shm_ok else "⚠️ disabled (low space, fallback to JPEG-over-Redis)",
            settings.shm_dir, settings.shm_slots_per_camera,
        )

    # 6. Start detection loop
    if _yolo_model:
        _detection_task = asyncio.create_task(_detection_loop())
        logger.info("🟢 omni-object ready — detection loop running.")
    else:
        logger.warning("🟡 omni-object ready (DEGRADED — no YOLO model)")

    yield

    # Shutdown
    logger.info("🔴 omni-object shutting down...")
    if _detection_task:
        _detection_task.cancel()
        try:
            await _detection_task
        except asyncio.CancelledError:
            pass
    _capture_pool.stop_all()
    _tracker_pool.clear()
    await _publisher.disconnect()
    await _db_engine.dispose()
    if _cpu_executor:
        _cpu_executor.shutdown(wait=False)
        _cpu_executor = None
    logger.info("👋 omni-object stopped.")


app = FastAPI(
    title="omni-object Service",
    description="RTSP Flow Master - YOLOv11m + ByteTrack + Redis Streams",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/rtsp/health")
async def health():
    yolo_ok = _yolo_model is not None
    redis_ok = _publisher.is_connected if _publisher else False
    cameras = len(_capture_pool.get_stats()) if _capture_pool else 0
    
    if yolo_ok and redis_ok:
        status = "healthy"
    elif redis_ok and not yolo_ok:
        # Service is UP and reachable, just no YOLO model loaded yet.
        # "degraded" tells frontend to show "Trực tuyến" (not "Ngoại tuyến")
        # while still indicating reduced capability.
        status = "degraded"
    else:
        status = "unhealthy"  # Redis dead — cannot function at all
    
    return {
        "status": status,
        "service": "omni-object",
        "version": "1.0.0",
        "yolo_loaded": yolo_ok,
        "yolo_model": get_settings().yolo_model,
        "device": _yolo_device,
        "cameras_active": cameras,
        "redis_connected": redis_ok,
    }


@app.get("/rtsp/snap/{camera_id}")
async def snapshot(camera_id: str):
    """
    Get the latest best-quality frame for a camera as JPEG.
    This is the primary API for downstream services to consume frames
    without managing their own RTSP connections.
    """
    if not _capture_pool:
        raise HTTPException(status_code=503, detail="Capture pool not ready")

    frame = await _run_cpu(_capture_pool.get_best_frame, camera_id)
    if frame is None:
        # Try Redis cache fallback
        if _publisher:
            cached = await _publisher.get_latest_frame(camera_id)
            if cached:
                return Response(content=cached, media_type="image/jpeg")
        raise HTTPException(status_code=404, detail=f"No frame for camera {camera_id[:8]}")

    jpeg = await _run_cpu(_frame_to_jpeg, frame, 90)
    return Response(content=jpeg, media_type="image/jpeg")


@app.get("/rtsp/stats")
async def stats():
    """Get capture pool and detection stats."""
    capture_stats = _capture_pool.get_stats() if _capture_pool else {}
    return {
        "cameras": capture_stats,
        "total_cameras": len(capture_stats),
        "healthy_cameras": sum(1 for v in capture_stats.values() if v["healthy"]),
        "yolo_loaded": _yolo_model is not None,
        "device": _yolo_device,
        "redis_connected": _publisher.is_connected if _publisher else False,
    }


@app.get("/rtsp/cameras")
async def list_cameras():
    """List all cameras in capture pool."""
    if not _capture_pool:
        return {"cameras": []}
    stats = _capture_pool.get_stats()
    return {
        "cameras": [
            {"camera_id": cam_id, **cam_stats}
            for cam_id, cam_stats in stats.items()
        ]
    }


@app.get("/rtsp/detections/latest")
async def latest_detections(cameraIds: str | None = None):
    """Return latest tracked detections per camera for realtime UI overlays."""
    try:
        if not bool(getattr(get_settings(), "enable_live_bbox_overlay", True)):
            return {"items": {}}
        requested_ids: Optional[set[str]] = None
        if cameraIds:
            requested_ids = {part.strip() for part in cameraIds.split(',') if part.strip()}

        if _latest_detections_lock is None:
            return {"items": {}}

        async with _latest_detections_lock:
            source = dict(_latest_detections)

        lpr_plate_items = await _get_latest_plate_overlays(requested_ids)

        items: Dict[str, dict] = {}
        all_camera_ids = set(source.keys()) | set(lpr_plate_items.keys())

        for cam_id in all_camera_ids:
            if requested_ids is not None and cam_id not in requested_ids:
                continue

            payload = source.get(cam_id) or {}
            frame_w = max(1, int(payload.get("frame_width") or 1))
            frame_h = max(1, int(payload.get("frame_height") or 1))
            detections = payload.get("detections") or []

            normalized: list[dict] = []
            for det in detections:
                bbox = det.get("bbox") or []
                if len(bbox) != 4:
                    continue
                x1, y1, x2, y2 = [int(v) for v in bbox]
                class_name = str(det.get("class_name", "unknown"))
                confidence = float(det.get("confidence", 0.0))
                normalized.append({
                    "track_id": int(det.get("track_id", -1)),
                    "class_name": class_name,
                    "class": class_name,
                    "confidence": confidence,
                    "bbox": [x1, y1, x2, y2],
                    "bbox_norm": [
                        max(0.0, min(1.0, x1 / frame_w)),
                        max(0.0, min(1.0, y1 / frame_h)),
                        max(0.0, min(1.0, x2 / frame_w)),
                        max(0.0, min(1.0, y2 / frame_h)),
                    ],
                    "source": "omni-object",
                })

            for plate_det in lpr_plate_items.get(cam_id, []):
                bbox = plate_det.get("bbox") or []
                if len(bbox) != 4:
                    continue
                x1, y1, x2, y2 = [int(v) for v in bbox]
                normalized.append({
                    "track_id": int(plate_det.get("track_id", -1)),
                    "class_name": "plate",
                    "class": "plate",
                    "confidence": float(plate_det.get("confidence", 0.0)),
                    "bbox": [x1, y1, x2, y2],
                    "bbox_norm": [
                        max(0.0, min(1.0, x1 / frame_w)),
                        max(0.0, min(1.0, y1 / frame_h)),
                        max(0.0, min(1.0, x2 / frame_w)),
                        max(0.0, min(1.0, y2 / frame_h)),
                    ],
                    "source": "omni-vehicle",
                })

            timestamp = float(payload.get("timestamp") or 0)
            if timestamp <= 0 and lpr_plate_items.get(cam_id):
                timestamps = [float(det.get("timestamp") or 0) for det in lpr_plate_items[cam_id]]
                timestamp = max(timestamps) if timestamps else 0.0

            boxes = [
                {
                    "class": det.get("class") or det.get("class_name") or "unknown",
                    "bbox": det.get("bbox"),
                    "confidence": det.get("confidence", 0.0),
                    "track_id": det.get("track_id", -1),
                }
                for det in normalized
                if isinstance(det.get("bbox"), list)
            ]

            items[cam_id] = {
                "timestamp": timestamp,
                "frame_width": frame_w,
                "frame_height": frame_h,
                "detections": normalized,
                "boxes": boxes,
            }

        return {"items": items}
    except Exception as e:
        logger.warning("latest_detections error (returning empty): %s", e)
        return {"items": {}}


@app.get("/rtsp/settings")
async def get_rtsp_settings():
    """Get current RTSP Ingress configuration (excluding sensitive infra fields)"""
    # Exclude database credentials and internal URLs from API response
    _SENSITIVE_FIELDS = {"database_url", "redis_streams_url"}
    return get_settings().model_dump(exclude=_SENSITIVE_FIELDS)


@app.get("/rtsp/settings/meta")
async def get_rtsp_settings_meta():
    """Get field descriptions, types, validation ranges, and tiers for UI rendering"""
    return get_settings().get_field_descriptions()


@app.post("/rtsp/settings")
async def update_rtsp_settings(new_settings: dict):
    """Update and persist RTSP Ingress configuration.
    Only mutable fields are accepted; infra fields (database_url, device, etc.)
    require a container restart and are intentionally excluded.
    """
    settings = get_settings()

    # Fields that should NOT be changed at runtime (need container restart)
    _READONLY_FIELDS = {
        "database_url", "redis_streams_url", "stream_prefix",
        "shm_dir", "mediamtx_rtsp_url", "mediamtx_api_url",
        "device", "yolo_model",
    }

    errors: list[str] = []
    applied: dict[str, object] = {}

    for key, value in new_settings.items():
        if key in _READONLY_FIELDS:
            errors.append(f"{key}: read-only, requires container restart")
            continue
        if not hasattr(settings, key):
            errors.append(f"{key}: unknown setting")
            continue

        valid = True
        field_info = settings.model_fields.get(key)
        if field_info:
            for m in field_info.metadata:
                if hasattr(m, 'ge') and isinstance(value, (int, float)) and value < m.ge:
                    errors.append(f"{key}: must be ≥ {m.ge}, got {value}")
                    valid = False
                    break
                if hasattr(m, 'le') and isinstance(value, (int, float)) and value > m.le:
                    errors.append(f"{key}: must be ≤ {m.le}, got {value}")
                    valid = False
                    break

        if not valid:
            continue

        setattr(settings, key, value)
        applied[key] = value

    settings.save_to_file()

    return {
        "success": len(errors) == 0,
        "applied": applied,
        "errors": errors,
        "settings": settings.model_dump(),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8555)
