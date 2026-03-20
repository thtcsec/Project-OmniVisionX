"""
Redis stream consumer for omni-human.

Listens to ``{prefix}:detections`` (omni-object), publishes to ``{prefix}:humans``.
Fetches frames from ``{prefix}:frames:{camera_id}``.

Improvements:
  - Track-level dedup: same track_id only processed once per cooldown window
  - Upscaling small face crops for better InsightFace recognition
  - Adaptive person crop padding (head is above body bbox)
  - Min crop size validation before running detection
  - Robust bounds checking on face crop coordinates
  - Configurable stale event threshold
"""
import asyncio
import logging
import os
import time
import uuid
from datetime import datetime
from typing import Optional, Dict, Tuple

import cv2
import numpy as np
import redis.asyncio as aioredis

from app.config import get_settings

logger = logging.getLogger("omni-human.stream_consumer")


def _stream_prefix() -> str:
    return getattr(get_settings(), "stream_prefix", "omni")


def _consumer_group() -> str:
    return getattr(get_settings(), "redis_consumer_group", "omni-human-group")


def _detections_stream() -> str:
    return f"{_stream_prefix()}:detections"


def _frames_stream(camera_id: str) -> str:
    return f"{_stream_prefix()}:frames:{camera_id}"


def _humans_output_stream() -> str:
    return f"{_stream_prefix()}:humans"

# ── Constants ──
INSIGHTFACE_MIN_SIZE = 112      # InsightFace expects at least 112x112 for good results


class HumanStreamConsumer:
    """
    Consumes person detections from Redis Stream, runs face detection +
    embedding, and publishes enriched face events back to Redis.
    """

    def __init__(self, face_detector=None, face_recognizer=None, db_engine=None):
        self._face_detector = face_detector
        self._face_recognizer = face_recognizer
        self._db_engine = db_engine
        self._redis: Optional[aioredis.Redis] = None
        self._connected = False
        self._running = False
        self._task: Optional[asyncio.Task] = None

        # Track dedup: {(camera_id, track_id): last_processed_timestamp}
        self._track_last_seen: Dict[Tuple[str, int], float] = {}
        self._dedup_cleanup_counter = 0

        self._stats = {
            "events_consumed": 0,
            "events_skipped_dedup": 0,
            "events_skipped_stale": 0,
            "events_skipped_no_frame": 0,
            "events_skipped_small_crop": 0,
            "events_skipped_low_conf": 0,
            "faces_detected": 0,
            "faces_identified": 0,
            "faces_upscaled": 0,
            "frames_dropped": 0,
            "errors": 0,
        }

    @property
    def stats(self):
        return self._stats

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def connect(self):
        settings = get_settings()
        redis_url = os.getenv("REDIS_STREAMS_URL", settings.redis_streams_url)
        try:
            self._redis = aioredis.from_url(redis_url, decode_responses=False)
            await self._redis.ping()
            self._connected = True
            logger.info("✅ omni-human connected to Redis Streams")
        except Exception as e:
            logger.warning("⚠️ Redis Streams unavailable: %s (omni-human will use API-only mode)", e)
            self._connected = False

    async def disconnect(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._redis:
            await self._redis.aclose()
            self._connected = False

    async def ensure_group(self):
        if not self._connected:
            return
        try:
            await self._redis.xgroup_create(
                _detections_stream(), _consumer_group(), id="$", mkstream=True
            )
        except aioredis.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                logger.warning("Group creation error: %s", e)

    async def start(self):
        if not self._connected or self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._consume_loop())
        _s = get_settings()
        logger.info("🔄 omni-human stream consumer started (dedup=%ds, stale=%ds)",
                     _s.track_dedup_cooldown, _s.stale_event_threshold)

    async def _consume_loop(self):
        """Main consumer loop: listen for person detections, run face recognition."""
        consumer_name = f"frs-worker-{os.getpid()}"
        _log_timer = time.time()

        while self._running:
            try:
                results = await self._redis.xreadgroup(
                    _consumer_group(), consumer_name,
                    streams={_detections_stream(): ">"},
                    count=20,
                    block=1000,
                )

                if not results:
                    continue

                for stream_name, messages in results:
                    for msg_id, data in messages:
                        try:
                            await self._process_detection(data)
                            self._stats["events_consumed"] += 1
                            # ACK only after successful processing
                            await self._redis.xack(stream_name, _consumer_group(), msg_id)
                        except Exception as e:
                            logger.warning("Process error (msg %s will retry from PEL): %s", msg_id, e, exc_info=True)
                            self._stats["errors"] += 1
                            # Do NOT ACK — message stays in PEL for retry

                # Periodic dedup cache cleanup (every 100 events)
                self._dedup_cleanup_counter += 1
                if self._dedup_cleanup_counter >= 100:
                    self._cleanup_dedup_cache()
                    self._dedup_cleanup_counter = 0

                # Periodic stats logging (every 60s)
                if time.time() - _log_timer > 60:
                    logger.info(
                        "📊 omni-human stats: consumed=%d faces=%d identified=%d "
                        "dedup=%d stale=%d low_conf=%d no_frame=%d small=%d errors=%d",
                        self._stats["events_consumed"],
                        self._stats["faces_detected"],
                        self._stats["faces_identified"],
                        self._stats["events_skipped_dedup"],
                        self._stats["events_skipped_stale"],
                        self._stats["events_skipped_low_conf"],
                        self._stats["events_skipped_no_frame"],
                        self._stats["events_skipped_small_crop"],
                        self._stats["errors"],
                    )
                    _log_timer = time.time()

            except asyncio.CancelledError:
                break
            except aioredis.ConnectionError as e:
                logger.error("Redis connection lost: %s — reconnecting in 3s", e)
                self._connected = False
                await asyncio.sleep(3)
                try:
                    await self.connect()
                    if self._connected:
                        await self.ensure_group()
                except Exception as re:
                    logger.warning("omni-human reconnect failed: %s", re)
            except aioredis.ResponseError as e:
                # NOGROUP: stream was wiped or consumer group was dropped
                if "NOGROUP" in str(e):
                    logger.warning("Consumer group gone (NOGROUP) — recreating group and retrying in 5s...")
                    await asyncio.sleep(5)
                    try:
                        await self.ensure_group()
                    except Exception as eg:
                        logger.warning("ensure_group failed: %s", eg)
                else:
                    logger.exception("Redis ResponseError in omni-human consumer: %s", e)
                    await asyncio.sleep(2)
            except Exception as e:
                logger.exception("Consumer loop error: %s", e)
                await asyncio.sleep(2)

    def _cleanup_dedup_cache(self):
        """Remove expired entries from track dedup cache to prevent memory leak."""
        now = time.time()
        # Use at least 5s TTL to avoid premature eviction of spatial dedup keys
        # (spatial cooldown is 1.5s, while track_dedup_cooldown could be as low as 0.3s)
        ttl = max(get_settings().track_dedup_cooldown * 3, 5.0)
        expired = [k for k, v in self._track_last_seen.items() if now - v > ttl]
        for k in expired:
            del self._track_last_seen[k]
        if expired:
            logger.debug("🧹 Cleaned %d expired dedup entries (ttl=%.0fs)", len(expired), ttl)

    async def _process_detection(self, data: dict):
        """Process a single person detection event."""
        class_name = data.get(b"class_name", b"").decode()

        # Only process persons
        if class_name != "person":
            return

        camera_id = data.get(b"camera_id", b"").decode()
        track_id = int(data.get(b"global_track_id", b"-1").decode())
        bbox_str = data.get(b"bbox", b"0,0,0,0").decode()
        confidence = float(data.get(b"confidence", b"0").decode())
        timestamp = float(data.get(b"timestamp", b"0").decode())

        # ── Parse bbox EARLY (needed by spatial dedup) ──
        parts = bbox_str.split(",")
        if len(parts) != 4:
            return
        x1, y1, x2, y2 = [int(float(p)) for p in parts]

        # ── Guard 1: Drop stale events ──
        settings = get_settings()
        age = time.time() - timestamp
        if age > settings.stale_event_threshold:
            self._stats["events_skipped_stale"] += 1
            if self._stats["events_skipped_stale"] % 50 == 1:
                logger.warning("⏰ Stale event (age=%.1fs > %.1fs): cam=%s track=%d — omni-human may be falling behind",
                               age, settings.stale_event_threshold, camera_id[:8], track_id)
            return

        # ── Guard 2: Track-level dedup ──
        if track_id != -1:
            dedup_key = (camera_id, track_id)
            last_seen = self._track_last_seen.get(dedup_key, 0)
            if time.time() - last_seen < settings.track_dedup_cooldown:
                self._stats["events_skipped_dedup"] += 1
                return
            self._track_last_seen[dedup_key] = time.time()
        else:
            # Spatial deduplication for untracked detections to prevent 30fps processing
            # Use actual center point with 50px grid (200 was too coarse — 100px cells
            # caused nearby people to hash to the same cell and get silently dropped)
            cx = ((x1 + x2) // 2) // 50
            cy = ((y1 + y2) // 2) // 50
            dedup_key = (camera_id, "spatial", cx, cy)
            last_seen = self._track_last_seen.get(dedup_key, 0)
            if time.time() - last_seen < 1.5:  # Shorter cooldown for spatial grid
                self._stats["events_skipped_dedup"] += 1
                return
            self._track_last_seen[dedup_key] = time.time()

        # ── Guard 3: Low confidence filter ──
        if confidence < settings.human_confidence_threshold:
            self._stats["events_skipped_low_conf"] += 1
            return

        # Fetch frame (match by exact stream ID if available, else by timestamp)
        # omni-object publishes frame_stream_id for O(1) exact-ID lookup
        frame_stream_id = data.get(b"frame_stream_id", b"").decode() or None
        frame_jpeg = await self._get_frame(camera_id, detection_ts=timestamp, frame_stream_id=frame_stream_id)
        if frame_jpeg is None:
            self._stats["events_skipped_no_frame"] += 1
            if self._stats["events_skipped_no_frame"] % 20 == 1:
                logger.warning("🖼️ No frame for camera %s (track=%d) — frames stream empty or JPEG missing",
                               camera_id[:8], track_id)
            return

        # Decode
        nparr = np.frombuffer(frame_jpeg, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return

        h, w = img_bgr.shape[:2]

        # ── Improved person crop padding ──
        # Head is above body bbox — need generous top padding
        # Side padding proportional to person width for better face capture
        person_h = y2 - y1
        person_w = x2 - x1
        pad_top = int(person_h * 0.35)       # 35% top (was 20% — too little for head)
        pad_bottom = int(person_h * 0.05)     # 5% bottom
        pad_side = max(20, int(person_w * 0.15))  # 15% sides, min 20px

        crop_x1 = max(0, x1 - pad_side)
        crop_y1 = max(0, y1 - pad_top)
        crop_x2 = min(w, x2 + pad_side)
        crop_y2 = min(h, y2 + pad_bottom)
        person_crop = img_bgr[crop_y1:crop_y2, crop_x1:crop_x2]

        # ── Guard 4: Min crop size ──
        if person_crop.size == 0 or person_crop.shape[0] < settings.min_person_crop_px or person_crop.shape[1] < settings.min_person_crop_px:
            self._stats["events_skipped_small_crop"] += 1
            return

        face_identity = "Unknown"
        face_confidence = 0.0
        face_embedding_id = None
        face_crop = None

        try:
            # InsightFace expects BGR (OpenCV convention) — do NOT convert to RGB
            img_for_insight = person_crop

            # ── Strategy: InsightFace preferred (detect + align + embed in one pass) ──
            if self._face_recognizer:
                # If crop is too small for InsightFace, upscale it first
                crop_h, crop_w = img_for_insight.shape[:2]
                scale_factor = 1.0
                if min(crop_h, crop_w) < INSIGHTFACE_MIN_SIZE:
                    scale_factor = INSIGHTFACE_MIN_SIZE / min(crop_h, crop_w)
                    new_w = int(crop_w * scale_factor)
                    new_h = int(crop_h * scale_factor)
                    img_for_insight = cv2.resize(img_for_insight, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                    self._stats["faces_upscaled"] += 1

                feats = await asyncio.to_thread(self._face_recognizer.extract_embedding, img_for_insight)
                if feats:
                    self._stats["faces_detected"] += 1
                    best_feat = max(
                        feats,
                        key=lambda x: (x["bbox"][2] - x["bbox"][0]) * (x["bbox"][3] - x["bbox"][1]),
                    )
                    embedding = best_feat["embedding"]
                    face_confidence = best_feat.get("score", best_feat.get("det_score", best_feat.get("confidence", 0.0)))

                    # Robust face crop with bounds checking
                    fx1, fy1, fx2, fy2 = [int(v) for v in best_feat["bbox"]]
                    # If we upscaled, convert coords back to original crop space
                    if scale_factor > 1.0:
                        fx1 = int(fx1 / scale_factor)
                        fy1 = int(fy1 / scale_factor)
                        fx2 = int(fx2 / scale_factor)
                        fy2 = int(fy2 / scale_factor)

                    # Clamp face coords to crop bounds
                    orig_h, orig_w = person_crop.shape[:2]
                    fx1 = max(0, min(fx1, orig_w - 1))
                    fy1 = max(0, min(fy1, orig_h - 1))
                    fx2 = max(fx1 + 1, min(fx2, orig_w))
                    fy2 = max(fy1 + 1, min(fy2, orig_h))

                    face_crop = person_crop[fy1:fy2, fx1:fx2]

                    # Discard tiny face crops
                    if face_crop.size == 0 or face_crop.shape[0] < settings.min_face_crop_px or face_crop.shape[1] < settings.min_face_crop_px:
                        face_crop = person_crop  # Fallback to whole person crop

                    # Search DB for face match
                    if self._db_engine:
                        identity, fid, is_known = await self._search_face_db(embedding)
                        if is_known:
                            face_identity = identity
                            face_embedding_id = fid
                            self._stats["faces_identified"] += 1

            # ── Fallback: YOLOv8-face (detection only, no embedding) ──
            elif self._face_detector:
                detections = await asyncio.to_thread(self._face_detector.detect, person_crop)
                if detections:
                    best = max(detections, key=lambda d: d["conf"])
                    self._stats["faces_detected"] += 1
                    face_confidence = float(best["conf"])
                    fx1, fy1, fx2, fy2 = [int(v) for v in best["box"]]

                    # Bounds check
                    ph, pw = person_crop.shape[:2]
                    fx1 = max(0, min(fx1, pw - 1))
                    fy1 = max(0, min(fy1, ph - 1))
                    fx2 = max(fx1 + 1, min(fx2, pw))
                    fy2 = max(fy1 + 1, min(fy2, ph))

                    face_crop = person_crop[fy1:fy2, fx1:fx2]
                    if face_crop.size == 0 or face_crop.shape[0] < settings.min_face_crop_px:
                        face_crop = person_crop

        except Exception as e:
            logger.warning("Face recognition error for track %d: %s", track_id, e)
            self._stats["errors"] += 1
            # Don't return — still publish a person event even if face extraction failed

        # ── Publish to humans stream ──
        # Publish if: face was detected OR high-confidence person detection
        face_found = face_confidence > 0.0
        high_conf_person = confidence > settings.person_publish_conf

        if face_found or high_conf_person:
            # Save face/person crop thumbnail (in thread to avoid blocking event loop)
            face_img = face_crop if (face_crop is not None and face_crop.size > 0) else person_crop
            face_crop_path = await asyncio.to_thread(
                self._save_face_thumbnail, face_img, camera_id, face_identity
            )
            await self._publish_face_event(
                camera_id=camera_id,
                track_id=track_id,
                bbox=(x1, y1, x2, y2),
                face_identity=face_identity,
                face_confidence=face_confidence,
                person_confidence=confidence,  # original person conf from omni-object
                face_embedding_id=face_embedding_id,
                timestamp=timestamp,
                face_crop_path=face_crop_path,
            )
            logger.info(
                "📸 Published face event: cam=%s track=%d identity=%s face_conf=%.2f person_conf=%.2f",
                camera_id[:8], track_id, face_identity, face_confidence, confidence,
            )
        else:
            logger.debug(
                "⏭️ Skipped person (no face & low person conf=%.2f < %.2f): cam=%s track=%d",
                confidence, settings.person_publish_conf, camera_id[:8], track_id,
            )

    def _save_face_thumbnail(
        self,
        face_crop: np.ndarray,
        camera_id: str,
        face_identity: str,
    ) -> str:
        """Save face crop to thumbnails directory. Returns filename."""
        try:
            thumb_dir = get_settings().thumbnail_path
            os.makedirs(thumb_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_name = "".join(c for c in face_identity if c.isalnum()) or "UNKNOWN"
            uid = uuid.uuid4().hex[:6]
            filename = f"{ts}_{camera_id[:8]}_{safe_name}_{uid}_face.jpg"
            filepath = os.path.join(thumb_dir, filename)

            # Resize large crops to save disk space (max 200px on longest side)
            h, w = face_crop.shape[:2]
            if max(h, w) > 200:
                scale = 200.0 / max(h, w)
                face_crop = cv2.resize(face_crop, (int(w * scale), int(h * scale)),
                                       interpolation=cv2.INTER_AREA)

            cv2.imwrite(filepath, face_crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
            return filename
        except Exception as e:
            logger.warning("Failed to save face thumbnail: %s", e)
            return ""

    async def _search_face_db(self, embedding: list) -> tuple:
        """Search pgvector for face match. Returns (name, face_id, is_known).
        
        Strategy: prefer known faces first, then fallback to all faces.
        This prevents unknown DB entries from shadowing actual known matches.
        """
        try:
            from sqlalchemy import text
            # pgvector expects '[0.1,0.2,...]' format without spaces
            emb_str = '[' + ','.join(f'{v:.6f}' for v in embedding) + ']'
            _threshold = get_settings().face_db_match_distance

            async with self._db_engine.connect() as conn:
                # 1. Try known faces first (higher priority)
                result = await conn.execute(text("""
                    SELECT "Id", "Name", "IsKnown",
                           "Embedding" <=> CAST(:emb AS vector) AS distance
                    FROM "Faces"
                    WHERE "IsDeleted" = FALSE AND "IsKnown" = TRUE
                    ORDER BY distance ASC
                    LIMIT 1
                """), {"emb": emb_str})
                row = result.fetchone()
                if row:
                    db_id, db_name, db_known, dist = row
                    if dist < _threshold:
                        logger.debug("🎯 Face matched (known): %s (dist=%.3f)", db_name, dist)
                        return (db_name or "Unknown", str(db_id), True)

                # 2. Fallback: search all faces (for re-ID of previously seen unknowns)
                result = await conn.execute(text("""
                    SELECT "Id", "Name", "IsKnown",
                           "Embedding" <=> CAST(:emb AS vector) AS distance
                    FROM "Faces"
                    WHERE "IsDeleted" = FALSE
                    ORDER BY distance ASC
                    LIMIT 1
                """), {"emb": emb_str})
                row = result.fetchone()
                if row:
                    db_id, db_name, db_known, dist = row
                    if dist < _threshold:
                        logger.debug("🔍 Face matched (any): %s known=%s (dist=%.3f)", db_name, db_known, dist)
                        return (db_name or "Unknown", str(db_id), db_known)
                    else:
                        logger.debug("❓ No match (best dist=%.3f > threshold %.2f)", dist, _threshold)
        except Exception as e:
            logger.warning("Face DB search error: %s", e)
        return ("Unknown", None, False)

    async def _get_frame(self, camera_id: str, detection_ts: float = 0.0,
                         frame_stream_id: Optional[str] = None) -> Optional[bytes]:
        """Fetch best-matching frame for camera from Redis.

        Strategy:
        1. If frame_stream_id is known (omni-object attaches it), do an O(1) exact-ID
           XRANGE lookup — fast and precise, avoids any temporal mismatch.
        2. Otherwise read last 5 entries and pick the one closest to detection_ts.
           Accept the frame as long as delta < stale_event_threshold (configurable,
           default 10 s) — previously this was hardcoded to 2.0 s which caused ALL
           events to be dropped whenever omni-human fell > 2 s behind.
        3. Fallback: use the newest frame if temporal matching fails.

        Args:
            camera_id: Camera UUID
            detection_ts: Detection event timestamp for temporal matching
            frame_stream_id: Redis stream message ID of the co-published frame
        """
        try:
            # ── Fast path: exact-ID lookup ──
            if frame_stream_id:
                try:
                    exact = await self._redis.xrange(
                        _frames_stream(camera_id),
                        min=frame_stream_id, max=frame_stream_id,
                        count=1,
                    )
                    if exact:
                        _, data = exact[0]
                        frame_jpeg = data.get(b"frame_jpeg")
                        if frame_jpeg:
                            return frame_jpeg
                except Exception:
                    pass  # Fall through to scan

            entries = await self._redis.xrevrange(
                _frames_stream(camera_id), count=5
            )
            if not entries:
                return None

            settings = get_settings()

            # Pick the frame closest to detection timestamp
            if detection_ts > 0 and len(entries) > 1:
                best_entry = None
                best_delta = float("inf")
                for _, fdata in entries:
                    try:
                        frame_ts = float(fdata.get(b"timestamp", b"0").decode())
                    except (ValueError, AttributeError):
                        frame_ts = 0.0
                    delta = abs(frame_ts - detection_ts)
                    if delta < best_delta:
                        best_delta = delta
                        best_entry = fdata
                # Accept any frame within stale_event_threshold (was hardcoded 2.0s
                # — too tight; omni-human can fall behind under load causing all frames to fail).
                if best_entry is None or best_delta > settings.stale_event_threshold:
                    return None
                data = best_entry
            else:
                _, data = entries[0]

            # Prefer JPEG (works across Docker containers)
            frame_jpeg = data.get(b"frame_jpeg")
            if frame_jpeg:
                return frame_jpeg

            # Fallback: read from shared memory ring buffer
            shm_path = data.get(b"shm_path")
            if shm_path:
                shm_path_str = shm_path.decode()
                shm_shape = data.get(b"shm_shape", b"").decode()
                shm_dtype = data.get(b"shm_dtype", b"uint8").decode()
                shm_index = int(data.get(b"shm_index", b"0").decode())
                if shm_path_str and shm_shape and os.path.exists(shm_path_str):
                    try:
                        parts = [int(x) for x in shm_shape.split(",") if x]
                        if len(parts) == 3:
                            h, w, c = parts
                            itemsize = int(np.dtype(shm_dtype).itemsize)
                            frame_bytes = h * w * c * itemsize
                            offset = shm_index * frame_bytes
                            frame_view = np.memmap(
                                shm_path_str, dtype=shm_dtype, mode='r',
                                offset=offset, shape=(h, w, c),
                            )
                            # MUST copy: omni-object can overwrite this SHM slot at any time
                            frame_bgr = np.array(frame_view)
                            _, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
                            if buf is not None:
                                return buf.tobytes()
                    except Exception as e:
                        logger.debug("SHM read failed for %s: %s", camera_id[:8], e)

            logger.debug("No frame data available for camera %s", camera_id[:8])
        except Exception as e:
            logger.warning("Frame fetch failed for %s: %s", camera_id[:8], e)
        return None

    async def _publish_face_event(
        self,
        camera_id: str,
        track_id: int,
        bbox: tuple,
        face_identity: str,
        face_confidence: float,
        person_confidence: float,
        face_embedding_id: Optional[str],
        timestamp: float,
        face_crop_path: str = "",
    ):
        if not self._connected:
            return
        try:
            entry = {
                b"camera_id": camera_id.encode(),
                b"global_track_id": str(track_id).encode(),
                b"class_name": b"person",
                b"bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}".encode(),
                b"confidence": f"{person_confidence:.3f}".encode(),
                b"timestamp": f"{timestamp:.3f}".encode(),
                b"face_identity": face_identity.encode(),
                b"face_confidence": f"{face_confidence:.3f}".encode(),
                b"face_embedding_id": (face_embedding_id or "").encode(),
                b"face_crop_path": face_crop_path.encode(),
            }
            await self._redis.xadd(_humans_output_stream(), entry, maxlen=500, approximate=True)
        except Exception as e:
            logger.warning("Failed to publish frs_event: %s", e)
