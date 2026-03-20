"""
Redis Stream Publisher for omni-object.
Stream layout (prefix defaults to ``omni``):
  {prefix}:detections — tracked objects (vehicle/person)
  {prefix}:frames:{camera_id} — latest frames for consumers / snap API
"""
import asyncio
import logging
import os
from typing import List, Optional

import cv2
import redis.asyncio as aioredis
import numpy as np

from app.config import get_settings

logger = logging.getLogger("omni-object.publisher")


class SharedFrameRing:
    """Zero-copy frame ring buffer using memory-mapped files.
    
    Automatically checks available disk space on the SHM directory
    and disables itself if insufficient, falling back to JPEG-over-Redis.
    """

    # Minimum free space required to keep SHM enabled (50MB)
    _MIN_FREE_BYTES = 50 * 1024 * 1024
    # Re-check interval (seconds) to avoid stat() on every write
    _CHECK_INTERVAL_S = 30.0

    def __init__(self, settings):
        self._enabled = bool(getattr(settings, "shm_enabled", False))
        self._dir = getattr(settings, "shm_dir", "/app/shm/omnivision")
        self._slots = int(getattr(settings, "shm_slots_per_camera", 4))
        self._buffers: dict[str, dict] = {}
        self._last_space_check = 0.0
        self._space_ok = True
        if self._enabled:
            os.makedirs(self._dir, exist_ok=True)
            self._check_shm_capacity()

    def _check_shm_capacity(self) -> bool:
        """Check if SHM directory has enough free space."""
        import time as _time
        now = _time.monotonic()
        if now - self._last_space_check < self._CHECK_INTERVAL_S:
            return self._space_ok
        self._last_space_check = now
        try:
            import shutil
            usage = shutil.disk_usage(self._dir)
            self._space_ok = usage.free >= self._MIN_FREE_BYTES
            if not self._space_ok:
                logger.warning(
                    "⚠️ SHM space low: %dMB free (need %dMB) at %s — "
                    "falling back to JPEG-over-Redis. Increase shm-size in docker-compose.",
                    usage.free // (1024 * 1024),
                    self._MIN_FREE_BYTES // (1024 * 1024),
                    self._dir,
                )
        except Exception as e:
            logger.warning("⚠️ Cannot check SHM space: %s — disabling SHM", e)
            self._space_ok = False
        return self._space_ok

    def write(self, camera_id: str, frame_bgr: Optional[np.ndarray]) -> Optional[dict]:
        if not self._enabled or frame_bgr is None:
            return None

        # Periodic capacity check — auto-disable if SHM full
        if not self._check_shm_capacity():
            return None

        if frame_bgr.ndim == 2:
            frame_bgr = frame_bgr[:, :, None]

        if frame_bgr.dtype != np.uint8:
            frame_bgr = frame_bgr.astype(np.uint8, copy=False)

        frame_bgr = np.ascontiguousarray(frame_bgr)
        h, w = frame_bgr.shape[:2]
        c = int(frame_bgr.shape[2]) if frame_bgr.ndim == 3 else 1
        dtype = frame_bgr.dtype.name
        frame_bytes = int(frame_bgr.nbytes)
        path = os.path.join(self._dir, f"{camera_id}.bin")

        buffer = self._buffers.get(camera_id)
        if buffer is None or buffer["shape"] != (h, w, c) or buffer["dtype"] != dtype:
            # Close old memmap before creating new (prevent fd leak on resolution change)
            if buffer is not None:
                try:
                    old_mm = buffer["memmap"]
                    old_mm.flush()
                    if hasattr(old_mm, '_mmap') and old_mm._mmap is not None:
                        old_mm._mmap.close()
                except Exception:
                    pass
            total_bytes = frame_bytes * self._slots
            os.makedirs(self._dir, exist_ok=True)
            if not os.path.exists(path) or os.path.getsize(path) != total_bytes:
                with open(path, "wb") as f:
                    f.truncate(total_bytes)
            memmap = np.memmap(path, dtype=np.uint8, mode="r+", shape=(total_bytes,))
            buffer = {
                "path": path,
                "shape": (h, w, c),
                "dtype": dtype,
                "frame_bytes": frame_bytes,
                "index": 0,
                "memmap": memmap,
            }
            self._buffers[camera_id] = buffer

        slot_index = int(buffer["index"])
        offset = slot_index * int(buffer["frame_bytes"])
        memmap = buffer["memmap"]
        memmap[offset:offset + buffer["frame_bytes"]] = frame_bgr.reshape(-1)
        memmap.flush()
        buffer["index"] = (slot_index + 1) % self._slots

        return {
            "path": buffer["path"],
            "index": slot_index,
            "shape": f"{h},{w},{c}",
            "dtype": buffer["dtype"],
        }

    def prune(self, active_camera_ids: set):
        """Remove SHM buffers for cameras no longer active."""
        stale = set(self._buffers.keys()) - active_camera_ids
        for cam_id in stale:
            buf = self._buffers.pop(cam_id, None)
            if buf:
                try:
                    mm = buf["memmap"]
                    mm.flush()
                    if hasattr(mm, '_mmap') and mm._mmap is not None:
                        mm._mmap.close()
                except Exception:
                    pass


class RedisStreamPublisher:
    """
    Publishes detection metadata + JPEG frame bytes to Redis Streams.
    Uses XADD with MAXLEN for automatic ring-buffer behavior.
    """

    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self._redis: Optional[aioredis.Redis] = None
        self._connected = False
        self._shm_ring = SharedFrameRing(self.settings)
        self._publish_queue: Optional[asyncio.Queue] = None
        self._publish_worker_task: Optional[asyncio.Task] = None
        self._last_idle_frame_publish_ts: dict[str, float] = {}
        # Exponential backoff state for reconnection
        self._reconnect_attempt = 0
        self._max_backoff = 30.0  # seconds
        self._base_backoff = 0.5  # seconds
        self._reconnecting = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def connect(self):
        """Connect to Redis Streams instance with exponential backoff."""
        try:
            self._redis = aioredis.from_url(
                self.settings.redis_streams_url,
                decode_responses=False,  # binary mode for frame bytes
            )
            await self._redis.ping()
            self._connected = True
            self._reconnect_attempt = 0  # reset backoff on success
            self._reconnecting = False
            logger.info("✅ Connected to Redis Streams: %s", self.settings.redis_streams_url)
            
            # Start background publish worker
            queue_size = getattr(self.settings, "stream_publish_queue_size", 50)
            # Drain old queue on reconnect to avoid publishing stale data
            if self._publish_queue is not None:
                drained = 0
                while not self._publish_queue.empty():
                    try:
                        self._publish_queue.get_nowait()
                        self._publish_queue.task_done()
                        drained += 1
                    except asyncio.QueueEmpty:
                        break
                if drained:
                    logger.info("Drained %d stale tasks from publish queue on reconnect", drained)
            self._publish_queue = asyncio.Queue(maxsize=queue_size)
            # Cancel old worker if running
            if self._publish_worker_task and not self._publish_worker_task.done():
                self._publish_worker_task.cancel()
                try:
                    await self._publish_worker_task
                except asyncio.CancelledError:
                    pass
            self._publish_worker_task = asyncio.create_task(self._publish_worker())
        except Exception as e:
            self._reconnect_attempt += 1
            backoff = min(self._max_backoff, self._base_backoff * (2 ** self._reconnect_attempt))
            logger.error("❌ Redis Streams connection failed (attempt %d, backoff %.1fs): %s",
                         self._reconnect_attempt, backoff, e)
            self._connected = False
            self._reconnecting = False
            await asyncio.sleep(backoff)

    async def _publish_worker(self):
        """Background worker that pulls from local Queue and publishes to Redis.
        
        IMPORTANT: Loop uses 'while True' instead of 'while self._connected' so
        the worker survives temporary Redis disconnects. Reconnection happens in
        _reconnect_with_backoff(); this worker just waits for the queue.
        """
        logger.info("Background Redis publisher worker started")
        while True:
            task = None
            try:
                # If disconnected, wait briefly before checking queue again
                if not self._connected:
                    await asyncio.sleep(0.5)
                    continue
                task = await asyncio.wait_for(self._publish_queue.get(), timeout=2.0)
                if task["type"] == "detections":
                    await self._do_publish_detections(*task["args"])
                elif task["type"] == "frame_only":
                    result = await self._do_publish_frame_only(*task["args"])
                    # Resolve the future with the stream ID, if caller is waiting
                    future = task.get("future")
                    if future and not future.done():
                        future.set_result(result)
                self._publish_queue.task_done()
            except asyncio.TimeoutError:
                # No task in 2s — loop back and recheck _connected
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Publish worker error: %s", e)
                # Resolve pending future with None on error
                if task is not None:
                    future = task.get("future")
                    if future and not future.done():
                        future.set_result(None)
                    self._publish_queue.task_done()
                await asyncio.sleep(0.5)

    async def disconnect(self):
        if self._publish_worker_task:
            self._publish_worker_task.cancel()
            try:
                await self._publish_worker_task
            except asyncio.CancelledError:
                pass

        if self._redis:
            await self._redis.aclose()
            self._connected = False
            logger.info("Redis Streams disconnected")

    def can_use_shm_only(self) -> bool:
        return bool(
            getattr(self.settings, "shm_enabled", False)
            and getattr(self._shm_ring, "_enabled", False)
            and getattr(self._shm_ring, "_space_ok", False)
        )

    def should_publish_idle_frame(self, camera_id: str, timestamp: float) -> bool:
        interval = float(getattr(self.settings, "idle_frame_publish_interval_s", 2.0))
        if interval <= 0:
            self._last_idle_frame_publish_ts[camera_id] = timestamp
            return True

        last_published = self._last_idle_frame_publish_ts.get(camera_id, 0.0)
        if timestamp - last_published >= interval:
            self._last_idle_frame_publish_ts[camera_id] = timestamp
            return True
        return False

    def prune_idle_timestamps(self, active_camera_ids: set[str]) -> None:
        stale = set(self._last_idle_frame_publish_ts.keys()) - active_camera_ids
        for cam_id in stale:
            self._last_idle_frame_publish_ts.pop(cam_id, None)

    @staticmethod
    def _decode_shm_frame_to_jpeg(data: dict) -> Optional[bytes]:
        try:
            shm_path = data.get(b"shm_path")
            if not shm_path:
                return None

            path = shm_path.decode()
            shape_raw = data.get(b"shm_shape", b"").decode()
            dtype = data.get(b"shm_dtype", b"uint8").decode()
            index = int(data.get(b"shm_index", b"0").decode())
            if not path or not shape_raw or not os.path.exists(path):
                return None

            parts = [int(x) for x in shape_raw.split(",") if x]
            if len(parts) != 3:
                return None

            h, w, c = parts
            itemsize = int(np.dtype(dtype).itemsize)
            frame_bytes = h * w * c * itemsize
            offset = index * frame_bytes
            frame_view = np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=(h, w, c))
            frame_bgr = np.array(frame_view)
            ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
            if not ok or buf is None:
                return None
            return buf.tobytes()
        except Exception as e:
            logger.debug("SHM JPEG reconstruction failed: %s", e)
            return None

    async def publish_detections(
        self,
        camera_id: str,
        detections: List[dict],
        frame_jpeg: Optional[bytes],
        timestamp: float,
        frame_bgr: Optional[np.ndarray] = None,
        frame_stream_id: Optional[str] = None,
    ):
        """Enqueue detections to be published."""
        if not self._connected or not self._publish_queue:
            return
            
        try:
            self._publish_queue.put_nowait({
                "type": "detections",
                "args": (camera_id, detections, frame_jpeg, timestamp, frame_bgr, frame_stream_id)
            })
        except asyncio.QueueFull:
            logger.warning("Redis publish queue FULL! Dropping frame for camera %s (Backpressure)", camera_id)

    async def _do_publish_detections(
        self,
        camera_id: str,
        detections: List[dict],
        frame_jpeg: Optional[bytes],
        timestamp: float,
        frame_bgr: Optional[np.ndarray] = None,
        frame_stream_id: Optional[str] = None,
    ):
        """
        Publish tracked detections + frame to Redis Stream.

        Each detection entry:
        {
            "camera_id": str,
            "global_track_id": int,
            "class_name": str (car/truck/motorcycle/person/bus),
            "class_id": int,
            "bbox": "x1,y1,x2,y2",
            "confidence": float,
            "timestamp": float,
            "frame_jpeg": bytes   (only on frame stream, not per-detection)
        }
        """
        if not self._connected or not self._redis:
            return

        prefix = self.settings.stream_prefix
        maxlen = self.settings.stream_maxlen

        try:
            # 1. Publish each detection to {prefix}:detections
            for det in detections:
                entry = {
                    b"camera_id": camera_id.encode(),
                    b"global_track_id": str(det["track_id"]).encode(),
                    b"class_name": det["class_name"].encode(),
                    b"class_id": str(det["class_id"]).encode(),
                    b"bbox": f'{det["bbox"][0]},{det["bbox"][1]},{det["bbox"][2]},{det["bbox"][3]}'.encode(),
                    b"confidence": f'{det["confidence"]:.3f}'.encode(),
                    b"timestamp": f'{timestamp:.3f}'.encode(),
                }
                # Attach frame_stream_id for exact frame correlation
                if frame_stream_id:
                    entry[b"frame_stream_id"] = frame_stream_id.encode() if isinstance(frame_stream_id, str) else frame_stream_id
                await self._redis.xadd(
                    f"{prefix}:detections",
                    entry,
                    maxlen=maxlen,
                    approximate=True,
                )

            # 2. Publish frame to {prefix}:frames:* ONLY if not already done by publish_frame_only.
            # When frame_stream_id is set, the caller already published the frame.
            if not frame_stream_id:
                shm_info = self._shm_ring.write(camera_id, frame_bgr)
                frame_entry = {
                    b"camera_id": camera_id.encode(),
                    b"timestamp": f'{timestamp:.3f}'.encode(),
                }
                if frame_jpeg:
                    frame_entry[b"frame_jpeg"] = frame_jpeg
                if shm_info:
                    frame_entry[b"shm_path"] = shm_info["path"].encode()
                    frame_entry[b"shm_index"] = str(shm_info["index"]).encode()
                    frame_entry[b"shm_shape"] = shm_info["shape"].encode()
                    frame_entry[b"shm_dtype"] = shm_info["dtype"].encode()
                await self._redis.xadd(
                    f"{prefix}:frames:{camera_id}",
                    frame_entry,
                    maxlen=3,  # keep only last 3 frames per camera
                    approximate=False,
                )

        except Exception as e:
            logger.warning("Redis publish failed: %s", e)
            self._connected = False
            # Attempt reconnection with backoff in background (guard against multiple concurrent tasks)
            if not self._reconnecting:
                self._reconnecting = True
                asyncio.create_task(self._reconnect_with_backoff())

    async def _reconnect_with_backoff(self):
        """Reconnect to Redis with exponential backoff — retries indefinitely.
        
        Never gives up: Redis is essential for the entire detection pipeline.
        Uses capped exponential backoff (max 30s between attempts).
        """
        attempt = 0
        while True:
            attempt += 1
            self._reconnect_attempt += 1
            backoff = min(self._max_backoff, self._base_backoff * (2 ** min(self._reconnect_attempt, 8)))
            logger.info("Redis reconnect attempt %d in %.1fs...", attempt, backoff)
            await asyncio.sleep(backoff)
            try:
                await self.connect()
                if self._connected:
                    logger.info("✅ Redis reconnected after %d attempts", attempt)
                    self._reconnecting = False
                    return
            except Exception as e:
                logger.error("Redis reconnect attempt %d failed: %s", attempt, e)
            # Log periodic warning for prolonged outages
            if attempt % 20 == 0:
                logger.warning("⚠️ Redis still unreachable after %d reconnect attempts", attempt)

    async def publish_frame_only(
        self,
        camera_id: str,
        frame_jpeg: Optional[bytes],
        timestamp: float,
        frame_bgr: Optional[np.ndarray] = None,
    ) -> Optional[str]:
        """Enqueue empty frame to be published. Returns frame_stream_id for correlation."""
        if not self._connected or not self._publish_queue:
            return None

        # Use a Future to get the stream_id back from the worker
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        try:
            self._publish_queue.put_nowait({
                "type": "frame_only",
                "args": (camera_id, frame_jpeg, timestamp, frame_bgr),
                "future": future,
            })
        except asyncio.QueueFull:
            return None  # Silent drop for empty frames

        try:
            # Wait briefly for the stream_id (should be fast)
            return await asyncio.wait_for(future, timeout=2.0)
        except asyncio.TimeoutError:
            return None

    async def _do_publish_frame_only(
        self,
        camera_id: str,
        frame_jpeg: Optional[bytes],
        timestamp: float,
        frame_bgr: Optional[np.ndarray] = None,
    ) -> Optional[str]:
        """Publish frame to {prefix}:frames:{camera_id} without detection entries.

        Keeps the frame stream fresh for omni-human / omni-vehicle.
        Returns the Redis stream message ID for frame correlation.
        """
        if not self._connected or not self._redis:
            return None

        prefix = self.settings.stream_prefix

        try:
            shm_info = self._shm_ring.write(camera_id, frame_bgr)
            frame_entry = {
                b"camera_id": camera_id.encode(),
                b"timestamp": f'{timestamp:.3f}'.encode(),
            }
            if frame_jpeg:
                frame_entry[b"frame_jpeg"] = frame_jpeg
            if shm_info:
                frame_entry[b"shm_path"] = shm_info["path"].encode()
                frame_entry[b"shm_index"] = str(shm_info["index"]).encode()
                frame_entry[b"shm_shape"] = shm_info["shape"].encode()
                frame_entry[b"shm_dtype"] = shm_info["dtype"].encode()
            msg_id = await self._redis.xadd(
                f"{prefix}:frames:{camera_id}",
                frame_entry,
                maxlen=3,
                approximate=False,
            )
            # Return the stream ID as string for correlation
            if isinstance(msg_id, bytes):
                return msg_id.decode()
            return str(msg_id) if msg_id else None
        except Exception as e:
            logger.warning("Redis frame-only publish failed: %s", e)
            # Trigger reconnect so next publish doesn't fail silently
            if "ConnectionError" in type(e).__name__ or "ConnectionRefusedError" in type(e).__name__:
                self._connected = False
                asyncio.ensure_future(self._reconnect_with_backoff())
            return None

    async def ensure_consumer_groups(self):
        """
        Create consumer groups for downstream services.
        Idempotent: ignores if group already exists.
        """
        if not self._connected or not self._redis:
            return

        prefix = self.settings.stream_prefix
        groups = [
            (f"{prefix}:detections", "omni-vehicle-group"),
            (f"{prefix}:detections", "omni-human-group"),
            (f"{prefix}:detections", "omni-fusion-group"),
        ]

        for stream, group in groups:
            try:
                await self._redis.xgroup_create(
                    stream, group, id="$", mkstream=True
                )
                logger.info("Created consumer group: %s on %s", group, stream)
            except aioredis.ResponseError as e:
                if "BUSYGROUP" in str(e):
                    pass  # group already exists
                else:
                    logger.warning("Failed to create group %s: %s", group, e)

    async def get_latest_frame(self, camera_id: str) -> Optional[bytes]:
        """Get the most recent frame JPEG for a camera (for /snap API)."""
        if not self._connected or not self._redis:
            return None

        prefix = self.settings.stream_prefix
        try:
            # XREVRANGE: get last entry
            entries = await self._redis.xrevrange(
                f"{prefix}:frames:{camera_id}", count=1
            )
            if entries:
                _msg_id, data = entries[0]
                frame_jpeg = data.get(b"frame_jpeg")
                if frame_jpeg:
                    return frame_jpeg
                return self._decode_shm_frame_to_jpeg(data)
        except Exception as e:
            logger.warning("Failed to get latest frame for %s: %s", camera_id[:8], e)
        return None
