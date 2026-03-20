import logging
import os
import time
from collections import OrderedDict
from typing import Optional

import cv2
import numpy as np
import redis.asyncio as aioredis

logger = logging.getLogger("omni-vehicle.stream_frame")


class LprFrameManager:
    def __init__(self, settings):
        self.settings = settings
        self._frame_cache: OrderedDict[str, tuple[float, np.ndarray]] = OrderedDict()
        self._last_cache_eviction: float = 0.0

    def _frames_stream(self, camera_id: str) -> str:
        pfx = getattr(self.settings, "stream_prefix", "omni")
        return f"{pfx}:frames:{camera_id}"

    @property
    def cache_entries(self) -> int:
        return len(self._frame_cache)

    def evict_stale_cache(self) -> int:
        now = time.monotonic()
        if now - self._last_cache_eviction < 30.0:
            return 0
        self._last_cache_eviction = now

        stale_frame_keys = [cam_id for cam_id, (ts, _) in self._frame_cache.items() if now - ts > 60.0]
        for cam_id in stale_frame_keys:
            del self._frame_cache[cam_id]

        max_cache_entries = int(getattr(self.settings, "lpr_frame_cache_max_entries", 64))
        if max_cache_entries > 0:
            while len(self._frame_cache) > max_cache_entries:
                self._frame_cache.popitem(last=False)

        return len(stale_frame_keys)

    def set_frame_cache(self, camera_id: str, frame: np.ndarray) -> None:
        self._frame_cache[camera_id] = (time.monotonic(), frame)
        self._frame_cache.move_to_end(camera_id)
        max_cache_entries = int(getattr(self.settings, "lpr_frame_cache_max_entries", 64))
        if max_cache_entries > 0:
            while len(self._frame_cache) > max_cache_entries:
                self._frame_cache.popitem(last=False)

    async def get_frame_entry(
        self,
        redis_client: Optional[aioredis.Redis],
        camera_id: str,
        detection_ts: float = 0.0,
        frame_stream_id: str = None,
    ) -> Optional[dict]:
        if redis_client is None:
            return None
        try:
            if frame_stream_id:
                try:
                    entries = await redis_client.xrange(
                        self._frames_stream(camera_id),
                        min=frame_stream_id,
                        max=frame_stream_id,
                        count=1,
                    )
                    if entries:
                        _, data = entries[0]
                        return data
                except Exception:
                    pass

            entries = await redis_client.xrevrange(self._frames_stream(camera_id), count=5)
            if not entries:
                return None

            if detection_ts <= 0:
                _, data = entries[0]
                return data

            best_entry = None
            best_delta = float("inf")
            for _, data in entries:
                try:
                    frame_ts = float(data.get(b"timestamp", b"0").decode())
                except (ValueError, AttributeError):
                    frame_ts = 0.0
                delta = abs(frame_ts - detection_ts)
                if delta < best_delta:
                    best_delta = delta
                    best_entry = data

            max_delta = float(getattr(self.settings, "lpr_frame_match_max_delta_s", 0.8))
            if best_entry is not None and best_delta <= max_delta:
                if best_delta > 0.3:
                    logger.debug("Frame-bbox temporal gap %.2fs for cam %s (>300ms)", best_delta, camera_id[:8])
                return best_entry

            latest_entry = entries[0][1] if entries else None
            latest_delta = float("inf")
            if latest_entry is not None:
                try:
                    latest_ts = float(latest_entry.get(b"timestamp", b"0").decode())
                    latest_delta = abs(latest_ts - detection_ts)
                except (ValueError, AttributeError):
                    latest_delta = float("inf")

            fallback_delta = float(getattr(self.settings, "lpr_frame_match_fallback_delta_s", 3.5))
            if latest_entry is not None and latest_delta <= fallback_delta:
                logger.debug(
                    "Frame fallback used for cam %s (best_delta=%.2fs, latest_delta=%.2fs)",
                    camera_id[:8], best_delta, latest_delta,
                )
                return latest_entry

            logger.debug(
                "No frame within %.1fs of detection ts=%.3f for cam %s (best_delta=%.2fs)",
                max_delta, detection_ts, camera_id[:8], best_delta,
            )
            return None
        except Exception as e:
            logger.warning("Frame fetch failed for %s: %s", camera_id[:8], e)
            return None

    async def get_frame_bgr(
        self,
        redis_client: Optional[aioredis.Redis],
        camera_id: str,
        detection_ts: float = 0.0,
        frame_stream_id: str = None,
    ) -> Optional[np.ndarray]:
        allow_cache = detection_ts <= 0 and frame_stream_id is None
        if allow_cache:
            ttl = max(0.05, min(0.5, float(getattr(self.settings, "lpr_camera_rate_window_s", 1.0)) * 0.5))
            now = time.monotonic()
            cached = self._frame_cache.get(camera_id)
            if cached and now - cached[0] <= ttl:
                self._frame_cache.move_to_end(camera_id)
                return cached[1]

        data = await self.get_frame_entry(redis_client, camera_id, detection_ts=detection_ts, frame_stream_id=frame_stream_id)
        if not data:
            fallback_ttl = float(getattr(self.settings, "lpr_frame_cache_fallback_ttl_s", 0.45))
            cached = self._frame_cache.get(camera_id)
            if cached and (time.monotonic() - cached[0]) <= fallback_ttl:
                self._frame_cache.move_to_end(camera_id)
                return cached[1]
            return None

        if self.settings.lpr_shm_enabled:
            shm_path = data.get(b"shm_path")
            if shm_path:
                try:
                    path = shm_path.decode()
                    shape_raw = data.get(b"shm_shape", b"").decode()
                    dtype = data.get(b"shm_dtype", b"uint8").decode()
                    index = int(data.get(b"shm_index", b"0").decode())
                    if shape_raw:
                        parts = [int(p) for p in shape_raw.split(",") if p]
                        if len(parts) == 3:
                            h, w, c = parts
                            itemsize = int(np.dtype(dtype).itemsize)
                            frame_bytes = int(h * w * c * itemsize)
                            offset = index * frame_bytes
                            if frame_bytes <= 0 or offset < 0 or not os.path.exists(path):
                                return None
                            file_size = os.path.getsize(path)
                            if offset + frame_bytes > file_size:
                                return None
                            frame_view = np.memmap(
                                path,
                                dtype=dtype,
                                mode="r",
                                offset=offset,
                                shape=(h, w, c),
                            )
                            frame_copy = np.array(frame_view)
                            del frame_view
                            self.set_frame_cache(camera_id, frame_copy)
                            return frame_copy
                except Exception as e:
                    logger.warning("SHM frame fetch failed for %s: %s", camera_id[:8], e)

        frame_jpeg = data.get(b"frame_jpeg")
        if not frame_jpeg:
            return None
        nparr = np.frombuffer(frame_jpeg, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return None
        self.set_frame_cache(camera_id, img_bgr)
        return img_bgr
