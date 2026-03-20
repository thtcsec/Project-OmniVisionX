import hashlib
import time
from typing import Optional

import redis.asyncio as aioredis


class LprBackpressurePolicy:
    def __init__(self, settings):
        self.settings = settings

    async def update_global_drop_ratio(
        self,
        redis_client: Optional[aioredis.Redis],
        last_backlog_check: float,
    ) -> tuple[Optional[float], float, Optional[int]]:
        if not getattr(self.settings, "lpr_adaptive_frame_skip", True):
            return 0.0, last_backlog_check, 0

        now = time.monotonic()
        interval_ms = int(getattr(self.settings, "lpr_stream_check_interval_ms", 500))
        if now - last_backlog_check < (interval_ms / 1000.0):
            return None, last_backlog_check, None

        if redis_client is None:
            return None, now, None

        backlog = await self._read_backlog(redis_client)
        if backlog is None:
            return None, now, None

        threshold = max(1, int(getattr(self.settings, "lpr_stream_backlog_threshold", 200)))
        min_ratio = float(getattr(self.settings, "lpr_stream_drop_ratio_min", 0.0))
        max_ratio = float(getattr(self.settings, "lpr_stream_drop_ratio_max", 0.35))
        min_ratio = max(0.0, min(1.0, min_ratio))
        max_ratio = max(min_ratio, min(1.0, max_ratio))

        if backlog <= threshold:
            return 0.0, now, backlog

        ratio = (backlog - threshold) / float(threshold)
        ratio = min(max_ratio, ratio)
        if ratio < min_ratio:
            ratio = min_ratio
        return ratio, now, backlog

    async def _read_backlog(self, redis_client: aioredis.Redis) -> Optional[int]:
        try:
            pfx = getattr(self.settings, "stream_prefix", "omni")
            grp = getattr(self.settings, "redis_consumer_group", "omni-vehicle-group")
            pending_info = await redis_client.xpending(f"{pfx}:detections", grp)
            if isinstance(pending_info, dict):
                return int(pending_info.get("pending", 0))
            if isinstance(pending_info, (list, tuple)) and len(pending_info) >= 1:
                return int(pending_info[0])
            return 0
        except Exception:
            try:
                pfx = getattr(self.settings, "stream_prefix", "omni")
                return int(await redis_client.xlen(f"{pfx}:detections"))
            except Exception:
                return None

    def update_camera_drop_ratio(self, entry: dict[str, float]) -> float:
        now = time.monotonic()
        window_s = float(getattr(self.settings, "lpr_camera_rate_window_s", 1.0))
        fps_mid = float(getattr(self.settings, "lpr_camera_fps_mid", 15.0))
        fps_high = float(getattr(self.settings, "lpr_camera_fps_high", 20.0))
        ratio_mid = float(getattr(self.settings, "lpr_camera_drop_ratio_mid", 0.15))
        ratio_high = float(getattr(self.settings, "lpr_camera_drop_ratio_high", 0.6))

        entry["count"] = float(entry.get("count", 0.0)) + 1.0
        elapsed = now - float(entry.get("start", now))
        if elapsed < window_s:
            return float(entry.get("ratio", 0.0))

        fps = float(entry.get("count", 0.0)) / max(elapsed, 0.001)
        entry["fps"] = fps
        if fps < fps_mid:
            ratio = 0.0
        elif fps < fps_high:
            ratio = max(0.0, min(1.0, ratio_mid))
        else:
            ratio = max(0.0, min(1.0, ratio_high))

        entry["start"] = now
        entry["count"] = 0.0
        entry["ratio"] = ratio
        entry["last_update"] = now
        return ratio

    @staticmethod
    def should_drop_event(
        camera_id: str,
        msg_id: str,
        drop_ratio: float,
        track_id: int | None = None,
        timestamp: float | None = None,
    ) -> bool:
        identity = str(track_id) if track_id is not None and track_id >= 0 else msg_id
        ts_bucket = str(int(timestamp * 2.0)) if timestamp is not None and timestamp > 0 else "na"
        sample_bytes = hashlib.md5(f"{camera_id}:{identity}:{ts_bucket}".encode()).digest()[:4]
        sample = int.from_bytes(sample_bytes, "big") / float(2**32)
        return sample < drop_ratio
