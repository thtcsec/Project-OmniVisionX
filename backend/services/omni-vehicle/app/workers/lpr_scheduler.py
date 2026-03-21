"""
Minimal LPR scheduler for omni-vehicle.
Pulls frames from online cameras and runs plate detection periodically.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from app.config import get_settings

logger = logging.getLogger("omni-vehicle.scheduler")


def _fix_url_for_docker(url: str) -> str:
    """Replace localhost/127.0.0.1 with host.docker.internal for Docker networking.
    Uses :// boundary to avoid mangling URLs like http://my-localhost-service:8080.
    """
    import re
    url = re.sub(r'(://)(localhost)((?::\d+)?)', r'\1host.docker.internal\3', url)
    url = re.sub(r'(://)(127\.0\.0\.1)((?::\d+)?)', r'\1host.docker.internal\3', url)
    return url


def _capture_rtsp_frame(url: str) -> Optional[np.ndarray]:
    cap = None
    try:
        cap = cv2.VideoCapture(_fix_url_for_docker(url), cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ret, frame = cap.read()
        if not ret or frame is None:
            return None
        return frame
    except Exception as exc:
        logger.warning("RTSP capture failed: %s", exc)
        return None
    finally:
        if cap is not None:
            cap.release()


def _parse_roi(roi_json: Optional[str]) -> Optional[List[Tuple[int, int]]]:
    if not roi_json:
        return None
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


class LprScheduler:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._engine: AsyncEngine = create_async_engine(
            self.settings.database_url,
            future=True,
            pool_pre_ping=True,
        )
        self._task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self, process_frame_fn) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop(process_frame_fn))
        logger.info("🔄 LPR scheduler started (interval=%ss)", self.settings.snapshot_interval)

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._engine:
            await self._engine.dispose()
        logger.info("⏹️ LPR scheduler stopped")

    async def _loop(self, process_frame_fn) -> None:
        interval = max(1, int(self.settings.snapshot_interval))
        while self._running:
            try:
                cameras = await self._get_active_cameras()
                if cameras:
                    await self._process_cameras(cameras, process_frame_fn)
            except Exception:
                logger.exception("LPR scheduler loop error")
            await asyncio.sleep(interval)

    async def _get_active_cameras(self) -> Dict[str, dict]:
        try:
            allow_raw = os.environ.get("LPR_CAMERA_ALLOWLIST", "").strip()
            allow = {x.strip() for x in allow_raw.split(",") if x.strip()} if allow_raw else None
            async with self._engine.connect() as conn:
                result = await conn.execute(
                    text(
                        'SELECT "Id", "StreamUrl", "EnablePlateOcr" '
                        'FROM "Cameras" WHERE LOWER(TRIM("Status")) = \'online\''
                    )
                )
                rows = result.fetchall()
                cameras: Dict[str, dict] = {}
                for row in rows:
                    camera_id = str(row[0])
                    if allow is not None and camera_id not in allow:
                        continue
                    stream_url = row[1]
                    enable_plate = row[2] if row[2] is not None else True
                    if not stream_url or not enable_plate:
                        continue
                    cameras[camera_id] = {
                        "url": stream_url,
                        "roi": None,
                    }
                if cameras:
                    logger.info("📹 LPR scheduler cameras: %s", len(cameras))
                return cameras
        except Exception:
            logger.exception("DB error fetching cameras")
            return {}

    async def _process_cameras(self, cameras: Dict[str, dict], process_frame_fn) -> None:
        for cam_id, cfg in cameras.items():
            # Run blocking RTSP capture in a thread to avoid blocking the event loop
            frame = await asyncio.to_thread(_capture_rtsp_frame, cfg["url"])
            if frame is None:
                continue
            try:
                await process_frame_fn(
                    img_bgr=frame,
                    camera_id=cam_id,
                    roi_points=cfg.get("roi"),
                    is_night=False,
                    use_tracking=True,
                    use_fortress=False,
                    dedup=True,
                )
            except Exception:
                logger.exception("LPR process error for camera %s", cam_id[:8])


def scheduler_enabled() -> bool:
    raw = os.environ.get("LPR_SCHEDULER_ENABLED", "1").strip().lower()
    return raw in {"1", "true", "yes"}
