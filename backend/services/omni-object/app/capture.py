"""
RTSP capture pool for omni-object
==============================
Persistent RTSP connections with best-frame selection.
Each camera gets a dedicated capture thread that runs at target FPS.
"""
import cv2
import logging
import os
import threading
import time
from collections import deque
from typing import Dict, Optional, Tuple

import numpy as np

from app.config import get_settings

logger = logging.getLogger("omni-object.capture")

# NOTE: OPENCV_FFMPEG_CAPTURE_OPTIONS is set dynamically in RTSPCapture._reconnect()
# using settings.rtsp_transport so the value respects live config changes.


class RTSPCapture:
    """
    Single-camera persistent RTSP reader running in a background thread.
    Maintains a ring buffer for best-frame selection.
    """

    BACKOFF_LEVELS = {10: 0.5, 30: 2.0, 999: 10.0}

    def __init__(self, camera_id: str, rtsp_url: str, settings=None):
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.settings = settings or get_settings()

        self._cap: Optional[cv2.VideoCapture] = None
        self._buffer: deque = deque(maxlen=self.settings.capture_buffer_size)
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._error_count = 0
        self._last_frame_time = 0.0
        self._resolution: Optional[Tuple[int, int]] = None

    def start(self) -> bool:
        if self._running:
            return True
        self._running = True
        self._thread = threading.Thread(
            target=self._capture_loop,
            name=f"rtsp-{self.camera_id[:8]}",
            daemon=True,
        )
        self._thread.start()
        return True

    def stop(self):
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        if self._cap:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    def _capture_loop(self):
        """Main capture thread: read frames at target FPS."""
        interval = 1.0 / self.settings.capture_fps
        while self._running:
            try:
                t0 = time.time()
                if self._cap is None or not self._cap.isOpened():
                    self._reconnect()
                    if self._cap is None:
                        continue

                ret, frame = self._cap.read()
                if not ret or frame is None:
                    self._error_count += 1
                    # FIX: Giảm tolerance từ 3 → 2 để reconnect nhanh hơn khi camera lag
                    # Tránh miss frames khi camera bị packet loss hoặc network hiccup
                    if self._error_count >= 2:
                        self._reconnect()
                    continue

                self._error_count = 0
                self._last_frame_time = time.time()
                if self._resolution is None:
                    h, w = frame.shape[:2]
                    self._resolution = (w, h)

                with self._lock:
                    self._buffer.append(frame)

                # Pace to target FPS (subtract elapsed read time)
                elapsed = time.time() - t0
                sleep_time = interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            except Exception as e:
                logger.warning("[%s] Capture error: %s", self.camera_id[:8], e)
                self._error_count += 1
                time.sleep(1)

    def _reconnect(self):
        """Reconnect with exponential backoff."""
        if self._cap:
            try:
                self._cap.release()
            except Exception:
                pass

        # Clear stale frames from before disconnect
        with self._lock:
            self._buffer.clear()

        # Reset resolution so it is re-read from the new connection
        self._resolution = None

        # Apply configured RTSP transport (BUG-4: was hardcoded to 'tcp' at module level)
        rtsp_transport = getattr(self.settings, "rtsp_transport", "tcp")
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = f"rtsp_transport;{rtsp_transport}"

        backoff = 0.5
        for threshold, delay in self.BACKOFF_LEVELS.items():
            if self._error_count <= threshold:
                backoff = delay
                break
        else:
            # error_count > 999: use maximum backoff to avoid flooding
            backoff = max(self.BACKOFF_LEVELS.values())
        # Respect configurable max backoff
        max_backoff = getattr(self.settings, 'reconnect_backoff_max', None)
        if max_backoff and max_backoff > 0:
            backoff = min(backoff, max_backoff)

        logger.info("[%s] Reconnecting (errors=%d, backoff=%.1fs)...",
                    self.camera_id[:8], self._error_count, backoff)
        time.sleep(backoff)

        # After sleeping, check if stop() was called — avoids creating a
        # VideoCapture that would never be released.
        if not self._running:
            return

        try:
            # Create capture BEFORE opening so timeouts take effect
            self._cap = cv2.VideoCapture()
            self._cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
            self._cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize stale frames
            self._cap.open(self.rtsp_url, cv2.CAP_FFMPEG)
            if self._cap.isOpened():
                self._error_count = 0
                logger.info("[%s] RTSP connected ✅", self.camera_id[:8])
            else:
                self._cap.release()
                self._cap = None
                self._error_count += 1
        except Exception as e:
            logger.warning("[%s] RTSP connect failed: %s", self.camera_id[:8], e)
            self._cap = None
            self._error_count += 1

    def get_best_frame(self) -> Optional[np.ndarray]:
        """Get sharpest frame from recent buffer slice."""
        with self._lock:
            if not self._buffer:
                return None
            frames = list(self._buffer)

        best_of = max(1, int(getattr(self.settings, "capture_best_of_last_n", 3)))
        if best_of < len(frames):
            frames = frames[-best_of:]

        # Pick frame with highest Laplacian variance (sharpness)
        best_frame = None
        best_score = -1
        for f in frames:
            try:
                gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                score = cv2.Laplacian(gray, cv2.CV_64F).var()
                if score > best_score:
                    best_score = score
                    best_frame = f
            except Exception:
                continue
        return best_frame

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Get most recent frame."""
        with self._lock:
            return self._buffer[-1] if self._buffer else None

    def is_healthy(self) -> bool:
        return (
            self._running
            and self._error_count < 10
            and (time.time() - self._last_frame_time) < 30
        )


class CapturePool:
    """
    Manages persistent RTSP connections for all active cameras.
    """

    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self._captures: Dict[str, RTSPCapture] = {}

    def add_camera(self, camera_id: str, rtsp_url: str) -> bool:
        existing = self._captures.get(camera_id)
        if existing:
            # Check if URL changed — if so, restart the capture
            if existing.rtsp_url != rtsp_url:
                logger.info("Camera %s URL changed, reconnecting", camera_id[:8])
                existing.stop()
                del self._captures[camera_id]
            else:
                return True
        cap = RTSPCapture(camera_id, rtsp_url, self.settings)
        if cap.start():
            self._captures[camera_id] = cap
            logger.info("Added camera %s to capture pool", camera_id[:8])
            return True
        return False

    def remove_camera(self, camera_id: str):
        cap = self._captures.pop(camera_id, None)
        if cap:
            cap.stop()
            logger.info("Removed camera %s from capture pool", camera_id[:8])

    def sync_cameras(self, camera_configs: Dict[str, str]):
        """
        Sync pool with DB camera list.
        Adds new cameras, removes stale ones, reconnects if URL changed.
        camera_configs: {camera_id: rtsp_url}
        """
        current = set(self._captures.keys())
        desired = set(camera_configs.keys())

        # Remove stale
        for cam_id in current - desired:
            self.remove_camera(cam_id)

        # Add new + update existing (add_camera handles URL changes)
        for cam_id, url in camera_configs.items():
            self.add_camera(cam_id, url)

    def get_best_frame(self, camera_id: str) -> Optional[np.ndarray]:
        cap = self._captures.get(camera_id)
        return cap.get_best_frame() if cap else None

    def get_latest_frame(self, camera_id: str) -> Optional[np.ndarray]:
        cap = self._captures.get(camera_id)
        return cap.get_latest_frame() if cap else None

    def get_stats(self) -> Dict:
        return {
            cam_id: {
                "healthy": cap.is_healthy(),
                "resolution": cap._resolution,
                "error_count": cap._error_count,
            }
            for cam_id, cap in self._captures.items()
        }

    def stop_all(self):
        for cap in self._captures.values():
            cap.stop()
        self._captures.clear()
