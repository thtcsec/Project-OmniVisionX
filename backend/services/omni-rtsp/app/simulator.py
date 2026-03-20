"""
RTSP Simulator - Loop video files as camera feeds
TODO: Implement cv2-based video loop simulation
"""

import cv2
from typing import Optional


class RTSPSimulator:
    """
    Simulates RTSP camera feeds by looping video files.
    Used for testing without actual cameras.
    """

    def __init__(self, video_path: str, fps: int = 30):
        self.video_path = video_path
        self.fps = fps
        self._cap = None

    async def start(self):
        """TODO: Open video file and start looping"""
        raise NotImplementedError("BOILERPLATE")

    async def read_frame(self) -> Optional[bytes]:
        """TODO: Read next frame as JPEG bytes"""
        raise NotImplementedError("BOILERPLATE")

    async def stop(self):
        """TODO: Release video resources"""
        raise NotImplementedError("BOILERPLATE")
