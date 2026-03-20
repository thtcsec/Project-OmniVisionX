"""
OmniDetector - YOLOv11 Wrapper for OmniVision
TODO: Implement vehicle/person detection using YOLOv11
"""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np


@dataclass
class Detection:
    track_id: int
    class_name: str
    confidence: float
    bbox: List[int]  # [x1, y1, x2, y2]


class OmniDetector:
    """
    YOLOv11 wrapper for OmniVision.
    Detects vehicles (car, motorcycle, truck, bus) and persons.
    """

    def __init__(self, model_path: str = "yolo11m.pt", device: str = "cuda:0"):
        self.model_path = model_path
        self.device = device
        self._model = None

    async def load(self):
        """TODO: Load YOLO model"""
        raise NotImplementedError("BOILERPLATE")

    async def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        TODO: Run YOLO inference on frame.
        Returns list of detections for vehicles and persons.
        """
        raise NotImplementedError("BOILERPLATE")

    def _filter_target_classes(self, results) -> List[Detection]:
        """TODO: Filter COCO classes to vehicles + persons only"""
        raise NotImplementedError("BOILERPLATE")
