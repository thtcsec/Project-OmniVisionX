import os
import logging
import numpy as np

logger = logging.getLogger("omni-human.detectors")

# Guard heavy ML imports — if torch or ultralytics is broken (CUDA mismatch,
# corrupted install, OOM during import), the service should still start in
# degraded mode instead of crashing with 502.
try:
    import torch
    TORCH_AVAILABLE = True
except Exception as e:
    torch = None
    TORCH_AVAILABLE = False
    logger.error("Failed to import torch: %s — detectors will be disabled", e)

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception as e:
    YOLO = None
    YOLO_AVAILABLE = False
    logger.error("Failed to import ultralytics: %s — detectors will be disabled", e)

class HumanDetector:
    def __init__(self, model_path: str, conf_thres: float = 0.5):
        self.model_path = model_path
        self.conf_thres = conf_thres
        self.model = None

        if not TORCH_AVAILABLE or not YOLO_AVAILABLE:
            logger.error("❌ HumanDetector disabled: torch=%s, YOLO=%s", TORCH_AVAILABLE, YOLO_AVAILABLE)
            return

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.half = self.device != 'cpu'  # Use FP16 on GPU
        
        logger.info("Loading Human Detector from %s on %s (FP16=%s)...", model_path, self.device, self.half)
        try:
            self.model = YOLO(model_path)
            # Warmup to stabilize latency
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            self.model.predict(dummy, device=self.device, half=self.half, verbose=False)
            logger.info("✅ Human Detector Loaded and Warmed Up.")
        except Exception as e:
            logger.error("❌ Error loading Human Detector: %s", e)
            self.model = None

    def detect(self, image: np.ndarray):
        """
        Detect humans in the image.
        Returns list of [x1, y1, x2, y2, conf]
        """
        if self.model is None:
            return []
        
        results = self.model.predict(
            image, 
            classes=[0], 
            conf=self.conf_thres, 
            iou=0.45,
            imgsz=640,
            half=self.half,
            device=self.device,
            agnostic_nms=True,   # Improves overlapping detections
            verbose=False,
            augment=False        # Disable TTA for speed
        )
        
        detections = []
        if results:
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    detections.append([int(x1), int(y1), int(x2), int(y2), conf])
        return detections

class FaceDetector:
    def __init__(self, model_path: str, conf_thres: float = 0.5):
        self.model_path = model_path
        self.conf_thres = conf_thres
        self.model = None

        if not TORCH_AVAILABLE or not YOLO_AVAILABLE:
            logger.error("❌ FaceDetector disabled: torch=%s, YOLO=%s", TORCH_AVAILABLE, YOLO_AVAILABLE)
            return

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.half = self.device != 'cpu'
        
        logger.info("Loading Face Detector from %s on %s (FP16=%s)...", model_path, self.device, self.half)
        try:
            self.model = YOLO(model_path)
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            self.model.predict(dummy, device=self.device, half=self.half, verbose=False)
            logger.info("✅ Face Detector Loaded and Warmed Up.")
        except Exception as e:
            logger.error("❌ Error loading Face Detector: %s", e)
            self.model = None

    def detect(self, image: np.ndarray):
        """
        Detect faces in the image.
        Returns list of {'box': [x1, y1, x2, y2], 'conf': conf, 'keypoints': kpts}
        """
        if self.model is None:
            return []
        
        results = self.model.predict(
            image, 
            conf=self.conf_thres, 
            iou=0.40,            # Slightly stricter IOU for faces
            imgsz=640,
            half=self.half,
            device=self.device,
            agnostic_nms=True,
            verbose=False,
            augment=False
        )
        
        detections = []
        if results:
            for r in results:
                boxes = r.boxes
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    
                    # Keypoints live on r.keypoints, not box.keypoints
                    kpts = None
                    if hasattr(r, 'keypoints') and r.keypoints is not None:
                        try:
                            kpts = r.keypoints.xy[i].cpu().numpy().tolist()
                        except (IndexError, AttributeError):
                            pass

                    detections.append({
                        "box": [int(x1), int(y1), int(x2), int(y2)],
                        "conf": conf,
                        "keypoints": kpts
                    })
        return detections
