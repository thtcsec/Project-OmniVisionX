"""
FORTRESS LPR PIPELINE
=========================================
3-Stage Pipeline tối ưu cho Vietnamese License Plates
- Stage 1: Vehicle Detection (YOLOv11m)
- Stage 2: Plate Detection + OBB (YOLOv11n-OBB) 
- Stage 3: Recognition (STN-LPRNet + Focal CTC)

Tối ưu cho:
- RTX GPU với TensorRT FP16
- Cả ban ngày lẫn ban đêm
- Speed + Accuracy cấp độ nhà nước

Author: OmniVision Team
Version: 3.0 Fortress Edition
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import logging
import os
import time
import asyncio
import threading
from enum import Enum

from app.services.ocr.adaptive_line_split import adaptive_split_2line, validate_split_result
from app.services.core.enhancer import ImageEnhancer
from app.services.ocr.gpu_ctc_decoder import decode_ctc_gpu

# Optional glare handling
try:
    from app.services.ocr.retroreflective_sim import detect_glare, suppress_glare
    GLARE_HANDLING_AVAILABLE = True
except ImportError:
    GLARE_HANDLING_AVAILABLE = False

# VN plate validator (optional)
try:
    from app.services.plate.vn_plate_validator import validate_and_correct_plate
    from app.services.plate.plate_constants import VALID_PROVINCE_CODES
    VN_VALIDATOR_AVAILABLE = True
except ImportError:
    VN_VALIDATOR_AVAILABLE = False
    VALID_PROVINCE_CODES = None

from app.config import get_settings
from app.services.pipeline.collectors.data_collection import DataCollector
from app.services.plate.plate_utils import normalize_plate_basic, is_valid_vn_plate_format, classify_plate_color_hsv

logger = logging.getLogger(__name__)


# ============================================
# CONFIGURATION
# ============================================
class PlateType(Enum):
    """Vietnamese plate layout types (line arrangement)"""
    ONE_LINE = "1line"      # Wide layout (e.g., 520x110)
    TWO_LINE = "2line"      # Compact layout (e.g., 330x165, 190x140)
    UNKNOWN = "unknown"


# Aspect ratio threshold to treat as wide (likely 1-line layout)
LONG_PLATE_AR = 3.2

# Two-line plate confidence weights: bottom line (serial number)
# carries more information than top line (province code), so it
# gets higher weight in the weighted average.
TWO_LINE_TOP_WEIGHT = 0.4
TWO_LINE_BOTTOM_WEIGHT = 0.6


class PlateColor(Enum):
    """Plate color = legal status"""
    WHITE = "private"       # Xe tư nhân
    BLUE = "government"     # Xe công
    YELLOW = "commercial"   # Xe kinh doanh (taxi, grab)
    RED = "military"        # Quân đội
    UNKNOWN = "unknown"


@dataclass
class PlateResult:
    """Single plate recognition result"""
    plate_text: str                 # "29A-12345" hoặc "29-A1/234.56"
    confidence: float               # 0.0 - 1.0
    plate_type: PlateType           # 1-line or 2-line
    plate_color: PlateColor         # Legal status
    bbox: Tuple[int, int, int, int] # (x1, y1, x2, y2) in original frame
    corners: Optional[np.ndarray]   # 4 corners for OBB
    plate_crop: Optional[np.ndarray] # Cropped plate image
    processing_time_ms: float       # Pipeline latency


@dataclass
class FrameResult:
    """All plates detected in a single frame"""
    plates: List[PlateResult]
    frame_id: int
    timestamp: float
    total_time_ms: float
    

# ============================================
# STAGE 1: VEHICLE DETECTION
# ============================================
class VehicleDetector:
    """
    Stage 1: Detect vehicles in full frame
    Using YOLOv11m for robust detection at distance
    
    Why detect vehicle first?
    - Constrains search space for plate detection
    - Reduces false positives (text on signs)
    - Enables tracking (DeepSORT)
    """
    
    VEHICLE_CLASSES = ['car', 'motorcycle', 'bus', 'truck']
    
    def __init__(self, model_path: Optional[str] = None, 
                 device: str = 'cuda',
                 conf_threshold: float = 0.4,
                 use_tensorrt: bool = True):
        """
        Args:
            model_path: Path to YOLOv11m weights
            device: 'cuda' or 'cpu'
            conf_threshold: Detection confidence threshold
            use_tensorrt: Use TensorRT FP16 if available
        """
        self.device = device
        self.conf_threshold = conf_threshold
        self.use_tensorrt = use_tensorrt
        self.model = None
        
        # Try to load model
        if model_path and Path(model_path).exists():
            self._load_model(model_path)
        else:
            logger.warning("⚠️ Vehicle detector model not found, will use plate-only mode")
    
    def _load_model(self, model_path: str):
        """Load YOLOv11 model"""
        try:
            from ultralytics import YOLO
            
            self.model = YOLO(model_path)
            
            # Export to TensorRT if available and requested
            if self.use_tensorrt and self.device == 'cuda':
                trt_path = model_path.replace('.pt', '_fp16.engine')
                if not Path(trt_path).exists():
                    logger.info("🔧 Exporting to TensorRT FP16...")
                    exported_path = self.model.export(format='engine', half=True, device=0)
                    # Reload the exported TensorRT engine (export() returns actual path)
                    actual_trt = exported_path if exported_path else trt_path
                    if Path(str(actual_trt)).exists():
                        self.model = YOLO(str(actual_trt))
                        logger.info("✅ TensorRT engine loaded: %s", actual_trt)
                    else:
                        logger.warning("⚠️ TensorRT export completed but engine not found at %s, using PyTorch", actual_trt)
                else:
                    self.model = YOLO(trt_path)
            
            logger.info(f"✅ Vehicle detector loaded: {model_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load vehicle detector: {e}")
            self.model = None
    
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect vehicles in frame
        
        Returns:
            List of {bbox, class, confidence, crop}
        """
        if self.model is None:
            # Fallback: return full frame as single "vehicle"
            h, w = frame.shape[:2]
            return [{
                'bbox': (0, 0, w, h),
                'class': 'unknown',
                'confidence': 1.0,
                'crop': frame
            }]
        
        # AmbientAdapter: EMA-smoothed confidence replaces binary is_night toggle
        base_conf = self.conf_threshold
        try:
            from app.services.core.ambient_adapter import AmbientAdapter
            _amb = AmbientAdapter.get_instance()
            camera_id = _amb.get_active_camera()
            adaptive_conf = _amb.get_threshold(camera_id, "fortress_vehicle_confidence", base_conf)
        except Exception:
            # Fallback to legacy binary is_night logic
            try:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                is_night = float(np.percentile(hsv[:, :, 2], 90)) < 80
            except Exception:
                is_night = False
            adaptive_conf = max(0.15, base_conf - 0.10) if is_night else min(0.9, base_conf + 0.05)

        # Optimize: force imgsz=640 to prevent native 1080p/4K processing
        # Greatly reduces GPU/CPU load and latency at no significant mAP loss for vehicles
        results = self.model(frame, conf=adaptive_conf, imgsz=640, verbose=False)
        
        vehicles = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                cls_name = result.names[cls_id]
                
                if cls_name in self.VEHICLE_CLASSES:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    
                    # Clamp to frame bounds and skip degenerate crops
                    h_f, w_f = frame.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w_f, x2), min(h_f, y2)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    
                    # Crop vehicle region
                    crop = frame[y1:y2, x1:x2]
                    
                    vehicles.append({
                        'bbox': (x1, y1, x2, y2),
                        'class': cls_name,
                        'confidence': conf,
                        'crop': crop
                    })
        
        return vehicles


# ============================================
# STAGE 2: PLATE DETECTION WITH OBB
# ============================================
class PlateDetectorOBB:
    """
    Stage 2: Detect license plates with Oriented Bounding Box
    Using YOLOv11n-OBB for fast + accurate plate localization
    
    Key features:
    - Predicts rotation angle θ for tilted plates
    - Classifies 1-line vs 2-line plates
    - Returns 4 corners for perspective correction
    """
    
    def __init__(self, model_path: Optional[str] = None,
                 device: str = 'cuda',
                 conf_threshold: float = 0.5,
                 use_tensorrt: bool = True):
        """
        Args:
            model_path: Path to YOLOv11-OBB weights
            device: 'cuda' or 'cpu'
            conf_threshold: Detection confidence
            use_tensorrt: Use TensorRT acceleration
        """
        self.device = device
        self.conf_threshold = conf_threshold
        self.use_tensorrt = use_tensorrt
        self.model = None
        self._adaptive_preprocessor = None  # lazy-init, reused across frames
        
        # Fallback to traditional CV if no model
        self.use_cv_fallback = True
        
        if model_path and Path(model_path).exists():
            self._load_model(model_path)
    
    def _load_model(self, model_path: str):
        """Load YOLOv11-OBB model"""
        try:
            from ultralytics import YOLO
            if self.use_tensorrt:
                engine_path = os.path.splitext(model_path)[0] + ".engine"
                if Path(engine_path).exists():
                    self.model = YOLO(engine_path)
                    self.use_cv_fallback = False
                    logger.info(f"✅ Plate OBB detector loaded (TensorRT): {engine_path}")
                    return

            self.model = YOLO(model_path)
            self.use_cv_fallback = False
            logger.info(f"✅ Plate OBB detector loaded: {model_path}")
        except Exception as e:
            logger.error(f"❌ Failed to load OBB model: {e}")
            self.use_cv_fallback = True
    
    def detect(self, vehicle_crop: np.ndarray, 
               vehicle_bbox: Tuple[int, int, int, int]) -> List[Dict]:
        """
        Detect plates in vehicle crop
        
        Args:
            vehicle_crop: Cropped vehicle image
            vehicle_bbox: Vehicle bbox in original frame (for coordinate mapping)
        
        Returns:
            List of {bbox, corners, angle, plate_type, confidence, crop}
        """
        if self.model is not None and not self.use_cv_fallback:
            return self._detect_yolo_obb(vehicle_crop, vehicle_bbox)
        else:
            return self._detect_cv_fallback(vehicle_crop, vehicle_bbox)
    
    def _detect_yolo_obb(self, vehicle_crop: np.ndarray,
                         vehicle_bbox: Tuple[int, int, int, int]) -> List[Dict]:
        """YOLOv11-OBB detection"""
        # Day/night adaptive confidence for plate localization
        # Preprocess the vehicle crop adaptively
        is_night = False
        try:
            if self._adaptive_preprocessor is None:
                self._adaptive_preprocessor = AdaptivePreprocessor()
            enhanced_crop = self._adaptive_preprocessor.preprocess(vehicle_crop.copy())
            is_night = self._adaptive_preprocessor.is_night_image(vehicle_crop)
        except Exception as e:
            logger.warning("AdaptivePreprocessor failed (night mode disabled for this frame): %s", e)
            enhanced_crop = vehicle_crop.copy()

        # AmbientAdapter: EMA-smoothed confidence replaces binary is_night toggle
        base_conf = self.conf_threshold
        try:
            from app.services.core.ambient_adapter import AmbientAdapter
            _amb = AmbientAdapter.get_instance()
            camera_id = _amb.get_active_camera()
            adaptive_conf = _amb.get_threshold(camera_id, "fortress_plate_confidence", base_conf)
        except Exception:
            # Fallback to legacy binary logic
            adaptive_conf = max(0.20, base_conf - 0.10) if is_night else min(0.9, base_conf + 0.05)

        # Detect using YOLO-OBB on the small visually-enhanced vehicle crop (saves 90% GPU power compared to full frame)
        results = self.model(enhanced_crop, conf=adaptive_conf, verbose=False)
        
        plates = []
        vx1, vy1, vx2, vy2 = vehicle_bbox
        
        for result in results:
            if hasattr(result, 'obb') and result.obb is not None:
                for obb in result.obb:
                    # OBB format: (cx, cy, w, h, angle)
                    cx, cy, w, h, angle = obb.xywhr[0]
                    conf = float(obb.conf[0])
                    cls_id = int(obb.cls[0])
                    
                    # Get 4 corners
                    corners = self._get_corners_from_obb(cx, cy, w, h, angle)
                    
                    # Map to original frame coordinates
                    corners_global = corners + np.array([vx1, vy1])
                    
                    # Determine plate layout from aspect ratio
                    aspect_ratio = w / h if h > 0 else 1.0
                    plate_type = PlateType.ONE_LINE if aspect_ratio > LONG_PLATE_AR else PlateType.TWO_LINE
                    
                    # Crop and rectify plate
                    plate_crop = self._rectify_plate(vehicle_crop, corners, angle)
                    
                    # Calculate bbox from corners
                    x_min, y_min = corners.min(axis=0)
                    x_max, y_max = corners.max(axis=0)
                    
                    plates.append({
                        'bbox': (int(vx1 + x_min), int(vy1 + y_min), 
                                int(vx1 + x_max), int(vy1 + y_max)),
                        'corners': corners_global,
                        'angle': float(angle),
                        'plate_type': plate_type,
                        'confidence': conf,
                        'crop': plate_crop
                    })

        # Regular YOLO box fallback (LP_detector.pt / non-OBB models)
        if not plates:
            for result in results:
                if hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
                    for box in result.boxes:
                        x1r, y1r, x2r, y2r = map(int, box.xyxy[0].cpu().numpy())
                        conf_r = float(box.conf[0])
                        w_box = max(1, x2r - x1r)
                        h_box = max(1, y2r - y1r)
                        aspect = w_box / h_box
                        if aspect < 1.2 or aspect > 7.0 or w_box < 20 or h_box < 10:
                            continue
                        corners_r = np.array([
                            [x1r, y1r], [x2r, y1r], [x2r, y2r], [x1r, y2r]
                        ], dtype=np.float32)
                        corners_global_r = corners_r + np.array([vx1, vy1])
                        plate_type_r = PlateType.ONE_LINE if aspect > LONG_PLATE_AR else PlateType.TWO_LINE
                        cy1 = max(0, y1r)
                        cy2 = min(vehicle_crop.shape[0], y2r)
                        cx1 = max(0, x1r)
                        cx2 = min(vehicle_crop.shape[1], x2r)
                        crop_r = vehicle_crop[cy1:cy2, cx1:cx2]
                        if crop_r.size == 0:
                            continue
                        plates.append({
                            'bbox': (vx1 + x1r, vy1 + y1r, vx1 + x2r, vy1 + y2r),
                            'corners': corners_global_r,
                            'angle': 0.0,
                            'plate_type': plate_type_r,
                            'confidence': conf_r,
                            'crop': crop_r
                        })

        return plates
    
    def _detect_cv_fallback(self, vehicle_crop: np.ndarray,
                            vehicle_bbox: Tuple[int, int, int, int]) -> List[Dict]:
        """
        Traditional CV fallback - CUDA accelerated
        Uses edge detection + contour analysis
        """
        h, w = vehicle_crop.shape[:2]
        vx1, vy1, vx2, vy2 = vehicle_bbox
        
        # Convert to grayscale
        if len(vehicle_crop.shape) == 3:
            gray = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = vehicle_crop
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Edge detection
        edges = cv2.Canny(enhanced, 50, 150)
        
        # Morphological operations to connect edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
        dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        plates = []
        for contour in contours:
            # Filter by area
            area = cv2.contourArea(contour)
            if area < 500 or area > (h * w * 0.5):
                continue
            
            # Get rotated rectangle
            rect = cv2.minAreaRect(contour)
            (cx, cy), (rw, rh), angle = rect
            
            # Ensure width > height
            if rw < rh:
                rw, rh = rh, rw
                angle += 90
            
            # Filter by aspect ratio (plates are 2:1 to 5:1)
            aspect_ratio = rw / rh if rh > 0 else 0
            if aspect_ratio < 1.5 or aspect_ratio > 6.0:
                continue
            
            # Get 4 corners
            corners = cv2.boxPoints(rect).astype(np.float32)
            corners_global = corners + np.array([vx1, vy1])
            
            # Determine plate layout
            plate_type = PlateType.ONE_LINE if aspect_ratio > LONG_PLATE_AR else PlateType.TWO_LINE
            
            # Rectify plate
            plate_crop = self._rectify_plate(vehicle_crop, corners, angle)
            
            # Calculate bbox
            x_min, y_min = corners.min(axis=0)
            x_max, y_max = corners.max(axis=0)
            
            plates.append({
                'bbox': (int(vx1 + x_min), int(vy1 + y_min),
                        int(vx1 + x_max), int(vy1 + y_max)),
                'corners': corners_global,
                'angle': float(angle),
                'plate_type': plate_type,
                'confidence': 0.7,  # CV fallback has lower confidence
                'crop': plate_crop
            })
        
        # Sort by area (largest first)
        plates.sort(key=lambda p: (p['bbox'][2] - p['bbox'][0]) * (p['bbox'][3] - p['bbox'][1]), reverse=True)
        
        return plates[:3]  # Max 3 plates per vehicle
    
    def _get_corners_from_obb(self, cx: float, cy: float, 
                               w: float, h: float, angle: float) -> np.ndarray:
        """Convert OBB (cx, cy, w, h, angle) to 4 corners.
        
        NOTE: Ultralytics OBB xywhr returns angle in RADIANS.
        Do NOT convert with np.radians() again.
        """
        # Rotation matrix (angle already in radians from YOLO OBB)
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        
        # Half dimensions
        hw, hh = w / 2, h / 2
        
        # Corner offsets (before rotation)
        offsets = np.array([
            [-hw, -hh],  # top-left
            [hw, -hh],   # top-right
            [hw, hh],    # bottom-right
            [-hw, hh],   # bottom-left
        ])
        
        # Rotate offsets
        rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        rotated = offsets @ rotation.T
        
        # Add center
        corners = rotated + np.array([cx, cy])
        
        return corners.astype(np.float32)
    
    def _rectify_plate(self, image: np.ndarray, 
                       corners: np.ndarray, angle: float) -> np.ndarray:
        """
        Rectify (unwarp) plate to frontal view
        Using perspective transform based on 4 corners
        """
        # Sort corners: TL, TR, BR, BL
        corners = self._order_corners(corners)

        # Expand corners outward to avoid tight crops
        h_img, w_img = image.shape[:2]
        center = corners.mean(axis=0)
        pad_ratio = 0.08
        corners = center + (corners - center) * (1.0 + pad_ratio)
        corners[:, 0] = np.clip(corners[:, 0], 0, w_img - 1)
        corners[:, 1] = np.clip(corners[:, 1], 0, h_img - 1)
        
        # Destination size based on plate type
        w = int(np.linalg.norm(corners[1] - corners[0]))
        h = int(np.linalg.norm(corners[3] - corners[0]))
        
        # Guard against degenerate OBB corners (near-zero area)
        if h < 1 or w < 1:
            logger.debug("Degenerate OBB corners (w=%d, h=%d), returning blank rectified", w, h)
            return np.zeros((140, 220, 3), dtype=np.uint8)
        
        # Standard output sizes (higher for OCR clarity)
        if w / h > LONG_PLATE_AR:  # wide layout
            dst_w, dst_h = 280, 70
        else:  # compact layout
            dst_w, dst_h = 220, 140
        
        dst_corners = np.array([
            [0, 0],
            [dst_w - 1, 0],
            [dst_w - 1, dst_h - 1],
            [0, dst_h - 1]
        ], dtype=np.float32)
        
        # Perspective transform
        M = cv2.getPerspectiveTransform(corners, dst_corners)
        rectified = cv2.warpPerspective(image, M, (dst_w, dst_h))
        
        return rectified
    
    def _order_corners(self, corners: np.ndarray) -> np.ndarray:
        """Order corners: top-left, top-right, bottom-right, bottom-left.

        Sorting by Y then splitting into top/bottom pairs fails for plates rotated
        ~45° — diagonally-opposite corners end up in the same group, producing a
        crossed quadrilateral and a mirrored/corrupted rectification.

        The standard document-scanner approach uses sum/diff of coordinates:
          - top-left     → min(x + y)   (smallest combined coordinate)
          - bottom-right → max(x + y)   (largest combined coordinate)
          - top-right    → min(y - x)   (smallest y-x, i.e. most to the right relative to top)
          - bottom-left  → max(y - x)   (largest y-x)

        This is rotation-invariant and handles 0°–90°+ tilt correctly.
        """
        pts = corners.reshape(4, 2)
        ordered = np.zeros((4, 2), dtype=np.float32)

        s = pts.sum(axis=1)            # x + y per point
        ordered[0] = pts[np.argmin(s)] # top-left
        ordered[2] = pts[np.argmax(s)] # bottom-right

        d = np.diff(pts, axis=1)       # y - x per point
        ordered[1] = pts[np.argmin(d)] # top-right (min y-x)
        ordered[3] = pts[np.argmax(d)] # bottom-left (max y-x)

        return ordered


# ============================================
# STAGE 3: PLATE RECOGNITION (STN-LPRNet)
# ============================================
class PlateRecognizer:
    """
    Stage 3: Read characters from plate crop
    Using STN-LPRNet with Focal CTC Loss
    
    Key features:
    - STN for residual perspective correction
    - Focal CTC for handling confusing chars (8/B, 0/D)
    - Split strategy for 2-line plates
    """
    
    # Vietnamese plate characters
    VOCAB = "0123456789ABCDEFGHKLMNPRSTUVXYZ-."
    
    def __init__(self, model_path: Optional[str] = None,
                 device: str = 'cuda',
                 use_tensorrt: bool = True):
        """
        Args:
            model_path: Path to STN-LPRNet weights
            device: 'cuda' or 'cpu'
            use_tensorrt: Use TensorRT acceleration
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.use_tensorrt = use_tensorrt
        self.model = None
        
        # Character mappings
        self.char2idx = {c: i + 1 for i, c in enumerate(self.VOCAB)}
        self.idx2char = {i + 1: c for i, c in enumerate(self.VOCAB)}
        self.idx2char[0] = ''  # blank
        
        # Try to load model
        if model_path and Path(model_path).exists():
            self._load_model(model_path)
        else:
            logger.warning("⚠️ STN-LPRNet not found, using PaddleOCR fallback")
            self._init_paddleocr_fallback()
    
    def _load_model(self, model_path: str):
        """Load STN-LPRNet model"""
        try:
            # Import LPRv3 from models
            from app.models.lprv3 import LPRv3
            
            self.model = LPRv3(num_classes=len(self.VOCAB) + 1)
            
            # Load weights
            state_dict = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            self.model.load_state_dict(state_dict)
            
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Switch to deploy mode (fuse BN)
            self.model.switch_to_deploy()
            
            logger.info(f"✅ STN-LPRNet loaded: {model_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load STN-LPRNet: {e}")
            self._init_paddleocr_fallback()
    
    def _init_paddleocr_fallback(self):
        """Initialize PaddleOCR as fallback"""
        try:
            from paddleocr import PaddleOCR
            # CRITICAL FIX: Use 'ch' (Chinese) model instead of 'en' (English)
            # Chinese model is trained on more diverse characters including numbers
            # and handles license plate fonts MUCH better than English model.
            # Research shows: Generic OCR (English) causes 8/B, 0/D, 5/S confusion.
            # Chinese model has better character discrimination for plate-like text.
            try:
                self.paddleocr = PaddleOCR(
                    lang='ch',  # FIXED: 'ch' instead of 'en' for better plate OCR
                    use_gpu=True, 
                    show_log=False,
                    use_angle_cls=True,
                    # Optimize for license plates (short text, high contrast)
                    det_db_thresh=0.2,  # Lower threshold for small text
                    det_db_box_thresh=0.4,  # More lenient box detection
                    rec_batch_num=6,  # Batch size for recognition
                )
            except Exception as e1:
                logger.warning(f"PaddleOCR GPU init failed, trying CPU: {e1}")
                self.paddleocr = PaddleOCR(
                    lang='ch',  # FIXED: 'ch' instead of 'en'
                    use_gpu=False, 
                    show_log=False,
                    use_angle_cls=True,
                    det_db_thresh=0.2,
                    det_db_box_thresh=0.4,
                )
            self.use_paddleocr = True
            logger.info("✅ PaddleOCR 2.9.x fallback initialized (lang='ch' for better plate OCR)")
        except Exception as e:
            self.paddleocr = None
            self.use_paddleocr = False
            logger.warning(f"⚠️ PaddleOCR not available: {e}")
    
    def recognize(self, plate_crop: np.ndarray, 
                  plate_type: PlateType = PlateType.ONE_LINE) -> Tuple[str, float]:
        """
        Recognize characters in plate crop
        
        Args:
            plate_crop: Rectified plate image
            plate_type: 1-line or 2-line
        
        Returns:
            (plate_text, confidence)
        """
        if plate_crop is None or plate_crop.size == 0:
            return "", 0.0
        
        # Preprocess crop for glare/blur
        plate_crop = self._prepare_crop(plate_crop)

        # Run both layouts and choose best by validity + confidence
        if plate_type == PlateType.TWO_LINE:
            primary = self._recognize_2line(plate_crop)
            secondary = self._recognize_1line(plate_crop)
        else:
            primary = self._recognize_1line(plate_crop)
            secondary = self._recognize_2line(plate_crop)

        return self._select_best_candidate(plate_crop, plate_type, primary, secondary)

    def recognize_batch(self, plate_items: List[Tuple[np.ndarray, PlateType]]) -> List[Tuple[str, float]]:
        """
        Batch recognize plates in a frame.
        Uses STN-LPRNet batch inference when available.
        """
        if not plate_items:
            return []

        # Fallback to per-plate if no model or OCR
        if self.model is None and not self.use_paddleocr:
            return [("", 0.0) for _ in plate_items]
        if self.model is None:
            return [self.recognize(crop, ptype) for crop, ptype in plate_items]

        # Preprocess all crops first
        preprocessed: List[np.ndarray] = []
        for crop, _ptype in plate_items:
            if crop is None or crop.size == 0:
                preprocessed.append(crop)
            else:
                preprocessed.append(self._prepare_crop(crop))

        results: List[Tuple[str, float]] = [("", 0.0) for _ in plate_items]

        # Build batch for 1-line plates
        one_line_indices: List[int] = []
        one_line_crops: List[np.ndarray] = []

        # Build batch for 2-line plates (top/bottom mapping)
        two_line_indices: List[int] = []
        two_line_crops: List[np.ndarray] = []
        two_line_map: List[Tuple[int, str]] = []  # (plate_idx, part)

        for idx, (crop, ptype) in enumerate(plate_items):
            if crop is None or crop.size == 0:
                continue

            plate_crop = preprocessed[idx]

            if ptype == PlateType.TWO_LINE:
                h, w = plate_crop.shape[:2]
                # Always use adaptive split to find the natural valley
                # between line 1 (province) and line 2 (serial). The old
                # fixed 50% cut was wrong because VN plate lines have
                # unequal heights.
                top_crop, bottom_crop, _ = adaptive_split_2line(plate_crop, fallback_ratio=0.45)
                top_crop = cv2.resize(top_crop, (200, 50))
                bottom_crop = cv2.resize(bottom_crop, (200, 50))

                two_line_indices.append(idx)
                two_line_crops.append(top_crop)
                two_line_map.append((idx, "top"))
                two_line_crops.append(bottom_crop)
                two_line_map.append((idx, "bottom"))
            else:
                one_line_indices.append(idx)
                one_line_crops.append(plate_crop)

        # Run batch inference
        one_line_results: Dict[int, Tuple[str, float]] = {}
        if one_line_crops:
            batch_results = self._recognize_lprnet_batch_chunked(one_line_crops)
            for i, plate_idx in enumerate(one_line_indices):
                one_line_results[plate_idx] = batch_results[i]

        two_line_results: Dict[int, Tuple[str, float]] = {}
        if two_line_crops:
            batch_results = self._recognize_lprnet_batch_chunked(two_line_crops)
            # Assemble top/bottom
            parts: Dict[int, Dict[str, Tuple[str, float]]] = {}
            for i, (plate_idx, part) in enumerate(two_line_map):
                parts.setdefault(plate_idx, {})[part] = batch_results[i]

            for plate_idx, lines in parts.items():
                top_text, top_conf = lines.get("top", ("", 0.0))
                bottom_text, bottom_conf = lines.get("bottom", ("", 0.0))

                is_valid, _reason = validate_split_result(top_text, bottom_text)
                if not is_valid:
                    two_line_results[plate_idx] = ("", 0.0)
                else:
                    combined_text = f"{top_text}/{bottom_text}"
                    # Weighted average: bottom line (serial) is more informative
                    combined_conf = TWO_LINE_TOP_WEIGHT * top_conf + TWO_LINE_BOTTOM_WEIGHT * bottom_conf
                    two_line_results[plate_idx] = (combined_text, combined_conf)

        # Final selection per plate (with secondary fallback if needed)
        for idx, (crop, ptype) in enumerate(plate_items):
            if crop is None or crop.size == 0:
                results[idx] = ("", 0.0)
                continue

            plate_crop = preprocessed[idx]
            if ptype == PlateType.TWO_LINE:
                primary = two_line_results.get(idx, ("", 0.0))
                secondary = ("", 0.0)
                if not primary[0] or not is_valid_vn_plate_format(normalize_plate_basic(primary[0])):
                    secondary = self._recognize_1line(plate_crop)
            else:
                primary = one_line_results.get(idx, ("", 0.0))
                secondary = ("", 0.0)
                if not primary[0] or not is_valid_vn_plate_format(normalize_plate_basic(primary[0])):
                    secondary = self._recognize_2line(plate_crop)

            results[idx] = self._select_best_candidate(plate_crop, ptype, primary, secondary)

        return results

    def _select_best_candidate(self, plate_crop: np.ndarray, 
                               plate_type: PlateType,
                               primary: Tuple[str, float],
                               secondary: Tuple[str, float]) -> Tuple[str, float]:
        candidates = []
        for text, conf in (primary, secondary):
            if not text:
                continue
            normalized = normalize_plate_basic(text)
            is_valid = is_valid_vn_plate_format(normalized)
            candidates.append((text, conf, is_valid))

        if not candidates:
            return "", 0.0

        valid_candidates = [c for c in candidates if c[2]]
        if valid_candidates:
            best = max(valid_candidates, key=lambda x: x[1])
        else:
            best = max(candidates, key=lambda x: x[1])

        # AmbientAdapter: EMA-smoothed OCR confidence floor
        try:
            from app.services.core.ambient_adapter import AmbientAdapter
            from app.config import get_settings
            _amb = AmbientAdapter.get_instance()
            camera_id = _amb.get_active_camera()
            base_ocr_conf = float(get_settings().ocr_confidence_threshold)
            min_conf = _amb.get_threshold(camera_id, "ocr_confidence_threshold", base_ocr_conf)
        except Exception:
            # Fallback to legacy binary is_night logic
            try:
                if len(plate_crop.shape) == 3:
                    hsv = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2HSV)
                    is_night = float(np.percentile(hsv[:, :, 2], 90)) < 80
                else:
                    is_night = float(np.mean(plate_crop)) < 80
            except Exception:
                is_night = False
            min_conf = 0.32 if is_night else 0.42

        normalized_best = normalize_plate_basic(best[0])
        is_valid_best = is_valid_vn_plate_format(normalized_best)
        if best[1] < min_conf:
            relaxed_floor = max(0.22, float(min_conf) - 0.12)
            if not (is_valid_best and best[1] >= relaxed_floor):
                return "", 0.0

        # Enforce minimum length for compact (2-line) layouts
        if plate_type == PlateType.TWO_LINE and len(normalized_best) < 7 and best[1] < 0.50:
            return "", 0.0

        return best[0], best[1]

    def _recognize_1line(self, plate_crop: np.ndarray) -> Tuple[str, float]:
        """Single-line recognition with multi-crop retry for siêu cấp accuracy."""
        if self.model is not None:
            text, conf = self._recognize_lprnet(plate_crop)
        elif self.use_paddleocr:
            text, conf = self._recognize_paddleocr(plate_crop)
        else:
            return "", 0.0

        # Siêu cấp: If low confidence, try padded & rotated crop variants
        if conf < 0.80 and plate_crop is not None and plate_crop.size > 0:
            candidates = [(text, conf)]
            h, w = plate_crop.shape[:2]

            # Variant 1: 5% padding (catches edge characters cut off by tight crop)
            pad5 = max(2, int(min(h, w) * 0.05))
            padded5 = cv2.copyMakeBorder(plate_crop, pad5, pad5, pad5, pad5,
                                          cv2.BORDER_REPLICATE)
            # Variant 2: 10% padding (more aggressive)
            pad10 = max(3, int(min(h, w) * 0.10))
            padded10 = cv2.copyMakeBorder(plate_crop, pad10, pad10, pad10, pad10,
                                           cv2.BORDER_REPLICATE)

            # Variant 3: Slight rotation correction (±2°)
            center = (w // 2, h // 2)
            M_pos = cv2.getRotationMatrix2D(center, 2, 1.0)
            M_neg = cv2.getRotationMatrix2D(center, -2, 1.0)
            rotated_pos = cv2.warpAffine(plate_crop, M_pos, (w, h),
                                          borderMode=cv2.BORDER_REPLICATE)
            rotated_neg = cv2.warpAffine(plate_crop, M_neg, (w, h),
                                          borderMode=cv2.BORDER_REPLICATE)

            variants = [padded5, padded10, rotated_pos, rotated_neg]

            if self.model is not None:
                batch_results = self._recognize_lprnet_batch(variants)
                candidates.extend(batch_results)
            else:
                for v in variants:
                    candidates.append(self._recognize_paddleocr(v))

            # Pick best: prefer valid VN format, then highest confidence
            valid = [(t, c) for t, c in candidates
                     if t and is_valid_vn_plate_format(normalize_plate_basic(t))]
            if valid:
                best = max(valid, key=lambda x: x[1])
            else:
                best = max(candidates, key=lambda x: x[1])

            if best[1] > conf:
                text, conf = best

        return text, conf

    def _prepare_crop(self, plate_crop: np.ndarray) -> np.ndarray:
        """Enhance plate crop for OCR (lightweight, colour-preserving).

        Pipeline:
        1. Upscale tiny crops (< 140px wide) to reduce blur
        2. Detect night/IR conditions once
        3. Glare mitigation: CLAHE on L channel (if glare detected)
        4. Night enhancement: multi-scale retinex on L channel (preserves colour)
        5. Day enhancement: CLAHE on L channel + mild sharpening

        NOTE: Previous pipeline (a) ran CLAHE for glare, THEN ran
        ImageEnhancer which applied CLAHE again (double enhancement),
        and (b) converted night crops to grayscale, destroying colour
        information. Both are fixed here — colour is always preserved
        so the model can distinguish red/yellow/blue plates at night.
        """
        if plate_crop is None or plate_crop.size == 0:
            return plate_crop

        # Upscale small crops to reduce blur
        h, w = plate_crop.shape[:2]
        if w < 140 or h < 40:
            plate_crop = cv2.resize(plate_crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

        is_color = len(plate_crop.shape) == 3
        is_night = False
        has_glare = False
        glare_ratio = 0.0

        # Detect night mode (once, reused)
        try:
            if is_color:
                # Use V-channel brightness for night detection (works on BGR)
                hsv = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2HSV)
                v_p90 = float(np.percentile(hsv[:, :, 2], 90))
                is_night = v_p90 < 80
            else:
                is_night = float(np.mean(plate_crop)) < 80
        except Exception:
            pass

        # Detect glare
        if GLARE_HANDLING_AVAILABLE:
            try:
                has_glare, glare_ratio = detect_glare(plate_crop)
            except Exception:
                pass

        # Work in LAB space to enhance luminance while preserving colour
        if is_color:
            lab = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2LAB)
            l_ch, a_ch, b_ch = cv2.split(lab)
        else:
            l_ch = plate_crop.copy()

        # --- (A) Glare mitigation: CLAHE on L channel ---
        if has_glare and glare_ratio > 0.1:
            clahe_glare = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
            l_ch = clahe_glare.apply(l_ch)

        # --- (B) Night vs Day enhancement (on L channel, colour preserved) ---
        if is_night:
            # Multi-scale retinex on L channel for low-light recovery
            l_float = l_ch.astype(np.float32) + 1.0
            retinex = np.zeros_like(l_float)
            for sigma in [15, 80, 250]:
                blur = cv2.GaussianBlur(l_float, (0, 0), sigma)
                retinex += np.log10(l_float) - np.log10(blur + 1.0)
            retinex /= 3.0
            l_ch = cv2.normalize(retinex, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            # Mild CLAHE (only once — NOT stacked with glare CLAHE)
            if not (has_glare and glare_ratio > 0.1):
                clahe_night = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4))
                l_ch = clahe_night.apply(l_ch)
        else:
            # Day: mild CLAHE if not already glare-corrected
            if not (has_glare and glare_ratio > 0.1):
                clahe_day = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                l_ch = clahe_day.apply(l_ch)

            # Mild sharpening (reduced gain to avoid noise amplification)
            sharpen_kernel = np.array([[0, -0.5, 0],
                                       [-0.5, 3.0, -0.5],
                                       [0, -0.5, 0]], dtype=np.float32)
            l_ch = cv2.filter2D(l_ch, -1, sharpen_kernel)

        # Reassemble colour image
        if is_color:
            enhanced = cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2BGR)
        else:
            enhanced = l_ch

        return enhanced
    
    def _recognize_2line(self, plate_crop: np.ndarray) -> Tuple[str, float]:
        """
        Split 2-line plate and recognize each line separately
        Format: "29-A1/234.56"
        """
        if self.model is not None and hasattr(self.model, "decoder") and hasattr(self.model, "ctc_head_bottom"):
            return self._recognize_lprnet(plate_crop)

        h, w = plate_crop.shape[:2]

        # Adaptive split (fallback to 45% if no valley found)
        top_crop, bottom_crop, _ = adaptive_split_2line(plate_crop, fallback_ratio=0.45)
        
        # Resize to standard size
        top_crop = cv2.resize(top_crop, (200, 50))
        bottom_crop = cv2.resize(bottom_crop, (200, 50))
        
        # Recognize each line (batched for speed on GPU)
        if self.model is not None:
            results = self._recognize_lprnet_batch([top_crop, bottom_crop])
            top_text, top_conf = results[0]
            bottom_text, bottom_conf = results[1]
        elif self.use_paddleocr:
            top_text, top_conf = self._recognize_paddleocr(top_crop)
            bottom_text, bottom_conf = self._recognize_paddleocr(bottom_crop)
        else:
            return "", 0.0
        
        # Validate split before merging
        is_valid, _reason = validate_split_result(top_text, bottom_text)
        if not is_valid:
            return "", 0.0

        # Combine results — use same weighted average as batch path
        # for consistency (bottom/serial line is more informative)
        combined_text = f"{top_text}/{bottom_text}"
        combined_conf = TWO_LINE_TOP_WEIGHT * top_conf + TWO_LINE_BOTTOM_WEIGHT * bottom_conf
        
        return combined_text, combined_conf
    
    def _recognize_lprnet(self, plate_crop: np.ndarray) -> Tuple[str, float]:
        """STN-LPRNet recognition (single crop)"""
        results = self._recognize_lprnet_batch([plate_crop])
        return results[0] if results else ("", 0.0)

    def _recognize_lprnet_batch(self, plate_crops: List[np.ndarray]) -> List[Tuple[str, float]]:
        """STN-LPRNet recognition (batched)"""
        if not plate_crops:
            return []

        tensors = []
        for plate_crop in plate_crops:
            img = cv2.resize(plate_crop, (94, 24))
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            img = img.astype(np.float32) / 255.0
            img = (img - 0.5) / 0.5  # Normalize to [-1, 1]
            tensors.append(torch.from_numpy(img.transpose(2, 0, 1)))

        batch = torch.stack(tensors, dim=0).to(self.device)

        # Inference (FP16 if CUDA is available)
        with torch.no_grad():
            is_cuda = (hasattr(self.device, 'type') and self.device.type == 'cuda') or str(self.device).startswith('cuda')
            if is_cuda:
                with torch.amp.autocast('cuda'):
                    output = self.model(batch)
            else:
                output = self.model(batch)

        if isinstance(output, dict):
            output_top = output.get('output_top')
            output_bottom = output.get('output_bottom')
            if output_top is not None and output_bottom is not None and hasattr(self.model, "decoder"):
                return self._decode_lprv3_batch(output_top, output_bottom)
            logits = output_top if output_top is not None else output.get('features')
        else:
            logits = output

        return self._decode_ctc_batch(logits)

    def _recognize_lprnet_batch_chunked(self, plate_crops: List[np.ndarray],
                                        max_batch: int = 16) -> List[Tuple[str, float]]:
        """Run batch inference in chunks to avoid GPU OOM."""
        results: List[Tuple[str, float]] = []
        if not plate_crops:
            return results

        for i in range(0, len(plate_crops), max_batch):
            chunk = plate_crops[i:i + max_batch]
            results.extend(self._recognize_lprnet_batch(chunk))

        return results

    def _decode_ctc_batch(self, logits: torch.Tensor) -> List[Tuple[str, float]]:
        """Decode CTC output for a batch"""
        results: List[Tuple[str, float]] = []
        if logits is None:
            return results

        try:
            decoded = decode_ctc_gpu(logits, use_beam_search=True)
            if isinstance(decoded, tuple):
                decoded = [decoded]
            for text, conf in decoded:
                results.append(self._postprocess_plate(text, conf))
            return results
        except Exception:
            batch_size = logits.size(1) if logits.dim() >= 3 else 1
            if batch_size == 1:
                return [self._postprocess_plate(*self._decode_ctc(logits))]
            for b in range(batch_size):
                results.append(self._postprocess_plate(*self._decode_ctc(logits[:, b:b + 1, :])))
            return results

    def _decode_lprv3_batch(self, output_top: torch.Tensor, output_bottom: torch.Tensor) -> List[Tuple[str, float]]:
        results: List[Tuple[str, float]] = []
        if output_top is None or output_bottom is None:
            return results

        probs_top = torch.exp(output_top)
        probs_bottom = torch.exp(output_bottom)
        batch_size = probs_top.size(1) if probs_top.dim() >= 3 else 1

        for b in range(batch_size):
            top = probs_top[:, b, :] if probs_top.dim() == 3 else probs_top
            bottom = probs_bottom[:, b, :] if probs_bottom.dim() == 3 else probs_bottom
            decoded = self.model.decoder.decode(top, bottom, beam_width=5, use_beam_search=True)
            plate_text = decoded.get("plate", "")
            conf = decoded.get("confidence", 0.0)
            results.append(self._postprocess_plate(plate_text, conf))

        return results
    
    def _recognize_paddleocr(self, plate_crop: np.ndarray) -> Tuple[str, float]:
        """PaddleOCR fallback recognition"""
        if self.paddleocr is None:
            return "", 0.0
        
        try:
            result = self.paddleocr.ocr(plate_crop, cls=True)
            
            if result and result[0]:
                lines = []
                confs = []
                for line in result[0]:
                    box = line[0]
                    text = line[1][0]
                    conf = line[1][1]
                    # Compute line center for sorting
                    y_center = sum([pt[1] for pt in box]) / len(box)
                    x_center = sum([pt[0] for pt in box]) / len(box)
                    lines.append((y_center, x_center, text))
                    confs.append(conf)

                # Determine if we have two lines based on vertical spread
                h = plate_crop.shape[0]
                y_values = [y for y, _, _ in lines]
                y_min, y_max = min(y_values), max(y_values)
                two_line = (y_max - y_min) > (h * 0.2)

                if two_line:
                    median_y = sorted(y_values)[len(y_values) // 2]
                    top_line = sorted([l for l in lines if l[0] <= median_y], key=lambda x: x[1])
                    bottom_line = sorted([l for l in lines if l[0] > median_y], key=lambda x: x[1])
                    text = ''.join([t for _, _, t in top_line] + [t for _, _, t in bottom_line]).upper()
                else:
                    # Single line: sort by X only
                    lines.sort(key=lambda x: x[1])
                    text = ''.join([t for _, _, t in lines]).upper()

                confidence = min(confs) if confs else 0.0
                
                # Clean up text
                text = self._clean_plate_text(text)
                
                return text, confidence
        except Exception as e:
            logger.debug("PaddleOCR recognition failed: %s", e)
        
        return "", 0.0
    
    def _decode_ctc(self, logits: torch.Tensor) -> Tuple[str, float]:
        """Decode CTC output with greedy decoding"""
        # Get predictions
        probs = F.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)
        
        # Ensure 2D shape (T, N) even for 1D input (T,)
        if preds.dim() == 1:
            preds = preds.unsqueeze(1)
            probs = probs.unsqueeze(1) if probs.dim() == 2 else probs
        
        # Greedy decode (collapse repeats and remove blanks)
        decoded = []
        confidences = []
        prev_idx = 0
        
        for t in range(preds.size(0)):
            idx = preds[t, 0].item()
            if idx != 0 and idx != prev_idx:  # not blank and not repeat
                char = self.idx2char.get(idx, '')
                if char:
                    decoded.append(char)
                    confidences.append(probs[t, 0, idx].item())
            prev_idx = idx
        
        text = ''.join(decoded)
        confidence = np.mean(confidences) if confidences else 0.0
        
        return text, float(confidence)

    def _postprocess_plate(self, text: str, confidence: float) -> Tuple[str, float]:
        """Basic normalization after OCR decode.

        NOTE: Full validation (validate_and_correct_plate, province check)
        is done in _validate_plate() which is called later in the pipeline.
        Do NOT penalise here to avoid double penalty — only normalize.
        """
        if not text:
            return "", 0.0

        cleaned = normalize_plate_basic(text)

        return cleaned, confidence
    
    def _clean_plate_text(self, text: str) -> str:
        """Clean and normalize plate text"""
        # Remove invalid characters
        cleaned = ''.join(c for c in text if c in self.VOCAB)
        return cleaned


# ============================================
# NIGHT/DAY ADAPTIVE PROCESSOR
# ============================================
class AdaptivePreprocessor:
    """
    Adaptive preprocessing for day/night conditions
    Automatically detects lighting and applies appropriate enhancement
    """
    
    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    def is_night_image(self, image: np.ndarray) -> bool:
        """Detect if image is taken at night using V-channel 90th percentile.

        Consistent with the rest of the pipeline (VehicleDetector,
        PlateRecognizer) which all use the same V-channel p90 < 80 check.
        """
        try:
            if len(image.shape) == 3:
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                return float(np.percentile(hsv[:, :, 2], 90)) < 80
            return float(np.mean(image)) < 80
        except Exception:
            return False
    
    def is_ir_image(self, image: np.ndarray) -> bool:
        """Detect if image is from IR camera"""
        if len(image.shape) != 3:
            return True  # Grayscale likely IR
        
        # Check color variance (IR images have low color variance)
        b, g, r = cv2.split(image)
        color_variance = np.std([np.mean(b), np.mean(g), np.mean(r)])
        
        return color_variance < 10
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Apply adaptive preprocessing based on conditions"""
        is_night = self.is_night_image(image)
        is_ir = self.is_ir_image(image)
        
        if is_ir:
            return self._preprocess_ir(image)
        elif is_night:
            return self._preprocess_night(image)
        else:
            return self._preprocess_day(image)
    
    def _preprocess_day(self, image: np.ndarray) -> np.ndarray:
        """Standard daytime preprocessing"""
        if len(image.shape) == 3:
            # Convert to LAB for better contrast handling
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = self.clahe.apply(l)
            lab = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            enhanced = self.clahe.apply(image)
        
        return enhanced
    
    def _preprocess_night(self, image: np.ndarray) -> np.ndarray:
        """Night image preprocessing — colour-preserving LAB-space enhancement.

        Previous version converted to grayscale, destroying colour information
        needed for plate colour classification and character contrast at night.
        Now works on L-channel only, preserving a/b colour channels.
        """
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_ch, a_ch, b_ch = cv2.split(lab)
        else:
            l_ch = image.copy()
            a_ch = b_ch = None

        # Multi-scale retinex on L channel for night enhancement
        enhanced = self._multi_scale_retinex(l_ch)

        # Boost contrast on L channel
        enhanced = self.clahe.apply(enhanced)

        # Reassemble colour image
        if a_ch is not None:
            return cv2.cvtColor(cv2.merge([enhanced, a_ch, b_ch]), cv2.COLOR_LAB2BGR)
        return enhanced
    
    def _preprocess_ir(self, image: np.ndarray) -> np.ndarray:
        """IR camera preprocessing - handle plate washout.

        IR images have minimal colour information so grayscale is acceptable.
        Uses background-vs-center comparison for reliable inversion.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        h, w = gray.shape[:2]
        edge_thickness = max(1, h // 10)
        bg_brightness = (
            np.mean(gray[:edge_thickness, :]) + np.mean(gray[-edge_thickness:, :]) +
            np.mean(gray[:, :edge_thickness]) + np.mean(gray[:, -edge_thickness:])
        ) / 4
        center_h1, center_h2 = h // 4, 3 * h // 4
        center_w1, center_w2 = w // 4, 3 * w // 4
        center_brightness = np.mean(gray[center_h1:center_h2, center_w1:center_w2])
        
        if bg_brightness > center_brightness + 15:
            gray = cv2.bitwise_not(gray)
        
        enhanced = self.clahe.apply(gray)
        
        # Mild denoise (bilateral preserves edges better than NLM for small crops)
        enhanced = cv2.bilateralFilter(enhanced, 5, 75, 75)
        
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    
    def _multi_scale_retinex(self, image: np.ndarray, 
                              sigmas: List[int] = [15, 80, 250]) -> np.ndarray:
        """Multi-scale retinex for low-light enhancement"""
        image_float = image.astype(np.float32) + 1.0
        
        retinex = np.zeros_like(image_float)
        
        for sigma in sigmas:
            blur = cv2.GaussianBlur(image_float, (0, 0), sigma)
            retinex += np.log10(image_float) - np.log10(blur + 1.0)
        
        retinex = retinex / len(sigmas)
        
        # Normalize to 0-255
        retinex = cv2.normalize(retinex, None, 0, 255, cv2.NORM_MINMAX)
        
        return retinex.astype(np.uint8)


# ============================================
# MAIN PIPELINE
# ============================================
class FortressLPR:
    """
    🏰 FORTRESS LPR - Complete 3-Stage Pipeline
    
    Usage:
        lpr = FortressLPR()
        results = lpr.process_frame(frame)
        
        for plate in results.plates:
            print(f"Plate: {plate.plate_text}, Conf: {plate.confidence:.2f}")
    """
    
    def __init__(self, 
                 vehicle_model: Optional[str] = None,
                 plate_model: Optional[str] = None,
                 ocr_model: Optional[str] = None,
                 device: str = 'cuda',
                 use_tensorrt: bool = True):
        """
        Initialize Fortress LPR Pipeline
        
        Args:
            vehicle_model: Path to YOLOv11m vehicle detector
            plate_model: Path to YOLOv11-OBB plate detector
            ocr_model: Path to STN-LPRNet recognizer
            device: 'cuda' or 'cpu'
            use_tensorrt: Use TensorRT FP16 optimization
        """
        self.device = device
        self.use_tensorrt = use_tensorrt
        self.frame_count = 0

        settings = get_settings()
        
        # P0.3 FIX: GPU backpressure - limit concurrent verifications
        # Prevents GPU queue congestion during high load (rainy nights, peak hours)
        self._max_concurrent_verifications = 5  # Max 5 concurrent GPU operations
        self._gpu_semaphore = threading.Semaphore(self._max_concurrent_verifications)
        self._async_semaphore: Optional[asyncio.Semaphore] = None  # Lazy init for async
        self._verification_queue_size = 0
        self._dropped_verifications = 0
        
        logger.info("🏰 Initializing FORTRESS LPR Pipeline...")
        
        # Stage 1: Vehicle Detection
        self.vehicle_detector = VehicleDetector(
            model_path=vehicle_model,
            device=device,
            conf_threshold=settings.fortress_vehicle_confidence,
            use_tensorrt=use_tensorrt
        )
        
        # Stage 2: Plate Detection (OBB)
        self.plate_detector = PlateDetectorOBB(
            model_path=plate_model,
            device=device,
            conf_threshold=settings.fortress_plate_confidence,
            use_tensorrt=use_tensorrt
        )
        
        # Stage 3: Recognition
        self.recognizer = PlateRecognizer(
            model_path=ocr_model,
            device=device,
            use_tensorrt=use_tensorrt
        )
        
        # Adaptive preprocessor
        self.preprocessor = AdaptivePreprocessor()

        # Active learning / data collection
        self.data_collector = DataCollector(
            enabled=settings.enable_lpr_data_collection,
            base_dir=settings.lpr_collection_dir,
            sample_rate=settings.lpr_collect_sample_rate,
            collect_vehicles=settings.lpr_collect_vehicle,
            collect_plates=settings.lpr_collect_plate,
            min_conf=settings.lpr_collect_min_conf,
            max_conf=settings.lpr_collect_max_conf,
            low_conf_only=settings.lpr_collect_low_conf_only,
            quality_filter=settings.lpr_collect_quality_filter,
            min_sharpness=settings.lpr_collect_min_sharpness,
            min_brightness=settings.lpr_collect_min_brightness,
            max_brightness=settings.lpr_collect_max_brightness,
            min_vehicle_area=settings.lpr_collect_min_vehicle_area,
            min_plate_area=settings.lpr_collect_min_plate_area,
        )
        
        logger.info("✅ FORTRESS LPR Pipeline ready!")
    
    def process_frame(self, frame: np.ndarray, 
                      preprocess: bool = True,
                      allow_drop: bool = False) -> FrameResult:
        """
        Process a single frame through the 3-stage pipeline
        
        P0.3 FIX: With GPU backpressure control.
        
        Args:
            frame: Input BGR image
            preprocess: Apply adaptive preprocessing
            allow_drop: If True, drop frame when GPU overloaded instead of blocking
        
        Returns:
            FrameResult with all detected plates
        """
        # P0.3 FIX: GPU backpressure check
        if allow_drop and not self._gpu_semaphore.acquire(blocking=False):
            # GPU overloaded - drop this frame
            self._dropped_verifications += 1
            if self._dropped_verifications % 10 == 0:
                logger.warning(f"⚠️ GPU backpressure: dropped {self._dropped_verifications} verifications")
            return FrameResult(
                plates=[],
                frame_id=self.frame_count,
                timestamp=time.time(),
                total_time_ms=0.0
            )
        elif not allow_drop:
            # Block until GPU available
            self._gpu_semaphore.acquire(blocking=True)
        
        try:
            return self._process_frame_internal(frame, preprocess)
        finally:
            self._gpu_semaphore.release()
    
    def _process_frame_internal(self, frame: np.ndarray, 
                                 preprocess: bool = True) -> FrameResult:
        """Internal frame processing (called after acquiring semaphore)"""
        start_time = time.time()
        self.frame_count += 1
        
        plates = []
        
        # Preprocessing on full frame causes severe CPU bottleneck (Multi-Scale Retinex)
        # We pass the raw frame to the vehicle detector. Preprocessing is handled at the plate crop level instead.
        processed_frame = frame
        
        # Stage 1: Detect vehicles
        vehicles = self.vehicle_detector.detect(processed_frame)

        plate_infos: List[Dict] = []

        # IMPROVED Fallback Strategy: Always run full-frame scan as supplementary detection
        # This catches vehicles missed by Stage 1 detector (motorcycles, distant vehicles, partial occlusion)
        # Run BEFORE per-vehicle detection to establish baseline, then deduplicate with NMS
        h_frame, w_frame = processed_frame.shape[:2]
        full_frame_bbox = (0, 0, w_frame, h_frame)
        fallback_plates = []
        
        # Run fallback ALWAYS: deduplication via NMS handles overlapping results.
        # Previous condition `len(vehicles) <= 2` caused misses when detector found 3+ cars
        # but missed a motorcycle behind them.
        should_run_fallback = True
        
        if should_run_fallback:
            try:
                fallback_plates = self.plate_detector.detect(processed_frame, full_frame_bbox)
                if fallback_plates:
                    logger.info(f"🔄 Fallback scan: found {len(fallback_plates)} plates (vehicles={len(vehicles)})")
            except Exception as e:
                logger.debug(f"Full-frame fallback failed: {e}")

        for vehicle in vehicles:
            vehicle_crop = vehicle['crop']
            vehicle_bbox = vehicle['bbox']
            vehicle_conf = vehicle.get('confidence')

            # Optional data collection: vehicle crops
            self.data_collector.save_vehicle(vehicle_crop, self.frame_count, vehicle_conf)

            # Stage 2: Detect plates in vehicle
            detected_plates = self.plate_detector.detect(vehicle_crop, vehicle_bbox)

            for plate_info in detected_plates:
                plate_infos.append(plate_info)
        
        # Merge fallback plates with per-vehicle plates, then deduplicate with NMS
        if fallback_plates:
            plate_infos.extend(fallback_plates)
            # NMS deduplication: remove overlapping plates (IoU > 0.5)
            plate_infos = self._deduplicate_plates_nms(plate_infos, iou_threshold=0.5)

        if plate_infos:
            batch_start = time.time()
            batch_inputs = [(p['crop'], p['plate_type']) for p in plate_infos]
            batch_results = self.recognizer.recognize_batch(batch_inputs)
            batch_time_ms = (time.time() - batch_start) * 1000
            per_plate_time = batch_time_ms / max(1, len(plate_infos))

            for plate_info, (plate_text, confidence) in zip(plate_infos, batch_results):
                plate_crop = plate_info['crop']
                plate_type = plate_info['plate_type']

                # Detect plate color (legal status)
                plate_color = self._detect_plate_color(plate_crop)

                # Validate plate format
                plate_text, confidence = self._validate_plate(
                    plate_text, confidence, plate_type
                )

                # Create result
                result = PlateResult(
                    plate_text=plate_text,
                    confidence=confidence,
                    plate_type=plate_type,
                    plate_color=plate_color,
                    bbox=plate_info['bbox'],
                    corners=plate_info.get('corners'),
                    plate_crop=plate_crop,
                    processing_time_ms=per_plate_time
                )

                # Optional data collection: plate crops
                self.data_collector.save_plate(plate_crop, self.frame_count, confidence, plate_text)

                plates.append(result)
        
        total_time = (time.time() - start_time) * 1000
        
        return FrameResult(
            plates=plates,
            frame_id=self.frame_count,
            timestamp=time.time(),
            total_time_ms=total_time
        )
    
    def _detect_plate_color(self, plate_crop: np.ndarray) -> PlateColor:
        """Detect plate color to determine legal status"""
        color = classify_plate_color_hsv(plate_crop)
        if color == "white":
            return PlateColor.WHITE
        if color == "blue":
            return PlateColor.BLUE
        if color == "yellow":
            return PlateColor.YELLOW
        if color == "red":
            return PlateColor.RED
        return PlateColor.UNKNOWN
    
    def _validate_plate(self, text: str, confidence: float,
                        plate_type: PlateType) -> Tuple[str, float]:
        """
        Validate and correct plate text based on Vietnamese format
        
        Format rules:
        - 1-line: XX[A-Z]-XXXXX (e.g., 29A-12345)
        - 2-line: XX-XX/XXX.XX (e.g., 29-A1/234.56)
        """
        if not text:
            return "", 0.0
        
        # Province codes (valid Vietnamese provinces)
        if VN_VALIDATOR_AVAILABLE and VALID_PROVINCE_CODES:
            VALID_PROVINCES = {f"{code:02d}" for code in VALID_PROVINCE_CODES}
        else:
            # Province metadata unavailable: skip province-based penalty,
            # keep only generic normalization/length logic.
            VALID_PROVINCES = set()
        
        # Clean text
        cleaned = normalize_plate_basic(text)

        # Context-aware correction (already handles province validation + penalty)
        if VN_VALIDATOR_AVAILABLE:
            try:
                result = validate_and_correct_plate(cleaned, confidence)
                cleaned = result.corrected
                confidence = max(0.0, confidence - result.confidence_penalty)
            except Exception as e:
                logger.debug("VN plate validator failed in _validate_plate: %s", e)
                # Fallback: check province manually when validator crashes
                if VALID_PROVINCES and len(cleaned) >= 2 and cleaned[:2] not in VALID_PROVINCES:
                    confidence *= 0.85
        else:
            # Only apply province check when VN validator is NOT available
            # (avoid double penalty — validate_and_correct_plate already penalises)
            if VALID_PROVINCES and len(cleaned) >= 2:
                province = cleaned[:2]
                if province not in VALID_PROVINCES:
                    confidence *= 0.85
        
        # Length-based confidence penalty (graduated, mutually exclusive)
        plate_len = len(cleaned)
        if plate_len <= 6:
            confidence *= 0.72   # Keep low-confidence short reads for temporal consensus
        elif plate_len > 12:
            confidence *= 0.90   # Mild penalty only; validator handles stronger corrections
        # 7-char plates are valid VN format (e.g., 65A1234 — 4-digit serial, older plates)
        # 8+ char plates are also valid — no penalty needed
        
        return cleaned, confidence
    
    def _deduplicate_plates_nms(self, plates: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
        """
        Remove duplicate plates using Non-Maximum Suppression (NMS)
        
        When fallback detection and per-vehicle detection both find the same plate,
        we need to deduplicate to avoid double-counting. This uses IoU-based NMS
        to keep only the highest-confidence detection for overlapping plates.
        
        Args:
            plates: List of plate dicts with 'bbox' and optional 'confidence'
            iou_threshold: IoU threshold for considering plates as duplicates (default 0.5)
        
        Returns:
            Deduplicated list of plates
        """
        if len(plates) <= 1:
            return plates
        
        # Extract bboxes and confidences
        bboxes = []
        confidences = []
        for plate in plates:
            bbox = plate.get('bbox')
            if bbox is None or len(bbox) != 4:
                continue
            bboxes.append(bbox)
            # Use confidence if available, otherwise assume 1.0
            conf = plate.get('confidence', 1.0)
            confidences.append(conf)
        
        if not bboxes:
            return plates
        
        # Convert to numpy arrays
        bboxes = np.array(bboxes, dtype=np.float32)
        confidences = np.array(confidences, dtype=np.float32)
        
        # Sort by confidence (descending)
        indices = np.argsort(confidences)[::-1]
        
        keep_indices = []
        while len(indices) > 0:
            # Keep the highest confidence detection
            current = indices[0]
            keep_indices.append(current)
            
            if len(indices) == 1:
                break
            
            # Calculate IoU with remaining boxes
            current_bbox = bboxes[current]
            remaining_bboxes = bboxes[indices[1:]]
            
            # Calculate intersection
            x1 = np.maximum(current_bbox[0], remaining_bboxes[:, 0])
            y1 = np.maximum(current_bbox[1], remaining_bboxes[:, 1])
            x2 = np.minimum(current_bbox[2], remaining_bboxes[:, 2])
            y2 = np.minimum(current_bbox[3], remaining_bboxes[:, 3])
            
            intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
            
            # Calculate union
            current_area = (current_bbox[2] - current_bbox[0]) * (current_bbox[3] - current_bbox[1])
            remaining_areas = (remaining_bboxes[:, 2] - remaining_bboxes[:, 0]) * \
                             (remaining_bboxes[:, 3] - remaining_bboxes[:, 1])
            union = current_area + remaining_areas - intersection
            
            # Calculate IoU
            iou = intersection / np.maximum(union, 1e-6)
            
            # Keep only boxes with IoU below threshold
            indices = indices[1:][iou < iou_threshold]
        
        # Return deduplicated plates
        return [plates[i] for i in keep_indices]
    
    def process_video(self, video_path: str, 
                      output_path: Optional[str] = None,
                      show_preview: bool = False) -> List[FrameResult]:
        """
        Process a video file
        
        Args:
            video_path: Path to input video
            output_path: Path to save annotated video (optional)
            show_preview: Show live preview window
        
        Returns:
            List of FrameResult for each frame
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.error(f"❌ Cannot open video: {video_path}")
            return []
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        all_results = []
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                result = self.process_frame(frame)
                all_results.append(result)
                
                # Draw annotations
                annotated = self._draw_annotations(frame, result)
                
                if writer:
                    writer.write(annotated)
                
                if show_preview:
                    cv2.imshow('Fortress LPR', annotated)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        finally:
            cap.release()
            if writer:
                writer.release()
            if show_preview:
                cv2.destroyAllWindows()
        
        return all_results
    
    def _draw_annotations(self, frame: np.ndarray, 
                          result: FrameResult) -> np.ndarray:
        """Draw detection results on frame"""
        annotated = frame.copy()
        
        for plate in result.plates:
            x1, y1, x2, y2 = plate.bbox
            
            # Color based on plate color
            colors = {
                PlateColor.WHITE: (255, 255, 255),
                PlateColor.BLUE: (255, 0, 0),
                PlateColor.YELLOW: (0, 255, 255),
                PlateColor.RED: (0, 0, 255),
                PlateColor.UNKNOWN: (0, 255, 0),
            }
            color = colors.get(plate.plate_color, (0, 255, 0))
            
            # Draw bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw corners if available
            if plate.corners is not None:
                for i, corner in enumerate(plate.corners):
                    cv2.circle(annotated, tuple(map(int, corner)), 5, (0, 0, 255), -1)
            
            # Draw text
            label = f"{plate.plate_text} ({plate.confidence:.0%})"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
            cv2.putText(annotated, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Draw FPS
        fps_text = f"FPS: {1000/max(result.total_time_ms, 1):.1f}"
        cv2.putText(annotated, fps_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return annotated


# ============================================
# CONVENIENCE FUNCTIONS
# ============================================
import threading
_fortress_lpr: Optional[FortressLPR] = None
_fortress_lock = threading.Lock()


def _resolve_model_path(weights_dir: str, primary_name: str, fallback_names: list) -> str:
    """Resolve model path: try primary name, then fallbacks, including TensorRT engines."""
    primary = os.path.join(weights_dir, primary_name)
    if os.path.exists(primary):
        return primary
    engine = os.path.splitext(primary)[0] + ".engine"
    if os.path.exists(engine):
        return engine
    for name in fallback_names:
        path = os.path.join(weights_dir, name)
        if os.path.exists(path):
            logger.info("📦 Model resolved: %s → %s", primary_name, name)
            return path
        eng = os.path.splitext(path)[0] + ".engine"
        if os.path.exists(eng):
            logger.info("📦 Model resolved: %s → %s (TRT)", primary_name, os.path.basename(eng))
            return eng
    return primary


def get_fortress_lpr(**kwargs) -> FortressLPR:
    """Get singleton instance of Fortress LPR.

    When called with no explicit model paths (the common case from stream_consumer),
    resolves weights from settings.weights_dir with smart fallback to actual available
    model files (yolo11m.pt, LP_detector.pt, lprnet_vn_best.pt, etc.).
    """
    global _fortress_lpr
    if _fortress_lpr is None:
        with _fortress_lock:
            if _fortress_lpr is None:
                if not kwargs.get("plate_model") and not kwargs.get("vehicle_model"):
                    try:
                        _cfg = get_settings()
                        _w = _cfg.weights_dir
                        kwargs.setdefault("plate_model", _resolve_model_path(
                            _w, "yolov11-obb-vnplate.pt",
                            ["LP_detector.pt", "plate_detector.pt", "best.pt"]
                        ))
                        kwargs.setdefault("vehicle_model", _resolve_model_path(
                            _w, "yolov11m-vehicle.pt",
                            ["yolo11m.pt", "yolov11m.pt", "yolov11s.pt", "yolo11s.pt"]
                        ))
                        kwargs.setdefault("ocr_model", _resolve_model_path(
                            _w, "stn_lprnet_best.pt",
                            ["lprnet_vn_best.pt", "lprnet_best.pt"]
                        ))
                    except Exception as _e:
                        logger.warning("get_fortress_lpr: could not resolve model paths from settings: %s", _e)
                _fortress_lpr = FortressLPR(**kwargs)
    return _fortress_lpr


def read_plate(image: np.ndarray) -> List[PlateResult]:
    """
    Quick function to read plates from image
    
    Args:
        image: BGR image
    
    Returns:
        List of PlateResult
    """
    lpr = get_fortress_lpr()
    result = lpr.process_frame(image)
    return result.plates


# ============================================
# CLI INTERFACE
# ============================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="🏰 Fortress LPR - Vietnamese License Plate Recognition")
    parser.add_argument('--image', type=str, help='Process single image')
    parser.add_argument('--video', type=str, help='Process video file')
    parser.add_argument('--output', type=str, help='Output path for annotated video')
    parser.add_argument('--preview', action='store_true', help='Show live preview')
    parser.add_argument('--device', type=str, default='cuda', help='Device: cuda or cpu')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    lpr = FortressLPR(device=args.device)
    
    if args.image:
        # Process single image
        img = cv2.imread(args.image)
        result = lpr.process_frame(img)
        
        print(f"\n🏰 Fortress LPR Results:")
        print(f"   Processing time: {result.total_time_ms:.2f}ms")
        print(f"   Plates detected: {len(result.plates)}")
        
        for i, plate in enumerate(result.plates):
            print(f"\n   Plate {i+1}:")
            print(f"     Text: {plate.plate_text}")
            print(f"     Confidence: {plate.confidence:.2%}")
            print(f"     Type: {plate.plate_type.value}")
            print(f"     Color: {plate.plate_color.value}")
        
        # Show result
        annotated = lpr._draw_annotations(img, result)
        cv2.imshow('Result', annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    elif args.video:
        # Process video
        results = lpr.process_video(
            args.video,
            output_path=args.output,
            show_preview=args.preview
        )
        
        # Summary
        total_plates = sum(len(r.plates) for r in results)
        avg_time = np.mean([r.total_time_ms for r in results])
        
        print(f"\n🏰 Video Processing Complete!")
        print(f"   Frames: {len(results)}")
        print(f"   Total plates: {total_plates}")
        print(f"   Avg time/frame: {avg_time:.2f}ms")
        print(f"   Avg FPS: {1000/avg_time:.1f}")
    
    else:
        print("🏰 Fortress LPR - Usage:")
        print("   python fortress_lpr.py --image path/to/image.jpg")
        print("   python fortress_lpr.py --video path/to/video.mp4 --preview")
        print("   python fortress_lpr.py --video input.mp4 --output output.mp4")
