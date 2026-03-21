"""
Vietnamese License Plate Reader - Fortress Edition v4.0
========================================================
UPGRADED: YOLOv11-OBB + STN-LPRNet + Focal CTC (Fortress Mode)
FALLBACK: YOLOv5 LP_detector + PaddleOCR (Legacy Mode)

Fortress Mode (NEW - Recommended):
1. YOLOv11-OBB → detect plate with 4-corner + rotation
2. STN-LPRNet → spatial transform + CTC recognition
3. Focal CTC → handles confusing chars (8/B, 0/D)
4. Night/Day adaptive preprocessing

Legacy Mode (Fallback):
1. YOLO LP_detector → detect plate bbox in image
2. TPS Rectification → fix curved/distorted plates
3. Retroreflective glare suppression → better night/IR accuracy
4. Crop + Deskew/Enhance plate region  
5. PaddleOCR → character recognition
6. Normalize + Validate Vietnamese plate format

Models:
- Fortress: yolov11-obb-vnplate.pt, stn_lprnet.pt
- Legacy: LP_detector.pt (fallback)
"""
import os
import sys
import math
import logging
import re
from typing import Optional, Tuple, List
from PIL import Image
import numpy as np
import cv2

# PaddleOCR for OCR (replaces LPRNet)
from app.services.pipeline.application.lpr_service import OCRService, normalize_vn_plate
from app.services.plate.plate_selector import PlateSelector
from app.services.plate.plate_utils import normalize_plate_basic

# PHASE 1: VN Plate Validator with grammar constraints
try:
    from app.services.plate.vn_plate_validator import validate_and_correct_plate, get_adjusted_confidence
    VN_VALIDATOR_AVAILABLE = True
except ImportError:
    VN_VALIDATOR_AVAILABLE = False

# v3 Imports: TPS Rectifier + GPU CTC + Glare handling
try:
    from app.services.ocr.tps_rectifier import get_tps_rectifier, rectify_plate
    TPS_AVAILABLE = True
except ImportError:
    TPS_AVAILABLE = False
    
try:
    from app.services.ocr.gpu_ctc_decoder import get_gpu_ctc_decoder, decode_ctc_gpu
    GPU_CTC_AVAILABLE = True
except ImportError:
    GPU_CTC_AVAILABLE = False

try:
    from app.services.ocr.retroreflective_sim import detect_glare, suppress_glare
    GLARE_HANDLING_AVAILABLE = True
except ImportError:
    GLARE_HANDLING_AVAILABLE = False

# Fortress v4 Import
try:
    from app.services.pipeline.orchestration.fortress_lpr import FortressLPR, get_fortress_lpr, PlateResult
    FORTRESS_AVAILABLE = True
except ImportError:
    FORTRESS_AVAILABLE = False

logger = logging.getLogger(__name__)

# Feature flags (can be disabled for debugging)
USE_TPS_RECTIFICATION = True and TPS_AVAILABLE
USE_GPU_CTC = True and GPU_CTC_AVAILABLE  
USE_GLARE_SUPPRESSION = True and GLARE_HANDLING_AVAILABLE
USE_FORTRESS_MODE = True and FORTRESS_AVAILABLE

logger.info(f"🏰 Fortress v4 Features: FORTRESS={USE_FORTRESS_MODE}, TPS={USE_TPS_RECTIFICATION}, GPU_CTC={USE_GPU_CTC}, Glare={USE_GLARE_SUPPRESSION}")

# Global model cache
os.environ.setdefault("YOLO_AUTOINSTALL", "0")

_lp_detector = None
_paddle_ocr = None
_fortress_lpr = None
_fortress_lock = __import__('threading').Lock()
_plate_selector = PlateSelector(temporal_window_seconds=5.0)
_last_selector_evict_ts = 0.0


def _is_fortress_enabled() -> bool:
    if not USE_FORTRESS_MODE:
        return False
    try:
        from app.config import get_settings
        settings = get_settings()
        return bool(getattr(settings, "enable_fortress_lpr", True))
    except Exception:
        return True


def get_fortress_pipeline():
    """
    Get Fortress LPR pipeline with smart model resolution.
    Uses _resolve_model_path to find actual available models on disk:
      yolov11m-vehicle.pt → yolo11m.pt, yolov11m.pt, ...
      yolov11-obb-vnplate.pt → LP_detector.pt, ...
      stn_lprnet_best.pt → lprnet_vn_best.pt, ...
    Thread-safe lazy initialization (entire init inside lock).
    """
    global _fortress_lpr

    if not _is_fortress_enabled():
        return None

    if _fortress_lpr is not None:
        return _fortress_lpr

    with _fortress_lock:
        if _fortress_lpr is not None:
            return _fortress_lpr

        from app.config import get_settings
        from app.services.pipeline.orchestration.fortress_lpr import _resolve_model_path
        settings = get_settings()
        _w = settings.weights_dir

        plate_model = _resolve_model_path(
            _w, "yolov11-obb-vnplate.pt",
            ["LP_detector.pt", "plate_detector.pt", "best.pt"]
        )
        vehicle_model = _resolve_model_path(
            _w, "yolov11m-vehicle.pt",
            ["yolo11m.pt", "yolov11m.pt", "yolov11s.pt", "yolo11s.pt"]
        )
        ocr_model = _resolve_model_path(
            _w, "stn_lprnet_best.pt",
            ["lprnet_vn_best.pt", "lprnet_best.pt"]
        )

        has_plate = os.path.exists(plate_model)
        has_vehicle = os.path.exists(vehicle_model)
        has_ocr = os.path.exists(ocr_model)

        if not has_plate and not has_vehicle:
            logger.info("📦 No Fortress models found (plate=%s, vehicle=%s), using Legacy LPR", plate_model, vehicle_model)
            return None

        try:
            _fortress_lpr = FortressLPR(
                vehicle_model=vehicle_model if has_vehicle else None,
                plate_model=plate_model if has_plate else None,
                ocr_model=ocr_model if has_ocr else None,
                device='cuda',
                use_tensorrt=True,
            )
            logger.info("✅ Fortress LPR v4 initialized (vehicle=%s, plate=%s, ocr=%s)",
                        has_vehicle, has_plate, "lprnet" if has_ocr else "paddle")
            return _fortress_lpr
        except Exception as e:
            logger.warning("⚠️ Fortress LPR init failed, using legacy: %s", e)
            return None


def get_lp_detector():
    """Lazy load plate detector model - supports YOLOv5 (LP_detector.pt) or YOLOv8/v11 (best.pt)
    
    Uses plate_detector_model and plate_detector_confidence from central config.
    """
    global _lp_detector
    if _lp_detector is not None:
        return _lp_detector
    
    import torch
    from app.config import get_settings
    from app.services.core.device_utils import resolve_device
    
    settings = get_settings()
    device = resolve_device(settings.device)  # "cuda" or "cpu"
    
    # Get model name and confidence from central config
    model_name = settings.plate_detector_model  # Default: "best.pt"
    conf_threshold = settings.plate_detector_confidence  # Default: 0.25
    
    # Priority: configured model > best.pt > LP_detector.pt
    primary_path = os.path.join(settings.weights_dir, model_name)
    best_model_path = os.path.join(settings.weights_dir, "best.pt")
    legacy_model_path = os.path.join(settings.weights_dir, "LP_detector.pt")
    
    # Try primary configured model first
    model_path = primary_path if os.path.exists(primary_path) else (
        best_model_path if os.path.exists(best_model_path) else legacy_model_path
    )
    
    if not os.path.exists(model_path):
        logger.error(f"No plate detector model found (checked: {model_name}, best.pt, LP_detector.pt)")
        return None
    
    logger.info(f"🔧 Loading plate detector: {model_path} on device={device}")

    # Try ultralytics loader first (supports YOLOv5/YOLOv8/YOLOv11 weights when deps are present)
    try:
        os.environ.setdefault("YOLO_AUTOINSTALL", "0")
        from ultralytics import YOLO
        _lp_detector = YOLO(model_path)
        _lp_detector.conf = conf_threshold
        logger.info(f"✅ Plate detector loaded (ultralytics): {model_path} (conf={conf_threshold}, device={device})")
        return _lp_detector
    except Exception as e:
        err = str(e)
        if "models.yolo" in err or "models.common" in err or "AutoInstall" in err:
            logger.info(f"Model {model_path} looks like YOLOv5/legacy, loading with torch.hub...")
        else:
            logger.warning(f"Failed to load {model_path} as YOLOv8/v11: {e}")
    
    # Fallback to YOLOv5 format (torch.hub) - pass device explicitly to avoid "Invalid device id"
    for attempt, force_reload in enumerate([False, True]):
        try:
            # Avoid app/models shadowing yolov5 "models" package
            shadowed_models = []
            for mod_name, mod in list(sys.modules.items()):
                if mod_name == "models" or mod_name.startswith("models."):
                    shadowed_models.append((mod_name, mod))
                    del sys.modules[mod_name]
            _lp_detector = torch.hub.load(
                'ultralytics/yolov5', 
                'custom', 
                path=model_path,
                device=device,
                force_reload=force_reload,
                trust_repo=True
            )
            for name, mod in shadowed_models:
                sys.modules.setdefault(name, mod)
            _lp_detector.conf = conf_threshold
            logger.info(f"✅ Plate detector loaded (YOLOv5): {model_path} (conf={conf_threshold}, device={device})")
            return _lp_detector
        except Exception as e:
            for name, mod in shadowed_models:
                sys.modules.setdefault(name, mod)
            err_msg = str(e).lower()
            if not force_reload and ("invalid device" in err_msg or "cache" in err_msg or "out of date" in err_msg):
                logger.warning(f"⚠️ torch.hub cache stale, retrying with force_reload=True: {e}")
                continue
            # If GPU fails, try CPU as last resort
            if device != "cpu" and ("cuda" in err_msg or "device" in err_msg or "gpu" in err_msg):
                logger.warning(f"⚠️ GPU loading failed, falling back to CPU: {e}")
                try:
                    _lp_detector = torch.hub.load(
                        'ultralytics/yolov5',
                        'custom',
                        path=model_path,
                        device='cpu',
                        force_reload=True,
                        trust_repo=True
                    )
                    _lp_detector.conf = conf_threshold
                    logger.info(f"✅ Plate detector loaded (YOLOv5/CPU fallback): {model_path}")
                    return _lp_detector
                except Exception as cpu_err:
                    logger.error(f"CPU fallback also failed: {cpu_err}")
            logger.error(f"Failed to load plate detector: {e}")
            return None
    return None





def change_contrast(img: np.ndarray) -> np.ndarray:
    """Apply CLAHE contrast enhancement"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    limg = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced


def compute_skew(src_img: np.ndarray, center_thres: int = 0) -> float:
    """Compute skew angle of plate"""
    # Ensure grayscale for Canny
    img = cv2.medianBlur(src_img, 3)
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
    h, w = img.shape[:2]
    edges = cv2.Canny(img, 30, 100, apertureSize=3, L2gradient=True)
    lines = cv2.HoughLinesP(edges, 1, math.pi/180, 30, minLineLength=w/1.5, maxLineGap=h/3.0)
    
    if lines is None:
        return 0.0
    
    min_line = 100
    min_line_pos = 0
    for i in range(len(lines)):
        for x1, y1, x2, y2 in lines[i]:
            center_point = [(x1+x2)/2, (y1+y2)/2]
            if center_thres == 1 and center_point[1] < 7:
                continue
            if center_point[1] < min_line:
                min_line = center_point[1]
                min_line_pos = i
    
    angle = 0.0
    cnt = 0
    for x1, y1, x2, y2 in lines[min_line_pos]:
        ang = np.arctan2(y2 - y1, x2 - x1)
        # Fix: So sánh radian (~0.52 rad = 30 độ)
        if abs(ang) <= np.deg2rad(30):
            angle += ang
            cnt += 1
    
    return (angle / cnt) * 180 / math.pi if cnt > 0 else 0.0


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate image by angle degrees"""
    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, (w, h), flags=cv2.INTER_LINEAR)
    return result


def deskew(src_img: np.ndarray, use_contrast: bool = True, center_thres: int = 0) -> np.ndarray:
    """Deskew (straighten) plate image"""
    if use_contrast:
        return rotate_image(src_img, compute_skew(change_contrast(src_img), center_thres))
    return rotate_image(src_img, compute_skew(src_img, center_thres))


def warp_plate_perspective(plate_img: np.ndarray) -> np.ndarray:
    """
    Detect 4 corners of license plate and apply perspective transform.
    This flattens tilted/rotated plates for better OCR accuracy.
    
    Uses contour detection to find the largest quadrilateral.
    Falls back to original image if detection fails.
    """
    if plate_img is None or plate_img.size == 0:
        return plate_img
    
    h, w = plate_img.shape[:2]
    
    # Convert to grayscale
    if len(plate_img.shape) == 3:
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = plate_img.copy()
    
    # Apply bilateral filter to reduce noise while preserving edges
    blurred = cv2.bilateralFilter(gray, 5, 75, 75)
    
    # Edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Dilate to connect edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return plate_img  # Fallback: return original
    
    # Find the largest contour by area
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Approximate to polygon
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    # If we got 4 corners, use perspective transform
    if len(approx) == 4:
        # Order points: top-left, top-right, bottom-right, bottom-left
        pts = approx.reshape(4, 2).astype(np.float32)
        
        # Sort by y-coordinate first
        pts_sorted_y = pts[np.argsort(pts[:, 1])]
        top_pts = pts_sorted_y[:2]  # Top two points
        bottom_pts = pts_sorted_y[2:]  # Bottom two points
        
        # Sort top by x: left first
        top_pts = top_pts[np.argsort(top_pts[:, 0])]
        # Sort bottom by x: left first
        bottom_pts = bottom_pts[np.argsort(bottom_pts[:, 0])]
        
        src_pts = np.array([
            top_pts[0],     # top-left
            top_pts[1],     # top-right
            bottom_pts[1],  # bottom-right
            bottom_pts[0]   # bottom-left
        ], dtype=np.float32)
        
        # Target dimensions (maintain aspect ratio approximately)
        target_w = max(
            np.linalg.norm(src_pts[0] - src_pts[1]),  # top edge
            np.linalg.norm(src_pts[2] - src_pts[3])   # bottom edge
        )
        target_h = max(
            np.linalg.norm(src_pts[0] - src_pts[3]),  # left edge
            np.linalg.norm(src_pts[1] - src_pts[2])   # right edge
        )
        
        # Ensure reasonable dimensions
        target_w = int(max(target_w, w * 0.8))
        target_h = int(max(target_h, h * 0.8))
        
        dst_pts = np.array([
            [0, 0],
            [target_w - 1, 0],
            [target_w - 1, target_h - 1],
            [0, target_h - 1]
        ], dtype=np.float32)
        
        # Compute perspective transform matrix
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        
        # Apply transform
        warped = cv2.warpPerspective(plate_img, M, (target_w, target_h))
        
        print(f"   📐 Applied perspective correction (4-corner warp)", flush=True)
        return warped
    
    # If not 4 corners, try minAreaRect as fallback
    rect = cv2.minAreaRect(largest_contour)
    angle = rect[2]
    
    # Only rotate if angle is significant
    if abs(angle) > 2 and abs(angle) < 88:
        # minAreaRect returns angles in [-90, 0) range
        if angle < -45:
            angle = 90 + angle
        
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(plate_img, M, (w, h))
        print(f"   📐 Applied rotation correction ({angle:.1f}°)", flush=True)
        return rotated
    
    return plate_img  # No correction needed


def detect_blur(image: np.ndarray) -> float:
    """
    Detect image blur using Laplacian variance.
    Lower value = more blurry.
    
    Returns:
        blur_score: Higher is sharper (typical range 50-500)
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def detect_night_mode(image: np.ndarray, threshold: int = 80, percentile: int = 90) -> bool:
    """Check if image is dark/night mode"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    p_value = float(np.percentile(gray, percentile))
    return p_value < threshold


def preprocess_plate_adaptive(plate_np: np.ndarray) -> np.ndarray:
    """
    Adaptive preprocessing based on image characteristics.
    Optimized for blurry/noisy Vietnamese plates.
    
    Pipeline:
    1. Detect & fix overexposure (glare from headlights)
    2. Upscale to larger size (96px height) with LANCZOS4
    3. Bilateral filter for noise reduction (preserves edges)
    4. CLAHE for contrast
    5. Unsharp mask for character edges
    6. Deskew if needed
    """
    if plate_np is None or plate_np.size == 0:
        return plate_np
    
    h, w = plate_np.shape[:2]
    
    # 0. Detect overexposure (glare/headlight reflection)
    # Check for percentage of very bright pixels (>240)
    if len(plate_np.shape) == 3:
        gray_check = cv2.cvtColor(plate_np, cv2.COLOR_BGR2GRAY)
    else:
        gray_check = plate_np
    
    bright_pixels = np.sum(gray_check > 240)
    total_pixels = gray_check.size
    bright_ratio = bright_pixels / total_pixels
    
    is_overexposed = bright_ratio > 0.20  # Lowered from 30% to 20%
    is_severely_overexposed = bright_ratio > 0.40  # New: severe glare
    
    if is_overexposed:
        # Apply stronger gamma correction to darken overexposed areas
        # gamma > 1 darkens, gamma < 1 brightens
        gamma = 2.2 if is_severely_overexposed else 1.8  # Increased from 1.5
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 
                         for i in np.arange(0, 256)]).astype("uint8")
        plate_np = cv2.LUT(plate_np, table)
        
        # For severe overexposure, also apply CLAHE to recover contrast
        if is_severely_overexposed:
            if len(plate_np.shape) == 3:
                lab = cv2.cvtColor(plate_np, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
                l = clahe.apply(l)
                plate_np = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
            else:
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
                plate_np = clahe.apply(plate_np)
            print(f"   🔆🔆 Fixed SEVERE overexposure (bright_ratio={bright_ratio:.1%})", flush=True)
        else:
            print(f"   🔆 Fixed overexposure (bright_ratio={bright_ratio:.1%})", flush=True)
        
        # NOTE: Morphological glare suppression REMOVED
        # It was damaging character strokes and making OCR worse
    
    h, w = plate_np.shape[:2]

    # 1. Detect if this is a night/IR image (grayscale, low saturation, OR low brightness)
    # MUST BE DONE BEFORE RESIZING for correct height selection
    is_night_ir = False
    gray_tmp = cv2.cvtColor(plate_np, cv2.COLOR_BGR2GRAY) if len(plate_np.shape) == 3 else plate_np
    avg_brightness = np.mean(gray_tmp)
    
    if len(plate_np.shape) == 2:  # Grayscale
        is_night_ir = True
    elif len(plate_np.shape) == 3:
        hsv_check = cv2.cvtColor(plate_np, cv2.COLOR_BGR2HSV)
        avg_sat = np.mean(hsv_check[:, :, 1])
        # Combined check: Low saturation OR (Low brightness AND low saturation threshold)
        if avg_sat < 30 or (avg_brightness < 90 and avg_sat < 45):
            is_night_ir = True
            if is_night_ir:
                print(f"   🌙 IR/Night detected: sat={avg_sat:.1f}, bright={avg_brightness:.0f}", flush=True)

    # 2. Upscale to larger height for better OCR
    # For IR/Night, use even larger height to preserve detail
    target_height = 120 if is_night_ir else 96
    if h < target_height:
        scale = target_height / h
        plate_np = cv2.resize(plate_np, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
    
    # 2.6 GLARE REMOVAL (IR/Night only)
    if is_night_ir:
        gray = cv2.cvtColor(plate_np, cv2.COLOR_BGR2GRAY) if len(plate_np.shape) == 3 else plate_np
        _, glare_mask = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY)
        if np.sum(glare_mask > 0) > 50:
            plate_np = cv2.inpaint(plate_np, glare_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
            print("   💡 Glare removed", flush=True)
    
    # 3. IR Image Inversion (Critical for IR cameras!)
    # Goal: Get WHITE text on BLACK background for PaddleOCR.
    if is_night_ir:
        gray = cv2.cvtColor(plate_np, cv2.COLOR_BGR2GRAY) if len(plate_np.shape) == 3 else plate_np
        curr_h, curr_w = gray.shape[:2]
        
        # Sample background (edges) vs text (center)
        # Use relative slicing to be height-independent
        edge_thickness = max(1, curr_h // 10)
        bg_brightness = (np.mean(gray[:edge_thickness, :]) + np.mean(gray[-edge_thickness:, :]) + 
                        np.mean(gray[:, :edge_thickness]) + np.mean(gray[:, -edge_thickness:])) / 4
        
        # Center region should contain the text
        center_h1, center_h2 = curr_h // 4, 3 * curr_h // 4
        center_w1, center_w2 = curr_w // 4, 3 * curr_w // 4
        center_brightness = np.mean(gray[center_h1:center_h2, center_w1:center_w2])
        
        print(f"   📊 Brightness: bg={bg_brightness:.0f}, center={center_brightness:.0f}", flush=True)
        
        # Logic: If background (edges) is significantly brighter than center (text)
        # then it is Black-on-White, so we INVERT to get White-on-Black (preferred by PaddleOCR).
        # Use a ratio (1.2x) or a fixed delta (15) for stability.
        if bg_brightness > center_brightness + 15:
            plate_np = cv2.bitwise_not(plate_np)
            print(f"   🌙 IR inversion applied (bg={bg_brightness:.0f} > center={center_brightness:.0f})", flush=True)
        else:
            print(f"   ⚠️ IR not inverted (center is already brighter or similar: {center_brightness:.0f})", flush=True)
    
    # 4. MEASURE BLUR BEFORE BILATERAL
    blur_score = detect_blur(plate_np)
    is_very_blurry = blur_score < 40
    
    # 5. Bilateral filter - reduces noise while preserving edges
    # EVEN LIGHTER filter for IR to avoid melting characters
    if is_night_ir:
        # Night: Very light denoising
        plate_np = cv2.bilateralFilter(plate_np, d=3, sigmaColor=25, sigmaSpace=25)
    else:
        # Day: Standard strength
        plate_np = cv2.bilateralFilter(plate_np, d=3, sigmaColor=25, sigmaSpace=25)
    
    # 6. Check skew angle - only deskew if angle > 2 degrees
    angle = compute_skew(plate_np)
    if abs(angle) > 2:
        plate_np = rotate_image(plate_np, angle)
    
    # 6.5. P0: PERSPECTIVE CORRECTION (4-point transform for angled plates)
    # Only apply if plate has significant perspective distortion (angle > 15°)
    try:
        gray_persp = cv2.cvtColor(plate_np, cv2.COLOR_BGR2GRAY) if len(plate_np.shape) == 3 else plate_np
        
        # Detect edges to find plate boundary
        edges = cv2.Canny(gray_persp, 50, 150, apertureSize=3)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get largest contour (likely the plate)
            largest = max(contours, key=cv2.contourArea)
            
            # Approximate to polygon
            epsilon = 0.02 * cv2.arcLength(largest, True)
            approx = cv2.approxPolyDP(largest, epsilon, True)
            
            # If we found a quadrilateral (4 corners), apply perspective transform
            # 6. Perspective correction (ONLY FOR DAY MODE - too risky for IR)
            if not is_night_ir and len(approx) == 4:
                # Order points: top-left, top-right, bottom-right, bottom-left
                pts = approx.reshape(4, 2).astype(np.float32)
                
                # Sort points by y, then by x
                pts = pts[np.argsort(pts[:, 1])]  # Sort by y
                top_pts = pts[:2][np.argsort(pts[:2, 0])]  # Top 2, sorted by x
                bottom_pts = pts[2:][np.argsort(pts[2:, 0])]  # Bottom 2, sorted by x
                ordered_pts = np.array([top_pts[0], top_pts[1], bottom_pts[1], bottom_pts[0]], dtype=np.float32)
                
                # Calculate target rectangle dimensions
                width_top = np.linalg.norm(ordered_pts[1] - ordered_pts[0])
                width_bottom = np.linalg.norm(ordered_pts[2] - ordered_pts[3])
                width = int(max(width_top, width_bottom))
                
                height_left = np.linalg.norm(ordered_pts[3] - ordered_pts[0])
                height_right = np.linalg.norm(ordered_pts[2] - ordered_pts[1])
                height = int(max(height_left, height_right))
                
                # Define destination points (rectangle)
                dst_pts = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
                
                # Get perspective transform matrix
                matrix = cv2.getPerspectiveTransform(ordered_pts, dst_pts)
                
                # Apply transform
                warped = cv2.warpPerspective(plate_np, matrix, (width, height))
                
                # Only use warped if it's reasonable size (not too distorted)
                if 30 < width < 400 and 15 < height < 150:
                    plate_np = warped
                    print(f"   📐 Perspective corrected: {width}x{height}", flush=True)
    except Exception as e:
        # Perspective correction is optional - don't fail if it doesn't work
        pass
    
    # 7. CLAHE for contrast
    if is_overexposed:
        lab = cv2.cvtColor(plate_np, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4))  # Lowered clip
        cl = clahe.apply(l_channel)
        plate_np = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
    elif is_night_ir:
        # Lighter contrast for IR
        lab = cv2.cvtColor(plate_np, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)) # Lowered clip
        cl = clahe.apply(l_channel)
        plate_np = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
    else:
        plate_np = change_contrast(plate_np)
    
    # 8. Unsharp mask sharpening
    # Mitigation: Skip sharpening or minimize it if SR was applied to IR to avoid "hallucinating" strokes
    applied_sr = plate_np.shape[0] > 100 and h < 80 # Heuristic for SR scale
    
    if is_night_ir:
        if applied_sr:
            # Skip heavy sharpening for SR images to avoid artifacts
            pass 
        else:
            # Very conservative sharpening for Night images
            gaussian = cv2.GaussianBlur(plate_np, (0, 0), 1.0)
            plate_np = cv2.addWeighted(plate_np, 1.05, gaussian, -0.05, 0)
    elif not is_very_blurry:
        # Stronger unsharp mask for character edges
        gaussian = cv2.GaussianBlur(plate_np, (0, 0), 2.0)
        plate_np = cv2.addWeighted(plate_np, 1.5, gaussian, -0.5, 0)
    else:
        # Light sharpening for very blurry images
        gaussian = cv2.GaussianBlur(plate_np, (0, 0), 1.0)
        plate_np = cv2.addWeighted(plate_np, 1.2, gaussian, -0.2, 0)
    
    return plate_np


def preprocess_plate_for_ocr(plate_np: np.ndarray) -> np.ndarray:
    """
    Preprocess plate image for better OCR accuracy.
    """
    # 1. Resize to standard height (improves OCR)
    target_height = 64
    h, w = plate_np.shape[:2]
    if h < target_height:
        scale = target_height / h
        plate_np = cv2.resize(plate_np, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    
    # 2. Apply CLAHE contrast enhancement
    plate_np = change_contrast(plate_np)
    
    # 3. Light sharpening
    kernel = np.array([[-0.5,-0.5,-0.5], [-0.5, 5,-0.5], [-0.5,-0.5,-0.5]])
    plate_np = cv2.filter2D(plate_np, -1, kernel)
    
    return plate_np


def normalize_vn_plate_legacy(text: str) -> str:
    """
    CONSERVATIVE normalization for Vietnamese license plates.
    Only fix OBVIOUS OCR confusions, do NOT auto-convert valid letters.
    
    Safe fixes:
    - O → 0 (in digit positions only)
    - I/L → 1 (in digit positions only)
    
    DO NOT fix:
    - T → 7 (T is valid series letter)
    - A → 4 (A is valid series letter)
    - F → anything (F is valid series letter)
    """
    if not text or len(text) < 7:
        return text
    
    text = text.upper().strip()
    # Remove common separators
    text = text.replace(" ", "").replace("-", "").replace(".", "")
    
    result = []
    
    # Determine series position (letter at position 2, optionally 3)
    series_end = 3
    if len(text) > 3 and text[3].isalpha():
        series_end = 4
    
    for i, char in enumerate(text):
        # First 2 chars should be digits (province code)
        if i < 2:
            # ONLY fix obvious letter->digit confusions
            if char == 'O':
                result.append('0')
            elif char == 'I' or char == 'L':
                result.append('1')
            else:
                result.append(char)
        # Position 2-3 is series code (letters) - keep as-is
        elif i < series_end:
            result.append(char)
        # Rest should be digits (serial number)
        else:
            # ONLY fix obvious letter->digit confusions
            if char == 'O':
                result.append('0')
            elif char == 'I' or char == 'L':
                result.append('1')
            else:
                result.append(char)
    
    return ''.join(result)


def validate_vn_plate(text: str) -> Tuple[bool, float]:
    """
    Validate if text matches Vietnamese plate format.
    
    Returns:
        (is_valid, confidence_boost)
        - is_valid: True if matches VN plate pattern
        - confidence_boost: 0.0-0.2 bonus for good format
    """
    if not text or len(text) < 7:
        return False, 0.0
    
    # Remove separators for validation
    clean = text.replace("-", "").replace(".", "").replace(" ", "").upper()
    
    # VN plate patterns:
    # 1. Standard: 2 digits + 1 letter + 5 digits (65A12345)
    # 2. Extended: 2 digits + 2 letters + 5 digits (65AB12345)
    # 3. New format: 2 digits + 1 letter + 3 digits + 2 digits (65A123.45)
    
    # First 2 chars MUST be digits (province code) - strict validation
    if len(clean) < 2 or not clean[0].isdigit() or not clean[1].isdigit():
        return False, 0.0
    
    # Province code must be in valid range (11-99)
    province = int(clean[:2])
    if province < 11 or province > 99:
        return False, 0.0
    
    patterns = [
        r'^[1-9][0-9][A-Z][0-9]{5}$',           # 65A12345
        r'^[1-9][0-9][A-Z]{2}[0-9]{5}$',        # 65AB12345
        r'^[1-9][0-9][A-Z][0-9]{4,6}$',         # Flexible digit count
        r'^[1-9][0-9][A-Z]{1,2}[0-9]{4,6}$',    # Most flexible
    ]
    
    for i, pattern in enumerate(patterns):
        if re.match(pattern, clean):
            # More specific patterns get higher boost
            boost = 0.15 - (i * 0.03)
            return True, max(boost, 0.05)
    
    # Has valid province code but doesn't match standard patterns
    # Accept with low boost (likely partial plate)
    if len(clean) >= 7:
        return True, 0.02
    
    return False, 0.0


def detect_plate(image: Image.Image) -> Optional[Tuple[np.ndarray, List[int], float]]:
    """
    Detect license plate in image using YOLO (supports YOLOv5, YOLOv8, YOLOv11).
    
    Returns:
        Tuple of (plate_crop_np, [x1, y1, x2, y2], confidence) or None
    """
    detector = get_lp_detector()
    if detector is None:
        logger.warning("LP_detector model not loaded")
        return None
    
    # Convert PIL to numpy BGR
    img_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    logger.debug(f"Running LP_detector on image {img_np.shape}")

    # Day/night adaptive thresholds
    is_night = detect_night_mode(img_np)
    min_plate_conf = 0.30 if is_night else 0.40
    base_conf = getattr(detector, "conf", None)
    if base_conf is not None:
        adaptive_conf = max(0.15, base_conf - 0.05) if is_night else min(0.9, base_conf + 0.05)
        if adaptive_conf != base_conf:
            detector.conf = adaptive_conf
    
    # Detect model type and run detection accordingly
    plates = []
    
    try:
        # Check if YOLOv8/v11 (ultralytics YOLO class)
        if hasattr(detector, 'predict'):
            # YOLOv8/v11 API
            results = detector.predict(img_np, imgsz=640, verbose=False)
            if results and len(results) > 0:
                boxes = results[0].boxes
                if boxes is not None and len(boxes) > 0:
                    for i in range(len(boxes)):
                        x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                        conf = float(boxes.conf[i].cpu().numpy())
                        plates.append([x1, y1, x2, y2, conf])
        else:
            # YOLOv5 API (torch.hub)
            results = detector(img_np, size=640)
            plates = results.pandas().xyxy[0].values.tolist()
    finally:
        # Always restore original confidence, even if detection crashes
        if base_conf is not None:
            detector.conf = base_conf

    if not plates:
        logger.debug(f"LP_detector found 0 plates in image {img_np.shape[:2]}")
        return None
    
    # PHASE 1 FIX: Early rejection of low confidence plates
    # Adaptive day/night threshold to balance recall vs precision
    plates = [p for p in plates if p[4] >= min_plate_conf]
    
    if not plates:
        logger.debug(f"LP_detector: All plates rejected (conf < {min_plate_conf})")
        return None
    
    logger.debug(f"LP_detector found {len(plates)} valid plates (conf >= {min_plate_conf})")
    
    # Take highest confidence plate
    best = max(plates, key=lambda x: x[4])  # x[4] is confidence
    x1, y1, x2, y2 = int(best[0]), int(best[1]), int(best[2]), int(best[3])
    conf = float(best[4])
    
    logger.debug(f"Best plate: [{x1},{y1},{x2},{y2}] conf={conf:.2f}")
    
    # P0.8 FIX: More generous padding to avoid cutting plate edges
    # Issue: Plates often get partially cut, losing first/last characters
    # Solution: Increase padding from 15% to 25%, with higher minimums
    h, w = img_np.shape[:2]
    plate_w = x2 - x1
    plate_h = y2 - y1
    
    # Calculate aspect ratio to detect potential issues
    aspect_ratio = plate_w / plate_h if plate_h > 0 else 1.0
    
    # Use more aggressive padding for compact plates (likely 2-line layouts)
    if aspect_ratio < 3.2:  # compact layout or tilted plate
        pad_x = max(20, int(plate_w * 0.30))  # 30% for compact plates
        pad_y = max(15, int(plate_h * 0.25))
    else:  # Normal 1-line plate
        pad_x = max(15, int(plate_w * 0.25))  # 25% padding (was 15%)
        pad_y = max(10, int(plate_h * 0.20))  # 20% vertical padding
    
    # Apply asymmetric padding - more on left/right where characters are
    pad_x_left = pad_x
    pad_x_right = pad_x
    
    # Extra padding on top/bottom for 2-line plates
    pad_y_top = pad_y
    pad_y_bottom = pad_y
    
    x1 = max(0, x1 - pad_x_left)
    y1 = max(0, y1 - pad_y_top)
    x2 = min(w, x2 + pad_x_right)
    y2 = min(h, y2 + pad_y_bottom)
    
    logger.debug(f"P0.8: Applied padding x={pad_x}, y={pad_y} (aspect={aspect_ratio:.2f})")
    
    # Crop plate
    plate_crop = img_np[y1:y2, x1:x2]
    
    return plate_crop, [x1, y1, x2, y2], conf


def _two_line_ocr_rescue(plate_np: np.ndarray) -> Tuple[str, float]:
    """
    Rescue OCR for 2-line layouts by splitting top/bottom lines.
    Returns (corrected_text, confidence) or ("", 0.0) if not applicable.
    """
    try:
        if plate_np is None:
            return "", 0.0

        h, w = plate_np.shape[:2]
        if h < 30 or w < 80:
            return "", 0.0

        aspect = w / h if h > 0 else 1.0
        if aspect > 3.2:
            # Likely wide layout
            return "", 0.0

        processed = preprocess_plate_adaptive(plate_np.copy())
        proc_h = processed.shape[0]
        mid = int(proc_h * 0.45)
        top = processed[:mid, :]
        bottom = processed[mid:, :]

        ocr_service = OCRService.get_instance()

        def run_ocr(img_np: np.ndarray) -> Tuple[str, float]:
            pil = Image.fromarray(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))
            result = ocr_service._run_ocr_sync(pil)
            if not result:
                return "", 0.0
            text, conf, _ = result
            text = normalize_vn_plate(text)
            return text, conf

        top_text, top_conf = run_ocr(top)
        bottom_text, bottom_conf = run_ocr(bottom)

        if not top_text or not bottom_text:
            return "", 0.0

        try:
            from app.services.ocr.adaptive_line_split import validate_split_result
            is_valid, _reason = validate_split_result(top_text, bottom_text)
            if not is_valid:
                return "", 0.0
        except Exception:
            pass

        merged = f"{top_text}{bottom_text}"

        from app.services.ocr.ocr_corrector import get_corrector
        corrector = get_corrector()
        corrected_text, corrected_conf, _ = corrector.correct(merged, (top_conf + bottom_conf) / 2)

        if not corrected_text or len(corrected_text) < 7:
            return "", 0.0

        return corrected_text, corrected_conf

    except Exception as e:
        logger.warning(f"Two-line OCR rescue failed: {e}")
        return "", 0.0


def _province_rescue(plate_np: np.ndarray, base_text: str, base_conf: float) -> Tuple[str, float]:
    """
    Try to re-read province digits from the left/top region and override if confident.
    """
    try:
        if plate_np is None or not base_text:
            return base_text, base_conf

        clean = re.sub(r'[^A-Z0-9]', '', base_text.upper())
        if len(clean) < 7:
            return base_text, base_conf

        h, w = plate_np.shape[:2]
        if h < 30 or w < 80:
            return base_text, base_conf

        # Province digits are in the top-left area
        roi = plate_np[: int(h * 0.55), : int(w * 0.45)]
        processed = preprocess_plate_adaptive(roi.copy())

        ocr_service = OCRService.get_instance()
        pil = Image.fromarray(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
        result = ocr_service._run_ocr_sync(pil)
        if not result:
            return base_text, base_conf

        text, conf, _ = result
        text = normalize_vn_plate(text)
        digits = ''.join([c for c in text if c.isdigit()])
        if len(digits) < 2:
            return base_text, base_conf

        province = digits[:2]
        try:
            prov_int = int(province)
        except ValueError:
            return base_text, base_conf

        if prov_int < 11 or prov_int > 99:
            return base_text, base_conf

        if province == clean[:2]:
            return base_text, base_conf

        # Only override when rescue is confident and base is not ultra-confident
        if conf < 0.80 or base_conf >= 0.95:
            return base_text, base_conf

        merged = province + clean[2:]
        from app.services.ocr.ocr_corrector import get_corrector
        corrector = get_corrector()
        corrected_text, corrected_conf, _ = corrector.correct(merged, min(base_conf, conf))

        if corrected_text:
            return corrected_text, corrected_conf

        return merged, min(base_conf, conf)

    except Exception as e:
        logger.warning(f"Province rescue failed: {e}")
        return base_text, base_conf


def _serial_rescue(plate_np: np.ndarray, base_text: str, base_conf: float) -> Tuple[str, float]:
    """
    Try to re-read last digits from the right/bottom region and override if confident.
    """
    try:
        if plate_np is None or not base_text:
            return base_text, base_conf

        clean = re.sub(r'[^A-Z0-9]', '', base_text.upper())
        if len(clean) < 7:
            return base_text, base_conf

        h, w = plate_np.shape[:2]
        if h < 30 or w < 80:
            return base_text, base_conf

        # Serial digits are on the right and lower area
        roi = plate_np[int(h * 0.35):, int(w * 0.45):]
        processed = preprocess_plate_adaptive(roi.copy())

        ocr_service = OCRService.get_instance()
        pil = Image.fromarray(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
        result = ocr_service._run_ocr_sync(pil)
        if not result:
            return base_text, base_conf

        text, conf, _ = result
        text = normalize_vn_plate(text)
        digits = ''.join([c for c in text if c.isdigit()])
        if len(digits) < 2:
            return base_text, base_conf

        serial = digits[-2:]
        if base_conf >= 0.95 or conf < 0.80:
            return base_text, base_conf

        merged = clean[:-2] + serial
        from app.services.ocr.ocr_corrector import get_corrector
        corrector = get_corrector()
        corrected_text, corrected_conf, _ = corrector.correct(merged, min(base_conf, conf))

        if corrected_text:
            return corrected_text, corrected_conf

        return merged, min(base_conf, conf)

    except Exception as e:
        logger.warning(f"Serial rescue failed: {e}")
        return base_text, base_conf


def _select_arbitrated_candidate(candidates: List[Tuple[str, float, str]], camera_id: str = "default") -> Tuple[str, float]:
    global _last_selector_evict_ts
    now = time.time()
    if now - _last_selector_evict_ts > 60.0:
        try:
            _plate_selector.evict_idle_cameras(max_idle_seconds=60.0)
        except Exception:
            pass
        _last_selector_evict_ts = now

    consensus_text = ""
    consensus_ratio = 0.0
    consensus = _plate_selector.get_consensus_plate(camera_id=camera_id)
    if consensus:
        consensus_text, consensus_ratio = consensus
    best_text = ""
    best_conf = 0.0
    best_source = ""
    best_score = -1.0
    for text, conf, source in candidates:
        if not text:
            continue
        _, boost = validate_vn_plate(text)
        temporal = 0.0
        if consensus_text and normalize_plate_basic(consensus_text) == normalize_plate_basic(text):
            temporal = consensus_ratio
        score = 0.6 * conf + 0.25 * boost + 0.15 * temporal
        if score > best_score:
            best_score = score
            best_text = text
            best_conf = conf
            best_source = source
    if best_text:
        _plate_selector.add_candidate(best_text, best_conf, best_source or "default", camera_id=camera_id)
    return best_text, best_conf


def read_plate_hybrid(image: Image.Image) -> Tuple[str, float, Optional[Image.Image], Optional[List[int]]]:
    """
    Hybrid pipeline: YOLO LP_detector + PaddleOCR.
    SIMPLE VERSION - no ensemble, no voting, just PaddleOCR with validation.
    
    Args:
        image: PIL Image (vehicle or full frame)
    
    Returns:
        (plate_text, confidence, plate_crop, plate_bbox) or ("", 0.0, None, None) if failed
    """
    from app.config import get_settings
    settings = get_settings()
    print(f"   🔍 [Legacy] read_plate_hybrid called, image={image.size}", flush=True)
    
    # Step 1: Detect plate with YOLO
    result = detect_plate(image)
    
    plate_np = None
    plate_bbox = None
    detect_conf = 0.5
    
    if result is None:
        print(f"   🔍 LP_detector: No plate found in {image.size}", flush=True)
        return "", 0.0, None, None

    plate_np, plate_bbox, detect_conf = result
    print(f"🔍 LP_detector: Found plate at {plate_bbox} conf={detect_conf:.2f}", flush=True)

    # Step 2: Check if plate is too small/blurry - apply SR if needed
    plate_h, plate_w = plate_np.shape[:2]
    plate_area = plate_h * plate_w
    
    # Apply SR for small or very blurry plates (common in IR)
    should_upscale = False
    sr_reason = ""
    
    if plate_w < 150:  # Very small plate
        should_upscale = True
        sr_reason = f"small width ({plate_w}px)"
    elif plate_area < 8000:  # Small area
        should_upscale = True
        sr_reason = f"small area ({plate_area}px²)"
    else:
        # Check blur
        blur_score = detect_blur(plate_np)
        if blur_score < 100:  # Very blurry
            should_upscale = True
            sr_reason = f"blurry (score={blur_score:.0f})"
    
    if should_upscale:
        try:
            from app.services.ocr.super_resolution import get_super_resolution_service
            
            sr_service = get_super_resolution_service()
            if sr_service.is_available():
                # Convert numpy to PIL (Image already imported at top)
                plate_pil_sr = Image.fromarray(cv2.cvtColor(plate_np, cv2.COLOR_BGR2RGB))
                # Upscale x2 using SYNC method (we're already in executor)
                upscaled_pil = sr_service.upscale_sync(plate_pil_sr, scale=2)
                
                if upscaled_pil is not None:
                    plate_np = cv2.cvtColor(np.array(upscaled_pil), cv2.COLOR_RGB2BGR)
                    print(f"   🔍 SR x2 applied to plate: {sr_reason}", flush=True)
        except Exception as e:
            print(f"   ⚠️ SR failed: {e}", flush=True)
    
    # ============================================
    # v3 UPGRADE: TPS Rectification for curved plates
    # ============================================
    if USE_TPS_RECTIFICATION:
        try:
            rectifier = get_tps_rectifier()
            curvature = rectifier.estimate_curvature(plate_np)
            
            if curvature > 0.15:  # Significant curvature detected
                plate_np_rectified = rectifier.rectify(plate_np, auto_detect_curvature=True)
                if plate_np_rectified is not None and plate_np_rectified.size > 0:
                    plate_np = plate_np_rectified
                    print(f"   🏰 TPS rectification applied (curvature={curvature:.2f})", flush=True)
        except Exception as e:
            print(f"   ⚠️ TPS rectification failed: {e}", flush=True)
    
    # ============================================
    # v3 UPGRADE: Retroreflective glare suppression
    # ============================================
    if USE_GLARE_SUPPRESSION:
        try:
            has_glare, glare_ratio = detect_glare(plate_np)
            if has_glare and glare_ratio > 0.1:
                plate_np = suppress_glare(plate_np, method='inpaint')
                print(f"   🏰 Glare suppressed (ratio={glare_ratio:.1%})", flush=True)
        except Exception as e:
            print(f"   ⚠️ Glare suppression failed: {e}", flush=True)
    
    # Step 3: Preprocess plate for OCR
    processed = preprocess_plate_adaptive(plate_np.copy())
    
    # Step 4: Run PaddleOCR (Production Engine - trained on millions of real images)
    # PaddleOCR handles 2-line plates automatically by sorting text lines by Y position
    
    # Check if night/IR mode for preprocessing hints
    is_night_ir = False
    if len(plate_np.shape) == 3:
        hsv_check = cv2.cvtColor(plate_np, cv2.COLOR_BGR2HSV)
        avg_sat = np.mean(hsv_check[:, :, 1])
        if avg_sat < 30:
            is_night_ir = True
    
    # Convert processed numpy to PIL for PaddleOCR
    processed_pil = Image.fromarray(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
    
    # Get OCR service (singleton with lazy loading)
    ocr_service = OCRService.get_instance()
    
    # Run PaddleOCR synchronously (we're already in executor)
    try:
        ocr_result = ocr_service._run_ocr_sync(processed_pil)
        
        if ocr_result:
            text, ocr_conf, ocr_box = ocr_result
            # Normalize: fix common OCR errors (O->0, I->1, etc.)
            text = normalize_vn_plate(text)
            print(f"   🔤 PaddleOCR result: '{text}' conf={ocr_conf:.2f}", flush=True)
        else:
            text = ""
            ocr_conf = 0.0
            print(f"   🔤 PaddleOCR: No text detected", flush=True)
    except Exception as e:
        print(f"   ⚠️ PaddleOCR failed: {e}", flush=True)
        text = ""
        ocr_conf = 0.0
    
    engine_used = "paddleocr"
    
    min_len = int(getattr(settings, "ocr_min_text_length", 6))
    if not text:
        print(f"   ⏭️ OCR result too short: '{text}' (0 chars, min={min_len})", flush=True)
        return "", 0.0, None, None
    if len(text) < min_len and (ocr_conf < 0.7 or len(text) < max(5, min_len - 1)):
        print(f"   ⏭️ OCR result too short: '{text}' ({len(text)} chars, min={min_len})", flush=True)
        return "", 0.0, None, None
    
    # Step 5: Intelligent OCR Correction (replaces normalize + validate)
    from app.services.ocr.ocr_corrector import get_corrector
    
    corrector = get_corrector()
    corrected_text, corrected_conf, fixes_applied = corrector.correct(text, ocr_conf)
    
    # Log corrections for debugging/audit
    if fixes_applied:
        print(f"   🔧 Applied {len(fixes_applied)} corrections:", flush=True)
        for fix in fixes_applied:
            print(f"      - {fix}", flush=True)
    
    candidates: List[Tuple[str, float, str]] = []
    if corrected_text:
        candidates.append((corrected_text, detect_conf * corrected_conf, "legacy"))

    if plate_np is not None and (len(corrected_text) < 8 or corrected_conf < 0.7):
        rescue_text, rescue_conf = _two_line_ocr_rescue(plate_np)
        if rescue_text:
            candidates.append((rescue_text, detect_conf * rescue_conf, "legacy_2line"))

    rescue_text, rescue_conf = _province_rescue(plate_np, corrected_text, corrected_conf)
    rescue_text, rescue_conf = _serial_rescue(plate_np, rescue_text, rescue_conf)
    if rescue_text:
        candidates.append((rescue_text, detect_conf * rescue_conf, "legacy_rescue"))

    final_text, final_conf = _select_arbitrated_candidate(candidates)
    if not final_text:
        return "", 0.0, None, None

    save_corrected = final_text.replace(".", "").replace("-", "").replace(" ", "").upper()
    print(f"   📝 Corrected : '{save_corrected}' (combined={final_conf:.2f})", flush=True)

    logger.info(f"🔍 LPR: {text} -> {final_text} (conf={final_conf:.2f})")

    plate_pil = Image.fromarray(cv2.cvtColor(plate_np, cv2.COLOR_BGR2RGB))

    return final_text, final_conf, plate_pil, plate_bbox


def read_plate_fortress(image: Image.Image) -> Tuple[str, float, Optional[Image.Image], Optional[List[int]]]:
    """
    🏰 Fortress Mode: Read plate using YOLOv11-OBB + STN-LPRNet
    
    Returns:
        (plate_text, confidence, plate_crop, plate_bbox)
    """
    fortress = get_fortress_pipeline()
    
    if fortress is None:
        # Fallback to legacy
        return read_plate_hybrid(image)
    
    # Convert PIL to numpy BGR
    if isinstance(image, Image.Image):
        img_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    else:
        img_np = image
    
    # Process with Fortress pipeline
    result = fortress.process_frame(img_np, preprocess=True)
    
    if not result.plates:
        return "", 0.0, None, None
    
    # Get best plate (highest confidence)
    best_plate = max(result.plates, key=lambda p: p.confidence)
    
    # Convert crop to PIL
    plate_pil = None
    if best_plate.plate_crop is not None:
        plate_pil = Image.fromarray(cv2.cvtColor(best_plate.plate_crop, cv2.COLOR_BGR2RGB))
    
    # Convert bbox
    plate_bbox = list(best_plate.bbox) if best_plate.bbox else None
    
    base_text = best_plate.plate_text
    base_conf = best_plate.confidence
    candidates: List[Tuple[str, float, str]] = []
    if base_text:
        candidates.append((base_text, base_conf, "fortress"))

    if plate_pil is not None and (len(base_text) < 8 or base_conf < 0.7):
        plate_bgr = cv2.cvtColor(np.array(plate_pil), cv2.COLOR_RGB2BGR)
        rescue_text, rescue_conf = _two_line_ocr_rescue(plate_bgr)
        if rescue_text:
            candidates.append((rescue_text, rescue_conf, "fortress_2line"))

    if plate_pil is not None:
        plate_bgr = cv2.cvtColor(np.array(plate_pil), cv2.COLOR_RGB2BGR)
        rescue_text, rescue_conf = _province_rescue(plate_bgr, base_text, base_conf)
        rescue_text, rescue_conf = _serial_rescue(plate_bgr, rescue_text, rescue_conf)
        if rescue_text:
            candidates.append((rescue_text, rescue_conf, "fortress_rescue"))

    from app.services.ocr.ocr_corrector import get_corrector
    corrector = get_corrector()
    corrected_candidates: List[Tuple[str, float, str]] = []
    for text, conf, source in candidates:
        corrected_text, corrected_conf, _ = corrector.correct(text, conf)
        if corrected_text:
            corrected_candidates.append((corrected_text, corrected_conf, source))

    if plate_pil is not None and (len(base_text) < 8 or base_conf < 0.85):
        try:
            ocr_service = OCRService.get_instance()
            ocr_result = ocr_service._run_ocr_sync(plate_pil)
            if ocr_result:
                paddle_text, paddle_conf, _ = ocr_result
                paddle_text = normalize_vn_plate(paddle_text)
                paddle_text, paddle_conf, _ = corrector.correct(paddle_text, paddle_conf)
                if paddle_text:
                    corrected_candidates.append((paddle_text, paddle_conf, "paddleocr"))
        except Exception as e:
            logger.warning(f"⚠️ PaddleOCR rescue failed: {e}")

    if plate_pil is not None and (len(base_text) < 8 or base_conf < 0.85):
        try:
            from app.services.ocr.tta_ocr import create_tta_ocr
            ocr_service = OCRService.get_instance()

            def _ocr_func(img_bgr: np.ndarray) -> Tuple[str, float]:
                pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
                ocr_result = ocr_service._run_ocr_sync(pil_img)
                if not ocr_result:
                    return "", 0.0
                text, conf, _ = ocr_result
                text = normalize_vn_plate(text)
                return text, conf

            tta = create_tta_ocr(_ocr_func, num_augmentations=6)
            plate_bgr = cv2.cvtColor(np.array(plate_pil), cv2.COLOR_RGB2BGR)
            tta_result = tta.recognize(plate_bgr)

            if tta_result.plate_text and len(tta_result.plate_text) >= 7:
                tta_text, tta_conf, _ = corrector.correct(tta_result.plate_text, tta_result.confidence)
                if tta_text:
                    corrected_candidates.append((tta_text, tta_conf, "tta"))
        except Exception as e:
            logger.warning(f"⚠️ TTA OCR rescue failed: {e}")

    final_text, final_conf = _select_arbitrated_candidate(corrected_candidates)
    if not final_text:
        return "", 0.0, None, None

    logger.info(f"🏰 Fortress: {final_text} (conf={final_conf:.2f}, time={result.total_time_ms:.1f}ms)")

    return final_text, final_conf, plate_pil, plate_bbox


class VNPlateReader:
    """
    Vietnamese License Plate Reader Service.
    v4.0: Fortress Mode (YOLOv11-OBB + STN-LPRNet) with Legacy fallback
    Singleton pattern for model caching.
    """
    _instance = None
    _instance_lock = __import__('threading').Lock()
    
    def __init__(self):
        self._detector = None
        self._ocr = None
        self._use_fortress = USE_FORTRESS_MODE
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = VNPlateReader()
        return cls._instance
    
    def is_available(self) -> bool:
        """Check if LPR models are available"""
        from app.config import get_settings
        settings = get_settings()
        
        # Check Fortress models first
        fortress_ready = (
            os.path.exists(os.path.join(settings.weights_dir, "yolov11-obb-vnplate.pt")) or
            os.path.exists(os.path.join(settings.weights_dir, "yolov11-obb-vnplate.engine"))
        )
        
        # Check legacy model
        legacy_ready = os.path.exists(os.path.join(settings.weights_dir, "LP_detector.pt"))
        
        return fortress_ready or legacy_ready
    
    def is_fortress_mode(self) -> bool:
        """Check if Fortress mode is active"""
        return self._use_fortress and _is_fortress_enabled() and get_fortress_pipeline() is not None
    
    async def predict(self, image: Image.Image) -> Tuple[str, float, Optional[Image.Image], Optional[List[int]]]:
        """
        Async wrapper for plate recognition.
        Uses Fortress mode if available, falls back to legacy.
        
        Returns:
            (plate_text, confidence, plate_crop, plate_bbox)
        """
        import asyncio
        loop = asyncio.get_running_loop()
        
        # Use Fortress mode if available
        if self._use_fortress and _is_fortress_enabled() and get_fortress_pipeline() is not None:
            return await loop.run_in_executor(None, read_plate_fortress, image)
        else:
            return await loop.run_in_executor(None, read_plate_hybrid, image)
    
    def predict_sync(self, image: Image.Image) -> Tuple[str, float, Optional[Image.Image], Optional[List[int]]]:
        """
        Synchronous prediction method.
        Uses Fortress mode if available.
        """
        if self._use_fortress and get_fortress_pipeline() is not None:
            return read_plate_fortress(image)
        else:
            return read_plate_hybrid(image)
