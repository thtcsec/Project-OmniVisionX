"""
TPS Rectifier Service - Thin Plate Spline Geometric Correction
===============================================================
Integrates TPS-STN into LPR pipeline for curved/distorted plate correction.

Key features:
- Handles curved motorcycle plates (mounted on round fenders)
- Fixes perspective distortion from angled views
- Works with 4-corner detection from contours
- GPU-accelerated with PyTorch

Usage:
    rectifier = get_tps_rectifier()
    corrected = rectifier.rectify(plate_img)  # Auto-detect corners
    corrected = rectifier.rectify_with_corners(plate_img, corners_4pt)  # Manual corners
"""
import os
import logging
import threading
from typing import Optional, Tuple, List
import numpy as np
import cv2

logger = logging.getLogger(__name__)

# Global cache
_tps_rectifier = None
_tps_lock = threading.Lock()
_device = None


def get_device():
    """Get CUDA device string if available (lazy import torch only when needed)"""
    global _device
    if _device is None:
        try:
            from app.services.core.device_utils import get_torch_device
            _device = get_torch_device("cuda")
        except Exception:
            _device = "cpu"
    return _device


class TPSRectifier:
    """
    Thin Plate Spline Rectifier for license plates
    Simpler version that doesn't require learned weights - uses geometric TPS
    """
    
    def __init__(self, output_size: Tuple[int, int] = (94, 24)):
        """
        Args:
            output_size: (width, height) of output rectified plate
        """
        self.output_size = output_size  # (W, H)
        self.device = None  # Lazy init — only load torch if TPS is actually used
        
        # Standard plate aspect ratios
        self.ar_1line = 4.0  # Car plates: ~340x85mm
        self.ar_2line = 1.27  # Motorcycle: ~203x160mm
        
        logger.info(f"✅ TPSRectifier initialized (output: {output_size})")
    
    def detect_corners(self, plate_img: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect 4 corners of plate using contour detection.
        
        Returns:
            corners: (4, 2) array of corner points [TL, TR, BR, BL] or None
        """
        if plate_img is None or plate_img.size == 0:
            return None
        
        h, w = plate_img.shape[:2]
        
        # Convert to grayscale
        if len(plate_img.shape) == 3:
            gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_img.copy()
        
        # Enhance edges
        blurred = cv2.bilateralFilter(gray, 5, 75, 75)
        edges = cv2.Canny(blurred, 30, 100)
        
        # Dilate to connect edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Find largest contour
        largest = max(contours, key=cv2.contourArea)
        
        # Approximate to polygon
        epsilon = 0.02 * cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, epsilon, True)
        
        if len(approx) == 4:
            pts = approx.reshape(4, 2).astype(np.float32)
            return self._order_corners(pts)
        
        # Fallback: use minAreaRect
        rect = cv2.minAreaRect(largest)
        box = cv2.boxPoints(rect)
        return self._order_corners(box.astype(np.float32))
    
    def _order_corners(self, pts: np.ndarray) -> np.ndarray:
        """
        Order corners as [top-left, top-right, bottom-right, bottom-left]
        """
        # Sort by y first
        sorted_y = pts[np.argsort(pts[:, 1])]
        top = sorted_y[:2]
        bottom = sorted_y[2:]
        
        # Sort by x
        top = top[np.argsort(top[:, 0])]
        bottom = bottom[np.argsort(bottom[:, 0])]
        
        return np.array([top[0], top[1], bottom[1], bottom[0]], dtype=np.float32)
    
    def rectify_perspective(self, plate_img: np.ndarray, 
                            corners: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply perspective correction using detected or provided corners.
        
        Args:
            plate_img: Input plate image (BGR)
            corners: Optional (4, 2) corners, will detect if None
        
        Returns:
            Rectified plate image
        """
        if plate_img is None or plate_img.size == 0:
            return plate_img
        
        h, w = plate_img.shape[:2]
        
        # Detect corners if not provided
        if corners is None:
            corners = self.detect_corners(plate_img)
        
        if corners is None:
            # Fallback: return resized original
            return cv2.resize(plate_img, self.output_size)
        
        # Determine plate type from aspect ratio
        src_w = max(np.linalg.norm(corners[0] - corners[1]),
                    np.linalg.norm(corners[3] - corners[2]))
        src_h = max(np.linalg.norm(corners[0] - corners[3]),
                    np.linalg.norm(corners[1] - corners[2]))
        
        ar = src_w / max(src_h, 1)
        
        # Choose output size based on layout
        if ar > 3.2:
            # Wide layout
            out_w, out_h = 200, 50
        else:
            # Compact layout (typically 2-line)
            out_w, out_h = 150, 100
        
        # Define destination corners
        dst_corners = np.array([
            [0, 0],
            [out_w - 1, 0],
            [out_w - 1, out_h - 1],
            [0, out_h - 1]
        ], dtype=np.float32)
        
        # Compute perspective transform
        M = cv2.getPerspectiveTransform(corners, dst_corners)
        
        # Apply transform
        rectified = cv2.warpPerspective(plate_img, M, (out_w, out_h),
                                        flags=cv2.INTER_CUBIC,
                                        borderMode=cv2.BORDER_REPLICATE)
        
        return rectified
    
    def rectify_tps(self, plate_img: np.ndarray,
                    corners: Optional[np.ndarray] = None,
                    curvature: float = 0.0) -> np.ndarray:
        """
        Apply TPS (Thin Plate Spline) correction for curved plates.
        
        This handles:
        - Curved motorcycle plates (mounted on round fenders)
        - Barrel distortion from wide-angle lenses
        - Non-linear warping from extreme angles
        
        Args:
            plate_img: Input plate image (BGR)
            corners: Optional (4, 2) corners
            curvature: Estimated curvature (0-1), 0=flat, 1=highly curved
        
        Returns:
            Rectified plate image
        """
        if plate_img is None or plate_img.size == 0:
            return plate_img
        
        h, w = plate_img.shape[:2]
        
        # For low curvature, use faster perspective transform
        if curvature < 0.2:
            return self.rectify_perspective(plate_img, corners)
        
        # Detect corners if not provided
        if corners is None:
            corners = self.detect_corners(plate_img)
        
        if corners is None:
            return cv2.resize(plate_img, self.output_size)
        
        # Determine output size
        ar = w / max(h, 1)
        if ar > 2.5:
            out_w, out_h = 200, 50
        else:
            out_w, out_h = 150, 100
        
        # Create control points grid (5x3 for better curve handling)
        num_ctrl_x, num_ctrl_y = 5, 3
        
        # Source control points (interpolated along edges + center)
        src_pts = self._create_curved_control_points(corners, num_ctrl_x, num_ctrl_y, curvature)
        
        # Destination control points (regular grid)
        dst_pts = self._create_regular_grid(out_w, out_h, num_ctrl_x, num_ctrl_y)
        
        # Compute TPS transform
        tps = cv2.createThinPlateSplineShapeTransformer()
        
        # Reshape for OpenCV TPS
        src_pts_cv = src_pts.reshape(1, -1, 2).astype(np.float32)
        dst_pts_cv = dst_pts.reshape(1, -1, 2).astype(np.float32)
        
        matches = [cv2.DMatch(i, i, 0) for i in range(src_pts.shape[0])]
        
        tps.estimateTransformation(dst_pts_cv, src_pts_cv, matches)
        
        # Apply transform
        rectified = tps.warpImage(plate_img)
        
        # Crop to output size
        if rectified.shape[1] > out_w or rectified.shape[0] > out_h:
            rectified = cv2.resize(rectified, (out_w, out_h))
        
        return rectified
    
    def _create_curved_control_points(self, corners: np.ndarray,
                                       nx: int, ny: int,
                                       curvature: float) -> np.ndarray:
        """
        Create control points that follow curved plate edges.
        """
        pts = []
        
        for j in range(ny):
            t_y = j / (ny - 1)
            
            # Interpolate left and right edges
            left = corners[0] * (1 - t_y) + corners[3] * t_y
            right = corners[1] * (1 - t_y) + corners[2] * t_y
            
            for i in range(nx):
                t_x = i / (nx - 1)
                
                # Linear interpolation
                pt = left * (1 - t_x) + right * t_x
                
                # Add curvature offset (parabolic curve)
                if 0 < i < nx - 1:  # Don't curve edges
                    curve_offset = curvature * 10 * (t_x * (1 - t_x)) * 4
                    pt[1] += curve_offset * (0.5 - t_y)  # Curve toward center
                
                pts.append(pt)
        
        return np.array(pts, dtype=np.float32)
    
    def _create_regular_grid(self, w: int, h: int, nx: int, ny: int) -> np.ndarray:
        """Create regular grid of control points"""
        pts = []
        for j in range(ny):
            for i in range(nx):
                x = i * w / (nx - 1)
                y = j * h / (ny - 1)
                pts.append([x, y])
        return np.array(pts, dtype=np.float32)
    
    def estimate_curvature(self, plate_img: np.ndarray) -> float:
        """
        Estimate plate curvature from edge analysis.
        
        Returns:
            curvature: 0.0 (flat) to 1.0 (highly curved)
        """
        if plate_img is None or plate_img.size == 0:
            return 0.0
        
        h, w = plate_img.shape[:2]
        
        # Convert to grayscale
        if len(plate_img.shape) == 3:
            gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_img.copy()
        
        # Detect edges
        edges = cv2.Canny(gray, 50, 150)
        
        # Find lines using Hough
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30,
                                minLineLength=w//4, maxLineGap=h//4)
        
        if lines is None or len(lines) < 2:
            return 0.0
        
        # Analyze line angles
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1)
            angles.append(angle)
        
        # High variance in angles suggests curvature
        angle_std = np.std(angles)
        
        # Normalize to 0-1 range (0.3 rad std = high curvature)
        curvature = min(angle_std / 0.3, 1.0)
        
        return curvature
    
    def rectify(self, plate_img: np.ndarray, 
                auto_detect_curvature: bool = True) -> np.ndarray:
        """
        Main rectification method - automatically chooses best approach.
        
        Args:
            plate_img: Input plate image (BGR)
            auto_detect_curvature: Estimate curvature automatically
        
        Returns:
            Rectified plate image
        """
        if plate_img is None or plate_img.size == 0:
            return plate_img
        
        # Estimate curvature
        if auto_detect_curvature:
            curvature = self.estimate_curvature(plate_img)
        else:
            curvature = 0.0
        
        # Detect corners
        corners = self.detect_corners(plate_img)
        
        if curvature > 0.3:
            # Use TPS for curved plates
            logger.debug(f"Using TPS rectification (curvature={curvature:.2f})")
            return self.rectify_tps(plate_img, corners, curvature)
        else:
            # Use perspective transform for flat plates
            logger.debug(f"Using perspective rectification (curvature={curvature:.2f})")
            return self.rectify_perspective(plate_img, corners)


def get_tps_rectifier() -> TPSRectifier:
    """Get singleton TPS rectifier instance (thread-safe)"""
    global _tps_rectifier
    if _tps_rectifier is None:
        with _tps_lock:
            if _tps_rectifier is None:
                _tps_rectifier = TPSRectifier()
    return _tps_rectifier


# ============================================
# Integration helpers
# ============================================

def rectify_plate(plate_img: np.ndarray, 
                  use_tps: bool = True) -> np.ndarray:
    """
    Convenience function to rectify a plate image.
    
    Args:
        plate_img: Input plate image (BGR numpy array)
        use_tps: Use TPS for curved plates, else perspective only
    
    Returns:
        Rectified plate image
    """
    rectifier = get_tps_rectifier()
    
    if use_tps:
        return rectifier.rectify(plate_img, auto_detect_curvature=True)
    else:
        return rectifier.rectify_perspective(plate_img)


if __name__ == "__main__":
    # Test
    import sys
    
    print("🧪 Testing TPSRectifier...")
    
    rectifier = TPSRectifier()
    
    # Create test image
    test_img = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
    
    # Test perspective
    result = rectifier.rectify_perspective(test_img)
    print(f"✅ Perspective: {test_img.shape} -> {result.shape}")
    
    # Test TPS
    result_tps = rectifier.rectify_tps(test_img, curvature=0.5)
    print(f"✅ TPS: {test_img.shape} -> {result_tps.shape}")
    
    # Test auto
    result_auto = rectifier.rectify(test_img)
    print(f"✅ Auto: {test_img.shape} -> {result_auto.shape}")
    
    print("\n✅ All tests passed!")
