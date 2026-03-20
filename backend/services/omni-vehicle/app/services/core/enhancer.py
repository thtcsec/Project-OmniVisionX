"""
Image Enhancement Service
Optimized for Nighttime LPR using OpenCV.

Phase 4 Features:
- Multi-scale retinex for low-light plates
- IR plate inversion (white-on-black)
- Conservative enhancement (plate crop only)
"""
import cv2
import numpy as np
from PIL import Image


class ImageEnhancer:
    """
    Image enhancement for OCR, optimized for license plates.

    IMPORTANT: All enhancement should be applied to PLATE CROP only,
    never to full frame (to avoid over-processing).
    """

    @staticmethod
    def preprocess_for_ocr(image: Image.Image, use_threshold: bool = True) -> Image.Image:
        """
        Apply targeted enhancement chain for OCR:
        1. Grayscale
        2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
        3. Sharpening (Unsharp mask for character edges)
        4. Optional Adaptive Threshold (for OCR input)

        Args:
            image: PIL Image (Crop of license plate)
            use_threshold: Apply adaptive threshold for binary output

        Returns:
            Enhanced PIL Image ready for OCR
        """
        # Convert PIL to numpy (RGB format)
        img_np = np.array(image)

        # Check if image is valid
        if img_np.size == 0:
            return image

        # 1. Convert to Grayscale
        if len(img_np.shape) == 3:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_np

        # 2. Apply CLAHE
        # Clip Limit: 2.0 (standard for low light), TileGridSize: 8x8
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # 3. Sharpening (enhances character edges)
        # Using sharpen kernel instead of blur for better OCR
        sharpen_kernel = np.array([[0, -1, 0],
                                   [-1, 5, -1],
                                   [0, -1, 0]])
        sharpened = cv2.filter2D(enhanced, -1, sharpen_kernel)

        # 4. Optional: Adaptive Threshold (binary output for OCR)
        if use_threshold:
            result = cv2.adaptiveThreshold(
                sharpened, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11, 2
            )
        else:
            # Light denoise if not using threshold
            result = cv2.GaussianBlur(sharpened, (3, 3), 0)

        # Convert grayscale result back to RGB for consistent downstream handling
        result_rgb = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(result_rgb)

    @staticmethod
    def is_night_time(image, threshold: int = 80, percentile: int = 90) -> bool:
        """
        Heuristic to detect night/dark scenes using brightness percentile.
        Uses percentile of V channel in HSV to avoid mean bias.
        """
        # Handle both PIL Image and numpy array
        if hasattr(image, 'shape'):  # numpy array
            img_np = image
        elif hasattr(image, 'size'):  # PIL Image
            img_np = np.array(image)
        else:
            return False  # Unknown type

        if len(img_np.shape) != 3:
            return False

        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
        v_channel = hsv[:, :, 2]
        p_value = float(np.percentile(v_channel, percentile))

        return p_value < threshold

    @staticmethod
    def preprocess_dual(image: Image.Image) -> tuple:
        """
        Return both base (non-threshold) and binary (threshold) versions.
        Use binary as fallback if base OCR score is low.

        Args:
            image: PIL Image (Crop of license plate)

        Returns:
            Tuple of (base_image, binary_image)
        """
        base = ImageEnhancer.preprocess_for_ocr(image, use_threshold=False)
        binary = ImageEnhancer.preprocess_for_ocr(image, use_threshold=True)
        return base, binary

    # ========================================
    # PHASE 4: Night Mode Enhancement
    # ========================================

    @staticmethod
    def single_scale_retinex(img: np.ndarray, sigma: float) -> np.ndarray:
        """Single Scale Retinex for low-light enhancement"""
        blur = cv2.GaussianBlur(img, (0, 0), sigma)
        # Avoid log(0)
        blur = np.maximum(blur, 1)
        img_safe = np.maximum(img, 1)
        retinex = np.log10(img_safe.astype(np.float64)) - np.log10(blur.astype(np.float64))
        return retinex

    @staticmethod
    def multi_scale_retinex(img: np.ndarray, sigmas: list = None) -> np.ndarray:
        """
        Multi-Scale Retinex (MSR) for night/low-light enhancement.

        This mimics human vision adaptation to varying light conditions.
        Use on PLATE CROP only, not full frame!

        Args:
            img: Grayscale numpy array
            sigmas: Gaussian blur scales

        Returns:
            Enhanced grayscale image
        """
        if sigmas is None:
            sigmas = [15, 80, 250]
        if not sigmas:
            return img

        retinex = np.zeros_like(img, dtype=np.float64)

        for sigma in sigmas:
            retinex += ImageEnhancer.single_scale_retinex(img, sigma)

        retinex = retinex / len(sigmas)

        # Normalize to 0-255
        retinex = (retinex - retinex.min()) / (retinex.max() - retinex.min() + 1e-6)
        retinex = (retinex * 255).astype(np.uint8)

        return retinex

    @staticmethod
    def detect_ir_plate(image) -> bool:
        """
        Detect if plate is from IR camera (white text on dark background).

        Heuristic: If plate has more dark pixels than light, it's likely IR.
        """
        # Handle both PIL Image and numpy array
        if hasattr(image, 'shape'):  # numpy array
            img_np = image
        elif hasattr(image, 'size'):  # PIL Image
            img_np = np.array(image)
        else:
            return False

        if len(img_np.shape) == 3:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_np

        # Count dark vs light pixels
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        dark_ratio = np.sum(binary == 0) / binary.size

        # If more than 60% is dark, likely IR plate
        return dark_ratio > 0.6

    @staticmethod
    def invert_ir_plate(image: Image.Image) -> Image.Image:
        """
        Invert IR plate colors (white-on-black → black-on-white).

        This helps OCR which expects dark text on light background.
        """
        img_np = np.array(image)
        if len(img_np.shape) == 3:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_np

        inverted = cv2.bitwise_not(gray)
        return Image.fromarray(inverted)

    @staticmethod
    def preprocess_night_plate(image: Image.Image) -> Image.Image:
        """
        Full night-mode enhancement pipeline for plate crops.

        Pipeline:
        1. Check if IR plate → invert
        2. Apply multi-scale retinex
        3. CLAHE (lighter clip limit)
        4. Light sharpening (no threshold - avoid losing thin chars)

        Args:
            image: PIL Image (Plate crop only!)

        Returns:
            Enhanced plate image
        """
        # Skip processing if plate is too small (useless for OCR)
        if image.width < 60 or image.height < 20:
            return image

        img_np = np.array(image)
        if img_np.size == 0:
            return image

        # Convert to grayscale
        if len(img_np.shape) == 3:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_np

        # 1. Check for IR plate and invert if needed
        _, binary_check = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        dark_ratio = np.sum(binary_check == 0) / binary_check.size
        if dark_ratio > 0.6:
            gray = cv2.bitwise_not(gray)

        # 2. Multi-scale retinex (lighter sigmas for small plate crops)
        enhanced = ImageEnhancer.multi_scale_retinex(gray, sigmas=[10, 50, 150])

        # 3. CLAHE with lighter clip (avoid over-enhancement)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4))
        enhanced = clahe.apply(enhanced)

        # 4. Light sharpening
        sharpen_kernel = np.array([[0, -0.5, 0],
                                   [-0.5, 3, -0.5],
                                   [0, -0.5, 0]])
        enhanced = cv2.filter2D(enhanced, -1, sharpen_kernel)

        # 5. Light denoise (preserve edges)
        enhanced = cv2.bilateralFilter(enhanced, 5, 75, 75)

        # Convert grayscale result back to RGB for consistent downstream handling
        enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(enhanced_rgb)

    @staticmethod
    def preprocess_smart(image: Image.Image, is_night: bool = False, debug: bool = False) -> Image.Image:
        """
        Smart preprocessing based on day/night mode.

        Args:
            image: PIL Image (Plate crop)
            is_night: Use night mode enhancement
            debug: Print debug info

        Returns:
            Preprocessed image
        """
        # Skip tiny plates
        if image.width < 60 or image.height < 20:
            return image

        # Detect IR for logging
        is_ir = ImageEnhancer.detect_ir_plate(image) if is_night else False

        if debug:
            print(f"🔧 LPR preprocess: night={is_night}, ir={is_ir}, size={image.width}x{image.height}")

        if is_night:
            return ImageEnhancer.preprocess_night_plate(image)
        else:
            return ImageEnhancer.preprocess_for_ocr(image, use_threshold=False)
