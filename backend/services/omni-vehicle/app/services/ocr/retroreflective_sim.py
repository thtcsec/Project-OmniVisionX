"""
Retroreflective Glare Simulator - 3M Sheeting Simulation
=========================================================
Simulates realistic headlight glare on Vietnamese license plates.

Key insight from research:
- Vietnamese plates use 3M reflective sheeting (Type XI, retroreflective)
- Under headlights, creates HEXAGONAL diffraction pattern (micro-prisms)
- NOT Gaussian blur - models trained on Gaussian fail at night

This module provides:
1. Retroreflective glare simulation for training data augmentation
2. Glare detection for preprocessing (identify affected areas)
3. Glare suppression for OCR improvement

Reference: Cook-Torrance BRDF model for retroreflection
"""
import logging
from typing import Tuple, Optional
import numpy as np
import cv2

logger = logging.getLogger(__name__)


class ReflectiveSheetingSimulator:
    """
    Simulates 3M Diamond Grade (Type XI) reflective sheeting behavior.
    Used for realistic synthetic data augmentation.
    """
    
    def __init__(self, retroreflective_coeff: float = 800.0):
        """
        Args:
            retroreflective_coeff: Retroreflection coefficient (cd/lux/m²)
                                   Type XI: ~800, Type I: ~70
        """
        self.retroreflective_coeff = retroreflective_coeff
        
        # Pre-compute hexagonal kernel for micro-prism diffraction
        self._hex_kernel = self._create_hexagonal_kernel(radius=7)
        self._hex_kernel_large = self._create_hexagonal_kernel(radius=15)
    
    def _create_hexagonal_kernel(self, radius: int = 7) -> np.ndarray:
        """
        Create hexagonal bloom kernel (micro-prism diffraction pattern).
        
        Real retroreflective tape has hexagonal micro-prisms that create
        6-pointed star diffraction pattern, NOT circular Gaussian.
        """
        size = radius * 2 + 1
        kernel = np.zeros((size, size), dtype=np.float32)
        center = radius
        
        # Create 6 radial lines (hexagonal star)
        for angle_deg in [0, 60, 120, 180, 240, 300]:
            angle_rad = np.deg2rad(angle_deg)
            for r in range(radius):
                x = int(center + r * np.cos(angle_rad))
                y = int(center + r * np.sin(angle_rad))
                if 0 <= x < size and 0 <= y < size:
                    # Intensity decreases with distance
                    kernel[y, x] = 1.0 - (r / radius) * 0.7
        
        # Add center bright spot
        cv2.circle(kernel, (center, center), radius // 3, 1.0, -1)
        
        # Slight Gaussian blur to soften edges
        kernel = cv2.GaussianBlur(kernel, (3, 3), 0.5)
        
        # Normalize
        kernel = kernel / kernel.sum()
        
        return kernel
    
    def apply_headlight_glare(self, plate_img: np.ndarray,
                               light_angle: float = 0.0,
                               intensity: float = 0.5) -> np.ndarray:
        """
        Simulate headlight glare on license plate.
        
        This is DIFFERENT from specular glare:
        - Retroreflection bounces light back toward source
        - Creates characteristic hexagonal diffraction pattern
        - Affects areas based on viewing angle
        
        Args:
            plate_img: Input plate image (BGR)
            light_angle: Angle of light source relative to camera (degrees)
            intensity: Glare intensity (0-1)
        
        Returns:
            Plate image with realistic retroreflective glare
        """
        if plate_img is None or plate_img.size == 0:
            return plate_img
        
        h, w = plate_img.shape[:2]
        result = plate_img.copy().astype(np.float32)
        
        # Create intensity map based on retroreflection model
        intensity_map = self._compute_retroreflection_map(w, h, light_angle)
        
        # Scale by requested intensity
        intensity_map = intensity_map * intensity
        
        # Apply hexagonal bloom filter (NOT Gaussian!)
        bloom = cv2.filter2D(intensity_map, -1, self._hex_kernel)
        
        # Expand bloom to 3 channels
        bloom_3ch = np.stack([bloom, bloom, bloom], axis=-1)
        
        # Blend with original (additive blending for glare)
        result = result + bloom_3ch * 255 * intensity
        
        # Clip and convert back
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result
    
    def _compute_retroreflection_map(self, w: int, h: int,
                                      light_angle: float) -> np.ndarray:
        """
        Compute retroreflection intensity based on viewing geometry.
        
        Uses simplified Cook-Torrance BRDF:
        - Maximum reflection when light source = viewer direction
        - Falls off with angle deviation
        """
        # Create coordinate grids
        x = np.linspace(-1, 1, w)
        y = np.linspace(-1, 1, h)
        xx, yy = np.meshgrid(x, y)
        
        # Light direction (from angle)
        light_x = np.sin(np.deg2rad(light_angle))
        
        # Distance from center weighted by light angle
        dist_from_light_axis = np.abs(xx - light_x)
        
        # Retroreflection intensity (maximum at center of light cone)
        # Using exponential falloff
        intensity = np.exp(-dist_from_light_axis ** 2 / 0.5)
        
        # Add some randomness (uneven reflective surface)
        noise = np.random.uniform(0.8, 1.0, intensity.shape)
        intensity = intensity * noise
        
        return intensity.astype(np.float32)
    
    def apply_flash_glare(self, plate_img: np.ndarray,
                          center: Optional[Tuple[int, int]] = None,
                          intensity: float = 0.7) -> np.ndarray:
        """
        Simulate camera flash glare (close-range, intense).
        
        Flash creates more concentrated, intense retroreflection
        compared to headlights.
        
        Args:
            plate_img: Input plate image
            center: Flash center point (default: image center)
            intensity: Flash intensity (0-1)
        
        Returns:
            Image with flash glare
        """
        if plate_img is None or plate_img.size == 0:
            return plate_img
        
        h, w = plate_img.shape[:2]
        result = plate_img.copy().astype(np.float32)
        
        if center is None:
            center = (w // 2, h // 2)
        
        # Create flash intensity map (radial from center)
        y_coords, x_coords = np.ogrid[:h, :w]
        dist = np.sqrt((x_coords - center[0])**2 + (y_coords - center[1])**2)
        
        # Flash falloff
        max_dist = np.sqrt(w**2 + h**2) / 2
        intensity_map = 1 - (dist / max_dist)
        intensity_map = np.clip(intensity_map, 0, 1) ** 2  # Square for sharper falloff
        
        # Apply larger hexagonal bloom for flash
        bloom = cv2.filter2D(intensity_map.astype(np.float32), -1, 
                            self._hex_kernel_large)
        
        # Blend
        bloom_3ch = np.stack([bloom, bloom, bloom], axis=-1)
        result = result + bloom_3ch * 255 * intensity
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result


class GlareDetector:
    """
    Detect retroreflective glare in plate images.
    Used for preprocessing to identify affected areas.
    """
    
    def __init__(self, bright_threshold: int = 240,
                 saturation_threshold: int = 30):
        """
        Args:
            bright_threshold: Pixel brightness threshold for glare detection
            saturation_threshold: Low saturation indicates white glare
        """
        self.bright_threshold = bright_threshold
        self.saturation_threshold = saturation_threshold
    
    def detect(self, plate_img: np.ndarray) -> Tuple[bool, np.ndarray, float]:
        """
        Detect glare in plate image.
        
        Returns:
            has_glare: Whether significant glare detected
            glare_mask: Binary mask of glare regions
            glare_ratio: Ratio of glare pixels (0-1)
        """
        if plate_img is None or plate_img.size == 0:
            return False, np.zeros((1, 1), dtype=np.uint8), 0.0
        
        h, w = plate_img.shape[:2]
        
        # Convert to HSV
        if len(plate_img.shape) == 3:
            hsv = cv2.cvtColor(plate_img, cv2.COLOR_BGR2HSV)
            v_channel = hsv[:, :, 2]
            s_channel = hsv[:, :, 1]
        else:
            v_channel = plate_img
            s_channel = np.zeros_like(plate_img)
        
        # Glare = bright + low saturation (white)
        bright_mask = v_channel > self.bright_threshold
        low_sat_mask = s_channel < self.saturation_threshold
        
        glare_mask = (bright_mask & low_sat_mask).astype(np.uint8) * 255
        
        # Calculate ratio
        glare_pixels = np.sum(glare_mask > 0)
        total_pixels = h * w
        glare_ratio = glare_pixels / total_pixels
        
        has_glare = glare_ratio > 0.1  # >10% is significant
        
        return has_glare, glare_mask, glare_ratio


class GlareSuppressor:
    """
    Suppress retroreflective glare for better OCR.
    """
    
    def __init__(self):
        self.detector = GlareDetector()
    
    def suppress(self, plate_img: np.ndarray,
                 method: str = 'inpaint') -> np.ndarray:
        """
        Suppress glare in plate image.
        
        Args:
            plate_img: Input plate image
            method: 'inpaint', 'adaptive', or 'gamma'
        
        Returns:
            Glare-suppressed image
        """
        if plate_img is None or plate_img.size == 0:
            return plate_img
        
        # Detect glare
        has_glare, glare_mask, glare_ratio = self.detector.detect(plate_img)
        
        if not has_glare or glare_ratio < 0.05:
            return plate_img  # No significant glare
        
        result = plate_img.copy()
        
        if method == 'inpaint':
            # Inpaint glare regions
            result = cv2.inpaint(result, glare_mask, 
                               inpaintRadius=3, 
                               flags=cv2.INPAINT_TELEA)
        
        elif method == 'adaptive':
            # Adaptive histogram equalization on glare regions
            if len(result.shape) == 3:
                lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
                l_channel = lab[:, :, 0]
            else:
                l_channel = result.copy()
            
            # Apply CLAHE only to glare regions
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
            l_eq = clahe.apply(l_channel)
            
            # Blend based on glare mask
            mask_float = glare_mask.astype(np.float32) / 255.0
            l_channel = (l_channel * (1 - mask_float) + l_eq * mask_float).astype(np.uint8)
            
            if len(result.shape) == 3:
                lab[:, :, 0] = l_channel
                result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            else:
                result = l_channel
        
        elif method == 'gamma':
            # Gamma correction for overexposed areas
            # gamma > 1 darkens bright pixels: (i/255)^2.0 maps 200→157
            gamma = 2.0
            table = np.array([((i / 255.0) ** gamma) * 255 
                             for i in range(256)]).astype(np.uint8)
            
            corrected = cv2.LUT(result, table)
            
            # Blend based on glare mask
            mask_float = glare_mask.astype(np.float32) / 255.0
            if len(result.shape) == 3:
                mask_float = np.stack([mask_float] * 3, axis=-1)
            
            result = (result * (1 - mask_float) + corrected * mask_float).astype(np.uint8)
        
        return result


# ============================================
# Convenience functions
# ============================================

def add_retroreflective_glare(plate_img: np.ndarray,
                               intensity: float = 0.5,
                               light_angle: float = 0.0) -> np.ndarray:
    """
    Add realistic retroreflective glare to plate image.
    For data augmentation during training.
    """
    simulator = ReflectiveSheetingSimulator()
    return simulator.apply_headlight_glare(plate_img, light_angle, intensity)


def suppress_glare(plate_img: np.ndarray,
                   method: str = 'inpaint') -> np.ndarray:
    """
    Suppress glare in plate image for better OCR.
    """
    suppressor = GlareSuppressor()
    return suppressor.suppress(plate_img, method)


def detect_glare(plate_img: np.ndarray) -> Tuple[bool, float]:
    """
    Check if plate has significant glare.
    
    Returns:
        has_glare: True if >10% glare
        glare_ratio: Percentage of glare pixels
    """
    detector = GlareDetector()
    has_glare, _, glare_ratio = detector.detect(plate_img)
    return has_glare, glare_ratio


if __name__ == "__main__":
    print("🧪 Testing Retroreflective Glare Simulator...")
    
    # Create test plate image (white text on dark background)
    test_img = np.zeros((80, 200, 3), dtype=np.uint8)
    test_img[:] = (40, 40, 40)  # Dark background
    cv2.putText(test_img, "65A-12345", (10, 55), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    
    simulator = ReflectiveSheetingSimulator()
    
    # Test headlight glare
    glare_img = simulator.apply_headlight_glare(test_img, intensity=0.5)
    print(f"✅ Headlight glare: {test_img.shape} -> {glare_img.shape}")
    
    # Test flash glare
    flash_img = simulator.apply_flash_glare(test_img, intensity=0.7)
    print(f"✅ Flash glare: {test_img.shape} -> {flash_img.shape}")
    
    # Test detection
    has_glare, ratio = detect_glare(glare_img)
    print(f"✅ Glare detection: has_glare={has_glare}, ratio={ratio:.1%}")
    
    # Test suppression
    suppressed = suppress_glare(glare_img, method='inpaint')
    print(f"✅ Glare suppression: {glare_img.shape} -> {suppressed.shape}")
    
    print("\n✅ All tests passed!")
