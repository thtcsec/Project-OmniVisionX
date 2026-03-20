"""
Super-Resolution Service using RealESRGAN
Conditional upscaling for low-quality frames before YOLO/OCR processing

Applied ONLY when needed (adaptive):
- Frame is blurry (Laplacian variance < threshold)
- Frame is dark (night mode)
- Vehicle is too small/far
"""
import os
import asyncio
import logging
from typing import Optional, Tuple
from PIL import Image
import numpy as np
import cv2

logger = logging.getLogger(__name__)

# Global cache for model (lazy load)
_esrgan_upscaler = None


class SuperResolutionService:
    """
    Conditional Super-Resolution using RealESRGAN.
    Singleton pattern - model shared across all requests.
    """
    _instance = None
    _instance_lock = __import__('threading').Lock()  # Must be initialized at class-definition time
    _model_loaded = False
    
    # Thresholds for conditional SR - MORE AGGRESSIVE for LPR
    BLUR_THRESHOLD = 200      # Was 100 - now apply SR for more blur levels
    BRIGHTNESS_THRESHOLD = 80 # Was 60 - earlier night detection
    MIN_VEHICLE_WIDTH = 400   # Was 200 - upscale vehicles up to 400px width
    
    def __init__(self):
        self._upscaler = None
        self._model_last_used = None
        self._model_idle_timeout = 300  # 5 minutes idle timeout
        
        import threading
        self._lock = threading.Lock()
    
    @classmethod
    def get_instance(cls):
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = SuperResolutionService()
            return cls._instance
    
    def _load_model(self, scale: int = 2):
        """
        Lazy load RealESRGAN model.
        Uses x2 by default for balance between quality and speed.
        Holds lock during entire load to prevent TOCTOU race.
        """
        import time
        
        with self._lock:
            # Check if model needs cleanup due to idle timeout
            if (self._upscaler is not None and 
                self._model_last_used is not None and 
                time.time() - self._model_last_used > self._model_idle_timeout):
                logger.info("🧹 Cleaning up idle RealESRGAN model")
                self._cleanup_model()
            
            # Check if existing model matches requested scale
            if self._upscaler is not None:
                current_scale = getattr(self._upscaler, 'scale', None)
                if current_scale != scale:
                    logger.info("🔄 Scale mismatch (loaded x%s, requested x%s), reloading", current_scale, scale)
                    self._cleanup_model()
                else:
                    self._model_last_used = time.time()
                    return self._upscaler
        
            try:
                from basicsr.archs.rrdbnet_arch import RRDBNet
                from realesrgan import RealESRGANer
                
                # Choose model based on scale
                if scale == 4:
                    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                                   num_block=23, num_grow_ch=32, scale=4)
                    model_path = os.path.join('/app/weights', 'RealESRGAN_x4plus.pth')
                else:
                    # x2 model (faster, still good quality)
                    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                                   num_block=23, num_grow_ch=32, scale=2)
                    model_path = os.path.join('/app/weights', 'RealESRGAN_x2plus.pth')
                
                # Check if model exists
                if not os.path.exists(model_path):
                    logger.warning(f"RealESRGAN model not found at {model_path}")
                    return None
                
                self._upscaler = RealESRGANer(
                    scale=scale,
                    model_path=model_path,
                    model=model,
                    tile=200,      # Tile processing for memory efficiency
                    tile_pad=10,
                    pre_pad=0,
                    half=True      # FP16 for faster inference
                )
                self._model_last_used = time.time()
                
                logger.info(f"✅ RealESRGAN x{scale} loaded from {model_path}")
                return self._upscaler
                
            except ImportError as e:
                logger.warning(f"RealESRGAN not available: {e}")
                return None
            except Exception as e:
                logger.error(f"Failed to load RealESRGAN: {e}")
                return None
    
    def _cleanup_model(self):
        """
        Clean up GPU memory by releasing the model.
        Called automatically after idle timeout or manually.
        """
        if self._upscaler is not None:
            try:
                # Clear GPU memory
                if hasattr(self._upscaler, 'device') and 'cuda' in str(self._upscaler.device):
                    import torch
                    torch.cuda.empty_cache()
                
                # Delete model reference
                del self._upscaler
                self._upscaler = None
                self._model_last_used = None
                
                logger.info("🧹 RealESRGAN model cleaned up, GPU memory freed")
                
            except Exception as e:
                logger.warning(f"Error during model cleanup: {e}")
    
    def cleanup(self):
        """
        Public method to manually cleanup model.
        Useful for batch processing or memory management.
        """
        with self._lock:
            self._cleanup_model()
    
    def is_available(self) -> bool:
        """Check if SR model is available"""
        x2_path = os.path.join('/app/weights', 'RealESRGAN_x2plus.pth')
        x4_path = os.path.join('/app/weights', 'RealESRGAN_x4plus.pth')
        return os.path.exists(x2_path) or os.path.exists(x4_path)
    
    @staticmethod
    def compute_blur_score(image: np.ndarray) -> float:
        """Compute blur score using Laplacian variance"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        return cv2.Laplacian(gray, cv2.CV_64F).var()
    
    @staticmethod
    def compute_brightness(image: np.ndarray) -> float:
        """Compute average brightness"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        return np.mean(gray)
    
    def should_apply_sr(
        self,
        image: np.ndarray,
        vehicle_width: Optional[int] = None
    ) -> Tuple[bool, str]:
        """
        Determine if Super-Resolution should be applied.
        
        Returns:
            (should_apply, reason)
        """
        # Check blur
        blur_score = self.compute_blur_score(image)
        if blur_score < self.BLUR_THRESHOLD:
            return True, f"blurry (score={blur_score:.0f})"
        
        # Check brightness (night mode)
        brightness = self.compute_brightness(image)
        if brightness < self.BRIGHTNESS_THRESHOLD:
            return True, f"dark (brightness={brightness:.0f})"
        
        # Check vehicle size
        if vehicle_width is not None and vehicle_width < self.MIN_VEHICLE_WIDTH:
            return True, f"small vehicle (width={vehicle_width}px)"
        
        return False, "good quality"
    
    def upscale_sync(self, image: Image.Image, scale: int = 2) -> Optional[Image.Image]:
        """
        Synchronous upscaling using RealESRGAN.
        
        Args:
            image: PIL Image to upscale
            scale: Upscale factor (2 or 4)
            
        Returns:
            Upscaled PIL Image or None if failed
        """
        upscaler = self._load_model(scale)
        if upscaler is None:
            return None
        
        try:
            # Convert PIL to numpy (RGB)
            img_np = np.array(image)
            
            # RealESRGAN expects BGR
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            
            # Run upscaling
            output_bgr, _ = upscaler.enhance(img_bgr, outscale=scale)
            
            # Convert back to RGB PIL
            output_rgb = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)
            
            # Update last used time
            import time
            with self._lock:
                self._model_last_used = time.time()
            
            return Image.fromarray(output_rgb)
            
        except Exception as e:
            logger.error(f"RealESRGAN upscale failed: {e}")
            return None
    
    async def upscale(self, image: Image.Image, scale: int = 2) -> Optional[Image.Image]:
        """Async wrapper for upscaling"""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.upscale_sync, image, scale)
    
    async def conditional_upscale(
        self,
        image: Image.Image,
        vehicle_width: Optional[int] = None,
        force: bool = False
    ) -> Tuple[Image.Image, bool, str]:
        """
        Apply Super-Resolution only if needed.
        
        Args:
            image: PIL Image
            vehicle_width: Optional width of vehicle bbox
            force: Force SR regardless of quality
            
        Returns:
            (processed_image, was_upscaled, reason)
        """
        img_np = np.array(image)
        
        # Check if SR is needed
        if not force:
            should_apply, reason = self.should_apply_sr(img_np, vehicle_width)
            if not should_apply:
                return image, False, reason
        else:
            reason = "forced"
        
        # Check if model available
        if not self.is_available():
            return image, False, "model not available"
        
        # Apply SR
        result = await self.upscale(image, scale=2)
        if result is None:
            return image, False, "upscale failed"
        
        logger.info(f"🔍 Applied SR x2: {reason}")
        return result, True, reason


# Convenience function
def get_super_resolution_service() -> SuperResolutionService:
    return SuperResolutionService.get_instance()


def cleanup_super_resolution():
    """
    Cleanup function for graceful shutdown.
    Call this when shutting down the application.
    """
    service = SuperResolutionService.get_instance()
    service.cleanup()
