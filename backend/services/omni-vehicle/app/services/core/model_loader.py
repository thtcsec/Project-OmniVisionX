import os
import logging
import threading
import time
from typing import Optional, Tuple

import numpy as np

from app.services.core.device_utils import resolve_device
from app.config import get_settings

logger = logging.getLogger("omni-vehicle.model_loader")


class ModelLoader:
    _instance: "ModelLoader | None" = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self._detector = None
        self._ocr = None
        self._init_lock = threading.Lock()
        self._last_used = 0.0

    @classmethod
    def get_instance(cls) -> "ModelLoader":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = ModelLoader()
            return cls._instance

    def load(self) -> Tuple[object, object]:
        with self._init_lock:
            if self._detector is not None and self._ocr is not None:
                self._touch()
                return self._detector, self._ocr

            self._ensure_cache_dirs()
            self._apply_cuda_fraction()

            # Verify model integrity before loading
            weights_path = os.environ.get("WEIGHTS_PATH", "/app/weights")
            try:
                from shared.model_integrity import verify_all_models
                verify_all_models(weights_path, strict=True)
            except SystemExit:
                raise
            except Exception as e:
                logger.warning("⚠️ Model integrity check skipped: %s", e)

            # Load YOLOv5 plate detector
            try:
                import torch

                detector_path = os.path.join(weights_path, "LP_detector.pt")
                if os.path.exists(detector_path):
                    preferred_device = resolve_device("cuda")
                    loaded = False
                    for force_reload in [False, True]:
                        try:
                            self._detector = torch.hub.load(
                                "ultralytics/yolov5",
                                "custom",
                                path=detector_path,
                                device=preferred_device,
                                force_reload=force_reload,
                                trust_repo=True,
                            )
                            self._detector.conf = float(os.environ.get(
                                "PLATE_DETECTOR_CONFIDENCE",
                                str(get_settings().plate_detector_confidence)
                            ))
                            logger.info("✅ LP_detector loaded from %s (%s, conf=%.2f)", detector_path, preferred_device, self._detector.conf)
                            loaded = True
                            break
                        except Exception as exc:
                            err = str(exc).lower()
                            if not force_reload and ("invalid device" in err or "cache" in err or "out of date" in err):
                                logger.warning("⚠️ torch.hub cache stale, retrying with force_reload=True: %s", exc)
                                continue
                            if preferred_device != "cpu" and (
                                "no kernel image" in err
                                or "cuda error" in err
                                or "device-side" in err
                            ):
                                logger.warning("⚠️ CUDA not supported for LP_detector, falling back to CPU: %s", exc)
                                try:
                                    self._detector = torch.hub.load(
                                        "ultralytics/yolov5",
                                        "custom",
                                        path=detector_path,
                                        device="cpu",
                                        force_reload=force_reload,
                                        trust_repo=True,
                                    )
                                    self._detector.conf = float(os.environ.get(
                                        "PLATE_DETECTOR_CONFIDENCE",
                                        str(get_settings().plate_detector_confidence)
                                    ))
                                    logger.info("✅ LP_detector loaded from %s (cpu, conf=%.2f)", detector_path, self._detector.conf)
                                    loaded = True
                                except Exception as cpu_exc:
                                    logger.error("❌ LP_detector CPU fallback also failed: %s", cpu_exc)
                                break
                            else:
                                raise
                    if not loaded and self._detector is None:
                        logger.error("❌ Failed to load LP_detector after all attempts")
                else:
                    logger.warning("⚠️ LP_detector not found at %s", detector_path)
            except Exception as exc:
                logger.error("❌ Failed to load LP_detector: %s", exc)
                self._detector = None

            # Load PaddleOCR
            try:
                os.environ["FLAGS_log_level"] = "3"
                os.environ["GLOG_minloglevel"] = "3"
                from paddleocr import PaddleOCR

                use_gpu = False
                try:
                    import paddle

                    if paddle.is_compiled_with_cuda():
                        try:
                            gpu_count = paddle.device.cuda.device_count()
                            if gpu_count > 0:
                                use_gpu = True
                            else:
                                logger.warning("⚠️ CUDA compiled but no GPU devices found (count=%d)", gpu_count)
                                use_gpu = False
                        except Exception:
                            use_gpu = False
                except Exception:
                    use_gpu = False

                # CRITICAL FIX: Use 'ch' (Chinese) model for better plate OCR
                # English model causes severe character confusion (6→1, 5→6, M→S, A→H)
                # Chinese model has better discrimination for alphanumeric characters
                self._ocr = PaddleOCR(
                    lang="ch",  # FIXED: 'ch' instead of 'en'
                    use_gpu=use_gpu,
                    show_log=False,
                    use_angle_cls=False,  # Plate crops already oriented from detection
                    det_db_thresh=0.2,  # Lower for small plate text
                    det_db_box_thresh=0.4,  # More lenient detection
                    rec_batch_num=6,
                )
                logger.info("✅ PaddleOCR loaded (%s)", "GPU" if use_gpu else "CPU")
            except Exception as exc:
                logger.error("❌ Failed to load PaddleOCR: %s", exc)
                self._ocr = None

            self._touch()
            return self._detector, self._ocr

    def warmup(self) -> None:
        """Warm up models to avoid first-request latency."""
        self._touch()

        if self._detector is not None:
            try:
                dummy = np.zeros((640, 640, 3), dtype=np.uint8)
                _ = self._detector(dummy, size=640)
                logger.info("🔥 LP_detector warmup done")
            except Exception as exc:
                logger.warning("LP_detector warmup failed: %s", exc)

        if self._ocr is not None:
            try:
                dummy = np.zeros((48, 160, 3), dtype=np.uint8)
                _ = self._ocr.ocr(dummy)
                logger.info("🔥 PaddleOCR warmup done")
            except Exception as exc:
                logger.warning("PaddleOCR warmup failed: %s", exc)

    def get_models(self) -> Tuple[object, object]:
        if self._detector is None or self._ocr is None:
            return self.load()
        self._touch()
        return self._detector, self._ocr

    def check_cuda_health(self) -> bool:
        """Check if CUDA models are still functional after potential OOM.
        
        If CUDA is in an error state, clear the cache and mark models
        for reload on next get_models() call.
        """
        try:
            import torch
            if torch.cuda.is_available() and self._detector is not None:
                # Quick tensor check on current device
                device = next(self._detector.model.parameters()).device
                if device.type == 'cuda':
                    test = torch.zeros(1, device=device)
                    del test
            return True
        except RuntimeError as e:
            err = str(e).lower()
            if "out of memory" in err or "cuda error" in err or "device-side" in err:
                logger.error("🔥 CUDA in error state: %s — clearing cache and marking for reload", e)
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                self._detector = None
                self._ocr = None
                return False
            return True
        except Exception:
            return True

    def _touch(self) -> None:
        self._last_used = time.time()

    @staticmethod
    def _ensure_cache_dirs() -> None:
        cache_dirs = [
            os.environ.get("TORCH_HOME"),
            os.environ.get("XDG_CACHE_HOME"),
            os.environ.get("ULTRALYTICS_CACHE"),
            os.environ.get("PADDLE_HOME"),
            os.environ.get("PADDLEOCR_HOME"),
            os.environ.get("PADDLE_PDX_CACHE_HOME"),
            os.environ.get("INSIGHTFACE_HOME"),
            os.environ.get("HF_HOME"),
            os.environ.get("TRANSFORMERS_CACHE"),
            os.environ.get("ONNXRUNTIME_CACHE_DIR"),
            os.environ.get("OPENCV_CACHE_DIR"),
        ]
        for path in cache_dirs:
            if not path:
                continue
            try:
                os.makedirs(path, exist_ok=True)
            except Exception:
                continue

    @staticmethod
    def _apply_cuda_fraction() -> None:
        try:
            import torch

            if not torch.cuda.is_available():
                return

            fraction_raw = os.environ.get("CUDA_MEMORY_FRACTION")
            if fraction_raw:
                fraction = float(fraction_raw)
                if 0 < fraction <= 1:
                    torch.cuda.set_per_process_memory_fraction(fraction)
                    logger.info("CUDA memory fraction set to %.2f", fraction)
        except Exception as exc:
            logger.warning("Failed to apply CUDA memory fraction: %s", exc)
