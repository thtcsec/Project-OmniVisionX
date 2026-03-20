"""
Settings API Router
Manage AI Engine configuration and status - REAL HARDWARE DATA
"""
import asyncio
import os
import glob
import platform
import re
import shutil
import subprocess
import sys
import importlib.util
from importlib import metadata
from datetime import datetime
from typing import Optional
try:
    import torch
    TORCH_AVAILABLE = True
except Exception as _torch_err:
    torch = None
    TORCH_AVAILABLE = False
    print(f"Warning: torch not available ({type(_torch_err).__name__}: {_torch_err}), GPU info disabled")
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.config import get_settings

# NVIDIA Management Library for GPU monitoring
try:
    import pynvml
    PYNVML_AVAILABLE = True
except Exception as _nvml_err:
    PYNVML_AVAILABLE = False
    print(f"Warning: pynvml not available ({type(_nvml_err).__name__}: {_nvml_err}), GPU temperature/power data disabled")

router = APIRouter()


# ============================================================
# RESPONSE MODELS
# ============================================================

class GPUInfo(BaseModel):
    id: int
    name: str
    vram_total_mb: int
    vram_used_mb: int
    vram_free_mb: int
    cuda_version: str
    driver_version: str
    temperature: Optional[int] = None  # Celsius
    power_usage: Optional[float] = None  # Watts
    power_limit: Optional[float] = None  # Watts


class ModelFile(BaseModel):
    filename: str
    path: str
    size_mb: float
    type: str  # "yolo", "face", "ocr", "other"


class SystemInfo(BaseModel):
    cuda_available: bool
    cuda_version: Optional[str]
    gpu_count: int
    gpus: list[GPUInfo]
    torch_version: str
    device_in_use: str


class AvailableModels(BaseModel):
    yolo_models: list[ModelFile]
    face_models: list[ModelFile]
    ocr_models: list[ModelFile]
    other_models: list[ModelFile]


class CurrentConfig(BaseModel):
    # Model Settings
    yolo_model: str
    device: str
    batch_size: int
    
    # Thresholds
    confidence_threshold: float
    nms_threshold: float
    plate_merge_iou_threshold: float
    face_confidence_threshold: float
    plate_confidence_threshold: float
    plate_detector_confidence: float
    
    # OCR Thresholds (live-tier)
    ocr_confidence_threshold: float
    ocr_min_text_length: int
    event_min_vote_count: int
    event_instant_confidence: float
    event_dedup_ttl: float
    lpr_consensus_history: int
    lpr_stream_backlog_threshold: int
    lpr_stream_drop_ratio_max: float
    
    # Processing
    snapshot_interval: int
    
    # Feature Flags
    enable_face_recognition: bool
    enable_plate_ocr: bool
    enable_night_lpr: bool

    # Adaptive Confidence (legacy pipeline)
    adaptive_confidence_base: float
    adaptive_confidence_alpha: float
    adaptive_confidence_min: float
    adaptive_confidence_max: float

    # Fortress tuning
    fortress_vehicle_confidence: float
    fortress_plate_confidence: float
    enable_fortress_lpr: bool

    # Visualization
    lpr_draw_vehicle_bbox: bool

    # Super-resolution before OCR rescue
    lpr_sr_enable: bool
    lpr_sr_min_height: int
    lpr_sr_min_width: int
    lpr_sr_conf_threshold: float

    # Data collection
    enable_lpr_data_collection: bool
    lpr_collection_dir: str
    lpr_collect_sample_rate: float
    lpr_collect_vehicle: bool
    lpr_collect_plate: bool
    lpr_collect_min_conf: float
    lpr_collect_max_conf: float
    lpr_collect_low_conf_only: bool
    lpr_collect_quality_filter: bool
    lpr_collect_min_sharpness: float
    lpr_collect_min_brightness: float
    lpr_collect_max_brightness: float
    lpr_collect_min_vehicle_area: int
    lpr_collect_min_plate_area: int


class DataCollectionStats(BaseModel):
    base_dir: str
    vehicle_count: int
    plate_count: int
    quality_updated_at: Optional[str] = None
    vehicle_quality: Optional[dict] = None
    plate_quality: Optional[dict] = None


class FullSettingsResponse(BaseModel):
    system: SystemInfo
    models: AvailableModels
    config: CurrentConfig
    collection: DataCollectionStats


class UpdateConfigRequest(BaseModel):
    yolo_model: Optional[str] = None
    device: Optional[str] = None
    batch_size: Optional[int] = Field(default=None, ge=1, le=32)
    confidence_threshold: Optional[float] = Field(default=None, ge=0.05, le=0.95)
    nms_threshold: Optional[float] = Field(default=None, ge=0.1, le=0.9)
    plate_merge_iou_threshold: Optional[float] = Field(default=None, ge=0.3, le=0.9)
    face_confidence_threshold: Optional[float] = Field(default=None, ge=0.1, le=0.99)
    plate_confidence_threshold: Optional[float] = Field(default=None, ge=0.05, le=0.95)
    plate_detector_confidence: Optional[float] = Field(default=None, ge=0.05, le=0.95)
    ocr_confidence_threshold: Optional[float] = Field(default=None, ge=0.1, le=0.99)
    ocr_min_text_length: Optional[int] = Field(default=None, ge=3, le=12)
    event_min_vote_count: Optional[int] = Field(default=None, ge=1, le=20)
    event_instant_confidence: Optional[float] = Field(default=None, ge=0.5, le=1.0)
    event_dedup_ttl: Optional[float] = Field(default=None, ge=1.0, le=300.0)
    lpr_consensus_history: Optional[int] = Field(default=None, ge=3, le=50)
    lpr_stream_backlog_threshold: Optional[int] = Field(default=None, ge=10, le=5000)
    lpr_stream_drop_ratio_max: Optional[float] = Field(default=None, ge=0.0, le=0.5)
    snapshot_interval: Optional[int] = Field(default=None, ge=1, le=120)
    enable_face_recognition: Optional[bool] = None
    enable_plate_ocr: Optional[bool] = None
    enable_night_lpr: Optional[bool] = None
    adaptive_confidence_base: Optional[float] = Field(default=None, ge=0.1, le=0.9)
    adaptive_confidence_alpha: Optional[float] = Field(default=None, ge=0.0, le=0.5)
    adaptive_confidence_min: Optional[float] = Field(default=None, ge=0.05, le=0.9)
    adaptive_confidence_max: Optional[float] = Field(default=None, ge=0.1, le=1.0)
    fortress_vehicle_confidence: Optional[float] = Field(default=None, ge=0.05, le=0.9)
    fortress_plate_confidence: Optional[float] = Field(default=None, ge=0.05, le=0.9)
    enable_fortress_lpr: Optional[bool] = None
    lpr_draw_vehicle_bbox: Optional[bool] = None
    lpr_sr_enable: Optional[bool] = None
    lpr_sr_min_height: Optional[int] = Field(default=None, ge=16, le=256)
    lpr_sr_min_width: Optional[int] = Field(default=None, ge=32, le=512)
    lpr_sr_conf_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    enable_lpr_data_collection: Optional[bool] = None
    lpr_collection_dir: Optional[str] = None
    lpr_collect_sample_rate: Optional[float] = Field(default=None, ge=0.01, le=1.0)
    lpr_collect_vehicle: Optional[bool] = None
    lpr_collect_plate: Optional[bool] = None
    lpr_collect_min_conf: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    lpr_collect_max_conf: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    lpr_collect_low_conf_only: Optional[bool] = None
    lpr_collect_quality_filter: Optional[bool] = None
    lpr_collect_min_sharpness: Optional[float] = Field(default=None, ge=0.0, le=200.0)
    lpr_collect_min_brightness: Optional[float] = Field(default=None, ge=0.0, le=255.0)
    lpr_collect_max_brightness: Optional[float] = Field(default=None, ge=0.0, le=255.0)
    lpr_collect_min_vehicle_area: Optional[int] = Field(default=None, ge=100, le=100000)
    lpr_collect_min_plate_area: Optional[int] = Field(default=None, ge=50, le=50000)


class LprTrainRequest(BaseModel):
    note: Optional[str] = None
    data_dir: Optional[str] = None
    epochs: Optional[int] = None
    batch: Optional[int] = None
    lr: Optional[float] = None
    output_dir: Optional[str] = None
    use_stn: Optional[bool] = None
    focal_ctc: Optional[bool] = None
    synthetic: Optional[bool] = None
    synthetic_num_plates: Optional[int] = None
    synthetic_variants: Optional[int] = None
    synthetic_output_dir: Optional[str] = None
    synthetic_ratio: Optional[float] = None
    synthetic_glare_prob: Optional[float] = None
    synthetic_rain_prob: Optional[float] = None
    synthetic_mud_prob: Optional[float] = None
    synthetic_motion_blur_prob: Optional[float] = None
    domain_bridge_enabled: Optional[bool] = None
    domain_bridge_mode: Optional[str] = None
    domain_bridge_strength: Optional[float] = None
    domain_bridge_output_dir: Optional[str] = None
    domain_bridge_max_images: Optional[int] = None
    domain_bridge_lowres_scale: Optional[float] = None
    domain_bridge_jpeg_quality: Optional[int] = None
    tensorrt_build: Optional[bool] = None
    tensorrt_onnx_path: Optional[str] = None
    tensorrt_engine_path: Optional[str] = None
    tensorrt_fp16: Optional[bool] = None
    tensorrt_int8: Optional[bool] = None


class TrainOutputFile(BaseModel):
    path: str
    mtime: str


class CollectionFile(BaseModel):
    path: str
    mtime: str
    size: int
    kind: str


class CollectionListResponse(BaseModel):
    base_dir: str
    total: int
    items: list[CollectionFile]


class LprTrainStatusResponse(BaseModel):
    status: str
    request_path: Optional[str] = None
    requested_at: Optional[str] = None
    note: Optional[str] = None
    collection_dir: Optional[str] = None
    vehicle_count: Optional[int] = None
    plate_count: Optional[int] = None
    status_updated_at: Optional[str] = None
    report_path: Optional[str] = None
    output_dirs: list[str]
    latest_outputs: list[TrainOutputFile]


class DependencyPackage(BaseModel):
    name: str
    import_name: str
    installed: bool
    version: Optional[str] = None
    note: Optional[str] = None


class DependencyDiagnostics(BaseModel):
    python_version: str
    os: str
    prefer_gpu: bool
    nvidia_smi_present: bool
    nvidia_driver_version: Optional[str] = None
    nvidia_cuda_version: Optional[str] = None
    packages: list[DependencyPackage]
    recommendations: list[str]
    install_allowed: bool


class DependencyInstallRequest(BaseModel):
    mode: Optional[str] = "auto"  # auto|gpu|cpu


class DependencyInstallResult(BaseModel):
    executed: bool
    commands: list[str]
    stdout: Optional[str] = None
    stderr: Optional[str] = None


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_gpu_info() -> list[GPUInfo]:
    """Get real GPU information from PyTorch/CUDA"""
    gpus = []

    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        return gpus
    
    # Initialize pynvml if available
    nvml_handle = None
    if PYNVML_AVAILABLE:
        try:
            pynvml.nvmlInit()
            nvml_handle = True
        except Exception as e:
            print(f"Warning: pynvml init failed: {e}")
            nvml_handle = None
    
    try:
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            
            # Memory info
            total_mem = props.total_memory / 1024 / 1024  # MB
            # Get current memory usage
            torch.cuda.set_device(i)
            allocated = torch.cuda.memory_allocated(i) / 1024 / 1024
            reserved = torch.cuda.memory_reserved(i) / 1024 / 1024
            
            # CUDA version
            cuda_ver = torch.version.cuda or "Unknown"
            
            # Enhanced data from pynvml
            temperature = None
            power_usage = None
            power_limit = None
            driver_version = ""
            
            if nvml_handle:
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW to W
                    power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
                    driver_version = pynvml.nvmlSystemGetDriverVersion()
                except Exception as e:
                    print(f"Warning: pynvml read failed for GPU {i}: {e}")
            
            gpus.append(GPUInfo(
                id=i,
                name=props.name,
                vram_total_mb=int(total_mem),
                vram_used_mb=int(reserved),  # Reserved is more accurate for "in use"
                vram_free_mb=int(total_mem - reserved),
                cuda_version=cuda_ver,
                driver_version=driver_version,
                temperature=temperature,
                power_usage=round(power_usage, 1) if power_usage else None,
                power_limit=round(power_limit, 1) if power_limit else None
            ))
    except Exception as e:
        print(f"Error getting GPU info: {e}")
    finally:
        if nvml_handle:
            try:
                pynvml.nvmlShutdown()
            except:
                pass
    
    return gpus


def scan_model_files(weights_path: str) -> AvailableModels:
    """Scan filesystem for available model files with metadata validation"""
    yolo_models = []
    face_models = []
    ocr_models = []
    other_models = []
    
    # Common model extensions
    extensions = ['*.pt', '*.pth', '*.onnx', '*.engine', '*.trt']
    
    for ext in extensions:
        pattern = os.path.join(weights_path, '**', ext)
        for filepath in glob.glob(pattern, recursive=True):
            filename = os.path.basename(filepath)
            size_mb = os.path.getsize(filepath) / 1024 / 1024
            
            model = ModelFile(
                filename=filename,
                path=filepath,
                size_mb=round(size_mb, 2),
                type="other"
            )
            
            # Categorize by filename only — DO NOT load models into GPU for validation
            # Loading every .pt via YOLO() causes OOM on GET /settings
            lower_name = filename.lower()
            if 'yolo' in lower_name or 'v8' in lower_name or 'v11' in lower_name:
                model.type = "yolo"
                yolo_models.append(model)
            elif 'face' in lower_name or 'buffalo' in lower_name or 'arcface' in lower_name:
                model.type = "face"
                face_models.append(model)
            elif 'ocr' in lower_name or 'paddle' in lower_name or 'plate' in lower_name:
                model.type = "ocr"
                ocr_models.append(model)
            else:
                other_models.append(model)
    
    # Also check for YOLO models that will be auto-downloaded
    builtin_yolo = [
        # YOLOv8 Detection
        "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt",
        # YOLOv8 Segmentation
        "yolov8n-seg.pt", "yolov8s-seg.pt", "yolov8m-seg.pt",
        # YOLOv11 Detection (latest)
        "yolov11n.pt", "yolov11s.pt", "yolov11m.pt", "yolov11l.pt", "yolov11x.pt",
        # YOLOv11 Segmentation
        "yolov11n-seg.pt", "yolov11s-seg.pt", "yolov11m-seg.pt",
    ]
    
    existing_names = [m.filename for m in yolo_models]
    for model_name in builtin_yolo:
        if model_name not in existing_names:
            yolo_models.append(ModelFile(
                filename=model_name,
                path=f"(auto-download) {model_name}",
                size_mb=0,
                type="yolo"
            ))
    
    return AvailableModels(
        yolo_models=yolo_models,
        face_models=face_models,
        ocr_models=ocr_models,
        other_models=other_models
    )


def get_collection_stats(base_dir: str) -> DataCollectionStats:
    """Get image counts for data collection."""
    vehicle_dir = os.path.join(base_dir, "vehicles")
    plate_dir = os.path.join(base_dir, "plates")
    stats_path = os.path.join(base_dir, "quality_stats.json")

    def count_images(path: str) -> int:
        if not os.path.exists(path):
            return 0
        patterns = ["*.jpg", "*.jpeg", "*.png", "*.webp"]
        count = 0
        for pattern in patterns:
            count += len(glob.glob(os.path.join(path, pattern)))
        return count

    quality_data = _read_json(stats_path) if os.path.exists(stats_path) else None

    return DataCollectionStats(
        base_dir=base_dir,
        vehicle_count=count_images(vehicle_dir),
        plate_count=count_images(plate_dir),
        quality_updated_at=quality_data.get("updated_at") if quality_data else None,
        vehicle_quality=quality_data.get("vehicle") if quality_data else None,
        plate_quality=quality_data.get("plate") if quality_data else None,
    )


def _read_json(path: str) -> Optional[dict]:
    try:
        if not path or not os.path.exists(path):
            return None
        import json
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _scan_output_dirs(dirs: list[str]) -> list[TrainOutputFile]:
    candidates = []
    patterns = ["*.pt", "*.pth", "*.onnx", "*.engine", "*.trt"]
    for base in dirs:
        if not base or not os.path.exists(base):
            continue
        for pattern in patterns:
            for filepath in glob.glob(os.path.join(base, "**", pattern), recursive=True):
                try:
                    mtime = datetime.utcfromtimestamp(os.path.getmtime(filepath)).isoformat() + "Z"
                    candidates.append(TrainOutputFile(path=filepath, mtime=mtime))
                except Exception:
                    continue
    candidates.sort(key=lambda x: x.mtime, reverse=True)
    return candidates[:10]


def _list_collection_files(base_dir: str, kind: str, limit: int, offset: int) -> CollectionListResponse:
    target_dirs = []
    if kind in ("plates", "all"):
        target_dirs.append(("plates", os.path.join(base_dir, "plates")))
    if kind in ("vehicles", "all"):
        target_dirs.append(("vehicles", os.path.join(base_dir, "vehicles")))

    files: list[CollectionFile] = []
    for label, folder in target_dirs:
        if not os.path.exists(folder):
            continue
        for pattern in ["*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"]:
            for filepath in glob.glob(os.path.join(folder, pattern)):
                try:
                    mtime = datetime.utcfromtimestamp(os.path.getmtime(filepath)).isoformat() + "Z"
                    size = os.path.getsize(filepath)
                    files.append(CollectionFile(path=filepath, mtime=mtime, size=size, kind=label))
                except Exception:
                    continue

    files.sort(key=lambda x: x.mtime, reverse=True)
    total = len(files)
    items = files[offset:offset + limit]

    return CollectionListResponse(base_dir=base_dir, total=total, items=items)


def _check_package(name: str, import_name: Optional[str] = None) -> DependencyPackage:
    target = import_name or name
    installed = importlib.util.find_spec(target) is not None
    version = None
    note = None

    if installed:
        try:
            version = metadata.version(name)
        except Exception:
            try:
                version = metadata.version(target)
            except Exception:
                version = None

    # Extra notes for certain packages
    if installed and target == "onnxruntime":
        try:
            import onnxruntime as ort
            note = f"device={ort.get_device()}"
        except Exception:
            note = None
    if installed and target == "paddle":
        try:
            import paddle
            note = f"cuda={paddle.is_compiled_with_cuda()}"
        except Exception:
            note = None

    return DependencyPackage(
        name=name,
        import_name=target,
        installed=installed,
        version=version,
        note=note
    )


def _get_nvidia_smi_info() -> dict:
    info = {
        "present": False,
        "driver_version": None,
        "cuda_version": None
    }
    if not shutil.which("nvidia-smi"):
        return info

    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode != 0:
            return info

        output = result.stdout or ""
        info["present"] = True

        driver_match = re.search(r"Driver Version:\s*([0-9.]+)", output)
        cuda_match = re.search(r"CUDA Version:\s*([0-9.]+)", output)
        if driver_match:
            info["driver_version"] = driver_match.group(1)
        if cuda_match:
            info["cuda_version"] = cuda_match.group(1)
    except Exception:
        return info

    return info


def _torch_index_url(cuda_version: Optional[str]) -> str:
    if not cuda_version:
        return "https://download.pytorch.org/whl/cu126"
    major_minor = ".".join(cuda_version.split(".")[:2])
    mapping = {
        "13.0": "cu130",
        "12.8": "cu128",
        "12.6": "cu126",
        "12.4": "cu124",
        "12.3": "cu123",
        "12.2": "cu122",
        "12.1": "cu121",
        "12.0": "cu120",
        "11.8": "cu118"
    }
    tag = mapping.get(major_minor, "cu126")
    return f"https://download.pytorch.org/whl/{tag}"


def _build_recommendations(prefer_gpu: bool, cuda_version: Optional[str]) -> list[str]:
    commands: list[str] = []
    if prefer_gpu:
        torch_index = _torch_index_url(cuda_version)
        commands.append(f"python -m pip install torch torchvision --index-url {torch_index}")
        commands.append("python -m pip install onnxruntime-gpu==1.16.3")
        commands.append("python -m pip install paddlepaddle-gpu==2.6.2")
    else:
        commands.append("python -m pip install torch torchvision")
        commands.append("python -m pip install onnxruntime==1.16.3")
        commands.append("python -m pip install paddlepaddle==2.6.2")

    commands.append("python -m pip install ultralytics==8.3.0 paddleocr==2.9.1")
    commands.append("python -m pip install insightface==0.7.3 nvidia-ml-py3==7.352.0")
    return commands


def _build_install_commands(prefer_gpu: bool, cuda_version: Optional[str]) -> list[list[str]]:
    commands: list[list[str]] = []
    if prefer_gpu:
        torch_index = _torch_index_url(cuda_version)
        commands.append([sys.executable, "-m", "pip", "install", "torch", "torchvision", "--index-url", torch_index])
        commands.append([sys.executable, "-m", "pip", "install", "onnxruntime-gpu==1.16.3"])
        commands.append([sys.executable, "-m", "pip", "install", "paddlepaddle-gpu==2.6.2"])
    else:
        commands.append([sys.executable, "-m", "pip", "install", "torch", "torchvision"])
        commands.append([sys.executable, "-m", "pip", "install", "onnxruntime==1.16.3"])
        commands.append([sys.executable, "-m", "pip", "install", "paddlepaddle==2.6.2"])

    commands.append([sys.executable, "-m", "pip", "install", "ultralytics==8.3.0", "paddleocr==2.9.1"])
    commands.append([sys.executable, "-m", "pip", "install", "insightface==0.7.3", "nvidia-ml-py3==7.352.0"])
    return commands


# ============================================================
# API ENDPOINTS
# ============================================================

@router.get("/vehicle/settings", response_model=FullSettingsResponse)
async def get_full_settings():
    """Get complete system info, available models, and current configuration"""
    settings = get_settings()
    
    # GPU Info
    gpus = get_gpu_info()
    cuda_available = False
    cuda_version = None
    gpu_count = 0
    torch_version = "not installed"

    if TORCH_AVAILABLE:
        cuda_available = torch.cuda.is_available()
        cuda_version = torch.version.cuda
        gpu_count = torch.cuda.device_count() if cuda_available else 0
        torch_version = torch.__version__

    system = SystemInfo(
        cuda_available=cuda_available,
        cuda_version=cuda_version,
        gpu_count=gpu_count,
        gpus=gpus,
        torch_version=torch_version,
        device_in_use=settings.device
    )
    
    # Available Models
    models = scan_model_files(settings.weights_path)
    
    # Current Config
    config = CurrentConfig(
        yolo_model=settings.yolo_model,
        device=settings.device,
        batch_size=settings.batch_size,
        confidence_threshold=settings.confidence_threshold,
        nms_threshold=settings.nms_threshold,
        plate_merge_iou_threshold=settings.plate_merge_iou_threshold,
        face_confidence_threshold=settings.face_confidence_threshold,
        plate_confidence_threshold=settings.plate_confidence_threshold,
        plate_detector_confidence=settings.plate_detector_confidence,
        ocr_confidence_threshold=settings.ocr_confidence_threshold,
        ocr_min_text_length=settings.ocr_min_text_length,
        event_min_vote_count=settings.event_min_vote_count,
        event_instant_confidence=settings.event_instant_confidence,
        event_dedup_ttl=settings.event_dedup_ttl,
        lpr_consensus_history=settings.lpr_consensus_history,
        lpr_stream_backlog_threshold=settings.lpr_stream_backlog_threshold,
        lpr_stream_drop_ratio_max=settings.lpr_stream_drop_ratio_max,
        snapshot_interval=settings.snapshot_interval,
        enable_face_recognition=settings.enable_face_recognition,
        enable_plate_ocr=settings.enable_plate_ocr,
        enable_night_lpr=settings.enable_night_lpr,
        adaptive_confidence_base=settings.adaptive_confidence_base,
        adaptive_confidence_alpha=settings.adaptive_confidence_alpha,
        adaptive_confidence_min=settings.adaptive_confidence_min,
        adaptive_confidence_max=settings.adaptive_confidence_max,
        fortress_vehicle_confidence=settings.fortress_vehicle_confidence,
        fortress_plate_confidence=settings.fortress_plate_confidence,
        enable_fortress_lpr=settings.enable_fortress_lpr,
        lpr_draw_vehicle_bbox=settings.lpr_draw_vehicle_bbox,
        lpr_sr_enable=settings.lpr_sr_enable,
        lpr_sr_min_height=settings.lpr_sr_min_height,
        lpr_sr_min_width=settings.lpr_sr_min_width,
        lpr_sr_conf_threshold=settings.lpr_sr_conf_threshold,
        enable_lpr_data_collection=settings.enable_lpr_data_collection,
        lpr_collection_dir=settings.lpr_collection_dir,
        lpr_collect_sample_rate=settings.lpr_collect_sample_rate,
        lpr_collect_vehicle=settings.lpr_collect_vehicle,
        lpr_collect_plate=settings.lpr_collect_plate,
        lpr_collect_min_conf=settings.lpr_collect_min_conf,
        lpr_collect_max_conf=settings.lpr_collect_max_conf,
        lpr_collect_low_conf_only=settings.lpr_collect_low_conf_only,
        lpr_collect_quality_filter=settings.lpr_collect_quality_filter,
        lpr_collect_min_sharpness=settings.lpr_collect_min_sharpness,
        lpr_collect_min_brightness=settings.lpr_collect_min_brightness,
        lpr_collect_max_brightness=settings.lpr_collect_max_brightness,
        lpr_collect_min_vehicle_area=settings.lpr_collect_min_vehicle_area,
        lpr_collect_min_plate_area=settings.lpr_collect_min_plate_area,
    )
    
    return FullSettingsResponse(
        system=system,
        models=models,
        config=config,
        collection=get_collection_stats(settings.lpr_collection_dir)
    )


@router.get("/vehicle/settings/meta")
async def get_settings_meta():
    """Get field descriptions, types, validation ranges, and tiers for UI rendering.
    Returns: {field_name: {description, tier, type, min, max, value, default}}
    """
    return get_settings().get_field_descriptions()


_settings_lock = asyncio.Lock()


def _require_superadmin(request: Request) -> None:
    role_header = (request.headers.get("X-User-Role") or "").strip().lower()
    if role_header != "superadmin":
        raise HTTPException(status_code=403, detail="SuperAdmin role required to update LPR settings")

@router.post("/vehicle/settings", response_model=CurrentConfig)
async def update_settings(request: UpdateConfigRequest, http_request: Request):
    """Update runtime configuration"""
    _require_superadmin(http_request)
    async with _settings_lock:
        settings = get_settings()
    
        # Apply updates (runtime only, won't persist across restarts)
        if request.yolo_model is not None:
            settings.yolo_model = request.yolo_model
        if request.device is not None:
            settings.device = request.device
        if request.batch_size is not None:
            settings.batch_size = request.batch_size
        if request.confidence_threshold is not None:
            settings.confidence_threshold = request.confidence_threshold
        if request.nms_threshold is not None:
            settings.nms_threshold = request.nms_threshold
        if request.plate_merge_iou_threshold is not None:
            settings.plate_merge_iou_threshold = request.plate_merge_iou_threshold
        if request.face_confidence_threshold is not None:
            settings.face_confidence_threshold = request.face_confidence_threshold
        if request.plate_confidence_threshold is not None:
            settings.plate_confidence_threshold = request.plate_confidence_threshold
            # Sync alias: plate_detector_confidence is what the pipeline actually uses
            settings.plate_detector_confidence = request.plate_confidence_threshold
        if request.plate_detector_confidence is not None:
            settings.plate_detector_confidence = request.plate_detector_confidence
            settings.plate_confidence_threshold = request.plate_detector_confidence
        if request.ocr_confidence_threshold is not None:
            settings.ocr_confidence_threshold = request.ocr_confidence_threshold
        if request.ocr_min_text_length is not None:
            settings.ocr_min_text_length = request.ocr_min_text_length
        if request.event_min_vote_count is not None:
            settings.event_min_vote_count = request.event_min_vote_count
        if request.event_instant_confidence is not None:
            settings.event_instant_confidence = request.event_instant_confidence
        if request.event_dedup_ttl is not None:
            settings.event_dedup_ttl = request.event_dedup_ttl
        if request.lpr_consensus_history is not None:
            settings.lpr_consensus_history = request.lpr_consensus_history
        if request.lpr_stream_backlog_threshold is not None:
            settings.lpr_stream_backlog_threshold = request.lpr_stream_backlog_threshold
        if request.lpr_stream_drop_ratio_max is not None:
            settings.lpr_stream_drop_ratio_max = request.lpr_stream_drop_ratio_max
        if request.snapshot_interval is not None:
            settings.snapshot_interval = request.snapshot_interval
        if request.enable_face_recognition is not None:
            settings.enable_face_recognition = request.enable_face_recognition
        if request.enable_plate_ocr is not None:
            settings.enable_plate_ocr = request.enable_plate_ocr
        if request.enable_night_lpr is not None:
            settings.enable_night_lpr = request.enable_night_lpr
        if request.adaptive_confidence_base is not None:
            settings.adaptive_confidence_base = request.adaptive_confidence_base
        if request.adaptive_confidence_alpha is not None:
            settings.adaptive_confidence_alpha = request.adaptive_confidence_alpha
        if request.adaptive_confidence_min is not None:
            settings.adaptive_confidence_min = request.adaptive_confidence_min
        if request.adaptive_confidence_max is not None:
            settings.adaptive_confidence_max = request.adaptive_confidence_max
        if request.fortress_vehicle_confidence is not None:
            settings.fortress_vehicle_confidence = request.fortress_vehicle_confidence
        if request.fortress_plate_confidence is not None:
            settings.fortress_plate_confidence = request.fortress_plate_confidence
        if request.enable_fortress_lpr is not None:
            settings.enable_fortress_lpr = request.enable_fortress_lpr
        if request.lpr_draw_vehicle_bbox is not None:
            settings.lpr_draw_vehicle_bbox = request.lpr_draw_vehicle_bbox
        if request.lpr_sr_enable is not None:
            settings.lpr_sr_enable = request.lpr_sr_enable
        if request.lpr_sr_min_height is not None:
            settings.lpr_sr_min_height = request.lpr_sr_min_height
        if request.lpr_sr_min_width is not None:
            settings.lpr_sr_min_width = request.lpr_sr_min_width
        if request.lpr_sr_conf_threshold is not None:
            settings.lpr_sr_conf_threshold = request.lpr_sr_conf_threshold
        if request.enable_lpr_data_collection is not None:
            settings.enable_lpr_data_collection = request.enable_lpr_data_collection
        if request.lpr_collection_dir is not None:
            requested = os.path.realpath(request.lpr_collection_dir)
            allowed_base = os.path.realpath(
                getattr(settings, "data_root", os.environ.get("DATA_ROOT", "/data"))
            )
            if not requested.startswith(allowed_base + os.sep):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid collection directory: path must be within DATA_ROOT ({allowed_base})"
                )
            settings.lpr_collection_dir = requested
        if request.lpr_collect_sample_rate is not None:
            settings.lpr_collect_sample_rate = request.lpr_collect_sample_rate
        if request.lpr_collect_vehicle is not None:
            settings.lpr_collect_vehicle = request.lpr_collect_vehicle
        if request.lpr_collect_plate is not None:
            settings.lpr_collect_plate = request.lpr_collect_plate
        if request.lpr_collect_min_conf is not None:
            settings.lpr_collect_min_conf = request.lpr_collect_min_conf
        if request.lpr_collect_max_conf is not None:
            settings.lpr_collect_max_conf = request.lpr_collect_max_conf
        if request.lpr_collect_low_conf_only is not None:
            settings.lpr_collect_low_conf_only = request.lpr_collect_low_conf_only
        if request.lpr_collect_quality_filter is not None:
            settings.lpr_collect_quality_filter = request.lpr_collect_quality_filter
        if request.lpr_collect_min_sharpness is not None:
            settings.lpr_collect_min_sharpness = request.lpr_collect_min_sharpness
        if request.lpr_collect_min_brightness is not None:
            settings.lpr_collect_min_brightness = request.lpr_collect_min_brightness
        if request.lpr_collect_max_brightness is not None:
            settings.lpr_collect_max_brightness = request.lpr_collect_max_brightness
        if request.lpr_collect_min_vehicle_area is not None:
            settings.lpr_collect_min_vehicle_area = request.lpr_collect_min_vehicle_area
        if request.lpr_collect_min_plate_area is not None:
            settings.lpr_collect_min_plate_area = request.lpr_collect_min_plate_area
    
        # Persist settings to file
        settings.save_to_file()
    
    # Return updated config
    return CurrentConfig(
        yolo_model=settings.yolo_model,
        device=settings.device,
        batch_size=settings.batch_size,
        confidence_threshold=settings.confidence_threshold,
        nms_threshold=settings.nms_threshold,
        plate_merge_iou_threshold=settings.plate_merge_iou_threshold,
        face_confidence_threshold=settings.face_confidence_threshold,
        plate_confidence_threshold=settings.plate_confidence_threshold,
        plate_detector_confidence=settings.plate_detector_confidence,
        ocr_confidence_threshold=settings.ocr_confidence_threshold,
        ocr_min_text_length=settings.ocr_min_text_length,
        event_min_vote_count=settings.event_min_vote_count,
        event_instant_confidence=settings.event_instant_confidence,
        event_dedup_ttl=settings.event_dedup_ttl,
        lpr_consensus_history=settings.lpr_consensus_history,
        lpr_stream_backlog_threshold=settings.lpr_stream_backlog_threshold,
        lpr_stream_drop_ratio_max=settings.lpr_stream_drop_ratio_max,
        snapshot_interval=settings.snapshot_interval,
        enable_face_recognition=settings.enable_face_recognition,
        enable_plate_ocr=settings.enable_plate_ocr,
        enable_night_lpr=settings.enable_night_lpr,
        adaptive_confidence_base=settings.adaptive_confidence_base,
        adaptive_confidence_alpha=settings.adaptive_confidence_alpha,
        adaptive_confidence_min=settings.adaptive_confidence_min,
        adaptive_confidence_max=settings.adaptive_confidence_max,
        fortress_vehicle_confidence=settings.fortress_vehicle_confidence,
        fortress_plate_confidence=settings.fortress_plate_confidence,
        enable_fortress_lpr=settings.enable_fortress_lpr,
        lpr_draw_vehicle_bbox=settings.lpr_draw_vehicle_bbox,
        lpr_sr_enable=settings.lpr_sr_enable,
        lpr_sr_min_height=settings.lpr_sr_min_height,
        lpr_sr_min_width=settings.lpr_sr_min_width,
        lpr_sr_conf_threshold=settings.lpr_sr_conf_threshold,
        enable_lpr_data_collection=settings.enable_lpr_data_collection,
        lpr_collection_dir=settings.lpr_collection_dir,
        lpr_collect_sample_rate=settings.lpr_collect_sample_rate,
        lpr_collect_vehicle=settings.lpr_collect_vehicle,
        lpr_collect_plate=settings.lpr_collect_plate,
        lpr_collect_min_conf=settings.lpr_collect_min_conf,
        lpr_collect_max_conf=settings.lpr_collect_max_conf,
        lpr_collect_low_conf_only=settings.lpr_collect_low_conf_only,
        lpr_collect_quality_filter=settings.lpr_collect_quality_filter,
        lpr_collect_min_sharpness=settings.lpr_collect_min_sharpness,
        lpr_collect_min_brightness=settings.lpr_collect_min_brightness,
        lpr_collect_max_brightness=settings.lpr_collect_max_brightness,
        lpr_collect_min_vehicle_area=settings.lpr_collect_min_vehicle_area,
        lpr_collect_min_plate_area=settings.lpr_collect_min_plate_area,
    )


@router.post("/vehicle/train")
async def request_lpr_training(request: LprTrainRequest):
    """Create a training request file for superadmin workflows."""
    settings = get_settings()
    stats = get_collection_stats(settings.lpr_collection_dir)

    os.makedirs(settings.lpr_collection_dir, exist_ok=True)
    payload = {
        "requested_at": datetime.utcnow().isoformat() + "Z",
        "note": request.note,
        "collection_dir": settings.lpr_collection_dir,
        "vehicle_count": stats.vehicle_count,
        "plate_count": stats.plate_count,
        "data_dir": request.data_dir,
        "epochs": request.epochs,
        "batch": request.batch,
        "lr": request.lr,
        "output_dir": request.output_dir,
        "use_stn": request.use_stn,
        "focal_ctc": request.focal_ctc,
        "synthetic": request.synthetic,
        "synthetic_num_plates": request.synthetic_num_plates,
        "synthetic_variants": request.synthetic_variants,
        "synthetic_output_dir": request.synthetic_output_dir,
        "synthetic_ratio": request.synthetic_ratio,
        "synthetic_glare_prob": request.synthetic_glare_prob,
        "synthetic_rain_prob": request.synthetic_rain_prob,
        "synthetic_mud_prob": request.synthetic_mud_prob,
        "synthetic_motion_blur_prob": request.synthetic_motion_blur_prob,
        "domain_bridge_enabled": request.domain_bridge_enabled,
        "domain_bridge_mode": request.domain_bridge_mode,
        "domain_bridge_strength": request.domain_bridge_strength,
        "domain_bridge_output_dir": request.domain_bridge_output_dir,
        "domain_bridge_max_images": request.domain_bridge_max_images,
        "domain_bridge_lowres_scale": request.domain_bridge_lowres_scale,
        "domain_bridge_jpeg_quality": request.domain_bridge_jpeg_quality,
        "tensorrt_build": request.tensorrt_build,
        "tensorrt_onnx_path": request.tensorrt_onnx_path,
        "tensorrt_engine_path": request.tensorrt_engine_path,
        "tensorrt_fp16": request.tensorrt_fp16,
        "tensorrt_int8": request.tensorrt_int8,
    }

    request_path = os.path.join(settings.lpr_collection_dir, "train_request.json")
    with open(request_path, "w", encoding="utf-8") as f:
        import json
        json.dump(payload, f, indent=2)

    status_path = os.path.join(settings.lpr_collection_dir, "train_status.json")
    with open(status_path, "w", encoding="utf-8") as f:
        import json
        json.dump({
            "status": "queued",
            "updated_at": datetime.utcnow().isoformat() + "Z",
            "request_path": request_path
        }, f, indent=2)

    return {
        "status": "queued",
        "request_path": request_path,
        "stats": payload
    }


@router.get("/vehicle/train/status", response_model=LprTrainStatusResponse)
async def get_lpr_training_status():
    settings = get_settings()
    collection_dir = settings.lpr_collection_dir

    request_path = os.path.join(collection_dir, "train_request.json")
    status_path = os.path.join(collection_dir, "train_status.json")
    report_path = os.path.join(collection_dir, "training_session_report.json")

    request_data = _read_json(request_path) if os.path.exists(request_path) else None
    status_data = _read_json(status_path) if os.path.exists(status_path) else None

    # Fallback report in repo (legacy)
    legacy_report_path = "/app/training_session_report.json"
    if not os.path.exists(report_path) and os.path.exists(legacy_report_path):
        report_path = legacy_report_path

    # Determine status
    status = "none"
    if status_data and status_data.get("status"):
        status = status_data.get("status")
    elif request_data:
        status = "queued"
    if report_path and os.path.exists(report_path):
        status = "completed"

    output_dirs = list(dict.fromkeys([
        settings.weights_dir,
        settings.weights_path,
        "/app/weights",
        "/app/runs",
        collection_dir
    ]))

    latest_outputs = _scan_output_dirs(output_dirs)

    return LprTrainStatusResponse(
        status=status,
        request_path=request_path if os.path.exists(request_path) else None,
        requested_at=request_data.get("requested_at") if request_data else None,
        note=request_data.get("note") if request_data else None,
        collection_dir=request_data.get("collection_dir") if request_data else collection_dir,
        vehicle_count=request_data.get("vehicle_count") if request_data else None,
        plate_count=request_data.get("plate_count") if request_data else None,
        status_updated_at=status_data.get("updated_at") if status_data else None,
        report_path=report_path if report_path and os.path.exists(report_path) else None,
        output_dirs=output_dirs,
        latest_outputs=latest_outputs
    )


@router.put("/vehicle/train/request")
async def update_lpr_training_request(request: LprTrainRequest):
    """Update existing training request (merge with current request file)."""
    settings = get_settings()
    collection_dir = settings.lpr_collection_dir
    request_path = os.path.join(collection_dir, "train_request.json")
    status_path = os.path.join(collection_dir, "train_status.json")

    current = _read_json(request_path) if os.path.exists(request_path) else {}
    payload = {
        **(current or {}),
        "note": request.note if request.note is not None else (current or {}).get("note"),
        "data_dir": request.data_dir if request.data_dir is not None else (current or {}).get("data_dir"),
        "epochs": request.epochs if request.epochs is not None else (current or {}).get("epochs"),
        "command": request.command if request.command is not None else (current or {}).get("command"),
        "batch": request.batch if request.batch is not None else (current or {}).get("batch"),
        "lr": request.lr if request.lr is not None else (current or {}).get("lr"),
        "output_dir": request.output_dir if request.output_dir is not None else (current or {}).get("output_dir"),
        "use_stn": request.use_stn if request.use_stn is not None else (current or {}).get("use_stn"),
        "focal_ctc": request.focal_ctc if request.focal_ctc is not None else (current or {}).get("focal_ctc"),
        "synthetic": request.synthetic if request.synthetic is not None else (current or {}).get("synthetic"),
        "synthetic_num_plates": request.synthetic_num_plates if request.synthetic_num_plates is not None else (current or {}).get("synthetic_num_plates"),
        "synthetic_variants": request.synthetic_variants if request.synthetic_variants is not None else (current or {}).get("synthetic_variants"),
        "synthetic_output_dir": request.synthetic_output_dir if request.synthetic_output_dir is not None else (current or {}).get("synthetic_output_dir"),
        "synthetic_ratio": request.synthetic_ratio if request.synthetic_ratio is not None else (current or {}).get("synthetic_ratio"),
        "tensorrt_build": request.tensorrt_build if request.tensorrt_build is not None else (current or {}).get("tensorrt_build"),
        "tensorrt_onnx_path": request.tensorrt_onnx_path if request.tensorrt_onnx_path is not None else (current or {}).get("tensorrt_onnx_path"),
        "tensorrt_engine_path": request.tensorrt_engine_path if request.tensorrt_engine_path is not None else (current or {}).get("tensorrt_engine_path"),
        "tensorrt_fp16": request.tensorrt_fp16 if request.tensorrt_fp16 is not None else (current or {}).get("tensorrt_fp16"),
        "tensorrt_int8": request.tensorrt_int8 if request.tensorrt_int8 is not None else (current or {}).get("tensorrt_int8"),
    }

    payload["updated_at"] = datetime.utcnow().isoformat() + "Z"
    if "requested_at" not in payload:
        payload["requested_at"] = datetime.utcnow().isoformat() + "Z"
    payload["collection_dir"] = collection_dir

    with open(request_path, "w", encoding="utf-8") as f:
        import json
        json.dump(payload, f, indent=2)

    with open(status_path, "w", encoding="utf-8") as f:
        import json
        json.dump({
            "status": "queued",
            "updated_at": datetime.utcnow().isoformat() + "Z",
            "request_path": request_path
        }, f, indent=2)

    return {"status": "queued", "request_path": request_path, "request": payload}


@router.get("/vehicle/collect/list", response_model=CollectionListResponse)
async def list_collected_files(kind: str = Query("all", pattern="^(plates|vehicles|all)$"),
                               limit: int = Query(50, ge=1, le=500),
                               offset: int = Query(0, ge=0)):
    settings = get_settings()
    return _list_collection_files(settings.lpr_collection_dir, kind, limit, offset)


@router.delete("/vehicle/collect/clear")
async def clear_collected_files(kind: str = Query("all", pattern="^(plates|vehicles|all)$")):
    settings = get_settings()
    base_dir = settings.lpr_collection_dir
    targets = []
    if kind in ("plates", "all"):
        targets.append(os.path.join(base_dir, "plates"))
    if kind in ("vehicles", "all"):
        targets.append(os.path.join(base_dir, "vehicles"))

    deleted = 0
    for folder in targets:
        if not os.path.exists(folder):
            continue
        for pattern in ["*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"]:
            for filepath in glob.glob(os.path.join(folder, pattern)):
                try:
                    os.remove(filepath)
                    deleted += 1
                except Exception:
                    continue

    # Optional stats reset when clearing all
    if kind == "all":
        stats_path = os.path.join(base_dir, "quality_stats.json")
        if os.path.exists(stats_path):
            try:
                os.remove(stats_path)
            except Exception:
                pass

    return {"deleted": deleted, "kind": kind}


@router.get("/vehicle/train/output")
async def download_lpr_training_output(path: str = Query(..., min_length=1)):
    settings = get_settings()
    allowed_dirs = [
        settings.weights_dir,
        settings.weights_path,
        "/app/weights",
        "/app/runs",
        settings.lpr_collection_dir,
        "/app/data/collect",
    ]

    try:
        target = os.path.realpath(path)
        if not os.path.exists(target) or not os.path.isfile(target):
            raise HTTPException(status_code=404, detail="File not found")

        allowed = False
        for base in allowed_dirs:
            if not base:
                continue
            base_abs = os.path.realpath(base)
            try:
                if os.path.commonpath([target, base_abs]) == base_abs:
                    allowed = True
                    break
            except Exception:
                continue

        if not allowed:
            raise HTTPException(status_code=403, detail="Access denied")

        return FileResponse(path=target, filename=os.path.basename(target))
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid path")


@router.get("/vehicle/settings/deps/diagnostics", response_model=DependencyDiagnostics)
async def get_dependency_diagnostics():
    nvidia_info = await asyncio.get_running_loop().run_in_executor(None, _get_nvidia_smi_info)

    prefer_gpu = bool(nvidia_info.get("present"))
    if TORCH_AVAILABLE:
        try:
            prefer_gpu = prefer_gpu or torch.cuda.is_available()
        except Exception:
            pass

    packages = [
        _check_package("torch"),
        _check_package("torchvision"),
        _check_package("ultralytics"),
        _check_package("paddleocr", "paddleocr"),
        _check_package("paddlepaddle-gpu", "paddle"),
        _check_package("onnxruntime", "onnxruntime"),
        _check_package("insightface"),
        _check_package("nvidia-ml-py3", "pynvml")
    ]

    recommendations = _build_recommendations(prefer_gpu, nvidia_info.get("cuda_version"))
    install_allowed = os.getenv("AI_ENGINE_ALLOW_PIP_INSTALL", "0") == "1"

    return DependencyDiagnostics(
        python_version=sys.version.split(" ")[0],
        os=f"{platform.system()} {platform.release()}",
        prefer_gpu=prefer_gpu,
        nvidia_smi_present=bool(nvidia_info.get("present")),
        nvidia_driver_version=nvidia_info.get("driver_version"),
        nvidia_cuda_version=nvidia_info.get("cuda_version"),
        packages=packages,
        recommendations=recommendations,
        install_allowed=install_allowed
    )


@router.post("/vehicle/settings/deps/install", response_model=DependencyInstallResult)
async def install_dependencies(payload: DependencyInstallRequest):
    install_allowed = os.getenv("AI_ENGINE_ALLOW_PIP_INSTALL", "0") == "1"
    if not install_allowed:
        raise HTTPException(status_code=403, detail="Auto-install disabled. Set AI_ENGINE_ALLOW_PIP_INSTALL=1")

    nvidia_info = await asyncio.get_running_loop().run_in_executor(None, _get_nvidia_smi_info)
    prefer_gpu = bool(nvidia_info.get("present"))
    if payload.mode == "cpu":
        prefer_gpu = False
    elif payload.mode == "gpu":
        prefer_gpu = True

    recommendations = _build_recommendations(prefer_gpu, nvidia_info.get("cuda_version"))
    install_commands = _build_install_commands(prefer_gpu, nvidia_info.get("cuda_version"))

    def _run_install():
        stdout_chunks = []
        stderr_chunks = []
        for command in install_commands:
            try:
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=900
                )
                stdout_chunks.append(result.stdout or "")
                stderr_chunks.append(result.stderr or "")
            except Exception as e:
                stderr_chunks.append(str(e))
        return stdout_chunks, stderr_chunks

    stdout_chunks, stderr_chunks = await asyncio.get_running_loop().run_in_executor(None, _run_install)

    return DependencyInstallResult(
        executed=True,
        commands=recommendations,
        stdout="\n".join(stdout_chunks).strip() or None,
        stderr="\n".join(stderr_chunks).strip() or None
    )


# Keep old endpoint for backward compatibility
@router.get("/vehicle/config")
async def get_config_legacy():
    """Legacy endpoint - redirects to new settings"""
    full = await get_full_settings()
    return {
        "device": full.config.device,
        "yolo_model": full.config.yolo_model,
        "snapshot_interval": full.config.snapshot_interval,
        "confidence_threshold": full.config.confidence_threshold,
        "enable_plate_ocr": full.config.enable_plate_ocr,
        "enable_face_recognition": full.config.enable_face_recognition,
        "enable_night_lpr": full.config.enable_night_lpr,
        "gpu_info": [
            {
                "id": g.id,
                "name": g.name,
                "vram_total": g.vram_total_mb,
                "vram_used": g.vram_used_mb,
                "vram_free": g.vram_free_mb
            } for g in full.system.gpus
        ] if full.system.gpus else None
    }


@router.get("/vehicle/ambient")
async def get_ambient_status():
    """Get per-camera ambient brightness / day-night status.

    Returns a dict keyed by camera_id with brightness_ema,
    ambient_ratio (0=night, 1=day), and seconds since last update.
    """
    try:
        from app.services.core.ambient_adapter import AmbientAdapter
        adapter = AmbientAdapter.get_instance()
        cameras = adapter.get_all_states()
        return {
            "count": len(cameras),
            "cameras": cameras,
        }
    except Exception as exc:
        return {"count": 0, "cameras": {}, "error": str(exc)}
