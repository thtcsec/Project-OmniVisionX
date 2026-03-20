
# omni-human: face recognition service (FastAPI)

import os
import logging
import io
import time
import asyncio
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from pydantic import BaseModel
import numpy as np
from PIL import Image

# Import new services
from app.services.detectors import FaceDetector, HumanDetector
from app.services.recognizer import FaceRecognizer
from app.config import get_settings
from app.workers.stream_consumer import HumanStreamConsumer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("omni-human")

# Global models
face_detector: Optional[FaceDetector] = None
human_detector: Optional[HumanDetector] = None
face_recognizer: Optional[FaceRecognizer] = None
_stream_consumer: Optional[HumanStreamConsumer] = None

# Constants
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# Data paths — lazy initialization to prevent module-level Pydantic crash
# If env vars are malformed, get_settings() throws ValidationError at import time,
# killing the entire service (502) before any routes can register.
_settings_boot = None
DATA_ROOT = None
WEIGHTS_DIR = None
INSIGHT_ROOT = None

def _ensure_boot_settings():
    """Initialize data paths lazily on first use."""
    global _settings_boot, DATA_ROOT, WEIGHTS_DIR, INSIGHT_ROOT
    if _settings_boot is not None:
        return
    try:
        _settings_boot = get_settings()
    except Exception as e:
        logger.error("❌ Failed to load settings: %s — using defaults", e)
        # Create a minimal fallback so the service can at least start
        class _Fallback:
            data_root = os.environ.get("DATA_ROOT", "/data")
        _settings_boot = _Fallback()
    DATA_ROOT = _settings_boot.data_root
    WEIGHTS_DIR = os.path.join(DATA_ROOT, "weights")
    INSIGHT_ROOT = os.getenv("INSIGHTFACE_ROOT", os.getenv("INSIGHTFACE_HOME", 
        "/cache/insightface" if os.path.exists("/cache") else os.path.join(WEIGHTS_DIR, "insightface")
    ))

def get_model_path(subdir, filename):
    _ensure_boot_settings()
    preferred = os.path.join(WEIGHTS_DIR, subdir, filename)
    if os.path.exists(preferred):
        return preferred
    fallback = os.path.join(WEIGHTS_DIR, filename)
    return fallback

async def validate_image_upload(file: UploadFile, max_size: int = MAX_FILE_SIZE_BYTES) -> bytes:
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}")
    contents = await file.read()
    if len(contents) > max_size:
        raise HTTPException(status_code=400, detail="File too large")
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Empty file")
    return contents

# --- Pydantic Models ---
class Box(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float

class FaceResult(BaseModel):
    box: Box
    landmarks: Optional[List[List[float]]] = None
    embedding: Optional[List[float]] = None

class DetectionResponse(BaseModel):
    success: bool
    faces: List[FaceResult] = []
    humans: List[Box] = []
    processing_time_ms: float

class CompareResponse(BaseModel):
    success: bool
    similarity: float
    is_same_person: bool
    threshold: float


class EmbeddingResponse(BaseModel):
    success: bool
    embeddings: List[dict] = []
    processing_time_ms: float

# --- Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global face_detector, human_detector, face_recognizer, _stream_consumer
    logger.info("🚀 omni-human service starting...")
    
    # Ensure boot settings are initialized (lazy, crash-safe)
    _ensure_boot_settings()
    settings = get_settings()

    # 1. Load Human Detector (YOLOv8n) — use config threshold
    human_path = get_model_path("human", "yolov8n.pt")
    try:
        human_detector = HumanDetector(human_path, conf_thres=settings.human_confidence_threshold)
    except Exception as e:
        logger.error(f"❌ Failed to load Human Detector: {e}")

    # 2. Load Face Detector (YOLOv8n-face) — use config threshold
    face_path = get_model_path("face", "yolov8n-face.pt")
    try:
        face_detector = FaceDetector(face_path, conf_thres=settings.face_confidence_threshold)
    except Exception as e:
        logger.error(f"❌ Failed to load Face Detector: {e}")

    # 3. Load Face Recognizer (InsightFace)
    try:
        face_recognizer = FaceRecognizer(root_dir=INSIGHT_ROOT)
    except Exception as e:
        face_recognizer = None
        logger.error(f"❌ Failed to initialize Face Recognizer: {e}")

    # 4. Redis Stream Consumer (Event-Driven Phase 1)
    db_engine = None
    try:
        from sqlalchemy.ext.asyncio import create_async_engine
        db_url = os.getenv("DATABASE_URL", "postgresql+asyncpg://postgres:postgres@omni-db:5432/omnivision")
        db_engine = create_async_engine(db_url, pool_size=3, max_overflow=2)
    except Exception as e:
        logger.warning(f"⚠️ DB engine for omni-human consumer not available: {e}")

    try:
        _stream_consumer = HumanStreamConsumer(
            face_detector=face_detector,
            face_recognizer=face_recognizer,
            db_engine=db_engine,
        )
    except Exception as e:
        logger.error("❌ HumanStreamConsumer constructor failed: %s (API-only mode)", e)
        _stream_consumer = None
    
    _consumer_started = False
    if _stream_consumer is not None:
        try:
            await _stream_consumer.connect()
            await _stream_consumer.ensure_group()
            await _stream_consumer.start()
            _consumer_started = _stream_consumer.is_connected
        except Exception as e:
            logger.warning(f"⚠️ Stream consumer failed to start: {e} (API-only mode)")

        # If consumer failed to connect at startup (Redis not yet ready),
        # launch a background retry loop — mirrors omni-fusion behaviour.
        if not _consumer_started:
            async def _retry_consumer_start():
                """Keep retrying until Redis is reachable."""
                attempt = 0
                while True:
                    await asyncio.sleep(5)
                    if _stream_consumer is None:
                        return
                    try:
                        await _stream_consumer.connect()
                        if _stream_consumer.is_connected:
                            await _stream_consumer.ensure_group()
                            await _stream_consumer.start()
                            if _stream_consumer.is_connected:
                                logger.info("✅ omni-human stream consumer connected after %d retries", attempt + 1)
                                return
                    except Exception as e:
                        attempt += 1
                        logger.warning("omni-human consumer retry %d failed: %s", attempt, e)
            asyncio.create_task(_retry_consumer_start())
            logger.warning("⚠️ omni-human stream consumer starting in background retry mode")

    # Accurate startup log — reflect actual state
    _models_ok = face_recognizer is not None
    _stream_ok = _stream_consumer is not None and _stream_consumer.is_connected
    if _models_ok and _stream_ok:
        logger.info("✅ omni-human service ready! (full stack + event-driven)")
    elif _models_ok:
        logger.warning("⚠️ omni-human service ready in API-only mode (stream consumer not connected yet)")
    else:
        logger.error("❌ omni-human service started but face recognizer failed to load — check model weights")
    yield
    logger.info("👋 omni-human service shutting down...")
    if _stream_consumer:
        await _stream_consumer.disconnect()
    if db_engine:
        await db_engine.dispose()

app = FastAPI(title="omni-human Service", lifespan=lifespan)

@app.get("/human/health")
async def health_check():
    models = {
        "human_detector": human_detector is not None,
        "face_detector": face_detector is not None,
        "face_recognizer": face_recognizer is not None,
    }
    stream_ok = _stream_consumer is not None and _stream_consumer.is_connected
    all_models_ok = all(models.values())
    
    if all_models_ok and stream_ok:
        status = "healthy"
    elif face_recognizer is not None:
        status = "degraded"  # recognizer OK but missing detector/stream
    else:
        status = "unhealthy"
    
    return {
        "status": status,
        "service": "omni-human",
        "version": "1.0.0",
        "models": models,
        "stream_consumer": stream_ok,
        "stream_stats": _stream_consumer.stats if _stream_consumer else {},
    }


@app.get("/human/settings")
async def get_settings_endpoint():
    """Return current omni-human runtime settings + field metadata for UI sliders"""
    settings = get_settings()
    return {
        "data_root": os.environ.get("DATA_ROOT", "/data"),
        "device": os.environ.get("DEVICE", "cuda"),
        "cuda_memory_fraction": float(os.environ.get("CUDA_MEMORY_FRACTION", "0.3")),
        "face_confidence_threshold": settings.face_confidence_threshold,
        "human_confidence_threshold": settings.human_confidence_threshold,
        "insightface_det_score": settings.insightface_det_score,
        "face_db_match_distance": settings.face_db_match_distance,
        "track_dedup_cooldown": settings.track_dedup_cooldown,
        "stale_event_threshold": settings.stale_event_threshold,
        "min_person_crop_px": settings.min_person_crop_px,
        "min_face_crop_px": settings.min_face_crop_px,
        "person_publish_conf": settings.person_publish_conf,
        "model_status": {
            "human_detector": human_detector is not None,
            "face_detector": face_detector is not None,
            "face_recognizer": face_recognizer is not None,
        },
        "stream_consumer_connected": _stream_consumer is not None and _stream_consumer.is_connected,
        "field_descriptions": settings.get_field_descriptions(),
    }


@app.post("/human/settings")
async def update_settings_endpoint(new_settings: dict):
    """Update and persist Face Recognition configuration.
    Only mutable (live-tunable) fields are accepted.
    """
    settings = get_settings()

    # SECURITY: Only allow known mutable fields (allowlist approach)
    _MUTABLE_FIELDS = {
        "face_confidence_threshold", "human_confidence_threshold",
        "insightface_det_score", "face_db_match_distance",
        "track_dedup_cooldown", "stale_event_threshold",
        "min_person_crop_px", "min_face_crop_px", "person_publish_conf",
    }

    errors: list[str] = []
    applied: dict[str, object] = {}

    for key, value in new_settings.items():
        if key not in _MUTABLE_FIELDS:
            errors.append(f"{key}: not a mutable setting")
            continue

        # Validate using field metadata (ge/le constraints)
        valid = True
        field_info = settings.model_fields.get(key)
        if field_info:
            for m in field_info.metadata:
                if hasattr(m, 'ge') and isinstance(value, (int, float)) and value < m.ge:
                    errors.append(f"{key}: must be ≥ {m.ge}, got {value}")
                    valid = False
                    break
                if hasattr(m, 'le') and isinstance(value, (int, float)) and value > m.le:
                    errors.append(f"{key}: must be ≤ {m.le}, got {value}")
                    valid = False
                    break

        if not valid:
            continue

        setattr(settings, key, value)
        applied[key] = value

    # Propagate live-tunable thresholds to running model instances
    if "face_confidence_threshold" in applied and face_detector is not None:
        face_detector.conf_thres = applied["face_confidence_threshold"]
        logger.info("🔄 Updated face_detector.conf_thres → %.2f", applied["face_confidence_threshold"])
    if "human_confidence_threshold" in applied and human_detector is not None:
        human_detector.conf_thres = applied["human_confidence_threshold"]
        logger.info("🔄 Updated human_detector.conf_thres → %.2f", applied["human_confidence_threshold"])

    settings.save_to_file()

    return {
        "success": len(errors) == 0,
        "applied": applied,
        "errors": errors,
        "settings": settings.model_dump(),
    }

@app.post("/human/detect", response_model=DetectionResponse)
async def detect(
    file: UploadFile = File(...),
    detect_faces: bool = Query(True),
    detect_humans: bool = Query(True),
    extract_embedding: bool = Query(False)
):
    start = time.time()
    contents = await validate_image_upload(file)
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        img_np = np.array(image)
        # Convert RGB → BGR for OpenCV / InsightFace
        img_bgr = img_np[:, :, ::-1].copy()
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid image")

    resp_faces = []
    resp_humans = []

    # 1. Face Detection
    if detect_faces and face_detector:
        # Detect using YOLOv8-face (expects BGR) — run in thread to avoid blocking event loop
        det_results = await asyncio.to_thread(face_detector.detect, img_bgr)
        
        # If embedding requested, we need to crop and pass to recognizer, 
        # OR just run recognizer on whole image and match boxes. 
        # Simplest efficient way: If embedding needed, run InsightFace on whole image 
        # as it does detection+align+embed in one go, usually better than simple crop.
        # BUT, the requirement is to use YOLOv8-face. 
        # Let's stick to YOLOv8-face for detection coordinates as requested.
        
        for d in det_results:
            box = d["box"]
            face_res = FaceResult(
                box=Box(x1=box[0], y1=box[1], x2=box[2], y2=box[3], confidence=d["conf"]),
                landmarks=d.get("keypoints")
            )
            
            if extract_embedding and face_recognizer:
                # Extract embedding for this specific crop
                # A proper face recognition pipeline aligns the face using landmarks before embedding.
                # YOLOv8-face provides 5 landmarks. We can use them.
                # However, InsightFace's `get` expects the full image and does its own detection.
                # To fuse them: we can force InsightFace to use the provided bbox/kps? 
                # Not easily via high-level API.
                
                # FALLBACK: Use InsightFace for embedding extraction on the crop.
                # This is less accurate than full alignment but acceptable for integration v1.
                x1, y1, x2, y2 = box
                crop = img_bgr[y1:y2, x1:x2]
                if crop.size > 0:
                     # We can call a specialized method in recognizer to embed a crop
                     # For now, let's just use the high level 'get' on the crop 
                     # (which will re-detect inside the crop, ensuring alignment)
                     feats = await asyncio.to_thread(face_recognizer.extract_embedding, crop)
                     if feats:
                         # Take the largest face in the crop
                         best_feat = max(feats, key=lambda x: (x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))
                         face_res.embedding = best_feat['embedding']
            
            resp_faces.append(face_res)

    # 2. Human Detection
    if detect_humans and human_detector:
        det_results = await asyncio.to_thread(human_detector.detect, img_bgr)
        for d in det_results:
            resp_humans.append(Box(x1=d[0], y1=d[1], x2=d[2], y2=d[3], confidence=d[4]))

    return DetectionResponse(
        success=True,
        faces=resp_faces,
        humans=resp_humans,
        processing_time_ms=(time.time() - start) * 1000
    )

@app.post("/human/compare", response_model=CompareResponse)
async def compare(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    if not face_recognizer:
        raise HTTPException(status_code=503, detail="Face Recognizer not available")
        
    start = time.time()
    c1 = await validate_image_upload(file1)
    c2 = await validate_image_upload(file2)
    
    # InsightFace expects BGR — convert PIL RGB → BGR
    img1 = np.array(Image.open(io.BytesIO(c1)).convert("RGB"))[:, :, ::-1].copy()
    img2 = np.array(Image.open(io.BytesIO(c2)).convert("RGB"))[:, :, ::-1].copy()
    
    # Extract — run in thread to avoid blocking event loop
    feats1 = await asyncio.to_thread(face_recognizer.extract_embedding, img1)
    feats2 = await asyncio.to_thread(face_recognizer.extract_embedding, img2)
    
    if not feats1 or not feats2:
        settings = get_settings()
        return CompareResponse(success=False, similarity=0.0, is_same_person=False, threshold=settings.face_db_match_distance)
        
    # Compare largest faces
    f1 = max(feats1, key=lambda x: (x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))
    f2 = max(feats2, key=lambda x: (x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))
    
    sim = face_recognizer.compare_faces(f1['embedding'], f2['embedding'])
    settings = get_settings()
    threshold = settings.face_db_match_distance
    # face_db_match_distance is cosine DISTANCE threshold (lower = stricter).
    # cosine_distance = 1 - cosine_similarity.
    # Match when distance < threshold, i.e. similarity > (1 - threshold).
    cosine_dist = 1.0 - sim
    
    return CompareResponse(
        success=True,
        similarity=sim,
        is_same_person=cosine_dist < threshold,
        threshold=threshold
    )


@app.post("/human/embed", response_model=EmbeddingResponse)
async def embed(file: UploadFile = File(...)):
    if not face_recognizer:
        raise HTTPException(status_code=503, detail="Face Recognizer not available")

    start = time.time()
    contents = await validate_image_upload(file)
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        img_np = np.array(image)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image")

    # InsightFace expects BGR — convert PIL RGB → BGR
    img_bgr = img_np[:, :, ::-1].copy()
    embeddings = await asyncio.to_thread(face_recognizer.extract_embedding, img_bgr)

    return EmbeddingResponse(
        success=True,
        embeddings=embeddings,
        processing_time_ms=(time.time() - start) * 1000
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
