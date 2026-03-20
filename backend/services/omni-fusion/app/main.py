"""
omni-fusion: spatial–temporal fusion service.

Consumes plate + face Redis streams, links vehicles and faces, pushes results to the API.
"""
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import List, Optional

import httpx
from fastapi import FastAPI

from app.config import get_settings
from app.consumer import FusionConsumer
from app.spatial_engine import DetectionEvent, FusedIdentity, SpatialFusionEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("omni-fusion")

# === Globals ===
_consumer: Optional[FusionConsumer] = None
_engine: Optional[SpatialFusionEngine] = None
_consume_task: Optional[asyncio.Task] = None
_http_client: Optional[httpx.AsyncClient] = None

# Stats
_stats = {
    "events_received": 0,
    "fusions_created": 0,
    "fusions_pushed": 0,
    "errors": 0,
}


async def _push_to_cms(fused: FusedIdentity, settings):
    """Push a fused identity card to Web CMS API with retry."""
    url = f"{settings.cms_base_url}/api/Events/fused"
    payload = fused.to_dict()

    headers = {}
    if settings.internal_secret:
        headers["X-Internal-Secret"] = settings.internal_secret

    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            resp = await _http_client.post(url, json=payload, headers=headers, timeout=5.0)
            if resp.status_code < 400:
                _stats["fusions_pushed"] += 1
                logger.info(
                    "✅ Pushed fused identity: %s | plate=%s | driver=%s",
                    fused.linked_id, fused.plate_text, fused.driver_identity,
                )
                return
            elif resp.status_code in (502, 503, 504):
                # Transient infrastructure error (CMS restart/overload) — retry
                logger.warning("CMS transient error (%d), attempt %d/%d", resp.status_code, attempt, max_attempts)
                if attempt < max_attempts:
                    await asyncio.sleep(0.5)
                else:
                    _stats["errors"] += 1
            else:
                # Business error (400, 401, 422, 500, etc.) — no retry
                _stats["errors"] += 1
                logger.warning("CMS push failed (%d): %s", resp.status_code, resp.text[:200])
                return
        except Exception as e:
            if attempt < max_attempts:
                logger.warning("CMS push attempt %d failed: %s — retrying", attempt, e)
                await asyncio.sleep(0.5)
            else:
                logger.warning("CMS push error (final): %s", e)
                _stats["errors"] += 1


async def _on_fusion_window(camera_id: str, events: List[DetectionEvent]):
    """
    Called when a time window expires for a camera.
    Runs spatial fusion and pushes results to CMS.
    """
    settings = get_settings()

    _stats["events_received"] += len(events)

    # Run spatial fusion
    fused_list = _engine.fuse_events(events)
    _stats["fusions_created"] += len(fused_list)

    if fused_list:
        logger.info(
            "🔗 [%s] Fused %d identities from %d events",
            camera_id[:8], len(fused_list), len(events),
        )

    # Push meaningful fusions (have plate OR face data OR face crop)
    for fused in fused_list:
        has_plate = bool(fused.plate_text)
        has_named_face = bool(fused.driver_identity) and fused.driver_identity != "Unknown"
        has_face_crop = bool(fused.face_crop_path)
        # Also push if face was detected (confidence > 0) even if unknown —
        # this ensures standalone person events with detected faces reach CMS
        has_face_detected = (fused.driver_face_confidence or 0) > 0
        if has_plate or has_named_face or has_face_crop or has_face_detected:
            await _push_to_cms(fused, settings)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _consumer, _engine, _consume_task, _http_client
    settings = get_settings()

    logger.info("=" * 60)
    logger.info("🚀 omni-fusion starting (spatial–temporal fusion)...")
    logger.info("   Redis: %s", settings.redis_streams_url)
    logger.info("   CMS: %s", settings.cms_base_url)
    logger.info("   Temporal Window: %.1fs", settings.temporal_window_sec)
    logger.info("   Motorcycle IoU: %.2f", settings.motorcycle_iou_threshold)
    logger.info("   Car Driver Region: top %.0f%%", settings.car_driver_top_region * 100)
    logger.info("=" * 60)

    # Initialize shared HTTP client (reuse connections)
    _http_client = httpx.AsyncClient()

    # Initialize spatial engine
    _engine = SpatialFusionEngine(settings)

    # Initialize Redis consumer (retry up to 5 times)
    _consumer = FusionConsumer(settings)
    for attempt in range(1, 6):
        await _consumer.connect()
        if _consumer.is_connected:
            break
        logger.warning("Redis connection attempt %d/5 failed, retrying in 3s...", attempt)
        await asyncio.sleep(3)
    if not _consumer.is_connected:
        logger.error("❌ Could not connect to Redis after 5 attempts — consumer will NOT run!")
    else:
        await _consumer.ensure_groups()

    # Start consumption loop only when Redis is connected
    # (otherwise the task spins in a retry loop, wasting CPU and flooding logs)
    if _consumer.is_connected:
        _consume_task = asyncio.create_task(
            _consumer.consume_loop(_on_fusion_window)
        )
    else:
        logger.warning("⚠️ Consume loop NOT started — Redis unavailable")
    logger.info("🟢 omni-fusion ready!")

    yield

    # Shutdown
    logger.info("🔴 omni-fusion shutting down...")
    if _consume_task:
        _consume_task.cancel()
        try:
            await _consume_task
        except asyncio.CancelledError:
            pass
    if _consumer:
        await _consumer.disconnect()
    if _http_client:
        await _http_client.aclose()
    logger.info("👋 omni-fusion stopped.")


app = FastAPI(
    title="omni-fusion Service",
    description="Spatial–temporal fusion — links vehicles, plates, and faces",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/fusion/health")
async def health():
    consumer_ok = _consumer.is_connected if _consumer else False
    
    return {
        "status": "healthy" if consumer_ok else "degraded",
        "service": "omni-fusion",
        "version": "1.0.0",
        "consumer_connected": consumer_ok,
        "stats": _stats,
    }


@app.get("/fusion/stats")
async def stats():
    return _stats


@app.get("/fusion/settings")
async def get_fusion_settings():
    """Return current omni-fusion runtime settings + field metadata for UI sliders"""
    settings = get_settings()

    # Expose mutable tuning knobs (exclude infra like redis_streams_url)
    _SENSITIVE_FIELDS = {"redis_streams_url", "cms_base_url", "internal_secret"}

    return {
        **settings.model_dump(exclude=_SENSITIVE_FIELDS),
        "consumer_connected": _consumer.is_connected if _consumer else False,
        "stats": _stats,
    }


@app.post("/fusion/settings")
async def update_fusion_settings(new_settings: dict):
    """Update and persist Fusion configuration.
    Only mutable (live-tunable) fields are accepted.
    """
    settings = get_settings()

    _MUTABLE_FIELDS = {
        "motorcycle_iou_threshold", "car_driver_top_region",
        "temporal_window_sec", "min_spatial_overlap", "event_buffer_ttl",
    }

    errors: list[str] = []
    applied: dict[str, object] = {}

    for key, value in new_settings.items():
        if key not in _MUTABLE_FIELDS:
            errors.append(f"{key}: not a mutable setting")
            continue
        if not hasattr(settings, key):
            errors.append(f"{key}: unknown setting")
            continue

        old_val = getattr(settings, key)
        setattr(settings, key, value)
        applied[key] = value
        logger.info("⚙️ fusion config updated: %s = %s (was %s)", key, value, old_val)

    return {
        "applied": applied,
        "errors": errors,
        "current": settings.model_dump(exclude={"redis_streams_url", "cms_base_url", "internal_secret"}),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
