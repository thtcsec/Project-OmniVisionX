"""
OmniSimulator - Mock Camera RTSP Server
======================================
Loop video files and serve as RTSP streams.
Usage: Drop video files into /videos, access via RTSP URL.

Example RTSP URL: rtsp://localhost:8554/cam-1
"""

import asyncio
import logging
import os
import subprocess
import uuid
from pathlib import Path
from typing import Optional, Dict
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("omni-simulator")


class CameraConfig(BaseModel):
    video_path: str
    loop: bool = True
    fps: int = 30
    transcode_h264: bool = True


class RelayConfig(BaseModel):
    source_url: str
    transcode_h264: bool = True


# Global state
_active_cameras: Dict[str, subprocess.Popen] = {}
_ffmpeg_processes: Dict[str, subprocess.Popen] = {}
_active_relays: Dict[str, subprocess.Popen] = {}
_videos_dirs = [
    p.strip()
    for p in os.getenv("OMNI_SIMULATOR_VIDEOS_DIRS", "/videos,/videos-legacy,/data,/data/videos").split(",")
    if p.strip()
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 OmniSimulator Starting...")
    logger.info(f"   Videos directories: {_videos_dirs}")
    
    # Auto-start cameras from videos directory
    for video_file in _iter_video_files():
        cam_id = f"cam-{video_file.stem}"
        await start_camera(cam_id, str(video_file))
    
    logger.info("✅ OmniSimulator Ready!")
    logger.info(f"   RTSP URLs available at: rtsp://localhost:8554/<camera-id>")

    yield

    # Cleanup
    logger.info("👋 OmniSimulator stopping...")
    for cam_id in list(_active_cameras.keys()):
        await stop_camera(cam_id)


app = FastAPI(
    title="OmniSimulator",
    description="Mock Camera RTSP Server - Loop video files as camera streams",
    version="1.0.0",
    lifespan=lifespan,
)

_cors_origins = [
    origin.strip()
    for origin in os.getenv(
        "OMNI_SIMULATOR_CORS_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def start_camera(camera_id: str, video_path: str, loop: bool = True, fps: int = 30, transcode_h264: bool = True) -> bool:
    """
    Start RTSP stream for a video file using ffmpeg.
    RTSP URL will be: rtsp://localhost:8554/{camera_id}
    """
    if camera_id in _active_cameras:
        logger.warning(f"Camera {camera_id} already running")
        return False

    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        return False

    publish_host = os.getenv("OMNI_MEDIAMTX_RTSP_PUBLISH_HOST", "omni-mediamtx")
    publish_port = os.getenv("OMNI_MEDIAMTX_RTSP_PUBLISH_PORT", "8554")
    publish_url = f"rtsp://{publish_host}:{publish_port}/{camera_id}"

    cmd = [
        "ffmpeg",
        "-re",                          # Read input at native frame rate
        "-stream_loop", "-1" if loop else "0",  # Loop forever
        "-i", video_path,               # Input file
    ]

    if transcode_h264:
        cmd += [
            "-an",
            "-c:v", "libx264",
            "-preset", os.getenv("OMNI_SIMULATOR_X264_PRESET", "veryfast"),
            "-tune", os.getenv("OMNI_SIMULATOR_X264_TUNE", "zerolatency"),
            "-pix_fmt", "yuv420p",
            "-r", str(fps),
        ]
    else:
        cmd += ["-c", "copy"]

    cmd += [
        "-f", "rtsp",                   # Output format RTSP
        "-rtsp_transport", "tcp",
        publish_url,
    ]

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid  # Create new process group for cleanup
        )
        _active_cameras[camera_id] = process
        logger.info("Started camera %s -> %s", camera_id, publish_url)
        return True
    except Exception as e:
        logger.error(f"Failed to start camera {camera_id}: {e}")
        return False


async def stop_camera(camera_id: str) -> bool:
    """Stop a camera stream."""
    if camera_id not in _active_cameras:
        return False

    process = _active_cameras.pop(camera_id)
    try:
        os.killpg(os.getpgid(process.pid), 9)
        process.wait(timeout=5)
    except Exception as e:
        logger.warning(f"Error stopping camera {camera_id}: {e}")
    
    logger.info(f"Stopped camera {camera_id}")
    return True


async def start_relay(camera_id: str, source_url: str, transcode_h264: bool = True) -> bool:
    if camera_id in _active_relays:
        logger.warning("Relay %s already running", camera_id)
        return False

    source = (source_url or "").strip()
    if not source:
        return False

    publish_host = os.getenv("OMNI_MEDIAMTX_RTSP_PUBLISH_HOST", "omni-mediamtx")
    publish_port = os.getenv("OMNI_MEDIAMTX_RTSP_PUBLISH_PORT", "8554")
    publish_url = f"rtsp://{publish_host}:{publish_port}/{camera_id}"

    cmd: list[str] = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "warning",
    ]

    if source.lower().startswith("rtsp://") or source.lower().startswith("rtsps://"):
        cmd += ["-rtsp_transport", os.getenv("OMNI_RELAY_RTSP_TRANSPORT", "tcp")]

    cmd += ["-i", source]

    if transcode_h264:
        fps = os.getenv("OMNI_RELAY_FPS", "").strip()
        gop = os.getenv("OMNI_RELAY_GOP", "").strip()
        cmd += [
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            os.getenv("OMNI_RELAY_X264_PRESET", "veryfast"),
            "-tune",
            os.getenv("OMNI_RELAY_X264_TUNE", "zerolatency"),
            "-pix_fmt",
            "yuv420p",
        ]
        if fps:
            cmd += ["-r", fps]
        if gop:
            cmd += ["-g", gop, "-keyint_min", gop, "-sc_threshold", "0"]
    else:
        cmd += ["-c", "copy"]

    cmd += [
        "-f",
        "rtsp",
        "-rtsp_transport",
        "tcp",
        publish_url,
    ]

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid,
        )
        _active_relays[camera_id] = process
        logger.info("Started relay %s -> %s", camera_id, publish_url)
        return True
    except Exception as e:
        logger.error("Failed to start relay %s: %s", camera_id, e)
        return False


async def stop_relay(camera_id: str) -> bool:
    if camera_id not in _active_relays:
        return False

    process = _active_relays.pop(camera_id)
    try:
        os.killpg(os.getpgid(process.pid), 9)
        process.wait(timeout=5)
    except Exception as e:
        logger.warning("Error stopping relay %s: %s", camera_id, e)
    return True


@app.get("/simulator/health")
async def health():
    """Health check."""
    return {
        "status": "healthy",
        "service": "omni-simulator",
        "version": "1.0.0",
        "active_cameras": list(_active_cameras.keys()),
        "active_relays": list(_active_relays.keys()),
    }


@app.post("/simulator/cameras/{camera_id}/start")
async def api_start_camera(camera_id: str, video_path: str, loop: bool = True, fps: int = 30, transcode_h264: bool = True):
    """Start a mock camera stream."""
    success = await start_camera(camera_id, video_path, loop, fps, transcode_h264)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to start camera")
    publish_host = os.getenv("OMNI_MEDIAMTX_RTSP_PUBLISH_HOST", "omni-mediamtx")
    publish_port = os.getenv("OMNI_MEDIAMTX_RTSP_PUBLISH_PORT", "8554")
    return {
        "camera_id": camera_id,
        "rtsp_url": f"rtsp://{publish_host}:{publish_port}/{camera_id}",
        "video_path": video_path,
        "loop": loop,
        "fps": fps,
        "transcode_h264": transcode_h264,
    }


@app.post("/simulator/cameras/{camera_id}/stop")
async def api_stop_camera(camera_id: str):
    """Stop a mock camera stream."""
    success = await stop_camera(camera_id)
    if not success:
        raise HTTPException(status_code=404, detail="Camera not found")
    return {"message": f"Camera {camera_id} stopped"}


@app.post("/simulator/relays/{camera_id}/start")
async def api_start_relay(camera_id: str, cfg: RelayConfig):
    success = await start_relay(camera_id, cfg.source_url, cfg.transcode_h264)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to start relay")
    publish_host = os.getenv("OMNI_MEDIAMTX_RTSP_PUBLISH_HOST", "omni-mediamtx")
    publish_port = os.getenv("OMNI_MEDIAMTX_RTSP_PUBLISH_PORT", "8554")
    return {
        "camera_id": camera_id,
        "publish_url": f"rtsp://{publish_host}:{publish_port}/{camera_id}",
        "source_url": cfg.source_url,
        "transcode_h264": cfg.transcode_h264,
    }


@app.post("/simulator/relays/{camera_id}/stop")
async def api_stop_relay(camera_id: str):
    success = await stop_relay(camera_id)
    if not success:
        raise HTTPException(status_code=404, detail="Relay not found")
    return {"message": f"Relay {camera_id} stopped"}


@app.get("/simulator/relays")
async def list_relays():
    relays = []
    for cam_id in _active_relays:
        relays.append({
            "camera_id": cam_id,
            "status": "running" if _active_relays[cam_id].poll() is None else "stopped",
        })
    return {"relays": relays}


@app.get("/simulator/cameras")
async def list_cameras():
    """List all cameras and their status."""
    cameras = []
    for cam_id in _active_cameras:
        publish_host = os.getenv("OMNI_MEDIAMTX_RTSP_PUBLISH_HOST", "omni-mediamtx")
        publish_port = os.getenv("OMNI_MEDIAMTX_RTSP_PUBLISH_PORT", "8554")
        cameras.append({
            "camera_id": cam_id,
            "status": "running" if _active_cameras[cam_id].poll() is None else "stopped",
            "rtsp_url": f"rtsp://{publish_host}:{publish_port}/{cam_id}"
        })
    return {"cameras": cameras}


@app.get("/simulator/videos")
async def list_videos():
    """List available video files."""
    videos = []
    for video_file in _iter_video_files():
        videos.append({
            "filename": video_file.name,
            "path": str(video_file),
            "size_mb": round(video_file.stat().st_size / (1024 * 1024), 2)
        })
    return {"videos": videos}


@app.post("/simulator/videos/rescan")
async def rescan_videos(auto_start: bool = False):
    if auto_start:
        for video_file in _iter_video_files():
            cam_id = f"cam-{video_file.stem}"
            if cam_id not in _active_cameras:
                await start_camera(cam_id, str(video_file))
    return await list_videos()


def _iter_video_files():
    exts = {".mp4", ".avi", ".mov", ".mkv"}
    seen: set[str] = set()
    for base in _videos_dirs:
        if not os.path.exists(base):
            continue
        for video_file in Path(base).glob("*"):
            if video_file.is_file() and video_file.suffix.lower() in exts:
                key = str(video_file)
                if key in seen:
                    continue
                seen.add(key)
                yield video_file


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8554)
