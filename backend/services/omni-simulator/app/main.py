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
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("omni-simulator")


class CameraConfig(BaseModel):
    video_path: str
    loop: bool = True
    fps: int = 30


# Global state
_active_cameras: Dict[str, subprocess.Popen] = {}
_ffmpeg_processes: Dict[str, subprocess.Popen] = {}
_videos_dir = "/videos"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 OmniSimulator Starting...")
    logger.info(f"   Videos directory: {_videos_dir}")
    
    # Auto-start cameras from videos directory
    if os.path.exists(_videos_dir):
        for video_file in Path(_videos_dir).glob("*.mp4"):
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


async def start_camera(camera_id: str, video_path: str, loop: bool = True) -> bool:
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

    # ffmpeg command to serve video as RTSP
    # - Stream copy for efficiency (no re-encoding)
    # - Loop for continuous playback
    cmd = [
        "ffmpeg",
        "-re",                          # Read input at native frame rate
        "-stream_loop", "-1" if loop else "0",  # Loop forever
        "-i", video_path,               # Input file
        "-c", "copy",                   # Copy streams (no re-encode)
        "-f", "rtsp",                   # Output format RTSP
        f"rtsp://localhost:8554/{camera_id}"
    ]

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid  # Create new process group for cleanup
        )
        _active_cameras[camera_id] = process
        logger.info(f"Started camera {camera_id} -> rtsp://localhost:8554/{camera_id}")
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


@app.get("/simulator/health")
async def health():
    """Health check."""
    return {
        "status": "healthy",
        "service": "omni-simulator",
        "version": "1.0.0",
        "active_cameras": list(_active_cameras.keys())
    }


@app.post("/simulator/cameras/{camera_id}/start")
async def api_start_camera(camera_id: str, video_path: str, loop: bool = True):
    """Start a mock camera stream."""
    success = await start_camera(camera_id, video_path, loop)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to start camera")
    return {
        "camera_id": camera_id,
        "rtsp_url": f"rtsp://localhost:8554/{camera_id}",
        "video_path": video_path
    }


@app.post("/simulator/cameras/{camera_id}/stop")
async def api_stop_camera(camera_id: str):
    """Stop a mock camera stream."""
    success = await stop_camera(camera_id)
    if not success:
        raise HTTPException(status_code=404, detail="Camera not found")
    return {"message": f"Camera {camera_id} stopped"}


@app.get("/simulator/cameras")
async def list_cameras():
    """List all cameras and their status."""
    cameras = []
    for cam_id in _active_cameras:
        cameras.append({
            "camera_id": cam_id,
            "status": "running" if _active_cameras[cam_id].poll() is None else "stopped",
            "rtsp_url": f"rtsp://localhost:8554/{cam_id}"
        })
    return {"cameras": cameras}


@app.get("/simulator/videos")
async def list_videos():
    """List available video files."""
    if not os.path.exists(_videos_dir):
        return {"videos": []}
    
    videos = []
    for video_file in Path(_videos_dir).glob("*"):
        if video_file.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
            videos.append({
                "filename": video_file.name,
                "path": str(video_file),
                "size_mb": round(video_file.stat().st_size / (1024 * 1024), 2)
            })
    return {"videos": videos}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8554)
