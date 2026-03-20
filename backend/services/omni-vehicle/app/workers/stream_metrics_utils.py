import time
from typing import Any


def create_camera_stat_entry(now: float | None = None) -> dict[str, float]:
    ts = time.monotonic() if now is None else now
    return {
        "start": ts,
        "count": 0.0,
        "ratio": 0.0,
        "fps": 0.0,
        "received": 0.0,
        "processed": 0.0,
        "dropped": 0.0,
        "processing_ms_total": 0.0,
        "e2e_lag_ms_total": 0.0,
        "last_processing_ms": 0.0,
        "last_e2e_lag_ms": 0.0,
        "last_update": ts,
    }


def update_camera_processing_stats(entry: dict[str, float], processed_ms: float, e2e_lag_ms: float) -> None:
    entry["processed"] = float(entry.get("processed", 0.0)) + 1.0
    entry["processing_ms_total"] = float(entry.get("processing_ms_total", 0.0)) + processed_ms
    entry["e2e_lag_ms_total"] = float(entry.get("e2e_lag_ms_total", 0.0)) + e2e_lag_ms
    entry["last_processing_ms"] = processed_ms
    entry["last_e2e_lag_ms"] = e2e_lag_ms
    entry["last_update"] = time.monotonic()


def build_camera_metrics(camera_stats: dict[str, dict[str, float]]) -> dict[str, Any]:
    cameras: dict[str, Any] = {}
    for camera_id, stats in camera_stats.items():
        processed = float(stats.get("processed", 0.0))
        received = float(stats.get("received", 0.0))
        dropped = float(stats.get("dropped", 0.0))
        cameras[camera_id] = {
            "fps": round(float(stats.get("fps", 0.0)), 3),
            "drop_ratio": round(float(stats.get("ratio", 0.0)), 3),
            "received": int(received),
            "processed": int(processed),
            "dropped": int(dropped),
            "drop_percent": round((dropped / max(received, 1.0)) * 100.0, 2),
            "avg_processing_ms": round(float(stats.get("processing_ms_total", 0.0)) / max(processed, 1.0), 2),
            "avg_e2e_lag_ms": round(float(stats.get("e2e_lag_ms_total", 0.0)) / max(processed, 1.0), 2),
            "last_processing_ms": round(float(stats.get("last_processing_ms", 0.0)), 2),
            "last_e2e_lag_ms": round(float(stats.get("last_e2e_lag_ms", 0.0)), 2),
        }
    return cameras


def aggregate_global_processing_metrics(stats: dict[str, float]) -> tuple[float, float]:
    processed_frames = int(stats.get("processed_frames", 0))
    avg_processing_ms = float(stats.get("processing_time_ms_total", 0.0)) / processed_frames if processed_frames > 0 else 0.0
    avg_e2e_lag_ms = float(stats.get("e2e_lag_ms_total", 0.0)) / processed_frames if processed_frames > 0 else 0.0
    return avg_processing_ms, avg_e2e_lag_ms
