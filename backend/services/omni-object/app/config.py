"""
OmniVision omni-object — configuration (YOLO, RTSP, Redis streams, SHM)

SINGLE SOURCE OF TRUTH for YOLO detection, ByteTrack tracking,
RTSP capture, and shared-memory settings.

Every tunable field carries:
  - description  : explains what it does (shown in UI tooltip)
  - ge / le      : validation range  (Pydantic will reject out-of-range)
  - json_schema_extra.tier : "live" | "restart" | "infra"
        live    → hot-reloadable without restart
        restart → needs capture-loop restart (not container)
        infra   → needs container restart
"""
import json
import logging
import os
from pydantic import Field
from pydantic_settings import BaseSettings
from typing import Optional, Any

SETTINGS_FILE = "/app/config/settings.json"
logger = logging.getLogger("omni-object.config")

# ── Tier constants (for UI grouping) ──────────────────────────
_LIVE = {"tier": "live"}
_RESTART = {"tier": "restart"}
_INFRA = {"tier": "infra"}


class Settings(BaseSettings):
    # ── Infrastructure (tier: infra) ──────────────────────────
    database_url: str = Field(
        default="postgresql+asyncpg://postgres:postgres@omni-db:5432/omnivision",
        description="Chuỗi kết nối PostgreSQL (thay đổi cần restart container)",
        json_schema_extra=_INFRA,
    )
    device: str = Field(
        default="cuda:0",
        description="Thiết bị GPU/CPU cho YOLO inference (cuda:0, cuda:1, cpu)",
        json_schema_extra=_INFRA,
    )
    yolo_model: str = Field(
        default="/app/weights/yolo11n.pt",
        description="Đường dẫn file model YOLO cho Master Detector. "
                    "yolo11n.pt (nano, ~6MB) nhẹ nhất — đủ cho tracking xe. "
                    "yolo11s.pt nếu muốn cân bằng accuracy/speed. "
                    "yolo11m.pt chỉ dùng khi VRAM ≥ 6GB.",
        json_schema_extra=_INFRA,
    )

    # ── YOLO Detection (tier: live) ───────────────────────────
    confidence_threshold: float = Field(
        default=0.35, ge=0.05, le=0.95,
        description="Ngưỡng tin cậy YOLO — dưới giá trị này sẽ bỏ qua detection. "
                    "Nên bằng track_high_thresh để đồng bộ.",
        json_schema_extra=_LIVE,
    )
    nms_threshold: float = Field(
        default=0.50, ge=0.1, le=0.9,
        description="Ngưỡng Non-Maximum Suppression — càng cao thì càng cho phép "
                    "nhiều bounding box chồng lấn (giảm nếu bị phát hiện trùng xe).",
        json_schema_extra=_LIVE,
    )

    # ── ByteTrack Tracking (tier: live) ───────────────────────
    track_high_thresh: float = Field(
        default=0.35, ge=0.05, le=0.95,
        description="Ngưỡng cao của ByteTrack — detection ≥ giá trị này được tạo "
                    "track mới. PHẢI ≥ confidence_threshold, nếu không xe mới sẽ "
                    "bị mất track. 0.35 giảm false-positive track cho yolo11n.",
        json_schema_extra=_LIVE,
    )
    track_low_thresh: float = Field(
        default=0.15, ge=0.01, le=0.5,
        description="Ngưỡng thấp của ByteTrack — detection giữa low và high chỉ "
                    "dùng để duy trì track đã có, không tạo mới.",
        json_schema_extra=_LIVE,
    )
    track_buffer: int = Field(
        default=30, ge=5, le=600,
        description="Số frame giữ track bị mất trước khi xóa. "
                    "30 frames = 2.5s @ 12 FPS — đủ cho occlusion ngắn, "
                    "tránh zombie track gây miss xe mới.",
        json_schema_extra=_LIVE,
    )
    match_thresh: float = Field(
        default=0.65, ge=0.3, le=0.99,
        description="Ngưỡng IoU để ghép detection vào track hiện có. "
                    "0.65 linh hoạt hơn khi xe chen/vượt nhau mà vẫn đủ chặt.",
        json_schema_extra=_LIVE,
    )
    frame_rate: int = Field(
        default=12, ge=1, le=60,
        description="FPS mà ByteTrack sử dụng để tính Kalman prediction. "
                    "Nên bằng capture_fps.",
        json_schema_extra=_RESTART,
    )

    # ── RTSP Capture (tier: restart) ──────────────────────────
    rtsp_transport: str = Field(
        default="tcp",
        description="Giao thức RTSP: tcp (ổn định, chậm hơn) hoặc udp (nhanh, có thể mất gói).",
        json_schema_extra=_RESTART,
    )
    capture_fps: int = Field(
        default=6, ge=1, le=30,
        description="Số khung hình/giây lấy từ camera. "
                    "6 FPS đủ cho LPR (biển số rõ sau 3-4 frame) và giảm tải GPU đáng kể. "
                    "Tăng lên 10-12 nếu có GPU ≥ 6GB và cần tracking xe tốc độ cao.",
        json_schema_extra=_RESTART,
    )
    capture_buffer_size: int = Field(
        default=10, ge=1, le=100,
        description="Số frame giữ trong bộ đệm để chọn frame chất lượng cao nhất "
                    "(best-of-N). Tăng = chọn frame rõ hơn, tốn RAM hơn.",
        json_schema_extra=_RESTART,
    )
    capture_best_of_last_n: int = Field(
        default=3, ge=1, le=10,
        description="Chọn frame sharp nhất trong N frame gần nhất (giảm miss xe chạy nhanh). "
                    "Giá trị nhỏ ưu tiên tính thời gian thực.",
        json_schema_extra=_LIVE,
    )
    max_concurrent_inference: int = Field(
        default=4, ge=1, le=32,
        description="Số Camera tối đa xử lý suy luận AI tại 1 thời điểm (Concurrency). "
                    "Ngăn chặn lún luồng (Thread-pool Thrashing).",
        json_schema_extra=_RESTART,
    )
    batch_inference_size: int = Field(
        default=2, ge=1, le=16,
        description="Số camera gộp vào một lần YOLO batch inference. "
                    "2 là tối ưu cho yolo11n với GPU 4GB. Tăng lên 4-8 nếu VRAM ≥ 6GB.",
        json_schema_extra=_RESTART,
    )
    batch_collect_window_ms: int = Field(
        default=10, ge=0, le=100,
        description="Cửa sổ gom frame micro-batching (ms) trước khi gọi YOLO batch. "
                    "Tăng nhẹ để cải thiện GPU utilization khi frame đến lệch nhịp.",
        json_schema_extra=_LIVE,
    )
    enable_live_bbox_overlay: bool = Field(
        default=True,
        description="Bật/tắt lưu bbox mới nhất phục vụ overlay realtime trên preview. "
                    "Tắt để giảm tải CPU/RAM khi không cần xem bbox.",
        json_schema_extra=_LIVE,
    )
    cpu_worker_threads: int = Field(
        default=8, ge=2, le=64,
        description="Số worker thread riêng cho encode JPEG / pre-process / tác vụ CPU nặng.",
        json_schema_extra=_RESTART,
    )
    reconnect_backoff_max: float = Field(
        default=30.0, ge=5.0, le=300.0,
        description="Thời gian tối đa (giây) chờ trước khi thử kết nối lại camera "
                    "bị mất tín hiệu.",
        json_schema_extra=_RESTART,
    )

    # ── Stationary vehicle dedup (tier: live) ─────────────────
    stationary_publish_interval_s: float = Field(
        default=12.0, ge=1.0, le=60.0,
        description="Thời gian (giây) tối thiểu giữa 2 lần publish detection cho "
                    "phương tiện đứng yên. Tăng để giảm spam xe đỗ (12s = ít spam hơn 5s).",
        json_schema_extra=_LIVE,
    )
    stationary_movement_threshold: int = Field(
        default=45, ge=5, le=200,
        description="Khoảng cách pixel tối đa giữa 2 frame để coi phương tiện đứng yên. "
                    "Tăng nếu camera rung hoặc xe đỗ vẫn bị spam (45 = chặt hơn 30).",
        json_schema_extra=_LIVE,
    )
    stationary_frame_count: int = Field(
        default=24, ge=3, le=120,
        description="Số frame liên tiếp bbox không di chuyển trước khi coi là đứng yên. "
                    "24 frames ≈ 2s @ 12 FPS — tăng để tránh spam xe đỗ do camera rung.",
        json_schema_extra=_LIVE,
    )

    # ── Redis Streams (tier: infra) ───────────────────────────
    redis_streams_url: str = Field(
        default="redis://omni-bus:6379",
        description="URL Redis Streams dùng làm event bus giữa các service.",
        json_schema_extra=_INFRA,
    )
    stream_maxlen: int = Field(
        default=500, ge=50, le=10000,
        description="MAXLEN tối đa mỗi Redis stream. Tăng nếu consumer xử lý chậm.",
        json_schema_extra=_LIVE,
    )
    stream_publish_queue_size: int = Field(
        default=50, ge=10, le=500,
        description="Độ dài Background Queue cho Redis Publisher. Tăng nếu Redis chậm.",
        json_schema_extra=_RESTART,
    )
    stream_prefix: str = Field(
        default="omni",
        description="Redis stream prefix (e.g. omni:detections, omni:frames:{camera_id}).",
        json_schema_extra=_INFRA,
    )

    # ── Shared Memory Ring Buffer (tier: restart) ─────────────
    shm_enabled: bool = Field(
        default=True,
        description="Bật/tắt shared memory ring buffer cho zero-copy frame passing "
                    "giữa omni-object và omni-vehicle.",
        json_schema_extra=_RESTART,
    )
    shm_dir: str = Field(
        default="/app/shm/omnivision",
        description="Thư mục shared memory (tmpfs mount).",
        json_schema_extra=_INFRA,
    )
    shm_slots_per_camera: int = Field(
        default=4, ge=2, le=32,
        description="Số slot ring buffer cho mỗi camera. Tăng nếu omni-vehicle xử lý "
                    "chậm hơn omni-object gửi frame.",
        json_schema_extra=_RESTART,
    )
    shm_publish_jpeg: bool = Field(
        default=True,
        description="Gửi JPEG qua shared memory (True) hoặc raw numpy (False). "
                    "JPEG tiết kiệm bandwidth, raw nhanh hơn.",
        json_schema_extra=_RESTART,
    )

    # ── MediaMTX RTSP Relay (tier: infra) ─────────────────────
    mediamtx_rtsp_url: str = Field(
        default="rtsp://omni-mediamtx:8554",
        description="URL RTSP relay của MediaMTX (Single Ingress Architecture).",
        json_schema_extra=_INFRA,
    )
    mediamtx_api_url: str = Field(
        default="http://omni-mediamtx:8997",
        description="URL API quản lý MediaMTX.",
        json_schema_extra=_INFRA,
    )
    use_mediamtx_relay: bool = Field(
        default=True,
        description="Sử dụng MediaMTX relay thay vì kết nối trực tiếp tới camera RTSP. "
                    "Bật = mỗi camera chỉ 1 luồng RTSP, tiết kiệm băng thông.",
        json_schema_extra=_RESTART,
    )
    mediamtx_source_on_demand: bool = Field(
        default=False,  # KHUYẾN NGHỊ: Giữ False để stream luôn sẵn sàng
        description="Bật để MediaMTX chỉ kéo RTSP khi có người xem. Tắt để giữ stream nóng,"
                    " giảm trễ khi mở preview/fullscreen. ⚠️ Nếu bật = True, video sẽ mất 3-6s để load lần đầu.",
        json_schema_extra=_RESTART,
    )
    mediamtx_source_on_demand_start_timeout: str = Field(
        default="6s",
        description="Timeout khởi động source on-demand của MediaMTX.",
        json_schema_extra=_RESTART,
    )
    mediamtx_source_on_demand_close_after: str = Field(
        default="10m",
        description="Thời gian giữ source sau khi hết client khi bật on-demand.",
        json_schema_extra=_RESTART,
    )

    # ── Snapshot API (tier: live) ─────────────────────────────
    snapshot_cache_ttl: float = Field(
        default=2.0, ge=0.5, le=30.0,
        description="Thời gian cache (giây) cho snapshot API. Giảm nếu cần "
                    "frame real-time hơn.",
        json_schema_extra=_LIVE,
    )
    idle_frame_publish_interval_s: float = Field(
        default=2.0, ge=0.2, le=30.0,
        description="Chu kỳ tối thiểu (giây) publish frame khi camera không có detection. "
                    "Tăng để giảm CPU encode/JPEG và tải Redis ở luồng nhàn rỗi.",
        json_schema_extra=_LIVE,
    )

    # ── Camera Poller (tier: live) ────────────────────────────
    camera_poll_interval: float = Field(
        default=30.0, ge=5.0, le=300.0,
        description="Chu kỳ (giây) kiểm tra danh sách camera từ database. "
                    "Giảm nếu cần cập nhật camera mới nhanh hơn.",
        json_schema_extra=_LIVE,
    )
    camera_query_timeout_s: float = Field(
        default=5.0, ge=1.0, le=30.0,
        description="Timeout cho mỗi lần truy vấn danh sách camera từ DB (giây).",
        json_schema_extra=_LIVE,
    )
    camera_query_retries: int = Field(
        default=2, ge=0, le=10,
        description="Số lần retry truy vấn camera khi DB lỗi tạm thời.",
        json_schema_extra=_LIVE,
    )
    camera_query_retry_delay_s: float = Field(
        default=1.0, ge=0.1, le=10.0,
        description="Khoảng nghỉ giữa các lần retry truy vấn camera (giây).",
        json_schema_extra=_LIVE,
    )

    class Config:
        env_file = ".env"
        extra = "ignore"

    def get_field_descriptions(self) -> dict[str, dict[str, Any]]:
        """Return field metadata for UI rendering: {field_name: {description, tier, type, min, max, value}}"""
        result: dict[str, dict[str, Any]] = {}
        for name, field_info in self.model_fields.items():
            meta: dict[str, Any] = {
                "description": field_info.description or "",
                "tier": (field_info.json_schema_extra or {}).get("tier", "infra"),
                "type": field_info.annotation.__name__ if hasattr(field_info.annotation, '__name__') else str(field_info.annotation),
                "value": getattr(self, name),
                "default": field_info.default,
            }
            # Extract validation constraints
            for m in field_info.metadata:
                if hasattr(m, 'ge'):
                    meta["min"] = m.ge
                if hasattr(m, 'le'):
                    meta["max"] = m.le
            result[name] = meta
        return result

    def save_to_file(self):
        """Save mutable settings to JSON file"""
        data = {
            "confidence_threshold": self.confidence_threshold,
            "nms_threshold": self.nms_threshold,
            "capture_fps": self.capture_fps,
            "frame_rate": self.frame_rate,
            "track_high_thresh": self.track_high_thresh,
            "track_low_thresh": self.track_low_thresh,
            "track_buffer": self.track_buffer,
            "match_thresh": self.match_thresh,
            "capture_buffer_size": self.capture_buffer_size,
            "max_concurrent_inference": self.max_concurrent_inference,
            "batch_inference_size": self.batch_inference_size,
            "batch_collect_window_ms": self.batch_collect_window_ms,
            "cpu_worker_threads": self.cpu_worker_threads,
            "reconnect_backoff_max": self.reconnect_backoff_max,
            "stream_maxlen": self.stream_maxlen,
            "stream_publish_queue_size": self.stream_publish_queue_size,
            "use_mediamtx_relay": self.use_mediamtx_relay,
            "mediamtx_source_on_demand": self.mediamtx_source_on_demand,
            "mediamtx_source_on_demand_start_timeout": self.mediamtx_source_on_demand_start_timeout,
            "mediamtx_source_on_demand_close_after": self.mediamtx_source_on_demand_close_after,
            "shm_enabled": self.shm_enabled,
            "shm_dir": self.shm_dir,
            "shm_slots_per_camera": self.shm_slots_per_camera,
            "shm_publish_jpeg": self.shm_publish_jpeg,
            "snapshot_cache_ttl": self.snapshot_cache_ttl,
            "idle_frame_publish_interval_s": self.idle_frame_publish_interval_s,
            "camera_poll_interval": self.camera_poll_interval,
            "camera_query_timeout_s": self.camera_query_timeout_s,
            "camera_query_retries": self.camera_query_retries,
            "camera_query_retry_delay_s": self.camera_query_retry_delay_s,
        }
        os.makedirs(os.path.dirname(SETTINGS_FILE), exist_ok=True)
        with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    def load_from_file(self):
        """Load settings from JSON file if exists, with Pydantic validation."""
        if os.path.exists(SETTINGS_FILE):
            try:
                with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                current = self.model_dump() if hasattr(self, 'model_dump') else self.dict()
                current.update({k: v for k, v in data.items() if hasattr(self, k)})
                try:
                    validated = type(self)(**current)
                    for key in data:
                        if hasattr(validated, key):
                            setattr(self, key, getattr(validated, key))
                except Exception as ve:
                    logger.warning("Settings file validation error: %s — using defaults", ve)
            except Exception as e:
                logger.warning("Failed to load settings from %s: %s", SETTINGS_FILE, e)


_settings_instance: Optional[Settings] = None

def get_settings() -> Settings:
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings()
        _settings_instance.load_from_file()
    return _settings_instance
