"""
OmniVision omni-vehicle — configuration

SINGLE SOURCE OF TRUTH for thresholds and model configs.

Every tunable field carries:
  - description  : explains what it does (shown in UI tooltip)
  - ge / le      : validation range  (Pydantic will reject out-of-range)
  - json_schema_extra.tier : "live" | "restart" | "infra"
        live    → hot-reloadable without restart
        restart → needs worker restart (not container)
        infra   → needs container restart
"""
import json
import logging
import os
import tempfile
import threading
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings
from typing import Any

SETTINGS_FILE = "/app/config/settings.json"
logger = logging.getLogger("omni-vehicle.config")

# ── Tier constants ────────────────────────────────────────────
_LIVE = {"tier": "live"}
_RESTART = {"tier": "restart"}
_INFRA = {"tier": "infra"}


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # ── Infrastructure (tier: infra) ──────────────────────────
    database_url: str = Field(
        default="postgresql+asyncpg://postgres:postgres@omni-db:5432/omnivision",
        description="Chuỗi kết nối PostgreSQL (thay đổi cần restart container)",
        json_schema_extra=_INFRA,
    )
    mediamtx_hls_url: str = Field(
        default="http://media-server:8888",
        description="URL HLS stream của MediaMTX (dùng cho live preview)",
        json_schema_extra=_INFRA,
    )
    thumbnail_path: str = Field(
        default="/app/thumbnails",
        description="Thư mục lưu thumbnail biển số đã nhận diện",
        json_schema_extra=_INFRA,
    )
    lpr_draw_vehicle_bbox: bool = Field(
        default=True,
        description="Vẽ khung xanh quanh xe trên ảnh full-frame (dùng cho preview/tải xuống).",
        json_schema_extra=_LIVE,
    )
    weights_path: str = Field(
        default="/app/weights",
        description="Thư mục chứa file weights model AI",
        json_schema_extra=_INFRA,
    )
    weights_dir: str = Field(
        default="/app/weights",
        description="Alias của weights_path (dùng trong training worker)",
        json_schema_extra=_INFRA,
    )
    device: str = Field(
        default="cuda:0",
        description="Thiết bị GPU/CPU cho inference (cuda:0, cuda:1, cpu)",
        json_schema_extra=_INFRA,
    )
    redis_streams_url: str = Field(
        default="redis://omni-bus:6379",
        description="URL Redis Streams dùng làm event bus",
        json_schema_extra=_INFRA,
    )
    stream_prefix: str = Field(
        default="omni",
        description="Tiền tố Redis stream (omni:detections, omni:frames:…, omni:vehicles)",
        json_schema_extra=_INFRA,
    )
    redis_consumer_group: str = Field(
        default="omni-vehicle-group",
        description="Consumer group cho XREADGROUP trên stream detections",
        json_schema_extra=_INFRA,
    )

    s3_enabled: bool = Field(
        default=False,
        description="Bật upload ảnh event/plate lên MinIO (S3-compatible).",
        json_schema_extra=_INFRA,
    )
    s3_internal_endpoint: str = Field(
        default="omni-minio:9000",
        description="Endpoint nội bộ (container→MinIO) dạng host:port.",
        json_schema_extra=_INFRA,
    )
    s3_public_endpoint: str = Field(
        default="localhost:9000",
        description="Endpoint public (browser) dạng host:port để tạo presigned URL.",
        json_schema_extra=_INFRA,
    )
    s3_access_key: str = Field(
        default="admin",
        description="Access key MinIO (S3).",
        json_schema_extra=_INFRA,
    )
    s3_secret_key: str = Field(
        default="change_this_minio_secret",
        description="Secret key MinIO (S3).",
        json_schema_extra=_INFRA,
    )
    s3_bucket: str = Field(
        default="events",
        description="Bucket để lưu ảnh event.",
        json_schema_extra=_INFRA,
    )
    s3_secure: bool = Field(
        default=False,
        description="Dùng HTTPS cho MinIO endpoint.",
        json_schema_extra=_INFRA,
    )
    s3_prefix: str = Field(
        default="",
        description="Prefix object key trong bucket (tuỳ chọn).",
        json_schema_extra=_INFRA,
    )

    # ── YOLO parity (tier: live) ──────────────────────────────
    yolo_model: str = Field(
        default="yolov11m.pt",
        description="Tên file YOLO model (settings API parity với omni-object)",
        json_schema_extra=_INFRA,
    )
    batch_size: int = Field(
        default=4, ge=1, le=32,
        description="Kích thước batch inference YOLO. Tăng = nhanh hơn nếu đủ VRAM.",
        json_schema_extra=_RESTART,
    )
    confidence_threshold: float = Field(
        default=0.3, ge=0.05, le=0.95,
        description="Ngưỡng tin cậy YOLO (parity với omni-object, dùng cho settings API).",
        json_schema_extra=_LIVE,
    )
    nms_threshold: float = Field(
        default=0.45, ge=0.1, le=0.9,
        description="Ngưỡng Non-Maximum Suppression cho YOLO.",
        json_schema_extra=_LIVE,
    )
    plate_merge_iou_threshold: float = Field(
        default=0.55, ge=0.3, le=0.9,
        description="Ngưỡng IoU để gộp biển trùng khi merge Fortress + Legacy. "
                    "0.55 = biển gần nhau coi là trùng. Tăng nếu có nhiều biển sát nhau.",
        json_schema_extra=_LIVE,
    )

    # ── Face (tier: live) ─────────────────────────────────────
    face_confidence_threshold: float = Field(
        default=0.5, ge=0.1, le=0.99,
        description="Ngưỡng tin cậy nhận diện khuôn mặt (omni-human).",
        json_schema_extra=_LIVE,
    )
    enable_face_recognition: bool = Field(
        default=True,
        description="Bật/tắt tính năng nhận diện khuôn mặt.",
        json_schema_extra=_LIVE,
    )

    # ── Plate Detector (tier: live) ───────────────────────────
    plate_detector_model: str = Field(
        default="best.pt",
        description="Tên file model phát hiện biển số (YOLO OBB).",
        json_schema_extra=_INFRA,
    )
    plate_detector_confidence: float = Field(
        default=0.25, ge=0.05, le=0.95,
        description="Ngưỡng tin cậy phát hiện biển số. Giảm = nhận nhiều hơn nhưng "
                    "có thể false positive.",
        json_schema_extra=_LIVE,
    )
    plate_confidence_threshold: float = Field(
        default=0.25, ge=0.05, le=0.95,
        description="Ngưỡng tin cậy cuối cùng cho biển số (alias plate_detector_confidence).",
        json_schema_extra=_LIVE,
    )

    # ── Fortress Pipeline (tier: live) ────────────────────────
    fortress_vehicle_confidence: float = Field(
        default=0.12, ge=0.05, le=0.9,
        description="Ngưỡng tin cậy Fortress Vehicle Detector. 0.12 tối ưu cho cả ngày lẫn đêm. "
                    "Xe ban đêm confidence thường 0.12-0.25. CRITICAL: Giảm từ 0.20→0.12.",
        json_schema_extra=_LIVE,
    )
    fortress_plate_confidence: float = Field(
        default=0.20, ge=0.05, le=0.9,
        description="Ngưỡng tin cậy Fortress Plate Detector (OBB). 0.20 bắt thêm ~15% biển mờ/lóa. "
                    "Biển số lóa đèn pha ban đêm confidence rất thấp.",
        json_schema_extra=_LIVE,
    )
    enable_fortress_lpr: bool = Field(
        default=True,
        description="Bật/tắt pipeline Fortress LPR (STN-LPRNet). Tắt = dùng legacy PaddleOCR.",
        json_schema_extra=_LIVE,
    )
    enable_paddleocr: bool = Field(
        default=False,
        description="Bật/tắt PaddleOCR (legacy). Tắt để tránh crash trên CPU không hỗ trợ AVX.",
        json_schema_extra=_INFRA,
    )

    # ── GPU & Batch Tuning (tier: restart) ────────────────────
    lpr_max_gpu_jobs: int = Field(
        default=5, ge=1, le=32,
        description="Số job GPU chạy đồng thời tối đa. Tăng nếu GPU mạnh (RTX 4090).",
        json_schema_extra=_RESTART,
    )
    lpr_max_concurrent_tasks: int = Field(
        default=10, ge=1, le=100,
        description="Số xe tối đa được xử lý OCR đồng thời (Concurrency). Ngăn nghẽn CPU/GPU.",
        json_schema_extra=_RESTART,
    )
    lpr_worker_queue_size: int = Field(
        default=400, ge=50, le=5000,
        description="Kích thước queue nội bộ giữa Redis consumer và worker OCR. Giúp làm mượt burst load.",
        json_schema_extra=_RESTART,
    )
    lpr_max_batch_size: int = Field(
        default=8, ge=1, le=64,
        description="Kích thước batch tối đa cho LPR inference.",
        json_schema_extra=_RESTART,
    )
    lpr_max_batch_wait_ms: int = Field(
        default=10, ge=1, le=500,
        description="Thời gian chờ (ms) để gom đủ batch trước khi inference.",
        json_schema_extra=_LIVE,
    )
    lpr_max_concurrent_cameras: int = Field(
        default=0, ge=0, le=100,
        description="Số camera xử lý đồng thời tối đa. 0 = không giới hạn.",
        json_schema_extra=_LIVE,
    )

    # ── Backpressure Control (tier: live) ─────────────────────
    lpr_adaptive_frame_skip: bool = Field(
        default=True,
        description="Tự động bỏ frame khi backpressure cao. Tắt nếu không muốn mất frame.",
        json_schema_extra=_LIVE,
    )
    lpr_stream_backlog_threshold: int = Field(
        default=300, ge=10, le=5000,
        description="Số message tồn đọng trong Redis stream trước khi bắt đầu drop frame. "
                    "300 = 2.5s lag tolerance @ 120 detections/s (10 cameras × 12 FPS). "
                    "CRITICAL: Giảm từ 700→300 để phản ứng backpressure nhanh hơn.",
        json_schema_extra=_LIVE,
    )
    lpr_stream_drop_ratio_min: float = Field(
        default=0.0, ge=0.0, le=0.5,
        description="Tỷ lệ drop frame tối thiểu khi backpressure nhẹ.",
        json_schema_extra=_LIVE,
    )
    lpr_stream_drop_ratio_max: float = Field(
        default=0.25, ge=0.0, le=0.5,
        description="Tỷ lệ drop frame TỐI ĐA khi backpressure nặng. "
                    "0.3 = giữ 70% frame dù đang bận. Constraint: tối đa 0.5.",
        json_schema_extra=_LIVE,
    )
    lpr_stream_check_interval_ms: int = Field(
        default=500, ge=100, le=5000,
        description="Chu kỳ (ms) kiểm tra trạng thái backpressure Redis stream.",
        json_schema_extra=_LIVE,
    )
    lpr_consensus_history: int = Field(
        default=12, ge=3, le=50,
        description="Số lần OCR gần nhất để tính majority vote (biển số cuối cùng).",
        json_schema_extra=_LIVE,
    )

    # ── Adaptive Batch (tier: live) ───────────────────────────
    lpr_batch_adaptive: bool = Field(
        default=True,
        description="Tự động điều chỉnh batch size theo backlog. Tắt = dùng cố định.",
        json_schema_extra=_LIVE,
    )
    lpr_batch_backlog_small: int = Field(
        default=5, ge=1, le=100,
        description="Ngưỡng backlog nhỏ — dưới giá trị này sử dụng batch nhỏ.",
        json_schema_extra=_LIVE,
    )
    lpr_batch_backlog_medium: int = Field(
        default=20, ge=5, le=500,
        description="Ngưỡng backlog trung bình — dưới giá trị này sử dụng batch vừa.",
        json_schema_extra=_LIVE,
    )
    lpr_batch_small_size: int = Field(
        default=1, ge=1, le=16,
        description="Kích thước batch khi backlog nhỏ.",
        json_schema_extra=_LIVE,
    )
    lpr_batch_small_wait_ms: int = Field(
        default=5, ge=1, le=100,
        description="Thời gian chờ batch (ms) khi backlog nhỏ.",
        json_schema_extra=_LIVE,
    )
    lpr_batch_medium_size: int = Field(
        default=4, ge=1, le=32,
        description="Kích thước batch khi backlog trung bình.",
        json_schema_extra=_LIVE,
    )
    lpr_batch_medium_wait_ms: int = Field(
        default=10, ge=1, le=200,
        description="Thời gian chờ batch (ms) khi backlog trung bình.",
        json_schema_extra=_LIVE,
    )
    lpr_batch_large_size: int = Field(
        default=8, ge=1, le=64,
        description="Kích thước batch khi backlog lớn (trên medium ngưỡng).",
        json_schema_extra=_LIVE,
    )
    lpr_batch_large_wait_ms: int = Field(
        default=20, ge=1, le=500,
        description="Thời gian chờ batch (ms) khi backlog lớn.",
        json_schema_extra=_LIVE,
    )

    # ── Per-Camera Rate Limiting (tier: live) ─────────────────
    lpr_camera_rate_window_s: float = Field(
        default=1.0, ge=0.1, le=10.0,
        description="Cửa sổ thời gian (giây) để tính tốc độ frame mỗi camera.",
        json_schema_extra=_LIVE,
    )
    lpr_camera_fps_mid: float = Field(
        default=15.0, ge=1.0, le=60.0,
        description="Ngưỡng FPS trung bình — trên giá trị này bắt đầu drop. "
                    "Default 15 (trước là 10 — quá hung hãn, miss xe ở 10 FPS camera).",
        json_schema_extra=_LIVE,
    )
    lpr_camera_fps_high: float = Field(
        default=30.0, ge=1.0, le=120.0,
        description="Ngưỡng FPS cao — trên giá trị này drop mạnh hơn. "
                    "Default 30 (trước là 20).",
        json_schema_extra=_LIVE,
    )
    lpr_camera_drop_ratio_mid: float = Field(
        default=0.15, ge=0.0, le=0.9,
        description="Tỷ lệ drop frame khi FPS camera ở mức mid. "
                    "Default 0.15 (trước là 0.3 — drop 30% ngay ở mid là quá nhiều).",
        json_schema_extra=_LIVE,
    )
    lpr_camera_drop_ratio_high: float = Field(
        default=0.4, ge=0.0, le=0.95,
        description="Tỷ lệ drop frame khi FPS camera ở mức high. "
                    "Default 0.4 (trước là 0.6 — giữ 60% frame thay vì 40%).",
        json_schema_extra=_LIVE,
    )

    # ── Warmup (tier: restart) ────────────────────────────────
    lpr_model_warmup_enabled: bool = Field(
        default=True,
        description="Chạy warmup model khi khởi động để CUDA JIT compile trước.",
        json_schema_extra=_RESTART,
    )
    lpr_model_warmup_runs: int = Field(
        default=5, ge=1, le=50,
        description="Số lần warmup inference khi khởi động.",
        json_schema_extra=_RESTART,
    )
    lpr_model_warmup_size: int = Field(
        default=640, ge=320, le=1920,
        description="Kích thước ảnh dummy cho warmup (pixel).",
        json_schema_extra=_RESTART,
    )

    # ── Shared Memory (tier: restart) ─────────────────────────
    lpr_shm_enabled: bool = Field(
        default=True,
        description="Đọc frame từ shared memory thay vì Redis. Nhanh hơn ~10x.",
        json_schema_extra=_RESTART,
    )
    lpr_shm_copy_on_read: bool = Field(
        default=False,
        description="Copy frame khi đọc từ SHM (an toàn hơn, chậm hơn).",
        json_schema_extra=_RESTART,
    )
    lpr_shm_crop_only: bool = Field(
        default=False,
        description="Khi bật, chỉ copy vùng crop từ SHM thay vì toàn bộ frame. "
                    "Giảm RAM bandwidth khi scale 15-20+ camera 1080p. "
                    "Yêu cầu bbox được truyền qua _get_frame_bgr.",
        json_schema_extra=_RESTART,
    )
    lpr_frame_cache_max_entries: int = Field(
        default=64, ge=4, le=512,
        description="Số camera tối đa giữ frame cache trong RAM. Vượt ngưỡng sẽ evict mục cũ nhất.",
        json_schema_extra=_LIVE,
    )
    lpr_frame_match_max_delta_s: float = Field(
        default=2.0, ge=0.1, le=5.0,
        description="Ngưỡng chênh lệch timestamp tối đa (giây) giữa detection và frame. "
                    "Frame lệch quá giá trị này sẽ bị drop. "
                    "2.0s cho phép xử lý delay khi hệ thống bận (10+ cameras) mà vẫn đủ chính xác. "
                    "Default cũ 0.8s quá chặt — gây miss khi processing delay cao.",
        json_schema_extra=_LIVE,
    )

    # ── OCR Thresholds (tier: live) ───────────────────────────
    ocr_confidence_threshold: float = Field(
        default=0.40, ge=0.1, le=0.99,
        description="Ngưỡng tin cậy OCR cuối cùng. Dưới giá trị này kết quả bị loại.",
        json_schema_extra=_LIVE,
    )
    ocr_min_text_length: int = Field(
        default=5, ge=3, le=12,
        description="Chiều dài tối thiểu chuỗi OCR để coi là biển số hợp lệ. "
                    "Biển số VN ngắn nhất là 7 ký tự (29A-1234), nhưng OCR có thể miss 1-2 char. "
                    "Min=5 cho phép consensus fix lại sau. CRITICAL: Giảm từ 7→5.",
        json_schema_extra=_LIVE,
    )
    lpr_ocr_rescue_enabled: bool = Field(
        default=True,
        description="Bật OCR rescue (chạy lại OCR legacy khi confidence thấp/không hợp lệ).",
        json_schema_extra=_LIVE,
    )
    lpr_ocr_rescue_conf_threshold: float = Field(
        default=0.78, ge=0.3, le=0.99,
        description="Ngưỡng confidence để kích hoạt OCR rescue (<= ngưỡng sẽ thử OCR legacy).",
        json_schema_extra=_LIVE,
    )
    lpr_sr_enable: bool = Field(
        default=True,
        description="Bật Super-Resolution có điều kiện trước OCR rescue cho crop biển nhỏ/khó.",
        json_schema_extra=_LIVE,
    )
    lpr_sr_min_height: int = Field(
        default=40, ge=16, le=256,
        description="Chiều cao (px) tối thiểu của crop biển trước khi kích hoạt SR.",
        json_schema_extra=_LIVE,
    )
    lpr_sr_min_width: int = Field(
        default=120, ge=32, le=512,
        description="Chiều rộng (px) tối thiểu của crop biển trước khi kích hoạt SR.",
        json_schema_extra=_LIVE,
    )
    lpr_sr_conf_threshold: float = Field(
        default=0.82, ge=0.0, le=1.0,
        description="Chỉ chạy SR khi confidence OCR hiện tại <= ngưỡng này.",
        json_schema_extra=_LIVE,
    )
    lpr_legacy_accept_conf_offset: float = Field(
        default=0.25, ge=0.0, le=0.5,
        description="Offset từ strict_conf để tính legacy_accept_conf: max(floor, strict_conf - offset). "
                    "Legacy OCR chấp nhận kết quả nếu conf >= legacy_accept_conf.",
        json_schema_extra=_LIVE,
    )
    lpr_legacy_accept_conf_floor: float = Field(
        default=0.12, ge=0.05, le=0.5,
        description="Sàn tối thiểu cho legacy_accept_conf (tránh quá thấp gây nhiễu).",
        json_schema_extra=_LIVE,
    )

    # ── Event Thresholds (tier: live) ─────────────────────────
    event_instant_confidence: float = Field(
        default=0.75, ge=0.5, le=1.0,
        description="Ngưỡng confidence để phát event ngay lập tức (không cần vote).",
        json_schema_extra=_LIVE,
    )
    event_min_vote_count: int = Field(
        default=2, ge=1, le=20,
        description="Số lần OCR tối thiểu trước khi phát event biển số. "
                    "2 lần cân bằng độ chính xác và tỷ lệ bắt xe nhanh.",
        json_schema_extra=_LIVE,
    )
    event_dedup_ttl: float = Field(
        default=45.0, ge=1.0, le=300.0,
        description="Thời gian (giây) chống trùng event cùng biển số. "
                    "45s tối ưu cho xe đứng lâu trước cổng. Tăng lên 60-90s nếu có barrier chậm.",
        json_schema_extra=_LIVE,
    )

    # ── Redis Event Stream (tier: live) ───────────────────────
    lpr_event_stream_maxlen: int = Field(
        default=1000, ge=100, le=50000,
        description="MAXLEN stream Redis cho LPR events.",
        json_schema_extra=_LIVE,
    )
    lpr_event_stream_approximate_trim: bool = Field(
        default=True,
        description="Dùng approximate trim (~) khi XADD MAXLEN để giảm tải Redis. Tắt để trim chính xác.",
        json_schema_extra=_LIVE,
    )
    lpr_metrics_window_s: float = Field(
        default=30.0, ge=5.0, le=300.0,
        description="Cửa sổ thời gian (giây) để tổng hợp metric camera và độ trễ xử lý.",
        json_schema_extra=_LIVE,
    )

    # ── Scheduler (tier: live) ────────────────────────────────
    snapshot_interval: int = Field(
        default=10, ge=1, le=120,
        description="Chu kỳ (giây) chụp snapshot tự động.",
        json_schema_extra=_LIVE,
    )

    # ── Feature Toggles (tier: live) ──────────────────────────
    enable_plate_ocr: bool = Field(
        default=True,
        description="Bật/tắt OCR biển số. Tắt = chỉ detect biển, không đọc ký tự.",
        json_schema_extra=_LIVE,
    )
    enable_night_lpr: bool = Field(
        default=True,
        description="Bật/tắt chế độ nhận diện ban đêm (CLAHE, retinex, IR inversion).",
        json_schema_extra=_LIVE,
    )

    # ── Adaptive Confidence (tier: live) ──────────────────────
    adaptive_confidence_base: float = Field(
        default=0.38, ge=0.1, le=0.9,
        description="Ngưỡng confidence cơ sở cho adaptive thresholding (legacy pipeline).",
        json_schema_extra=_LIVE,
    )
    adaptive_confidence_alpha: float = Field(
        default=0.16, ge=0.0, le=0.5,
        description="Hệ số giảm threshold theo quality score (legacy pipeline).",
        json_schema_extra=_LIVE,
    )
    adaptive_confidence_min: float = Field(
        default=0.18, ge=0.05, le=0.9,
        description="Ngưỡng confidence tối thiểu sau adaptive (legacy pipeline).",
        json_schema_extra=_LIVE,
    )
    adaptive_confidence_max: float = Field(
        default=0.9, ge=0.1, le=1.0,
        description="Ngưỡng confidence tối đa sau adaptive (legacy pipeline).",
        json_schema_extra=_LIVE,
    )

    # ── Active Learning / Data Collection (tier: live) ────────
    enable_lpr_data_collection: bool = Field(
        default=True,
        description="Tự động thu thập ảnh xe/biển số cho active learning.",
        json_schema_extra=_LIVE,
    )
    lpr_collection_dir: str = Field(
        default="/app/data/collect",
        description="Thư mục lưu ảnh thu thập (vehicles/ và plates/).",
        json_schema_extra=_INFRA,
    )
    lpr_collect_sample_rate: float = Field(
        default=0.5, ge=0.01, le=1.0,
        description="Tỷ lệ sampling (0.5 = lấy 50% detection). Giảm nếu đầy ổ đĩa.",
        json_schema_extra=_LIVE,
    )
    lpr_collect_vehicle: bool = Field(
        default=True,
        description="Thu thập ảnh crop xe.",
        json_schema_extra=_LIVE,
    )
    lpr_collect_plate: bool = Field(
        default=True,
        description="Thu thập ảnh crop biển số.",
        json_schema_extra=_LIVE,
    )
    lpr_collect_min_conf: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Ngưỡng confidence tối thiểu để thu thập.",
        json_schema_extra=_LIVE,
    )
    lpr_collect_max_conf: float = Field(
        default=0.7, ge=0.0, le=1.0,
        description="Ngưỡng confidence tối đa — chỉ thu thập mẫu khó (low confidence).",
        json_schema_extra=_LIVE,
    )
    lpr_collect_low_conf_only: bool = Field(
        default=True,
        description="Chỉ thu thập mẫu có confidence thấp (giữa min và max).",
        json_schema_extra=_LIVE,
    )
    lpr_collect_quality_filter: bool = Field(
        default=True,
        description="Lọc ảnh theo chất lượng (sharpness, brightness) trước khi lưu.",
        json_schema_extra=_LIVE,
    )
    lpr_collect_min_sharpness: float = Field(
        default=40.0, ge=0.0, le=200.0,
        description="Độ sắc nét tối thiểu (Laplacian variance) để thu thập ảnh.",
        json_schema_extra=_LIVE,
    )
    lpr_collect_min_brightness: float = Field(
        default=35.0, ge=0.0, le=255.0,
        description="Độ sáng tối thiểu (mean pixel) để thu thập ảnh.",
        json_schema_extra=_LIVE,
    )
    lpr_collect_max_brightness: float = Field(
        default=230.0, ge=0.0, le=255.0,
        description="Độ sáng tối đa — trên giá trị này ảnh quá sáng/lóa.",
        json_schema_extra=_LIVE,
    )
    lpr_collect_min_vehicle_area: int = Field(
        default=2500, ge=100, le=100000,
        description="Diện tích pixel tối thiểu crop xe để thu thập.",
        json_schema_extra=_LIVE,
    )
    lpr_collect_min_plate_area: int = Field(
        default=600, ge=50, le=50000,
        description="Diện tích pixel tối thiểu crop biển số để thu thập.",
        json_schema_extra=_LIVE,
    )

    class Config:
        env_file = ".env"
        case_sensitive = False

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
            for m in field_info.metadata:
                if hasattr(m, 'ge'):
                    meta["min"] = m.ge
                if hasattr(m, 'le'):
                    meta["max"] = m.le
            result[name] = meta
        return result

    def save_to_file(self):
        """Save mutable settings to JSON file"""
        data: dict[str, Any] = {
            "yolo_model": self.yolo_model,
            "batch_size": self.batch_size,
            "confidence_threshold": self.confidence_threshold,
            "nms_threshold": self.nms_threshold,
            "plate_merge_iou_threshold": self.plate_merge_iou_threshold,
            "face_confidence_threshold": self.face_confidence_threshold,
            "enable_face_recognition": self.enable_face_recognition,
            "plate_detector_model": self.plate_detector_model,
            "plate_detector_confidence": self.plate_detector_confidence,
            "plate_confidence_threshold": self.plate_confidence_threshold,
            "fortress_vehicle_confidence": self.fortress_vehicle_confidence,
            "fortress_plate_confidence": self.fortress_plate_confidence,
            "enable_fortress_lpr": self.enable_fortress_lpr,
            "lpr_max_gpu_jobs": self.lpr_max_gpu_jobs,
            "lpr_max_concurrent_tasks": self.lpr_max_concurrent_tasks,
            "lpr_worker_queue_size": self.lpr_worker_queue_size,
            "lpr_max_batch_size": self.lpr_max_batch_size,
            "lpr_max_batch_wait_ms": self.lpr_max_batch_wait_ms,
            "lpr_max_concurrent_cameras": self.lpr_max_concurrent_cameras,
            "lpr_adaptive_frame_skip": self.lpr_adaptive_frame_skip,
            "lpr_stream_backlog_threshold": self.lpr_stream_backlog_threshold,
            "lpr_stream_drop_ratio_min": self.lpr_stream_drop_ratio_min,
            "lpr_stream_drop_ratio_max": self.lpr_stream_drop_ratio_max,
            "lpr_stream_check_interval_ms": self.lpr_stream_check_interval_ms,
            "lpr_consensus_history": self.lpr_consensus_history,
            "lpr_batch_adaptive": self.lpr_batch_adaptive,
            "lpr_batch_backlog_small": self.lpr_batch_backlog_small,
            "lpr_batch_backlog_medium": self.lpr_batch_backlog_medium,
            "lpr_batch_small_size": self.lpr_batch_small_size,
            "lpr_batch_small_wait_ms": self.lpr_batch_small_wait_ms,
            "lpr_batch_medium_size": self.lpr_batch_medium_size,
            "lpr_batch_medium_wait_ms": self.lpr_batch_medium_wait_ms,
            "lpr_batch_large_size": self.lpr_batch_large_size,
            "lpr_batch_large_wait_ms": self.lpr_batch_large_wait_ms,
            "lpr_camera_rate_window_s": self.lpr_camera_rate_window_s,
            "lpr_camera_fps_mid": self.lpr_camera_fps_mid,
            "lpr_camera_fps_high": self.lpr_camera_fps_high,
            "lpr_camera_drop_ratio_mid": self.lpr_camera_drop_ratio_mid,
            "lpr_camera_drop_ratio_high": self.lpr_camera_drop_ratio_high,
            "lpr_model_warmup_enabled": self.lpr_model_warmup_enabled,
            "lpr_model_warmup_runs": self.lpr_model_warmup_runs,
            "lpr_model_warmup_size": self.lpr_model_warmup_size,
            "lpr_shm_enabled": self.lpr_shm_enabled,
            "lpr_shm_copy_on_read": self.lpr_shm_copy_on_read,
            "lpr_shm_crop_only": self.lpr_shm_crop_only,
            "lpr_frame_cache_max_entries": self.lpr_frame_cache_max_entries,
            "lpr_frame_match_max_delta_s": self.lpr_frame_match_max_delta_s,
            "ocr_confidence_threshold": self.ocr_confidence_threshold,
            "ocr_min_text_length": self.ocr_min_text_length,
            "event_instant_confidence": self.event_instant_confidence,
            "event_min_vote_count": self.event_min_vote_count,
            "event_dedup_ttl": self.event_dedup_ttl,
            "lpr_event_stream_maxlen": self.lpr_event_stream_maxlen,
            "lpr_event_stream_approximate_trim": self.lpr_event_stream_approximate_trim,
            "lpr_metrics_window_s": self.lpr_metrics_window_s,
            "enable_plate_ocr": self.enable_plate_ocr,
            "enable_night_lpr": self.enable_night_lpr,
            "adaptive_confidence_base": self.adaptive_confidence_base,
            "adaptive_confidence_alpha": self.adaptive_confidence_alpha,
            "adaptive_confidence_min": self.adaptive_confidence_min,
            "adaptive_confidence_max": self.adaptive_confidence_max,
            "enable_lpr_data_collection": self.enable_lpr_data_collection,
            "snapshot_interval": self.snapshot_interval,
            "lpr_collection_dir": self.lpr_collection_dir,
            "lpr_collect_sample_rate": self.lpr_collect_sample_rate,
            "lpr_collect_vehicle": self.lpr_collect_vehicle,
            "lpr_collect_plate": self.lpr_collect_plate,
            "lpr_collect_min_conf": self.lpr_collect_min_conf,
            "lpr_collect_max_conf": self.lpr_collect_max_conf,
            "lpr_collect_low_conf_only": self.lpr_collect_low_conf_only,
            "lpr_collect_quality_filter": self.lpr_collect_quality_filter,
            "lpr_collect_min_sharpness": self.lpr_collect_min_sharpness,
            "lpr_collect_min_brightness": self.lpr_collect_min_brightness,
            "lpr_collect_max_brightness": self.lpr_collect_max_brightness,
            "lpr_collect_min_vehicle_area": self.lpr_collect_min_vehicle_area,
            "lpr_collect_min_plate_area": self.lpr_collect_min_plate_area,
            "lpr_sr_enable": self.lpr_sr_enable,
            "lpr_sr_min_height": self.lpr_sr_min_height,
            "lpr_sr_min_width": self.lpr_sr_min_width,
            "lpr_sr_conf_threshold": self.lpr_sr_conf_threshold,
        }
        os.makedirs(os.path.dirname(SETTINGS_FILE), exist_ok=True)
        # Atomic write: write to temp file first, then rename
        dir_name = os.path.dirname(SETTINGS_FILE)
        fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".json.tmp")
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            # os.replace is atomic on POSIX; on Windows it's as close as we get
            os.replace(tmp_path, SETTINGS_FILE)
        except Exception:
            # Clean up temp file on failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def load_from_file(self):
        """Load settings from JSON file if exists, with Pydantic validation."""
        if os.path.exists(SETTINGS_FILE):
            try:
                with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                # Merge file data with current values, validate via Pydantic
                current = self.model_dump() if hasattr(self, 'model_dump') else self.dict()
                current.update({k: v for k, v in data.items() if hasattr(self, k)})
                try:
                    validated = type(self)(**current)
                    for key in data:
                        if hasattr(validated, key):
                            setattr(self, key, getattr(validated, key))
                except Exception as ve:
                    logger.warning("Settings file validation error: %s — using defaults", ve)
            except Exception:
                logger.warning("Failed to load settings from %s", SETTINGS_FILE, exc_info=True)


_settings_instance: Settings | None = None
_settings_lock = threading.Lock()


def _resolve_thumbnail_path(path: str) -> str:
    candidates: list[str] = []
    if path:
        candidates.append(path)
    if path == "/app/thumbnails":
        parents = Path(__file__).resolve().parents
        project_root = parents[3] if len(parents) > 3 else parents[-1]
        candidates.append(str(project_root / "data" / "thumbnails"))
    temp_path = os.path.join(tempfile.gettempdir(), "thumbnails")
    if temp_path not in candidates:
        candidates.append(temp_path)
    for candidate in candidates:
        try:
            os.makedirs(candidate, exist_ok=True)
            probe = tempfile.NamedTemporaryFile(dir=candidate, delete=True)
            probe.close()
            return candidate
        except Exception:
            continue
    return candidates[0] if candidates else "/app/thumbnails"


def get_settings() -> Settings:
    global _settings_instance
    if _settings_instance is None:
        with _settings_lock:
            if _settings_instance is None:
                s = Settings()
                s.load_from_file()
                s.thumbnail_path = _resolve_thumbnail_path(s.thumbnail_path)
                _settings_instance = s
    return _settings_instance

