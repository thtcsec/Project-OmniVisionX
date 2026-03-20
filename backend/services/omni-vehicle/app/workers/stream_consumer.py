"""
Redis stream consumer for omni-vehicle.

Reads ``{stream_prefix}:detections`` (from omni-object) via consumer group,
runs LPR, publishes to ``{stream_prefix}:vehicles`` (plate + vehicle metadata).
"""
import asyncio
import logging
import os
import time
from typing import Any, Optional

import redis.asyncio as aioredis

from app.services.core.ambient_adapter import AmbientAdapter
from app.workers.stream_detection_pipeline import DetectionPorts, LprDetectionPipeline
from app.workers.stream_detection_orchestrator import LprDetectionOrchestrator
from app.workers.stream_event_publisher import LprEventPublisher
from app.workers.stream_vehicle_vision import VehicleBboxSmoother, estimate_vehicle_color, refine_vehicle_type
from app.workers.stream_frame_manager import LprFrameManager
from app.workers.stream_backpressure_policy import LprBackpressurePolicy
from app.workers.stream_metrics_utils import (
    aggregate_global_processing_metrics,
    build_camera_metrics,
    create_camera_stat_entry,
)
from app.workers.stream_thumbnail_service import LprThumbnailService
from app.workers.stream_plate_workflow import LprPlateWorkflow
from app.workers.tracking_consensus import TrackingConsensus

logger = logging.getLogger("omni-vehicle.stream_consumer")


class LprStreamConsumer:
    """
    Consumes vehicle detections from Redis Stream, runs LPR pipeline,
    and publishes enriched plate events back to Redis.
    """

    def __init__(self, settings, process_frame_fn=None):
        """
        Args:
            settings: omni-vehicle Settings
            process_frame_fn: async callable(img_bgr, camera_id, ...) -> List[PlateResult]
        """
        self.settings = settings
        self._process_frame = process_frame_fn
        self._redis: Optional[aioredis.Redis] = None
        self._connected = False
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._stats = {
            "events_consumed": 0,
            "plates_detected": 0,
            "frames_dropped": 0,
            "errors": 0,
            "processed_frames": 0,
            "processing_time_ms_total": 0.0,
            "e2e_lag_ms_total": 0.0,
        }
        self._drop_reasons: dict[str, int] = {
            "backpressure": 0,
            "age": 0,
            "no_frame": 0,
            "invalid_bbox": 0,
            "oversized_bbox": 0,
            "blur_frame": 0,
            "empty_vehicle_crop": 0,
            "consensus_skip": 0,
        }
        self._consensus = TrackingConsensus(settings)
        self._ambient = AmbientAdapter.get_instance()
        self._drop_ratio = 0.0
        self._last_backlog_check = 0.0
        self._last_backlog_len = 0
        self._camera_stats: dict[str, dict[str, float]] = {}
        self._frame_manager = LprFrameManager(settings)
        self._thumbnail_service = LprThumbnailService(settings)
        self._event_publisher = LprEventPublisher(settings)
        self._backpressure = LprBackpressurePolicy(settings)
        self._detection = LprDetectionOrchestrator(settings, process_frame_fn=process_frame_fn)
        self._plate_workflow = LprPlateWorkflow(
            consensus=self._consensus,
            drop_reasons=self._drop_reasons,
            save_thumbnails_fn=self._thumbnail_service.save_thumbnails,
            publish_plate_event_fn=self._publish_plate_event,
            refine_vehicle_type_fn=refine_vehicle_type,
            estimate_vehicle_color_fn=estimate_vehicle_color,
        )
        self._reclaim_task: Optional[asyncio.Task] = None
        self._worker_queue: Optional[asyncio.Queue] = None
        self._worker_tasks: list[asyncio.Task] = []
        self._track_keep_budget: dict[str, tuple[float, int]] = {}
        self._bbox_smoother = VehicleBboxSmoother(
            alpha=float(os.getenv("LPR_BBOX_EMA_ALPHA", "0.4")),
            ttl_s=float(os.getenv("LPR_BBOX_EMA_TTL_S", "4.0")),
        )
        self._frame_blur_min_var = max(0.0, float(os.getenv("LPR_FRAME_BLUR_MIN_VAR", "60")))

        async def _get_frame_bgr_port(camera_id: str, detection_ts: float = 0.0, frame_stream_id: str = None):
            return await self._frame_manager.get_frame_bgr(
                self._redis,
                camera_id,
                detection_ts=detection_ts,
                frame_stream_id=frame_stream_id,
            )

        self._detection_pipeline = LprDetectionPipeline(
            settings=settings,
            ambient=self._ambient,
            process_frame_fn=self._process_frame,
            stats=self._stats,
            track_keep_budget=self._track_keep_budget,
            bbox_smoother=self._bbox_smoother,
            frame_blur_min_var=self._frame_blur_min_var,
            ports=DetectionPorts(
                get_camera_stat_entry=self._get_camera_stat_entry,
                update_camera_drop_ratio=self._update_camera_drop_ratio,
                should_drop_event=self._should_drop_event,
                record_drop=self._record_drop,
                get_frame_bgr=_get_frame_bgr_port,
                detect_plates_on_vehicle=self._detection.detect_plates_on_vehicle,
                legacy_detect_plates=self._detection.legacy_detect_plates,
                merge_plate_candidates=self._detection.merge_plate_candidates,
                handle_plate_candidate=self._plate_workflow.handle_candidate,
                get_backlog_len=lambda: int(self._last_backlog_len),
            ),
        )

    @property
    def _sp(self) -> str:
        return getattr(self.settings, "stream_prefix", "omni")

    @property
    def _detections_stream(self) -> str:
        return f"{self._sp}:detections"

    @property
    def _consumer_group(self) -> str:
        return getattr(self.settings, "redis_consumer_group", "omni-vehicle-group")

    @property
    def stats(self):
        return self._stats

    def _record_drop(self, reason: str, cam_stat: Optional[dict[str, float]] = None) -> None:
        self._stats["frames_dropped"] += 1
        self._drop_reasons[reason] = int(self._drop_reasons.get(reason, 0)) + 1
        if cam_stat is not None:
            cam_stat["dropped"] = float(cam_stat.get("dropped", 0.0)) + 1.0

    async def connect(self):
        redis_url = getattr(self.settings, 'redis_streams_url', None) or os.getenv("REDIS_STREAMS_URL", "redis://redis-streams:6379")
        try:
            self._redis = aioredis.from_url(redis_url, decode_responses=False)
            await self._redis.ping()
            self._connected = True
            logger.info("✅ omni-vehicle connected to Redis Streams")
        except Exception as e:
            logger.warning("⚠️ Redis Streams unavailable: %s (LPR will use pull-mode only)", e)
            self._connected = False

    async def disconnect(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._reclaim_task:
            self._reclaim_task.cancel()
            try:
                await self._reclaim_task
            except asyncio.CancelledError:
                pass
        for worker_task in self._worker_tasks:
            worker_task.cancel()
        if self._worker_tasks:
            await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        self._worker_tasks = []
        self._worker_queue = None
        if self._redis:
            await self._redis.aclose()
            self._connected = False

    async def ensure_group(self):
        if not self._connected:
            return
        try:
            await self._redis.xgroup_create(
                self._detections_stream, self._consumer_group, id="$", mkstream=True
            )
        except aioredis.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                logger.warning("Group creation error: %s", e)

    async def start(self):
        if not self._connected or self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._consume_loop())
        self._reclaim_task = asyncio.create_task(self._reclaim_loop())
        logger.info("🔄 LPR Stream Consumer started")

    async def _consume_loop(self):
        """XREADGROUP detections → LPR → publish plates stream."""
        consumer_name = f"lpr-worker-{os.getpid()}"
        max_concurrent = int(getattr(self.settings, "lpr_max_concurrent_tasks", 10))
        queue_size = int(getattr(self.settings, "lpr_worker_queue_size", 400))
        queue_size = max(queue_size, max_concurrent * 2)
        self._worker_queue = asyncio.Queue(maxsize=queue_size)

        self._worker_tasks = [
            asyncio.create_task(self._worker_loop(consumer_name, worker_idx))
            for worker_idx in range(max_concurrent)
        ]

        while self._running:
            try:
                results = await self._redis.xreadgroup(
                    self._consumer_group, consumer_name,
                    streams={self._detections_stream: ">"},
                    count=20,
                    block=1000,
                )

                if not results:
                    continue

                await self._update_backpressure_state()
                self._evict_stale_caches()
                for stream_name, messages in results:
                    for msg_id, data in messages:
                        await self._worker_queue.put((stream_name, msg_id, data))

            except asyncio.CancelledError:
                break
            except aioredis.ConnectionError as e:
                logger.error("Redis connection lost: %s — reconnecting in 3s", e)
                self._connected = False
                await asyncio.sleep(3)
                try:
                    await self.connect()
                    if self._connected:
                        await self.ensure_group()
                except Exception as re:
                    logger.warning("LPR reconnect failed: %s", re)
            except aioredis.ResponseError as e:
                # NOGROUP: stream was wiped or consumer group was dropped
                # (happens when Redis restarts or omni-object hasn't published yet)
                if "NOGROUP" in str(e):
                    logger.warning("Consumer group gone (NOGROUP) — recreating group and retrying in 5s...")
                    await asyncio.sleep(5)
                    try:
                        await self.ensure_group()
                    except Exception as eg:
                        logger.warning("ensure_group failed: %s", eg)
                else:
                    logger.exception("Redis ResponseError in consumer loop: %s", e)
                    await asyncio.sleep(2)
            except Exception as e:
                logger.exception("Consumer loop error: %s", e)
                await asyncio.sleep(2)

        if self._worker_queue is not None:
            try:
                await asyncio.wait_for(self._worker_queue.join(), timeout=5.0)
            except Exception:
                pass
        if self._worker_tasks:
            for worker_task in self._worker_tasks:
                worker_task.cancel()
            await asyncio.gather(*self._worker_tasks, return_exceptions=True)
            self._worker_tasks = []

    async def _worker_loop(self, consumer_name: str, worker_idx: int):
        worker_name = f"{consumer_name}-{worker_idx}"
        while self._running or (self._worker_queue is not None and not self._worker_queue.empty()):
            if self._worker_queue is None:
                await asyncio.sleep(0.1)
                continue
            try:
                stream_name, msg_id, data = await asyncio.wait_for(self._worker_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            try:
                await self._process_detection(msg_id, data)
                self._stats["events_consumed"] += 1
                await self._redis.xack(stream_name, self._consumer_group, msg_id)
            except Exception as e:
                logger.warning("Process error in %s (msg %s will retry): %s", worker_name, msg_id, e)
                self._stats["errors"] += 1
            finally:
                self._worker_queue.task_done()

    async def _process_detection(self, msg_id, data: dict):
        await self._detection_pipeline.process_detection(msg_id, data, global_drop_ratio=self._drop_ratio)

    async def _reclaim_loop(self):
        """Background task: reclaim messages stuck in PEL (crashed workers).

        Uses XAUTOCLAIM to reclaim messages idle > 30s, then re-processes them.
        Without this, messages from crashed workers stay in PEL forever.
        """
        consumer_name = f"lpr-reclaim-{os.getpid()}"
        reclaim_idle_ms = 30_000  # 30 seconds
        reclaim_interval = 10.0   # check every 10 seconds

        while self._running:
            try:
                await asyncio.sleep(reclaim_interval)
                if not self._connected or not self._redis:
                    continue

                # XAUTOCLAIM: claim messages idle > reclaim_idle_ms
                # Returns: (next_start_id, [(msg_id, data), ...], [deleted_ids])
                result = await self._redis.xautoclaim(
                    self._detections_stream,
                    self._consumer_group,
                    consumer_name,
                    min_idle_time=reclaim_idle_ms,
                    start_id="0-0",
                    count=50,
                )
                if not result or len(result) < 2:
                    continue

                claimed_messages = result[1]
                if not claimed_messages:
                    continue

                logger.info("Reclaimed %d idle PEL messages", len(claimed_messages))
                for msg_id, data in claimed_messages:
                    if not data:  # deleted message (tombstone)
                        await self._redis.xack(self._detections_stream, self._consumer_group, msg_id)
                        continue
                    try:
                        await self._process_detection(msg_id, data)
                        await self._redis.xack(self._detections_stream, self._consumer_group, msg_id)
                    except Exception as e:
                        logger.warning("Reclaim process error (msg %s): %s", msg_id, e)
                        # Leave in PEL for next reclaim cycle

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("Reclaim loop error: %s", e)
                await asyncio.sleep(5)

    async def _update_backpressure_state(self):
        ratio, checked_at, backlog = await self._backpressure.update_global_drop_ratio(
            self._redis,
            self._last_backlog_check,
        )
        self._last_backlog_check = checked_at
        if backlog is not None:
            self._last_backlog_len = int(backlog)
        if ratio is not None:
            self._drop_ratio = float(ratio)

    def _update_camera_drop_ratio(self, camera_id: str) -> float:
        entry = self._get_camera_stat_entry(camera_id)
        return self._backpressure.update_camera_drop_ratio(entry)

    def _should_drop_event(
        self,
        camera_id: str,
        msg_id: str,
        drop_ratio: float,
        track_id: int | None = None,
        timestamp: float | None = None,
    ) -> bool:
        return self._backpressure.should_drop_event(
            camera_id=camera_id,
            msg_id=msg_id,
            drop_ratio=drop_ratio,
            track_id=track_id,
            timestamp=timestamp,
        )

    def _get_camera_stat_entry(self, camera_id: str) -> dict[str, float]:
        now = time.monotonic()
        entry = self._camera_stats.get(camera_id)
        if entry is None:
            entry = create_camera_stat_entry(now)
            self._camera_stats[camera_id] = entry
        return entry

    def _evict_stale_caches(self) -> None:
        now = time.monotonic()
        stale_frame_count = self._frame_manager.evict_stale_cache()
        stale_stat_keys = [
            cam_id for cam_id, entry in self._camera_stats.items()
            if now - float(entry.get("last_update", entry.get("start", now))) > 120.0
        ]
        for cam_id in stale_stat_keys:
            del self._camera_stats[cam_id]

        stale_track_keys = [
            track_key for track_key, (updated_at, _) in self._track_keep_budget.items()
            if now - float(updated_at) > 180.0
        ]
        for track_key in stale_track_keys:
            del self._track_keep_budget[track_key]

        stale_bbox_count = self._bbox_smoother.evict_stale(now)

        if stale_frame_count or stale_stat_keys or stale_track_keys or stale_bbox_count:
            logger.debug(
                "Cache eviction: %d frame caches, %d camera stats, %d track budgets, %d bbox EMA entries removed",
                stale_frame_count, len(stale_stat_keys), len(stale_track_keys), stale_bbox_count,
            )

        self._ambient.evict_stale()


    async def _publish_plate_event(
        self,
        camera_id: str,
        track_id: int,
        class_name: str,
        bbox: tuple,
        plate_text: str,
        plate_confidence: float,
        timestamp: float,
        plate_crop_path: str = "",
        full_frame_path: str = "",
        cache_entry: dict = None,
        plate_bbox: tuple | list | None = None,
        vehicle_color: str | None = None,
    ):
        await self._event_publisher.publish_plate_event(
            redis_client=self._redis,
            redis_connected=self._connected,
            camera_id=camera_id,
            track_id=track_id,
            class_name=class_name,
            bbox=bbox,
            plate_text=plate_text,
            plate_confidence=plate_confidence,
            timestamp=timestamp,
            plate_crop_path=plate_crop_path,
            full_frame_path=full_frame_path,
            cache_entry=cache_entry,
            plate_bbox=plate_bbox,
            vehicle_color=vehicle_color,
        )

    def get_metrics_snapshot(self) -> dict[str, Any]:
        avg_processing_ms, avg_e2e_lag_ms = aggregate_global_processing_metrics(self._stats)

        queue_depth = self._worker_queue.qsize() if self._worker_queue is not None else 0
        queue_max = self._worker_queue.maxsize if self._worker_queue is not None else 0

        cameras = build_camera_metrics(self._camera_stats)

        return {
            "consumer_running": self._running,
            "redis_connected": self._connected,
            "backlog_len": int(self._last_backlog_len),
            "global_drop_ratio": round(float(self._drop_ratio), 3),
            "drop_reasons": {k: int(v) for k, v in self._drop_reasons.items()},
            "queue_depth": int(queue_depth),
            "queue_capacity": int(queue_max),
            "frame_cache_entries": self._frame_manager.cache_entries,
            "avg_processing_ms": round(avg_processing_ms, 2),
            "avg_e2e_lag_ms": round(avg_e2e_lag_ms, 2),
            "workers": len(self._worker_tasks),
            "detection_telemetry": self._detection.get_telemetry_snapshot(),
            "cameras": cameras,
        }
