"""
Redis stream consumer for omni-fusion.
Consumes ``{prefix}:vehicles`` and ``{prefix}:humans``.
"""
import asyncio
import logging
import time
from collections import defaultdict
from typing import Dict, List, Optional

import redis.asyncio as aioredis

from app.config import get_settings
from app.spatial_engine import DetectionEvent

logger = logging.getLogger("omni-fusion.consumer")


class FusionConsumer:
    """
    Consumes enriched events from omni-vehicle and omni-human.

    Stream layout (prefix ``omni`` by default):
      omni:vehicles — plate + vehicle metadata
      omni:humans — face / person recognition
    """

    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self._redis: Optional[aioredis.Redis] = None
        self._connected = False
        # Buffer: camera_id -> list of DetectionEvent
        self._event_buffer: Dict[str, List[DetectionEvent]] = defaultdict(list)
        self._buffer_timestamps: Dict[str, float] = {}
        # Track pending ACKs per camera: camera_id -> list of (stream_name, msg_id)
        self._camera_pending_acks: Dict[str, List[tuple]] = defaultdict(list)

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def connect(self):
        try:
            self._redis = aioredis.from_url(
                self.settings.redis_streams_url,
                decode_responses=False,
            )
            await self._redis.ping()
            self._connected = True
            logger.info("✅ Connected to Redis Streams")
        except Exception as e:
            logger.error("❌ Redis connection failed: %s", e)
            self._connected = False

    async def disconnect(self):
        if self._redis:
            await self._redis.aclose()
            self._connected = False

    async def ensure_groups(self):
        """Create consumer groups if not exist."""
        if not self._connected:
            return

        prefix = self.settings.stream_prefix
        streams = [
            f"{prefix}:vehicles",
            f"{prefix}:humans",
        ]
        group = self.settings.consumer_group

        for stream in streams:
            try:
                await self._redis.xgroup_create(stream, group, id="$", mkstream=True)
                logger.info("Created group %s on %s", group, stream)
            except aioredis.ResponseError as e:
                if "BUSYGROUP" not in str(e):
                    logger.warning("Group creation error: %s", e)

    async def consume_loop(self, callback):
        """
        Main consumption loop. Reads from multiple streams via consumer group.
        Calls callback(camera_id, events) when a time window expires.
        
        If not initially connected, retries every 5s instead of returning silently.
        
        Args:
            callback: async function(camera_id: str, events: List[DetectionEvent])
        """
        # Retry connection if not initially connected (BUG FIX: was returning silently)
        while not self._connected:
            logger.warning("Fusion consumer not connected — retrying in 5s...")
            await asyncio.sleep(5)
            try:
                await self.connect()
                if self._connected:
                    await self.ensure_groups()
            except Exception as e:
                logger.warning("Fusion reconnect failed: %s", e)

        prefix = self.settings.stream_prefix
        group = self.settings.consumer_group
        consumer = self.settings.consumer_name

        stream_keys = [
            f"{prefix}:vehicles",
            f"{prefix}:humans",
        ]

        # ── PEL recovery: re-read previously delivered but un-ACK'd messages ──
        try:
            pel_streams = {s: "0" for s in stream_keys}
            pel_results = await self._redis.xreadgroup(
                group, consumer,
                streams=pel_streams,
                count=200,
            )
            pel_count = 0
            if pel_results:
                for stream_name, messages in pel_results:
                    stream_str = stream_name.decode() if isinstance(stream_name, bytes) else stream_name
                    for msg_id, data in messages:
                        event = self._parse_event(stream_str, data)
                        if event:
                            self._event_buffer[event.camera_id].append(event)
                            if event.camera_id not in self._buffer_timestamps:
                                self._buffer_timestamps[event.camera_id] = time.time()
                        # PEL recovery: accumulate in _camera_pending_acks (same dict
                        # used by the main loop) so flush→ACK actually finds them.
                        cam_key = event.camera_id if event else "__unknown__"
                        self._camera_pending_acks[cam_key].append((stream_name, msg_id))
                        pel_count += 1
            if pel_count:
                logger.info("♻️ Recovered %d un-ACK'd messages from PEL", pel_count)
        except Exception as e:
            logger.warning("PEL recovery skipped: %s", e)

        streams = {s: ">" for s in stream_keys}

        logger.info("🔄 Fusion consumer loop started (window=%.1fs)", self.settings.temporal_window_sec)

        while True:
            try:
                # XREADGROUP: block for 1s, read up to 50 messages
                results = await self._redis.xreadgroup(
                    group, consumer,
                    streams=streams,
                    count=50,
                    block=1000,
                )

                if results:
                    for stream_name, messages in results:
                        stream_str = stream_name.decode() if isinstance(stream_name, bytes) else stream_name
                        for msg_id, data in messages:
                            event = self._parse_event(stream_str, data)
                            if event:
                                self._event_buffer[event.camera_id].append(event)
                                if event.camera_id not in self._buffer_timestamps:
                                    self._buffer_timestamps[event.camera_id] = time.time()
                            # Defer ACK: accumulate msg_id per-camera for callback-tied ACK
                            cam_key = event.camera_id if event else "__unknown__"

                            self._camera_pending_acks[cam_key].append((stream_name, msg_id))

                # Check time windows and flush
                now = time.time()
                cameras_to_flush = []
                for cam_id, start_time in list(self._buffer_timestamps.items()):
                    if now - start_time >= self.settings.temporal_window_sec:
                        cameras_to_flush.append(cam_id)

                for cam_id in cameras_to_flush:
                    events = self._event_buffer.pop(cam_id, [])
                    self._buffer_timestamps.pop(cam_id, None)
                    if events:
                        try:
                            await callback(cam_id, events)
                            # ACK only after successful callback for this camera
                            cam_acks = self._camera_pending_acks.pop(cam_id, [])
                            for stream_key, mid in cam_acks:
                                try:
                                    await self._redis.xack(stream_key, group, mid)
                                except Exception as e:
                                    logger.warning("ACK failed for %s: %s", mid, e)
                        except Exception as e:
                            logger.error("Fusion callback error for %s: %s — msgs will retry from PEL", cam_id[:8], e)
                            # Do NOT ACK — messages stay in PEL for retry

                # ACK unknown (no-event) messages that don't belong to any camera
                unknown_acks = self._camera_pending_acks.pop("__unknown__", [])
                for stream_key, mid in unknown_acks:
                    try:
                        await self._redis.xack(stream_key, group, mid)
                    except Exception:
                        pass

                # Cleanup stale buffers (ACK their pending messages to prevent PEL leak)
                for cam_id in list(self._buffer_timestamps.keys()):
                    if now - self._buffer_timestamps[cam_id] > self.settings.event_buffer_ttl:
                        self._event_buffer.pop(cam_id, None)
                        self._buffer_timestamps.pop(cam_id, None)
                        stale_acks = self._camera_pending_acks.pop(cam_id, [])
                        for stream_key, mid in stale_acks:
                            try:
                                await self._redis.xack(stream_key, group, mid)
                            except Exception:
                                pass
                        if stale_acks:
                            logger.warning("Evicted stale buffer for camera %s (%d msgs ACKed)", cam_id[:8], len(stale_acks))

            except asyncio.CancelledError:
                break
            except aioredis.ConnectionError as e:
                logger.error("Redis connection lost: %s — reconnecting in 3s", e)
                self._connected = False
                await asyncio.sleep(3)
                try:
                    await self.connect()
                    if self._connected:
                        await self.ensure_groups()
                except Exception as re:
                    logger.warning("Fusion reconnect failed: %s", re)
            except aioredis.ResponseError as e:
                if "NOGROUP" in str(e):
                    logger.warning("Consumer group gone (NOGROUP) — recreating in 5s...")
                    await asyncio.sleep(5)
                    try:
                        await self.ensure_groups()
                    except Exception as eg:
                        logger.warning("ensure_groups failed: %s", eg)
                else:
                    logger.exception("Redis ResponseError in fusion consumer: %s", e)
                    await asyncio.sleep(2)
            except Exception as e:
                logger.exception("Consumer loop error: %s", e)
                await asyncio.sleep(2)

        logger.info("Fusion consumer loop stopped")

    def _parse_event(self, stream_name: str, data: dict) -> Optional[DetectionEvent]:
        """Parse a Redis Stream message into a DetectionEvent."""
        try:
            camera_id = data.get(b"camera_id", b"").decode()
            if not camera_id:
                return None

            bbox_str = data.get(b"bbox", b"").decode()
            parts = bbox_str.split(",") if bbox_str else []
            if len(parts) == 4:
                bbox = tuple(round(float(p)) for p in parts)
                # Reject zero-area or invalid bboxes
                if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                    logger.debug("Skipping event with zero-area bbox: %s", bbox_str)
                    return None
            else:
                logger.debug("Skipping event with missing/invalid bbox: '%s'", bbox_str)
                return None

            # Handle empty or invalid global_track_id gracefully
            raw_track_id = data.get(b"global_track_id", b"-1").decode().strip()
            try:
                track_id = int(raw_track_id) if raw_track_id else -1
            except (ValueError, TypeError):
                track_id = -1

            event = DetectionEvent(
                camera_id=camera_id,
                global_track_id=track_id,
                class_name=data.get(b"class_name", b"unknown").decode(),
                bbox=bbox,
                confidence=float(data.get(b"confidence", b"0").decode()),
                timestamp=float(data.get(b"timestamp", b"0").decode()),
            )

            # Enrichment fields (from LPR/FRS streams)
            if b"plate_text" in data:
                event.plate_text = data[b"plate_text"].decode()
                event.plate_confidence = float(data.get(b"plate_confidence", b"0").decode())
                event.plate_crop_path = data.get(b"plate_crop_path", b"").decode() or None
                event.full_frame_path = data.get(b"full_frame_path", b"").decode() or None

            if b"face_identity" in data:
                event.face_identity = data[b"face_identity"].decode()
                event.face_confidence = float(data.get(b"face_confidence", b"0").decode())
                event.face_embedding_id = data.get(b"face_embedding_id", b"").decode() or None
                event.face_crop_path = data.get(b"face_crop_path", b"").decode() or None

            return event

        except Exception as e:
            logger.warning("Failed to parse event: %s", e)
            return None
