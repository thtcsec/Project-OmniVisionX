"""
Dahua SDK Manager - Auto-initialize Dahua LPR Bridge for all cameras
=====================================================================
Migrated from root ai-engine to omni-vehicle microservice.

Reads camera list from database, parses RTSP URLs to extract credentials,
and starts Dahua SDK event listeners for each Dahua camera.

Integration Flow:
1. Query "Cameras" table for all Online cameras with RTSP URLs
2. Parse RTSP URL to extract IP, username, password
3. Start DahuaLPRBridge for each camera
4. Bridge pushes LPR events to HybridBroker (omni-vehicle internal)
"""
import asyncio
import logging
import os
import re
from typing import Optional, Dict, List
from urllib.parse import urlparse, unquote
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text

from app.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class DahuaCameraConfig:
    """Parsed camera configuration for Dahua SDK"""
    camera_id: str
    ip: str
    port: int
    username: str
    password: str
    channel: int = 0
    rtsp_url: str = ""


def parse_rtsp_url(rtsp_url: str) -> Optional[Dict]:
    """
    Parse RTSP URL to extract credentials.

    Supports formats:
    - rtsp://user:pass@192.168.1.100:554/cam/realmonitor?channel=1&subtype=0
    - rtsp://admin:Admin123@10.0.0.50/stream1
    """
    if not rtsp_url or not rtsp_url.startswith("rtsp://"):
        return None

    try:
        parsed = urlparse(rtsp_url)
        username = unquote(parsed.username) if parsed.username else "admin"
        password = unquote(parsed.password) if parsed.password else ""
        ip = parsed.hostname
        port = parsed.port or 554

        if not ip:
            return None

        channel = 0
        if parsed.query:
            match = re.search(r'channel=(\d+)', parsed.query)
            if match:
                channel = int(match.group(1)) - 1

        return {
            "ip": ip,
            "port": port,
            "username": username,
            "password": password,
            "channel": channel
        }
    except Exception as e:
        logger.error(f"Failed to parse RTSP URL: {e}")
        return None


class DahuaSDKManager:
    """
    Manages Dahua SDK connections for all cameras in the system.
    Auto-discovers cameras from database and starts LPR bridges.
    """

    SDK_PORT = int(os.getenv("DAHUA_SDK_PORT", "37777"))

    def __init__(self, hybrid_broker=None):
        self.settings = get_settings()
        self.db_engine = create_async_engine(self.settings.database_url)
        self._bridges: Dict[str, 'DahuaLPRBridge'] = {}
        self._running = False
        self._refresh_task: Optional[asyncio.Task] = None
        self._refresh_interval = 300  # 5 minutes
        self._hybrid_broker = hybrid_broker

    def set_hybrid_broker(self, broker):
        """Set HybridBroker for all bridges"""
        self._hybrid_broker = broker
        for bridge in self._bridges.values():
            bridge.set_hybrid_broker(broker)

    async def start(self):
        """Start all Dahua bridges for cameras in database"""
        if self._running:
            return

        self._running = True
        try:
            self.main_loop = asyncio.get_running_loop()
        except RuntimeError:
            self.main_loop = None
            
        logger.info("🔌 Starting Dahua SDK Manager (omni-vehicle)...")

        await self._discover_and_connect_cameras()

        self._refresh_task = asyncio.create_task(self._refresh_loop())

        logger.info(f"✅ Dahua SDK Manager started ({len(self._bridges)} cameras)")

    async def stop(self):
        """Stop all Dahua bridges"""
        self._running = False

        if self._refresh_task:
            self._refresh_task.cancel()
            try:
                await self._refresh_task
            except asyncio.CancelledError:
                pass

        for camera_id, bridge in self._bridges.items():
            try:
                bridge.stop()
                logger.info(f"⏹️ Stopped Dahua bridge: {camera_id[:8]}")
            except Exception as e:
                logger.error(f"Error stopping bridge {camera_id[:8]}: {e}")

        self._bridges.clear()
        # Dispose DB engine to release connections
        try:
            await self.db_engine.dispose()
        except Exception as e:
            logger.error("Error disposing DB engine: %s", e)
        logger.info("🔴 Dahua SDK Manager stopped")

    async def _refresh_loop(self):
        """Periodically check for new cameras"""
        while self._running:
            await asyncio.sleep(self._refresh_interval)
            try:
                await self._discover_and_connect_cameras()
            except Exception as e:
                logger.error(f"Camera refresh error: {e}")

    async def _discover_and_connect_cameras(self):
        """Query database and connect to new Dahua cameras"""
        cameras = await self._get_dahua_cameras()

        if not cameras:
            logger.warning("No Dahua cameras found in database")
            return

        for config in cameras:
            if config.camera_id in self._bridges:
                continue

            success = await self._connect_camera(config)
            if success:
                logger.info(f"✅ Connected Dahua camera: {config.camera_id[:8]} @ {config.ip}")
            else:
                logger.warning(f"⚠️ Failed to connect: {config.camera_id[:8]} @ {config.ip}")

    async def _get_dahua_cameras(self) -> List[DahuaCameraConfig]:
        """Get list of Dahua cameras from database."""
        cameras = []

        try:
            async with self.db_engine.connect() as conn:
                result = await conn.execute(text('''
                    SELECT "Id", "StreamUrl", "Name"
                    FROM "Cameras"
                    WHERE LOWER(TRIM("Status")) = 'online' AND "StreamUrl" IS NOT NULL
                '''))
                rows = result.fetchall()

                for row in rows:
                    camera_id = str(row[0])
                    stream_url = row[1]
                    name = row[2] or ""

                    parsed = parse_rtsp_url(stream_url)
                    if not parsed:
                        continue

                    is_dahua = (
                        "dahua" in name.lower() or
                        "realmonitor" in stream_url.lower() or
                        "/cam/realmonitor" in stream_url.lower()
                    )

                    if is_dahua:
                        cameras.append(DahuaCameraConfig(
                            camera_id=camera_id,
                            ip=parsed["ip"],
                            port=self.SDK_PORT,
                            username=parsed["username"],
                            password=parsed["password"],
                            channel=parsed["channel"],
                            rtsp_url=stream_url
                        ))

        except Exception as e:
            logger.error(f"Database query error: {e}")

        return cameras

    async def _connect_camera(self, config: DahuaCameraConfig) -> bool:
        """Connect to a single Dahua camera and start LPR bridge"""
        try:
            from app.services.integration.camera.dahua_lpr_bridge import DahuaLPRBridge

            bridge = DahuaLPRBridge(
                camera_ip=config.ip,
                camera_port=config.port,
                username=config.username,
                password=config.password,
                channel=config.channel,
                use_fortress_verify=True,
                save_images=True
            )

            bridge._camera_ip_map[config.ip] = config.camera_id

            # Wire HybridBroker and Loop
            if self._hybrid_broker:
                bridge.set_hybrid_broker(self._hybrid_broker)
            
            if hasattr(self, 'main_loop') and self.main_loop:
                bridge.main_loop = self.main_loop

            def on_plate_detected(event):
                logger.info(f"🚗 [{config.camera_id[:8]}] Dahua LPR: {event.plate_number} (conf={event.confidence:.2f})")

            bridge.on_plate_detected(on_plate_detected)

            loop = asyncio.get_running_loop()

            def start_bridge():
                try:
                    success = bridge.start_listening()
                    return success
                except Exception as e:
                    logger.error(f"Bridge start error: {e}")
                    return False

            success = await loop.run_in_executor(None, start_bridge)

            if success:
                self._bridges[config.camera_id] = bridge
                return True
            else:
                return False

        except ImportError as e:
            logger.error(f"Dahua SDK not available: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to connect camera {config.camera_id[:8]}: {e}")
            return False

    def get_stats(self) -> Dict:
        """Get statistics for all bridges"""
        stats = {
            "total_cameras": len(self._bridges),
            "cameras": {}
        }

        for camera_id, bridge in self._bridges.items():
            try:
                stats["cameras"][camera_id[:8]] = bridge.get_stats()
            except Exception:
                pass

        return stats


# Singleton instance
_sdk_manager: Optional[DahuaSDKManager] = None


def get_dahua_sdk_manager(hybrid_broker=None) -> DahuaSDKManager:
    """Get or create singleton DahuaSDKManager instance"""
    global _sdk_manager
    if _sdk_manager is None:
        _sdk_manager = DahuaSDKManager(hybrid_broker=hybrid_broker)
    return _sdk_manager
