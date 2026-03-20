"""
🌉 Dahua LPR Bridge - Tích hợp Dahua Camera LPR với Fortress LPR
================================================================
Migrated from root ai-engine to omni-vehicle microservice.

Hybrid mode: Sử dụng LPR onboard của camera Dahua + verify bằng Fortress

Workflow:
1. Camera Dahua detect biển số → push event qua SDK
2. Nếu confidence cao (>=85%) → accept ngay
3. Nếu confidence thấp → gọi Fortress LPR verify
4. Sync với database OmniVision

Usage:
    bridge = DahuaLPRBridge(camera_ip="192.168.1.100")
    bridge.start_listening()
"""

import os
import sys
import time
import logging
import threading
from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, List, Tuple
from datetime import datetime
from pathlib import Path
from queue import Empty as _QueueEmpty, Queue
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DahuaPlateEvent:
    """Event từ camera Dahua khi detect biển số"""
    timestamp: datetime
    plate_number: str
    plate_color: str  # white, yellow, blue, black, green
    vehicle_type: str  # car, motorcycle, truck, bus
    vehicle_color: str
    confidence: float  # 0.0 - 1.0
    channel: int

    # Images
    global_image: Optional[bytes] = None
    plate_image: Optional[bytes] = None

    # Camera info
    camera_ip: str = ""
    camera_name: str = ""

    # Fortress verification
    fortress_verified: bool = False
    fortress_plate: str = ""
    fortress_confidence: float = 0.0


class DahuaLPRBridge:
    """
    Bridge giữa Dahua Camera SDK và OmniVision (omni-vehicle microservice)

    Features:
    - Subscribe traffic events từ camera Dahua
    - Auto-verify với Fortress LPR khi confidence thấp
    - Callback khi có plate detection
    - Inject events into HybridBroker for consensus
    - Lưu ảnh local nếu cần
    """

    # Confidence threshold để accept without verification
    HIGH_CONFIDENCE_THRESHOLD = float(os.getenv("DAHUA_HIGH_CONFIDENCE_THRESHOLD", "0.85"))

    def __init__(
        self,
        camera_ip: str,
        camera_port: int = 37777,
        username: str = "admin",
        password: str = "admin123",
        channel: int = 0,
        use_fortress_verify: bool = True,
        save_images: bool = True,
        image_save_dir: str = "captured_plates",
    ):
        self.camera_ip = camera_ip
        self.camera_port = camera_port
        self.username = username
        self.password = password
        self.channel = channel
        self.use_fortress_verify = use_fortress_verify
        self.save_images = save_images
        self.image_save_dir = Path(image_save_dir)

        # SDK handles
        self.sdk = None
        self.login_id = None
        self.attach_id = None

        # Callbacks
        self._on_plate_detected: Optional[Callable[[DahuaPlateEvent], None]] = None

        # Map camera IP -> camera ID (set by manager)
        self._camera_ip_map: Dict[str, str] = {}

        # Event queue for async processing
        self.event_queue: Queue = Queue()

        # Fortress LPR instance (lazy load)
        self._fortress_lpr = None

        # HybridBroker reference (set by manager)
        self._hybrid_broker = None

        # PHASE 0 FIX: Auto-reconnect with exponential backoff
        self._reconnect_thread: Optional[threading.Thread] = None
        self._reconnect_running = False
        self._reconnect_attempt = 0
        self._max_reconnect_attempts = 10
        self._base_backoff = 2.0
        self._max_backoff = 300.0
        self._is_connected = False

        # Processing thread (initialized in _start_processing_thread)
        self._processing_running = False
        self._processing_thread: Optional[threading.Thread] = None

        # Stats
        self.stats = {
            'total_events': 0,
            'high_confidence': 0,
            'low_confidence': 0,
            'fortress_verified': 0,
            'fortress_corrected': 0,
            'reconnect_attempts': 0,
            'reconnect_success': 0,
        }

        if save_images:
            self.image_save_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"🌉 DahuaLPRBridge initialized for {camera_ip}")

    def set_hybrid_broker(self, broker):
        """Set HybridBroker reference for event injection"""
        self._hybrid_broker = broker

    @property
    def fortress_lpr(self):
        """Lazy load Fortress LPR"""
        if self._fortress_lpr is None and self.use_fortress_verify:
            try:
                from app.services.pipeline.orchestration.fortress_lpr import FortressLPR
                device = "cuda"
                try:
                    import torch
                    if not torch.cuda.is_available():
                        device = "cpu"
                except Exception:
                    device = "cpu"
                self._fortress_lpr = FortressLPR(device=device)
                logger.info("✅ Fortress LPR loaded for verification")
            except Exception as e:
                logger.warning(f"⚠️ Cannot load Fortress LPR: {e}")
                self.use_fortress_verify = False
        return self._fortress_lpr

    def on_plate_detected(self, callback: Callable[[DahuaPlateEvent], None]):
        """Register callback khi detect được biển số"""
        self._on_plate_detected = callback

    def _init_sdk(self) -> bool:
        """Initialize Dahua NetSDK"""
        try:
            self._ensure_sdk_on_path()
            from NetSDK.NetSDK import NetClient
            from NetSDK.SDK_Callback import fDisConnect, fHaveReConnect

            self.sdk = NetClient()

            def disconnect_cb(lLoginID, pchDVRIP, nDVRPort, dwUser):
                logger.warning(f"⚠️ Camera disconnected: {self.camera_ip}")
                self._is_connected = False
                self._trigger_reconnect()

            def reconnect_cb(lLoginID, pchDVRIP, nDVRPort, dwUser):
                logger.info(f"✅ Camera reconnected: {self.camera_ip}")
                self._is_connected = True
                self._reconnect_attempt = 0
                self.stats['reconnect_success'] += 1

            self._disconnect_cb = fDisConnect(disconnect_cb)
            self._reconnect_cb = fHaveReConnect(reconnect_cb)

            self.sdk.InitEx(self._disconnect_cb)
            self.sdk.SetAutoReconnect(self._reconnect_cb)

            logger.info("✅ Dahua NetSDK initialized")
            return True

        except ImportError:
            logger.error("❌ NetSDK not found. Install: pip install NetSDK")
            return False
        except Exception as e:
            logger.error(f"❌ SDK init failed: {e}")
            return False

    def _ensure_sdk_on_path(self) -> None:
        """Ensure Dahua NetSDK wheel is discoverable in PYTHONPATH."""
        sdk_dir = os.getenv("DAHUA_SDK_PATH", "/app/dahua_sdk")
        if not os.path.isdir(sdk_dir):
            return

        for filename in os.listdir(sdk_dir):
            if filename.endswith(".whl"):
                wheel_path = os.path.join(sdk_dir, filename)
                if wheel_path not in sys.path:
                    sys.path.insert(0, wheel_path)
                if "win_amd64" in filename and sys.platform != "win32":
                    logger.warning("⚠️ Dahua SDK wheel is Windows-only; running on non-Windows may fail")
                return

        if sdk_dir not in sys.path:
            sys.path.insert(0, sdk_dir)

    def connect(self) -> bool:
        """Connect to camera"""
        if not self._init_sdk():
            return False

        try:
            from NetSDK.SDK_Struct import (
                NET_IN_LOGIN_WITH_HIGHLEVEL_SECURITY,
                NET_OUT_LOGIN_WITH_HIGHLEVEL_SECURITY,
                sizeof
            )
            from NetSDK.SDK_Enum import EM_LOGIN_SPAC_CAP_TYPE

            in_param = NET_IN_LOGIN_WITH_HIGHLEVEL_SECURITY()
            in_param.dwSize = sizeof(NET_IN_LOGIN_WITH_HIGHLEVEL_SECURITY)
            in_param.szIP = self.camera_ip.encode()
            in_param.nPort = self.camera_port
            in_param.szUserName = self.username.encode()
            in_param.szPassword = self.password.encode()
            in_param.emSpecCap = EM_LOGIN_SPAC_CAP_TYPE.TCP

            out_param = NET_OUT_LOGIN_WITH_HIGHLEVEL_SECURITY()
            out_param.dwSize = sizeof(NET_OUT_LOGIN_WITH_HIGHLEVEL_SECURITY)

            self.login_id, device_info, error_msg = self.sdk.LoginWithHighLevelSecurity(
                in_param, out_param
            )

            if self.login_id:
                logger.info(f"✅ Connected to camera {self.camera_ip}")
                logger.info(f"   Channels: {device_info.nChanNum}")
                self._is_connected = True
                self._reconnect_attempt = 0
                return True
            else:
                logger.error(f"❌ Login failed: {error_msg}")
                return False

        except Exception as e:
            logger.error(f"❌ Connect failed: {e}")
            return False

    def _trigger_reconnect(self):
        """Trigger auto-reconnect with exponential backoff."""
        if self._reconnect_running:
            return
        if self._reconnect_attempt >= self._max_reconnect_attempts:
            logger.error(f"❌ Max reconnect attempts ({self._max_reconnect_attempts}) reached for {self.camera_ip}")
            return

        self._reconnect_running = True
        self._reconnect_thread = threading.Thread(
            target=self._reconnect_loop,
            daemon=True,
            name=f"DahuaReconnect-{self.camera_ip}"
        )
        self._reconnect_thread.start()

    def _reconnect_loop(self):
        """Reconnect loop with exponential backoff."""
        import random
        while self._reconnect_running and not self._is_connected:
            self._reconnect_attempt += 1
            self.stats['reconnect_attempts'] += 1

            backoff = min(
                self._base_backoff * (2 ** (self._reconnect_attempt - 1)),
                self._max_backoff
            )
            backoff *= (1 + random.random() * 0.25)

            logger.info(f"🔄 Reconnect attempt {self._reconnect_attempt}/{self._max_reconnect_attempts} "
                        f"for {self.camera_ip} in {backoff:.1f}s")

            time.sleep(backoff)

            if self._reconnect_attempt >= self._max_reconnect_attempts:
                logger.error(f"❌ Giving up reconnect to {self.camera_ip} after {self._max_reconnect_attempts} attempts")
                break

            try:
                if self.connect():
                    logger.info(f"✅ Reconnected to {self.camera_ip} after {self._reconnect_attempt} attempts")
                    if self.start_listening():
                        logger.info(f"✅ Re-subscribed to traffic events on {self.camera_ip}")
                    break
            except Exception as e:
                logger.warning(f"⚠️ Reconnect attempt {self._reconnect_attempt} failed: {e}")

        self._reconnect_running = False

    def start_listening(self) -> bool:
        """Start listening for traffic/plate events"""
        if not self.login_id:
            if not self.connect():
                return False

        try:
            from NetSDK.SDK_Enum import EM_EVENT_IVS_TYPE
            from NetSDK.SDK_Callback import CB_FUNCTYPE
            from NetSDK.SDK_Struct import (
                DEV_EVENT_TRAFFICJUNCTION_INFO,
                C_LLONG, C_DWORD, C_LDWORD,
                POINTER, c_void_p, c_ubyte, c_int, cast
            )

            @CB_FUNCTYPE(None, C_LLONG, C_DWORD, c_void_p, POINTER(c_ubyte),
                         C_DWORD, C_LDWORD, c_int, c_void_p)
            def analyzer_callback(lHandle, dwAlarmType, pAlarmInfo, pBuffer,
                                  dwBufSize, dwUser, nSequence, reserved):
                if dwAlarmType == EM_EVENT_IVS_TYPE.TRAFFICJUNCTION:
                    self._handle_traffic_event(pAlarmInfo, pBuffer, dwBufSize)

            self._analyzer_callback = analyzer_callback

            self.attach_id = self.sdk.RealLoadPictureEx(
                self.login_id,
                self.channel,
                EM_EVENT_IVS_TYPE.TRAFFICJUNCTION,
                True,
                self._analyzer_callback,
                0,
                None
            )

            if self.attach_id:
                logger.info(f"✅ Subscribed to traffic events on channel {self.channel}")
                self._start_processing_thread()
                return True
            else:
                logger.error(f"❌ Subscribe failed: {self.sdk.GetLastErrorMessage()}")
                return False

        except Exception as e:
            logger.error(f"❌ Start listening failed: {e}")
            return False

    def _handle_traffic_event(self, pAlarmInfo, pBuffer, dwBufSize):
        """Handle traffic junction event from camera"""
        try:
            from NetSDK.SDK_Struct import DEV_EVENT_TRAFFICJUNCTION_INFO, cast, POINTER, c_ubyte

            alarm_info = cast(pAlarmInfo, POINTER(DEV_EVENT_TRAFFICJUNCTION_INFO)).contents

            plate_number = alarm_info.stTrafficCar.szPlateNumber.decode('gb2312', errors='ignore').strip()
            plate_color = alarm_info.stTrafficCar.szPlateColor.decode('utf-8', errors='ignore').strip()
            vehicle_type = alarm_info.stuVehicle.szObjectSubType.decode('utf-8', errors='ignore').strip()
            vehicle_color = alarm_info.stTrafficCar.szVehicleColor.decode('utf-8', errors='ignore').strip()

            timestamp = datetime(
                alarm_info.UTC.dwYear,
                alarm_info.UTC.dwMonth,
                alarm_info.UTC.dwDay,
                alarm_info.UTC.dwHour,
                alarm_info.UTC.dwMinute,
                alarm_info.UTC.dwSecond
            )

            global_image = None
            plate_image = None

            if alarm_info.stuObject.bPicEnble and dwBufSize > 0:
                offset = alarm_info.stuObject.stPicInfo.dwOffSet
                length = alarm_info.stuObject.stPicInfo.dwFileLenth

                if offset > 0:
                    global_buf = cast(pBuffer, POINTER(c_ubyte * offset)).contents
                    global_image = bytes(global_buf)

                if length > 0:
                    plate_buf = pBuffer[offset:offset + length]
                    plate_image = bytes(plate_buf)

            # Extract plate confidence from SDK struct (0-100 → 0.0-1.0)
            # Fallback to 0.75 (below HIGH_CONFIDENCE_THRESHOLD) to trigger Fortress verify
            try:
                sdk_confidence = getattr(alarm_info.stTrafficCar, 'nPlateConfidence', 0)
                plate_confidence = max(0.1, min(1.0, sdk_confidence / 100.0)) if sdk_confidence > 0 else 0.75
            except Exception:
                plate_confidence = 0.75

            event = DahuaPlateEvent(
                timestamp=timestamp,
                plate_number=plate_number,
                plate_color=plate_color,
                vehicle_type=vehicle_type,
                vehicle_color=vehicle_color,
                confidence=plate_confidence,
                channel=self.channel,
                global_image=global_image,
                plate_image=plate_image,
                camera_ip=self.camera_ip,
            )

            self.event_queue.put(event)
            self.stats['total_events'] += 1
            logger.debug(f"📥 Traffic event: {plate_number}")

        except Exception as e:
            logger.error(f"❌ Handle traffic event failed: {e}")

    def _start_processing_thread(self):
        """Start background thread to process events"""
        self._processing_running = True

        def process_loop():
            while self._processing_running:
                try:
                    event = self.event_queue.get(timeout=1.0)
                    self._process_event(event)
                except _QueueEmpty:
                    continue
                except Exception:
                    logger.exception("Event processing error")
                    continue

        self._processing_thread = threading.Thread(target=process_loop, daemon=True)
        self._processing_thread.start()
        logger.info("✅ Event processing thread started")

    def _process_event(self, event: DahuaPlateEvent):
        """Process a plate detection event"""
        if self.save_images and event.plate_image:
            self._save_plate_image(event)

        validated_plate, validation_passed = self._validate_plate(event.plate_number)

        if not validation_passed:
            logger.warning(f"⚠️ Plate failed validation: {event.plate_number}")
            event.confidence = min(event.confidence, 0.5)
            if self.use_fortress_verify and event.plate_image:
                self._verify_with_fortress(event)
        else:
            if validated_plate != event.plate_number:
                logger.info(f"🔧 Validator corrected: {event.plate_number} → {validated_plate}")
                event.plate_number = validated_plate

            if event.confidence >= self.HIGH_CONFIDENCE_THRESHOLD:
                self.stats['high_confidence'] += 1
                logger.info(f"✅ High confidence plate: {event.plate_number}")
            else:
                self.stats['low_confidence'] += 1
                if self.use_fortress_verify and event.plate_image:
                    self._verify_with_fortress(event)

        # Inject into HybridBroker (omni-vehicle internal)
        self._inject_to_hybrid_broker(event)

        if self._on_plate_detected:
            self._on_plate_detected(event)

    def _verify_with_fortress(self, event: DahuaPlateEvent):
        """Verify plate with Fortress LPR"""
        if not self.fortress_lpr or not event.plate_image:
            return

        try:
            import cv2
            import numpy as np

            img_array = np.frombuffer(event.plate_image, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            if img is None:
                return

            result = self.fortress_lpr.process_frame(img, preprocess=True)

            if result.plates:
                best = max(result.plates, key=lambda p: p.confidence)
                event.fortress_verified = True
                event.fortress_plate = best.plate_text
                event.fortress_confidence = best.confidence

                self.stats['fortress_verified'] += 1

                if event.fortress_plate != event.plate_number:
                    self.stats['fortress_corrected'] += 1
                    logger.info(
                        f"🔧 Fortress corrected: {event.plate_number} → {event.fortress_plate}"
                    )
                else:
                    logger.info(f"✅ Fortress confirmed: {event.plate_number}")

        except Exception as e:
            logger.error(f"❌ Fortress verify failed: {e}")

    def _inject_to_hybrid_broker(self, event: DahuaPlateEvent):
        """
        🌉 Inject detection into HybridBroker (omni-vehicle internal).
        Replaces old BatchProcessor injection with direct HybridBroker call.
        """
        if not self._hybrid_broker:
            return

        try:
            import asyncio
            import cv2
            import numpy as np
            from PIL import Image

            plate_pil = None
            global_pil = None

            if event.plate_image:
                img_array = np.frombuffer(event.plate_image, dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if img is not None:
                    plate_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            if event.global_image:
                img_array = np.frombuffer(event.global_image, dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if img is not None:
                    global_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            camera_id = self._get_camera_id_by_ip(event.camera_ip)

            is_sensitive = event.plate_color.lower() in ['red', 'blue', 'green', 'đỏ', 'xanh']
            needs_verify = event.confidence < self.HIGH_CONFIDENCE_THRESHOLD or is_sensitive

            track_id = event.channel * 1000000 + hash(event.plate_number) % 1000000

            async def _inject():
                success, final_plate = await self._hybrid_broker.inject_external_detection(
                    camera_id=camera_id,
                    plate_text=event.plate_number,
                    confidence=event.confidence,
                    plate_image=plate_pil,
                    global_image=global_pil,
                    track_id=track_id,
                    source="dahua_sdk",
                    plate_color=event.plate_color,
                    vehicle_type=event.vehicle_type or "car",
                    timestamp=event.timestamp,
                    needs_verification=needs_verify
                )

                if success:
                    event.fortress_verified = True
                    event.fortress_plate = final_plate or event.plate_number
                    self.stats['fortress_verified'] += 1

                    if final_plate and final_plate != event.plate_number:
                        self.stats['fortress_corrected'] += 1

            try:
                if hasattr(self, 'main_loop') and self.main_loop and self.main_loop.is_running():
                    asyncio.run_coroutine_threadsafe(_inject(), self.main_loop)
                else:
                    # Avoid asyncio.run() which creates+destroys a loop each call.
                    # Try to find an existing running loop first.
                    try:
                        loop = asyncio.get_running_loop()
                        asyncio.run_coroutine_threadsafe(_inject(), loop)
                    except RuntimeError:
                        asyncio.run(_inject())
            except Exception as e:
                logger.error(f"❌ Coroutine injection failed: {e}")

        except Exception as e:
            logger.error(f"❌ Hybrid Broker injection failed: {e}")

    def _get_camera_id_by_ip(self, camera_ip: str) -> str:
        """Map camera IP to Camera UUID from database."""
        if hasattr(self, '_camera_ip_map') and camera_ip in self._camera_ip_map:
            return self._camera_ip_map[camera_ip]

        if not hasattr(self, '_camera_ip_map'):
            self._camera_ip_map = {}

        # Reuse a single sync engine instead of creating one per call
        if not hasattr(self, '_sync_engine') or self._sync_engine is None:
            try:
                from sqlalchemy import create_engine
                from app.config import get_settings
                settings = get_settings()
                self._sync_engine = create_engine(
                    settings.database_url.replace('+asyncpg', ''),
                    pool_size=2, max_overflow=0,
                )
            except Exception as e:
                logger.debug(f"Failed to create sync engine: {e}")
                self._sync_engine = None

        try:
            from sqlalchemy import text

            if self._sync_engine is not None:
                with self._sync_engine.connect() as conn:
                    result = conn.execute(text(
                        'SELECT "Id" FROM "Cameras" WHERE "StreamUrl" LIKE :ip_pattern LIMIT 1'
                    ), {'ip_pattern': f'%{camera_ip}%'})
                    row = result.fetchone()
                    if row:
                        camera_id = str(row[0])
                        self._camera_ip_map[camera_ip] = camera_id
                        return camera_id
        except Exception as e:
            logger.debug(f"DB lookup failed for camera IP {camera_ip}: {e}")

        import hashlib
        synthetic_id = hashlib.md5(camera_ip.encode()).hexdigest()
        camera_id = f"{synthetic_id[:8]}-{synthetic_id[8:12]}-{synthetic_id[12:16]}-{synthetic_id[16:20]}-{synthetic_id[20:32]}"
        self._camera_ip_map[camera_ip] = camera_id
        return camera_id

    def _validate_plate(self, plate_text: str) -> Tuple[str, bool]:
        """Validate plate through vn_plate_validator."""
        try:
            from app.services.plate.vn_plate_validator import validate_and_correct_plate

            result = validate_and_correct_plate(plate_text)
            return result.corrected, result.is_valid

        except ImportError:
            import re
            clean = re.sub(r'[^A-Z0-9]', '', plate_text.upper())
            pattern = r'^[0-9]{2}[A-Z]{1,2}[0-9]{4,6}$'
            is_valid = bool(re.match(pattern, clean)) and len(clean) >= 7
            return clean, is_valid
        except Exception as e:
            logger.warning(f"Validation error: {e}")
            return plate_text, True

    def _save_plate_image(self, event: DahuaPlateEvent):
        """Save plate image to disk"""
        try:
            from app.services.plate.plate_utils import normalize_plate_basic

            timestamp_str = event.timestamp.strftime("%Y%m%d_%H%M%S")
            plate_clean = normalize_plate_basic(event.plate_number)

            if event.plate_image:
                filename = f"{timestamp_str}_{plate_clean}_plate.jpg"
                filepath = self.image_save_dir / filename
                with open(filepath, 'wb') as f:
                    f.write(event.plate_image)

            if event.global_image:
                filename = f"{timestamp_str}_{plate_clean}_global.jpg"
                filepath = self.image_save_dir / filename
                with open(filepath, 'wb') as f:
                    f.write(event.global_image)

        except Exception as e:
            logger.error(f"❌ Save image failed: {e}")

    def stop(self):
        """Stop listening and disconnect"""
        try:
            if self.attach_id:
                self.sdk.StopLoadPic(self.attach_id)
                self.attach_id = None

            if self.login_id:
                self.sdk.Logout(self.login_id)
                self.login_id = None

            if self.sdk:
                self.sdk.Cleanup()
                self.sdk = None

            logger.info("✅ DahuaLPRBridge stopped")

        except Exception as e:
            logger.error(f"❌ Stop failed: {e}")

    def get_stats(self) -> Dict:
        """Get statistics"""
        return {
            **self.stats,
            'correction_rate': (
                self.stats['fortress_corrected'] / max(self.stats['fortress_verified'], 1)
            ),
        }

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
