"""
DB persistence for omni-vehicle plate events.
"""
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from app.config import get_settings
from app.services.plate.plate_utils import fuzzy_plate_match
from app.services.plate.vehicle_types import normalize_vehicle_type

logger = logging.getLogger("omni-vehicle.event_repository")


import threading

class PlateEventRepository:
    _instance: "PlateEventRepository | None" = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        settings = get_settings()
        self._engine: AsyncEngine = create_async_engine(
            settings.database_url,
            future=True,
            pool_pre_ping=True,
        )
        self._settings = settings
        self._schema_ready = False
        self._schema_lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "PlateEventRepository":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = PlateEventRepository()
        return cls._instance

    async def dispose(self) -> None:
        """Dispose the async engine on shutdown to release DB connections."""
        try:
            await self._engine.dispose()
            logger.info("DB engine disposed")
        except Exception:
            logger.exception("Error disposing DB engine")

    # _fuzzy_plate_match → moved to plate_utils.fuzzy_plate_match

    async def save_plate_event(
        self,
        event_id: uuid.UUID,
        plate_id: uuid.UUID,
        camera_id: str,
        plate_number: str,
        vehicle_type: str,
        confidence: float,
        thumbnail_filename: Optional[str],
        full_frame_filename: Optional[str] = None,
        bbox: Optional[list] = None,
        tracking_id: Optional[str] = None,
        is_update: bool = False,
        metadata: Optional[dict] = None,
        color: Optional[str] = None,
    ) -> None:
        """Save plate detection to DB (Insert or Update)."""
        try:
            vehicle_type = normalize_vehicle_type(vehicle_type, plate_number)

            async with self._engine.begin() as conn:
                await self._ensure_schema(conn)
                bbox_json = json.dumps({
                    "coords": bbox,
                    "space": "full_frame",
                    "format": "[x1, y1, x2, y2]",
                }) if bbox else None

                if is_update:
                    await conn.execute(text("""
                        UPDATE "PlateRecords"
                        SET "PlateNumber" = :plate,
                            "ThumbnailPath" = :thumb,
                            "FullFramePath" = :full_frame,
                            "Confidence" = :conf,
                            "BoundingBox" = :bbox,
                            "VehicleType" = :type,
                            "Color" = :color
                        WHERE "Id" = :id
                    """), {
                        "id": plate_id,
                        "plate": plate_number,
                        "thumb": thumbnail_filename,
                        "full_frame": full_frame_filename,
                        "conf": confidence,
                        "bbox": bbox_json,
                        "type": vehicle_type,
                        "color": color,
                    })

                    await conn.execute(text("""
                        UPDATE "Events"
                        SET "ThumbnailPath" = :thumb,
                            "Description" = :desc,
                            "Metadata" = :metadata
                        WHERE "Id" = :id
                    """), {
                        "id": event_id,
                        "thumb": thumbnail_filename,
                        "desc": f"Phát hiện xe {vehicle_type} - {plate_number}",
                        "metadata": json.dumps(metadata) if metadata else None,
                    })

                else:
                    ts = datetime.utcnow()
                    dedup_seconds = float(self._settings.event_dedup_ttl)
                    cutoff = ts - timedelta(seconds=dedup_seconds)

                    existing = await conn.execute(text("""
                        SELECT 1
                        FROM "PlateRecords"
                        WHERE "CameraId" = :cam_id
                          AND "PlateNumber" = :plate
                          AND "Timestamp" > :cutoff
                          AND "IsDeleted" = FALSE
                        LIMIT 1
                    """), {
                        "cam_id": camera_id,
                        "plate": plate_number,
                        "cutoff": cutoff,
                    })
                    if existing.first() is not None:
                        return

                    if len(plate_number) >= 5:
                        prefix3 = plate_number[:3]
                        similar = await conn.execute(text("""
                            SELECT "PlateNumber"
                            FROM "PlateRecords"
                            WHERE "CameraId" = :cam_id
                              AND "PlateNumber" LIKE :prefix
                              AND LENGTH("PlateNumber") BETWEEN :min_len AND :max_len
                              AND "Timestamp" > :cutoff
                              AND "IsDeleted" = FALSE
                            LIMIT 5
                        """), {
                            "cam_id": camera_id,
                            "prefix": prefix3 + "%",
                            "min_len": len(plate_number) - 1,
                            "max_len": len(plate_number) + 1,
                            "cutoff": cutoff,
                        })
                        for row in similar:
                            existing_plate = row[0]
                            if fuzzy_plate_match(plate_number, existing_plate):
                                logger.debug("Fuzzy dedup: '%s' ≈ existing '%s'", plate_number, existing_plate)
                                return

                    await conn.execute(text("""
                        INSERT INTO "PlateRecords" (
                            "Id", "PlateNumber", "Timestamp", "CameraId",
                            "ThumbnailPath", "FullFramePath",
                            "VehicleType", "Confidence", "Color",
                            "TrackingId", "BoundingBox", "Direction",
                            "IsDeleted"
                        )
                        VALUES (
                            :id, :plate, :ts, :cam_id,
                            :thumb, :full_frame,
                            :type, :conf, :color,
                            :track_id, :bbox, :direction,
                            FALSE
                        )
                    """), {
                        "id": plate_id,
                        "plate": plate_number,
                        "ts": ts,
                        "cam_id": camera_id,
                        "thumb": thumbnail_filename,
                        "full_frame": full_frame_filename,
                        "type": vehicle_type,
                        "conf": confidence,
                        "color": color,
                        "track_id": tracking_id,
                        "bbox": bbox_json,
                        "direction": "unknown",
                    })

                    await conn.execute(text("""
                        INSERT INTO "Events" (
                            "Id", "Type", "CameraId", "Timestamp",
                            "ThumbnailPath", "Description", "PlateRecordId", "Metadata", "IsDeleted"
                        )
                        VALUES (
                            :id, 'Plate', :cam_id, :ts,
                            :thumb, :desc, :plate_id, :metadata, FALSE
                        )
                    """), {
                        "id": event_id,
                        "cam_id": camera_id,
                        "ts": ts,
                        "thumb": thumbnail_filename,
                        "desc": f"Phát hiện xe {vehicle_type} - {plate_number}",
                        "plate_id": plate_id,
                        "metadata": json.dumps(metadata) if metadata else None,
                    })

        except Exception:
            logger.exception("DB Save Error")
            raise  # Let callers know the write failed (prevents phantom cache entries)

    async def _ensure_schema(self, conn) -> None:
        if self._schema_ready:
            return
        with self._schema_lock:
            if self._schema_ready:
                return
            self._schema_ready = True

        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS "PlateRecords" (
                "Id" uuid PRIMARY KEY,
                "PlateNumber" text NOT NULL,
                "Timestamp" timestamptz NOT NULL,
                "CameraId" text NOT NULL,
                "ThumbnailPath" text NULL,
                "FullFramePath" text NULL,
                "Confidence" real NOT NULL DEFAULT 0,
                "BoundingBox" text NULL,
                "VehicleType" text NOT NULL DEFAULT 'unknown',
                "Color" text NULL,
                "TrackingId" text NULL,
                "Direction" text NULL,
                "IsDeleted" boolean NOT NULL DEFAULT false
            );
        """))
        await conn.execute(text("""ALTER TABLE "PlateRecords" ADD COLUMN IF NOT EXISTS "TrackingId" text NULL;"""))
        await conn.execute(text("""CREATE INDEX IF NOT EXISTS "IX_PlateRecords_Timestamp" ON "PlateRecords" ("Timestamp");"""))
        await conn.execute(text("""CREATE INDEX IF NOT EXISTS "IX_PlateRecords_PlateNumber" ON "PlateRecords" ("PlateNumber");"""))

        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS "Events" (
                "Id" uuid PRIMARY KEY,
                "Type" text NOT NULL,
                "CameraId" text NULL,
                "Timestamp" timestamptz NOT NULL,
                "ThumbnailPath" text NULL,
                "Description" text NULL,
                "PlateRecordId" uuid NULL,
                "Metadata" text NULL,
                "IsDeleted" boolean NOT NULL DEFAULT false
            );
        """))
        await conn.execute(text("""CREATE INDEX IF NOT EXISTS "IX_Events_Timestamp" ON "Events" ("Timestamp");"""))
