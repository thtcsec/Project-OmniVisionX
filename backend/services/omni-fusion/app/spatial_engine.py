"""
Spatial-Temporal Fusion Engine
===============================
Core logic for linking vehicle plates with driver/rider faces.

Two strategies by vehicle type:
1. Motorcycle: IoU overlap > threshold between person bbox and vehicle bbox
2. Car/Truck/Bus: Person center-Y must be in top-region of vehicle bbox
   (driver sits in the upper portion of the vehicle bounding box)

Uses Shapely for precise geometric operations.
"""
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

from shapely.geometry import box as shapely_box

logger = logging.getLogger("omni-fusion.spatial")


@dataclass
class DetectionEvent:
    """A detection event from Redis Streams."""
    camera_id: str
    global_track_id: int
    class_name: str           # car, truck, motorcycle, bus, person
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    timestamp: float
    # Enrichment from downstream services (LPR/FRS)
    plate_text: Optional[str] = None
    plate_confidence: Optional[float] = None
    plate_crop_path: Optional[str] = None
    full_frame_path: Optional[str] = None
    face_identity: Optional[str] = None
    face_confidence: Optional[float] = None
    face_embedding_id: Optional[str] = None
    face_crop_path: Optional[str] = None


@dataclass
class FusedIdentity:
    """
    Linked identity: Vehicle + Plate + Driver/Rider Face.
    This is the final output sent to Web CMS as a unified event card.
    """
    camera_id: str
    timestamp: float
    vehicle_track_id: int
    vehicle_type: str        # car, truck, motorcycle, bus
    vehicle_bbox: Tuple[int, int, int, int]

    plate_text: Optional[str] = None
    plate_confidence: Optional[float] = None
    plate_crop_path: Optional[str] = None
    full_frame_path: Optional[str] = None

    driver_identity: Optional[str] = None
    driver_face_confidence: Optional[float] = None
    driver_track_id: Optional[int] = None
    driver_bbox: Optional[Tuple[int, int, int, int]] = None
    face_crop_path: Optional[str] = None
    face_detected: bool = False
    person_only: bool = False

    linked_id: Optional[str] = None  # Composite ID: "VEH_{track_id}_FACE_{track_id}"

    def to_dict(self) -> dict:
        return {
            "camera_id": self.camera_id,
            "timestamp": self.timestamp,
            "vehicle_track_id": self.vehicle_track_id,
            "vehicle_type": self.vehicle_type,
            "vehicle_bbox": list(self.vehicle_bbox),
            "plate_text": self.plate_text,
            "plate_confidence": self.plate_confidence,
            "plate_crop_path": self.plate_crop_path,
            "full_frame_path": self.full_frame_path,
            "driver_identity": self.driver_identity,
            "driver_face_confidence": self.driver_face_confidence,
            "driver_track_id": self.driver_track_id,
            "driver_bbox": list(self.driver_bbox) if self.driver_bbox else None,
            "face_crop_path": self.face_crop_path,
            "face_detected": self.face_detected,
            "person_only": self.person_only,
            "linked_id": self.linked_id,
        }


class SpatialFusionEngine:
    """
    Performs spatial-temporal fusion on buffered detection events.

    Algorithm:
    1. Group events by camera_id + time window
    2. For each time window, find vehicle-person pairs:
       a. Motorcycle: IoU(person_bbox, motorcycle_bbox) > threshold
       b. Car/Truck/Bus: person center-Y in top 30% of vehicle bbox
    3. If plate + face both found → create FusedIdentity → push to CMS
    """

    MOTORCYCLE_TYPES = {"motorcycle"}
    LARGE_VEHICLE_TYPES = {"car", "truck", "bus"}

    def __init__(self, settings):
        self.settings = settings

    def compute_iou(
        self,
        bbox_a: Tuple[int, int, int, int],
        bbox_b: Tuple[int, int, int, int],
    ) -> float:
        """Compute IoU using Shapely for precision."""
        box_a = shapely_box(bbox_a[0], bbox_a[1], bbox_a[2], bbox_a[3])
        box_b = shapely_box(bbox_b[0], bbox_b[1], bbox_b[2], bbox_b[3])

        if box_a.is_empty or box_b.is_empty:
            return 0.0

        intersection = box_a.intersection(box_b).area
        union = box_a.union(box_b).area
        return intersection / union if union > 0 else 0.0

    def is_driver_in_vehicle(
        self,
        person_bbox: Tuple[int, int, int, int],
        vehicle_bbox: Tuple[int, int, int, int],
        vehicle_type: str,
    ) -> bool:
        """
        Determine if a person is the driver/rider of a vehicle.

        Motorcycle: IoU-based (rider overlaps with motorcycle)
        Car/Truck/Bus: Person head (center-Y) must be in top region of vehicle bbox
        """
        if vehicle_type in self.MOTORCYCLE_TYPES:
            iou = self.compute_iou(person_bbox, vehicle_bbox)
            return iou > self.settings.motorcycle_iou_threshold

        if vehicle_type in self.LARGE_VEHICLE_TYPES:
            # Person center-Y position
            person_cy = (person_bbox[1] + person_bbox[3]) / 2.0
            # Vehicle top region
            veh_y1, veh_y2 = vehicle_bbox[1], vehicle_bbox[3]
            veh_height = veh_y2 - veh_y1
            top_boundary = veh_y1 + (veh_height * self.settings.car_driver_top_region)

            # Person center-X must also be within vehicle horizontal bounds
            person_cx = (person_bbox[0] + person_bbox[2]) / 2.0
            veh_x1, veh_x2 = vehicle_bbox[0], vehicle_bbox[2]

            margin = veh_height * 0.15
            in_top_region = (veh_y1 - margin) <= person_cy <= top_boundary
            in_horizontal = veh_x1 <= person_cx <= veh_x2

            return in_top_region and in_horizontal

        return False

    def fuse_events(
        self,
        events: List[DetectionEvent],
    ) -> List[FusedIdentity]:
        """
        Run spatial-temporal fusion on a batch of events from the same camera
        within the same time window.

        Returns list of FusedIdentity objects ready to be pushed to CMS.
        """
        if not events:
            return []

        # Separate by type
        vehicles = [e for e in events if e.class_name in (self.MOTORCYCLE_TYPES | self.LARGE_VEHICLE_TYPES)]
        persons = [e for e in events if e.class_name == "person"]

        fused_results: List[FusedIdentity] = []
        matched_person_ids: set = set()  # Track persons matched to vehicles
        matched_person_indices: set = set()  # Track by list index for sentinel -1 IDs

        for vehicle in vehicles:
            # Skip vehicles with zero-area bbox (invalid/default)
            vx1, vy1, vx2, vy2 = vehicle.bbox
            if vx2 <= vx1 or vy2 <= vy1:
                continue

            # Find best matching person (driver/rider)
            best_person: Optional[DetectionEvent] = None
            best_score = 0.0
            best_person_idx: Optional[int] = None

            for p_idx, person in enumerate(persons):
                # Prevent one-to-many: skip persons already matched to a vehicle
                if p_idx in matched_person_indices:
                    continue
                # Also guard by track_id for tracked persons (legacy path)
                if person.global_track_id >= 0 and person.global_track_id in matched_person_ids:
                    continue
                # Skip persons with zero-area bbox
                px1, py1, px2, py2 = person.bbox
                if px2 <= px1 or py2 <= py1:
                    continue
                if self.is_driver_in_vehicle(person.bbox, vehicle.bbox, vehicle.class_name):
                    if vehicle.class_name in self.MOTORCYCLE_TYPES:
                        score = self.compute_iou(person.bbox, vehicle.bbox)
                    else:
                        # For car/truck: prefer person with highest Y overlap in top region
                        person_cy = (person.bbox[1] + person.bbox[3]) / 2.0
                        veh_y1 = vehicle.bbox[1]
                        veh_height = vehicle.bbox[3] - vehicle.bbox[1]
                        # Closer to top = higher score
                        score = 1.0 - ((person_cy - veh_y1) / (veh_height * self.settings.car_driver_top_region + 1e-9))

                    if score > best_score:
                        best_score = score
                        best_person = person
                        best_person_idx = p_idx

            # Build linked identity
            linked_id = f"VEH_{vehicle.global_track_id}"
            if best_person:
                linked_id += f"_FACE_{best_person.global_track_id}"
                # Mark matched by both track_id (for tracked) and index (for all)
                if best_person.global_track_id >= 0:
                    matched_person_ids.add(best_person.global_track_id)
                if best_person_idx is not None:
                    matched_person_indices.add(best_person_idx)

            fused = FusedIdentity(
                camera_id=vehicle.camera_id,
                timestamp=vehicle.timestamp,
                vehicle_track_id=vehicle.global_track_id,
                vehicle_type=vehicle.class_name,
                vehicle_bbox=vehicle.bbox,
                plate_text=vehicle.plate_text,
                plate_confidence=vehicle.plate_confidence,
                plate_crop_path=vehicle.plate_crop_path,
                full_frame_path=vehicle.full_frame_path,
                driver_identity=best_person.face_identity if best_person else None,
                driver_face_confidence=best_person.face_confidence if best_person else None,
                driver_track_id=best_person.global_track_id if best_person else None,
                driver_bbox=best_person.bbox if best_person else None,
                face_crop_path=best_person.face_crop_path if best_person else None,
                face_detected=bool(best_person and (best_person.face_confidence or 0) > 0),
                person_only=False,
                linked_id=linked_id,
            )
            fused_results.append(fused)

        # ── Standalone persons (not matched to any vehicle) ──
        # These are pedestrians or people detected without a nearby vehicle.
        # If they have face data (identity or crop), create a standalone FusedIdentity.
        for p_idx, person in enumerate(persons):
            # Skip persons already matched to a vehicle (by index or track_id)
            if p_idx in matched_person_indices:
                continue
            if person.global_track_id >= 0 and person.global_track_id in matched_person_ids:
                continue  # Already linked to a vehicle above

            has_face_data = (
                (person.face_identity and person.face_identity != "Unknown")
                or person.face_crop_path
                or (person.face_confidence and person.face_confidence > 0)
            )
            if not has_face_data:
                continue  # No face detected for this person, skip

            linked_id = f"PERSON_{person.global_track_id}"
            fused = FusedIdentity(
                camera_id=person.camera_id,
                timestamp=person.timestamp,
                vehicle_track_id=-1,  # -1 = no vehicle for standalone person
                vehicle_type="person",
                vehicle_bbox=person.bbox,
                plate_text=None,
                plate_confidence=None,
                plate_crop_path=None,
                full_frame_path=None,
                driver_identity=person.face_identity,
                driver_face_confidence=person.face_confidence,
                driver_track_id=person.global_track_id,
                driver_bbox=person.bbox,
                face_crop_path=person.face_crop_path,
                face_detected=bool((person.face_confidence or 0) > 0),
                person_only=True,
                linked_id=linked_id,
            )
            fused_results.append(fused)
            logger.info(
                "👤 Standalone person detected: track=%d identity=%s conf=%.2f",
                person.global_track_id,
                person.face_identity or "Unknown",
                person.face_confidence or 0.0,
            )

        return fused_results
