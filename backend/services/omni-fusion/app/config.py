"""
OmniVision omni-fusion — configuration.
Consumes vehicle (LPR) + human streams, performs geometric linkage.
"""
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Redis Streams (env var: REDIS_STREAMS_URL)
    redis_streams_url: str = "redis://omni-bus:6379"
    stream_prefix: str = "omni"

    # Consumer identity
    consumer_group: str = "omni-fusion-group"
    consumer_name: str = "fusion-worker-1"

    # Spatial-Temporal Fusion parameters
    # Motorcycle: IoU > threshold → link rider to plate
    motorcycle_iou_threshold: float = 0.3
    # Car/Truck: person center-Y in top N% of vehicle bbox → driver
    car_driver_top_region: float = 0.30   # top 30%
    # Temporal window: LPR and FRS events within N seconds are candidates
    temporal_window_sec: float = 2.0
    # Minimum overlap for Shapely spatial analysis
    min_spatial_overlap: float = 0.1

    # Web CMS push (env var: CMS_BASE_URL)
    cms_base_url: str = "http://omni-api:8080"
    internal_secret: str = ""

    # Cleanup
    event_buffer_ttl: float = 10.0   # seconds to keep events in memory

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
