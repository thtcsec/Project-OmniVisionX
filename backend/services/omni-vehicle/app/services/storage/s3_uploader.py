import io
import json
import threading
from typing import Optional, Tuple

from minio import Minio
from minio.error import S3Error


class S3Uploader:
    def __init__(
        self,
        *,
        enabled: bool,
        internal_endpoint: str,
        public_endpoint: str,
        access_key: str,
        secret_key: str,
        bucket: str,
        secure: bool,
        prefix: str = "",
    ):
        self.enabled = bool(enabled)
        self.bucket = bucket
        self.prefix = prefix.strip().strip("/")
        self._internal_endpoint = internal_endpoint
        self._public_endpoint = public_endpoint
        self._access_key = access_key
        self._secret_key = secret_key
        self._secure = bool(secure)
        self._bucket_ready = False
        self._lock = threading.Lock()

    def _client_internal(self) -> Minio:
        return Minio(
            self._internal_endpoint,
            access_key=self._access_key,
            secret_key=self._secret_key,
            secure=self._secure,
        )

    def _ensure_bucket(self) -> None:
        if self._bucket_ready or not self.enabled:
            return
        with self._lock:
            if self._bucket_ready or not self.enabled:
                return
            client = self._client_internal()
            try:
                if not client.bucket_exists(self.bucket):
                    client.make_bucket(self.bucket)
                policy = {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Principal": {"AWS": ["*"]},
                            "Action": ["s3:GetObject"],
                            "Resource": [f"arn:aws:s3:::{self.bucket}/*"],
                        }
                    ],
                }
                client.set_bucket_policy(self.bucket, json.dumps(policy))
            except S3Error:
                raise
            self._bucket_ready = True

    def _key(self, key: str) -> str:
        k = key.strip().lstrip("/")
        return f"{self.prefix}/{k}" if self.prefix else k

    def put_jpeg_and_url(self, key: str, data: bytes) -> Tuple[str, str]:
        if not self.enabled:
            return "", ""
        if not data:
            return "", ""
        self._ensure_bucket()
        k = self._key(key)
        internal = self._client_internal()
        internal.put_object(
            self.bucket,
            k,
            io.BytesIO(data),
            length=len(data),
            content_type="image/jpeg",
        )
        scheme = "https" if self._secure else "http"
        url = f"{scheme}://{self._public_endpoint.strip().strip('/')}/{self.bucket}/{k}"
        return k, url


def try_build_uploader(settings) -> Optional[S3Uploader]:
    enabled = bool(getattr(settings, "s3_enabled", False))
    if not enabled:
        return None
    internal = str(getattr(settings, "s3_internal_endpoint", "")).strip()
    public = str(getattr(settings, "s3_public_endpoint", "")).strip()
    access_key = str(getattr(settings, "s3_access_key", "")).strip()
    secret_key = str(getattr(settings, "s3_secret_key", "")).strip()
    bucket = str(getattr(settings, "s3_bucket", "")).strip()
    secure = bool(getattr(settings, "s3_secure", False))
    prefix = str(getattr(settings, "s3_prefix", "")).strip()

    if not internal or not public or not access_key or not secret_key or not bucket:
        return None
    return S3Uploader(
        enabled=True,
        internal_endpoint=internal,
        public_endpoint=public,
        access_key=access_key,
        secret_key=secret_key,
        bucket=bucket,
        secure=secure,
        prefix=prefix,
    )
