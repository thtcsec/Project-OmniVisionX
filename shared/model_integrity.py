"""
Model Integrity Verification for OmniVision
==========================================
Validates AI model files at startup by comparing MD5 hashes
against a manifest file.

Usage:
    from shared.model_integrity import verify_model, register_model_hash

    # At startup - crash if mismatch
    verify_model("/app/weights/yolo11m.pt", manifest_path="/app/weights/model_manifest.json")

    # One-time: generate manifest entry for a new model
    register_model_hash("/app/weights/yolo11m.pt", manifest_path="/app/weights/model_manifest.json")
"""
import hashlib
import json
import logging
import os
import sys

logger = logging.getLogger("model_integrity")

MANIFEST_FILENAME = "model_manifest.json"


def md5_file(path: str, chunk_size: int = 8192) -> str:
    """Calculate MD5 hash of a file using chunked reading."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def register_model_hash(model_path: str, manifest_path: str | None = None) -> dict:
    """Calculate and register a model's MD5 hash in the manifest."""
    if manifest_path is None:
        manifest_path = os.path.join(os.path.dirname(model_path), MANIFEST_FILENAME)

    filename = os.path.basename(model_path)
    file_hash = md5_file(model_path)
    file_size = os.path.getsize(model_path)

    manifest = {}
    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

    manifest[filename] = {"md5": file_hash, "size_bytes": file_size}

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Registered model: {filename} -> MD5: {file_hash}")
    return manifest[filename]


def verify_model(model_path: str, manifest_path: str | None = None, strict: bool = True) -> bool:
    """Verify a model file's MD5 hash matches the manifest."""
    if not os.path.exists(model_path):
        msg = f"Model file not found: {model_path}"
        if strict:
            logger.critical(msg)
            sys.exit(1)
        logger.warning(msg)
        return False

    if manifest_path is None:
        manifest_path = os.path.join(os.path.dirname(model_path), MANIFEST_FILENAME)

    filename = os.path.basename(model_path)

    if not os.path.exists(manifest_path):
        logger.warning(f"No manifest for {filename} - auto-registering")
        register_model_hash(model_path, manifest_path)
        return True

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    if filename not in manifest:
        logger.warning(f"No entry for {filename} - auto-registering")
        register_model_hash(model_path, manifest_path)
        return True

    expected_md5 = manifest[filename]["md5"]
    actual_md5 = md5_file(model_path)

    if actual_md5 != expected_md5:
        msg = f"Model integrity FAILED: {filename}"
        if strict:
            logger.critical(msg)
            sys.exit(1)
        logger.warning(msg)
        return False

    logger.info(f"Model verified: {filename}")
    return True
