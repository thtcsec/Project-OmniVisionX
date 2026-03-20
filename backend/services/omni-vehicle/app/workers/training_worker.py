"""
Background worker to run LPR training based on request files.
Migrated from root ai-engine/app/workers/training_worker.py to omni-vehicle.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import shlex
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

from app.config import get_settings

logger = logging.getLogger(__name__)


class LprTrainingWorker:
    def __init__(self, poll_interval: float = 5.0):
        self.settings = get_settings()
        self.poll_interval = poll_interval
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._train_task: Optional[asyncio.Task] = None
        self._last_request_mtime: Optional[float] = None

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info("🧪 LPR training worker started (interval=%ss)", self.poll_interval)

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._train_task:
            self._train_task.cancel()
        logger.info("🧪 LPR training worker stopped")

    async def _loop(self) -> None:
        while self._running:
            try:
                await self._tick()
            except Exception:
                logger.exception("Training worker tick failed")
            await asyncio.sleep(self.poll_interval)

    async def _tick(self) -> None:
        if self._train_task and not self._train_task.done():
            return

        collection_dir = self.settings.lpr_collection_dir
        request_path = os.path.join(collection_dir, "train_request.json")
        status_path = os.path.join(collection_dir, "train_status.json")

        if not os.path.exists(request_path):
            return

        try:
            req_mtime = os.path.getmtime(request_path)
        except Exception:
            req_mtime = None

        if self._last_request_mtime is not None and req_mtime is not None:
            if req_mtime <= self._last_request_mtime:
                return

        status = self._read_json(status_path) or {}
        if status.get("status") == "running":
            return

        request = self._read_json(request_path) or {}
        self._last_request_mtime = req_mtime

        self._train_task = asyncio.create_task(self._run_training(request))

    async def _run_training(self, request: dict) -> None:
        collection_dir = self.settings.lpr_collection_dir
        os.makedirs(collection_dir, exist_ok=True)
        request_path = os.path.join(collection_dir, "train_request.json")
        status_path = os.path.join(collection_dir, "train_status.json")
        report_path = os.path.join(collection_dir, "training_session_report.json")

        data_dir = request.get("data_dir") or collection_dir
        epochs = request.get("epochs") or 50

        def _is_within_allowed_roots(target_path: str, roots: list[str]) -> bool:
            target = os.path.realpath(target_path)
            for root in roots:
                root_abs = os.path.realpath(root)
                try:
                    if os.path.commonpath([target, root_abs]) == root_abs:
                        return True
                except Exception:
                    continue
            return False

        # SECURITY: Validate data_dir to prevent path traversal
        data_dir_resolved = os.path.realpath(data_dir)
        allowed_roots = ["/app", "/data", os.path.realpath(collection_dir)]
        if not _is_within_allowed_roots(data_dir_resolved, allowed_roots):
            logger.warning("Rejected data_dir path traversal attempt: %s → %s", data_dir, data_dir_resolved)
            self._write_status(
                os.path.join(collection_dir, "train_status.json"),
                "failed", os.path.join(collection_dir, "train_request.json"),
                error="data_dir outside allowed directories",
            )
            return
        batch_size = request.get("batch")
        lr = request.get("lr")
        output_dir = request.get("output_dir") or self.settings.weights_dir
        # SECURITY: Validate output_dir to prevent path traversal (mirrors data_dir check above)
        output_dir_resolved = os.path.realpath(output_dir)
        if not _is_within_allowed_roots(output_dir_resolved, allowed_roots):
            logger.warning("Rejected output_dir path traversal attempt: %s \u2192 %s", output_dir, output_dir_resolved)
            self._write_status(
                os.path.join(collection_dir, "train_status.json"),
                "failed", os.path.join(collection_dir, "train_request.json"),
                error="output_dir outside allowed directories",
            )
            return
        use_stn = True if request.get("use_stn") is None else bool(request.get("use_stn"))
        focal_ctc = True if request.get("focal_ctc") is None else bool(request.get("focal_ctc"))

        synthetic_enabled = bool(request.get("synthetic"))
        domain_bridge_enabled = bool(request.get("domain_bridge_enabled"))
        domain_bridge_mode = request.get("domain_bridge_mode") or "night"
        domain_bridge_strength = float(request.get("domain_bridge_strength") or 0.7)
        domain_bridge_output = request.get("domain_bridge_output_dir") or os.path.join(collection_dir, "domain_bridge")
        domain_bridge_max_images = request.get("domain_bridge_max_images")
        domain_bridge_lowres_scale = request.get("domain_bridge_lowres_scale")
        domain_bridge_jpeg_quality = request.get("domain_bridge_jpeg_quality")
        mixed_dataset_dir = None
        mixed_stats = None
        if synthetic_enabled:
            synthetic_dir = request.get("synthetic_output_dir") or os.path.join(collection_dir, "synthetic_dataset")
            num_plates = int(request.get("synthetic_num_plates") or 2000)
            variants = int(request.get("synthetic_variants") or 4)
            synth_ratio = float(request.get("synthetic_ratio") or 0.4)
            procedural_noise = {
                "glare_prob": request.get("synthetic_glare_prob"),
                "rain_prob": request.get("synthetic_rain_prob"),
                "mud_prob": request.get("synthetic_mud_prob"),
                "motion_blur_prob": request.get("synthetic_motion_blur_prob"),
            }
            try:
                await asyncio.to_thread(
                    self._generate_synthetic_dataset, synthetic_dir, num_plates, variants, procedural_noise
                )
                mixed_dataset_dir = os.path.join(collection_dir, "mixed_dataset")
                mixed_stats = self._build_mixed_dataset(data_dir, synthetic_dir, mixed_dataset_dir, synth_ratio)
                if mixed_stats and mixed_stats.get("total", 0) > 0:
                    data_dir = mixed_dataset_dir
                else:
                    data_dir = synthetic_dir
            except Exception as exc:
                logger.exception("Synthetic dataset generation failed: %s", exc)
        if domain_bridge_enabled:
            try:
                from app.services.training.domain_bridge import bridge_dataset
                bridge_dataset(
                    data_dir,
                    domain_bridge_output,
                    mode=domain_bridge_mode,
                    strength=domain_bridge_strength,
                    max_images=domain_bridge_max_images,
                    lowres_scale=domain_bridge_lowres_scale,
                    jpeg_quality=domain_bridge_jpeg_quality,
                )
                data_dir = domain_bridge_output
            except Exception as exc:
                logger.exception("Domain bridging failed: %s", exc)

        # SECURITY: Never accept arbitrary commands from request JSON.
        # Only whitelisted training scripts are allowed to prevent RCE.
        user_command = request.get("command")
        if user_command:
            logger.warning(
                "Ignoring user-supplied 'command' field in training request "
                "(arbitrary command execution is disabled for security)"
            )

        # NOTE: In omni-vehicle context, scripts are at /app/scripts/
        project_root = Path("/app")
        ALLOWED_SCRIPTS = {
            project_root / "scripts" / "train_stn_lprnet.py",
            project_root / "training" / "train_lprnet.py",
        }

        if use_stn or synthetic_enabled:
            script_path = project_root / "scripts" / "train_stn_lprnet.py"
            cmd = [
                "python",
                str(script_path),
                "--data",
                str(data_dir),
                "--epochs",
                str(epochs),
                "--output",
                str(output_dir),
            ]
            if batch_size:
                cmd += ["--batch", str(batch_size)]
            if lr:
                cmd += ["--lr", str(lr)]
            if not use_stn:
                cmd += ["--no-stn"]
            if not focal_ctc:
                cmd += ["--no-focal-ctc"]
        else:
            script_path = project_root / "training" / "train_lprnet.py"
            cmd = [
                "python",
                str(script_path),
                "--data_dir",
                str(data_dir),
                "--epochs",
                str(epochs),
            ]

        if not cmd:
            self._write_status(status_path, "failed", request_path, error="Missing training command")
            return

        start_time = datetime.utcnow().isoformat() + "Z"
        self._write_status(status_path, "running", request_path, extra={"started_at": start_time})

        log_path = os.path.join(collection_dir, f"train_log_{datetime.utcnow():%Y%m%d_%H%M%S}.log")
        exit_code = None
        error_msg = None

        try:
            logger.info("🧪 LPR training started: %s", " ".join(cmd))
            with open(log_path, "w", encoding="utf-8") as log_file:
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    cwd="/app",
                    stdout=log_file,
                    stderr=log_file,
                )
                exit_code = await process.wait()

            if exit_code != 0:
                error_msg = f"Training failed with exit code {exit_code}"
                status = "failed"
            else:
                status = "completed"
        except FileNotFoundError as exc:
            error_msg = f"Command not found: {exc}"
            status = "failed"
        except Exception as exc:
            error_msg = str(exc)
            status = "failed"

        end_time = datetime.utcnow().isoformat() + "Z"
        self._write_status(
            status_path,
            status,
            request_path,
            extra={
                "updated_at": end_time,
                "started_at": start_time,
                "finished_at": end_time,
                "exit_code": exit_code,
                "log_path": log_path,
                "data_dir": data_dir,
                "command": cmd,
                "error": error_msg,
            },
        )

        report = {
            "status": status,
            "requested_at": request.get("requested_at"),
            "started_at": start_time,
            "finished_at": end_time,
            "exit_code": exit_code,
            "data_dir": data_dir,
            "command": cmd,
            "log_path": log_path,
            "error": error_msg,
            "request": request,
            "focal_ctc": focal_ctc,
            "synthetic": {
                "enabled": synthetic_enabled,
                "mixed_dir": mixed_dataset_dir,
                "stats": mixed_stats,
            },
        }

        tensorrt_result = None
        if request.get("tensorrt_build"):
            tensorrt_result = self._build_tensorrt_engine(request, collection_dir)
            report["tensorrt"] = tensorrt_result
        try:
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
        except Exception:
            logger.exception("Failed to write training report")

        if status == "completed":
            logger.info("🧪 LPR training completed")
        else:
            logger.warning("🧪 LPR training failed: %s", error_msg)

    def _generate_synthetic_dataset(self, output_dir: str, num_plates: int, variants: int, procedural_noise: Optional[dict] = None) -> None:
        os.makedirs(output_dir, exist_ok=True)
        # Avoid os.chdir (process-global, race condition in async context).
        # Import directly — plate_generator should not depend on CWD.
        from app.services.training.plate_generator import DatasetGenerator
        gen = DatasetGenerator(output_dir=output_dir, procedural_noise=procedural_noise)
        gen.generate_dataset(num_plates=num_plates, variants_per_plate=variants)

    def _build_mixed_dataset(self, real_dir: str, synthetic_dir: str, output_dir: str, synthetic_ratio: float) -> dict:
        real_pairs = self._collect_image_label_pairs(real_dir)
        synth_pairs = self._collect_image_label_pairs(synthetic_dir)

        if not real_pairs and not synth_pairs:
            return {"total": 0, "real": 0, "synthetic": 0}

        os.makedirs(output_dir, exist_ok=True)
        images_dir = os.path.join(output_dir, "images")
        labels_dir = os.path.join(output_dir, "labels")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        if os.path.exists(images_dir):
            for path in Path(images_dir).glob("*"):
                try:
                    path.unlink()
                except Exception:
                    pass
        if os.path.exists(labels_dir):
            for path in Path(labels_dir).glob("*"):
                try:
                    path.unlink()
                except Exception:
                    pass

        total_available = len(real_pairs) + len(synth_pairs)
        target_synth = int(total_available * max(0.0, min(1.0, synthetic_ratio)))
        target_real = max(0, total_available - target_synth)

        chosen_real = real_pairs if len(real_pairs) <= target_real else random.sample(real_pairs, target_real)
        chosen_synth = synth_pairs if len(synth_pairs) <= target_synth else random.sample(synth_pairs, target_synth)

        def copy_pairs(pairs, prefix):
            for idx, (img_path, label_path) in enumerate(pairs):
                ext = os.path.splitext(img_path)[1]
                stem = f"{prefix}_{idx:06d}"
                dst_img = os.path.join(images_dir, stem + ext)
                dst_lbl = os.path.join(labels_dir, stem + ".txt")
                shutil.copy2(img_path, dst_img)
                shutil.copy2(label_path, dst_lbl)

        copy_pairs(chosen_real, "real")
        copy_pairs(chosen_synth, "syn")

        return {
            "total": len(chosen_real) + len(chosen_synth),
            "real": len(chosen_real),
            "synthetic": len(chosen_synth),
            "ratio": synthetic_ratio,
        }

    @staticmethod
    def _collect_image_label_pairs(base_dir: str) -> list[tuple[str, str]]:
        if not base_dir or not os.path.exists(base_dir):
            return []
        images_dir = Path(base_dir) / "images"
        labels_dir = Path(base_dir) / "labels"
        if not images_dir.exists():
            images_dir = Path(base_dir)
        if not labels_dir.exists():
            labels_dir = Path(base_dir)

        pairs = []
        for img_path in images_dir.glob("*"):
            if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
                continue
            label_path = labels_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                pairs.append((str(img_path), str(label_path)))
        return pairs

    def _build_tensorrt_engine(self, request: dict, collection_dir: str) -> dict:
        onnx_path = request.get("tensorrt_onnx_path")
        engine_path = request.get("tensorrt_engine_path")
        fp16 = request.get("tensorrt_fp16") if request.get("tensorrt_fp16") is not None else True
        int8 = bool(request.get("tensorrt_int8"))

        if not onnx_path:
            onnx_path = self._find_latest_file([self.settings.weights_dir, self.settings.weights_path, collection_dir], ["*.onnx"])
        if not onnx_path:
            return {"status": "skipped", "reason": "onnx_not_found"}
        if not engine_path:
            engine_path = os.path.splitext(onnx_path)[0] + ".engine"

        trtexec = shutil.which("trtexec")
        if not trtexec:
            return {"status": "skipped", "reason": "trtexec_not_found", "onnx": onnx_path}

        cmd = [trtexec, f"--onnx={onnx_path}", f"--saveEngine={engine_path}"]
        if fp16:
            cmd.append("--fp16")
        if int8:
            cmd.append("--int8")

        try:
            subprocess.run(cmd, check=True)
            return {"status": "ok", "onnx": onnx_path, "engine": engine_path, "fp16": fp16, "int8": int8}
        except Exception as exc:
            return {"status": "failed", "onnx": onnx_path, "engine": engine_path, "error": str(exc)}

    @staticmethod
    def _find_latest_file(base_dirs: list[str], patterns: list[str]) -> Optional[str]:
        latest_path = None
        latest_mtime = -1.0
        for base in base_dirs:
            if not base or not os.path.exists(base):
                continue
            for pattern in patterns:
                for path in Path(base).rglob(pattern):
                    try:
                        mtime = path.stat().st_mtime
                        if mtime > latest_mtime:
                            latest_mtime = mtime
                            latest_path = str(path)
                    except Exception:
                        continue
        return latest_path

    @staticmethod
    def _read_json(path: str) -> Optional[dict]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    @staticmethod
    def _write_status(path: str, status: str, request_path: str, extra: Optional[dict] = None, error: Optional[str] = None) -> None:
        payload = {
            "status": status,
            "updated_at": datetime.utcnow().isoformat() + "Z",
            "request_path": request_path,
        }
        if extra:
            payload.update(extra)
        if error:
            payload["error"] = error
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception:
            logger.exception("Failed to write training status")
