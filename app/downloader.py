"""Async install/update/remove lifecycle with in-memory + on-disk job tracking."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from . import ConfigError, DownloadError, SwapctlError
from . import hf_client, registry, swap_config, system
from .bootstrap import _extract_quant
from .config import settings
from .validation import (
    max_install_bytes,
    safe_child_dir,
    valid_filename,
    valid_model_id,
    valid_repo_id,
)

log = logging.getLogger("swapctl.downloader")

_LARGE_FILE_BYTES = 30 * 1024 ** 3
_JOBS_FILE = settings.registry_path.parent / "jobs.json"
_MAX_JOBS_PERSISTED = 50
_TERMINAL_STATUSES = {"done", "error", "interrupted"}

_JOBS: dict[str, dict[str, Any]] = {}
_TASKS: dict[str, asyncio.Task] = {}
_ACTIVE_MODELS: set[str] = set()
_INSTALL_SEMAPHORE = asyncio.Semaphore(1)
_CANCEL_FLAGS: dict[str, str] = {}  # job_id -> "cancel" | "pause"


def _persist() -> None:
    """Atomic dump of _JOBS to disk (keeps newest _MAX_JOBS_PERSISTED)."""
    try:
        _JOBS_FILE.parent.mkdir(parents=True, exist_ok=True)
        jobs_sorted = sorted(
            _JOBS.values(),
            key=lambda j: j.get("started_at", ""),
            reverse=True,
        )[:_MAX_JOBS_PERSISTED]
        fd, tmp = tempfile.mkstemp(prefix="jobs.", suffix=".tmp", dir=str(_JOBS_FILE.parent))
        try:
            with os.fdopen(fd, "w") as f:
                json.dump({"jobs": jobs_sorted}, f, indent=2)
            os.replace(tmp, _JOBS_FILE)
        except OSError:
            try:
                Path(tmp).unlink()
            except OSError:
                pass
    except OSError as exc:
        log.warning("job persist failed: %s", exc)


def load_persisted() -> int:
    """Load jobs from disk; mark any active ones as 'interrupted'. Call from app startup."""
    if not _JOBS_FILE.exists():
        return 0
    try:
        with open(_JOBS_FILE) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("jobs.json read failed: %s", exc)
        return 0
    count = 0
    for j in data.get("jobs", []):
        jid = j.get("job_id")
        if not jid:
            continue
        if j.get("status") in ("running", "queued"):
            j["status"] = "interrupted"
            j["error"] = "swapctl restarted while job was active"
            j["message"] = None
        _JOBS[jid] = j
        count += 1
    _persist()
    return count


def _new_job(kind: str, model_id: str, params: dict | None = None) -> str:
    job_id = uuid4().hex
    _JOBS[job_id] = {
        "job_id": job_id,
        "kind": kind,
        "status": "queued",
        "model_id": model_id,
        "bytes_done": 0,
        "bytes_total": 0,
        "error": None,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "params": params or {},
    }
    _persist()
    return job_id


def _set(job_id: str, **fields: Any) -> None:
    job = _JOBS.get(job_id)
    if job is None:
        return
    prev_status = job.get("status")
    job.update(fields)
    if "status" in fields and fields["status"] != prev_status:
        _persist()


def get_status(job_id: str) -> dict:
    """Return a copy of the job status dict."""
    job = _JOBS.get(job_id)
    if job is None:
        return {"status": "error", "error": "unknown job"}
    return dict(job)


def list_jobs() -> list[dict]:
    return [dict(j) for j in _JOBS.values()]


def clear_job(job_id: str) -> bool:
    """Remove a terminal job from history. Returns True if removed."""
    job = _JOBS.get(job_id)
    if job is None:
        return False
    if job.get("status") not in _TERMINAL_STATUSES:
        raise ConfigError(f"job {job_id} is not terminal (status={job.get('status')})")
    _JOBS.pop(job_id, None)
    _persist()
    return True


def clear_terminal_jobs() -> int:
    """Drop all done/error/interrupted jobs. Returns count removed."""
    to_drop = [jid for jid, j in _JOBS.items() if j.get("status") in _TERMINAL_STATUSES]
    for jid in to_drop:
        _JOBS.pop(jid, None)
    if to_drop:
        _persist()
    return len(to_drop)


async def _download_part(
    job_id: str,
    repo_id: str,
    filename: str,
    dest_dir: Path,
) -> Path:
    """Download one file from HF. Progress is owned by a parent _poll_dir task."""
    def _cancel_check() -> str | None:
        return _CANCEL_FLAGS.get(job_id)

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, lambda: hf_client.download_file(repo_id, filename, dest_dir, None, _cancel_check)
    )


def _dir_total_bytes(path: Path) -> int:
    """Sum of all file sizes anywhere under path (incl .cache/.../*.incomplete)."""
    total = 0
    if not path.exists():
        return 0
    for entry in path.rglob("*"):
        try:
            if entry.is_file():
                total += entry.stat().st_size
        except OSError:
            continue
    return total


async def _poll_dir(job_id: str, dest_dir: Path) -> None:
    """Polling task: bytes_done = total bytes on disk in dest_dir (incl. HF download cache)."""
    while True:
        try:
            _set(job_id, bytes_done=_dir_total_bytes(dest_dir))
        except OSError:
            pass
        await asyncio.sleep(1.0)


def _preflight(
    repo_id: str, filenames: list[str], mmproj_filename: str | None
) -> int:
    """Fetch expected total size for all files and verify disk headroom."""
    files = {f.filename: f for f in hf_client.list_repo_files(repo_id)}
    total = 0
    for fn in filenames:
        target = files.get(fn)
        if target is None:
            raise DownloadError(f"{repo_id} has no file named {fn}")
        total += target.size
    if mmproj_filename:
        mmp = files.get(mmproj_filename)
        if mmp is None:
            raise DownloadError(f"{repo_id} has no file named {mmproj_filename}")
        total += mmp.size
    cap = max_install_bytes()
    if total > cap:
        raise DownloadError(f"install exceeds cap: {total} > {cap} bytes")
    free_gb, _ = system.disk_free_gb(settings.models_dir)
    free_bytes = free_gb * (1024 ** 3)
    if total > free_bytes * 0.9:
        raise DownloadError(
            f"insufficient disk: need {total // 1024**3} GB, have {free_gb} GB free"
        )
    return total


async def _do_install(
    job_id: str,
    repo_id: str,
    filename: str,
    mmproj_filename: str | None,
    model_id: str,
    ctx_size: int,
) -> None:
    _set(job_id, status="running")
    try:
        model_dir = safe_child_dir(settings.models_dir, model_id)
    except ConfigError as exc:
        _set(job_id, status="error", error=str(exc))
        return
    bak_path: Path | None = None
    config_written = False
    registry_written = False
    dir_created = False
    poll_task: asyncio.Task | None = None
    try:
        part_filenames = hf_client.expand_split(repo_id, filename)
        total_expected = _preflight(repo_id, part_filenames, mmproj_filename)
        _set(job_id, status="queued", bytes_total=total_expected, bytes_done=0)
        async with _INSTALL_SEMAPHORE:
            signal = _CANCEL_FLAGS.pop(job_id, None)
            if signal:
                status = "error" if signal == "cancel" else "interrupted"
                error = "cancelled by user" if signal == "cancel" else "paused by user"
                _set(job_id, status=status, error=error)
                return
            _set(job_id, status="running")
            if not model_dir.exists():
                model_dir.mkdir(parents=True)
                dir_created = True
            poll_task = asyncio.create_task(_poll_dir(job_id, model_dir))
            gguf_path: Path | None = None
            total_weight_bytes = 0
            for idx, part in enumerate(part_filenames, start=1):
                _set(
                    job_id,
                    message=f"downloading part {idx}/{len(part_filenames)}",
                )
                path = await _download_part(job_id, repo_id, part, model_dir)
                total_weight_bytes += path.stat().st_size
                if part == filename:
                    gguf_path = path
            if gguf_path is None:
                gguf_path = Path(model_dir) / Path(filename).name
            mmproj_path: Path | None = None
            if mmproj_filename:
                _set(job_id, message="downloading mmproj")
                mmproj_path = await _download_part(
                    job_id, repo_id, mmproj_filename, model_dir
                )
            final_size = _dir_total_bytes(model_dir)
            _set(job_id, bytes_done=final_size, bytes_total=max(final_size, total_expected))
            size_bytes = total_weight_bytes
            no_mmap = size_bytes > _LARGE_FILE_BYTES
            port = swap_config.next_free_port(settings.port_base)
            bak_path = swap_config.backup()
            swap_config.add_model(
                model_id=model_id,
                gguf_path=gguf_path,
                mmproj_path=mmproj_path,
                port=port,
                ctx_size=ctx_size,
                no_mmap=no_mmap,
            )
            config_written = True
            try:
                hf_lm = hf_client.repo_last_modified(repo_id).isoformat()
            except DownloadError:
                hf_lm = None
            try:
                from .gguf_meta import read_model_tags as _read_tags
                _tags = _read_tags(gguf_path)
            except Exception:
                _tags = []
            record = registry.ModelRecord(
                model_id=model_id,
                repo_id=repo_id,
                filename=filename,
                mmproj=mmproj_filename,
                installed_at=datetime.now(timezone.utc).isoformat(),
                hf_last_modified=hf_lm,
                size_bytes=size_bytes,
                ctx_size=ctx_size,
                quant=_extract_quant(filename),
                model_tags=_tags,
            )
            registry.upsert(record)
            registry_written = True
            system.restart_swap()
        _set(job_id, status="done")
    except hf_client.DownloadCancelled as exc:
        signal = str(exc)
        _CANCEL_FLAGS.pop(job_id, None)
        if signal == "cancel":
            if dir_created and not registry_written:
                shutil.rmtree(model_dir, ignore_errors=True)
            _set(job_id, status="error", error="cancelled by user")
        else:
            _set(job_id, status="interrupted", error="paused by user")
    except Exception as exc:
        message = str(exc) or exc.__class__.__name__
        if config_written and bak_path is not None:
            try:
                swap_config.restore_backup()
                system.restart_swap()
            except SwapctlError:
                pass
        if registry_written:
            try:
                registry.delete(model_id)
            except SwapctlError:
                pass
        if dir_created and not registry_written:
            shutil.rmtree(model_dir, ignore_errors=True)
        _set(job_id, status="error", error=message)
    finally:
        if poll_task is not None:
            poll_task.cancel()
            try:
                await poll_task
            except asyncio.CancelledError:
                pass


async def _do_update(job_id: str, model_id: str) -> None:
    _set(job_id, status="running")
    try:
        record = registry.get(model_id)
        if record is None:
            raise DownloadError(f"unknown model_id: {model_id}")
        try:
            remote = hf_client.repo_last_modified(record.repo_id)
        except DownloadError as exc:
            raise exc
        remote_iso = remote.isoformat()
        if record.hf_last_modified == remote_iso:
            _set(job_id, status="done", error=None, message="up-to-date")
            return
        await _do_install(
            job_id=job_id,
            repo_id=record.repo_id,
            filename=record.filename,
            mmproj_filename=record.mmproj,
            model_id=model_id,
            ctx_size=record.ctx_size,
        )
    except Exception as exc:
        _set(job_id, status="error", error=str(exc) or exc.__class__.__name__)


def _release(model_id: str, job_id: str):
    def _done(_task: asyncio.Task) -> None:
        _ACTIVE_MODELS.discard(model_id)
        _TASKS.pop(job_id, None)
    return _done


def start_install(
    repo_id: str,
    filename: str,
    mmproj_filename: str | None,
    model_id: str,
    ctx_size: int,
) -> str:
    """Schedule an install coroutine and return the job id."""
    valid_repo_id(repo_id)
    valid_filename(filename)
    if mmproj_filename:
        valid_filename(mmproj_filename)
    valid_model_id(model_id)
    if model_id in _ACTIVE_MODELS:
        raise ConfigError(f"model_id {model_id!r} already has an active job")
    _ACTIVE_MODELS.add(model_id)
    params = {
        "repo_id": repo_id,
        "filename": filename,
        "mmproj_filename": mmproj_filename,
        "model_id": model_id,
        "ctx_size": int(ctx_size),
    }
    job_id = _new_job("install", model_id, params=params)
    task = asyncio.create_task(
        _do_install(job_id, repo_id, filename, mmproj_filename, model_id, ctx_size)
    )
    task.add_done_callback(_release(model_id, job_id))
    _TASKS[job_id] = task
    return job_id


def resume_job(job_id: str) -> str:
    """Resume an interrupted/error job in-place; returns same job_id."""
    job = _JOBS.get(job_id)
    if job is None:
        raise ConfigError(f"unknown job: {job_id}")
    if job.get("status") not in ("interrupted", "error"):
        raise ConfigError(f"job {job_id} is not resumable (status={job.get('status')})")
    params = job.get("params") or {}
    model_id = job.get("model_id", "")

    if model_id in _ACTIVE_MODELS:
        raise ConfigError(f"model_id {model_id!r} already has an active job")

    if job.get("kind") == "install" and all(k in params for k in ("repo_id", "filename", "model_id", "ctx_size")):
        _ACTIVE_MODELS.add(model_id)
        _set(job_id, status="queued", error=None, message="resuming")
        task = asyncio.create_task(
            _do_install(
                job_id, params["repo_id"], params["filename"],
                params.get("mmproj_filename"), params["model_id"],
                int(params["ctx_size"]),
            )
        )
        task.add_done_callback(_release(model_id, job_id))
        _TASKS[job_id] = task
        return job_id

    if job.get("kind") == "update" and params.get("model_id"):
        _ACTIVE_MODELS.add(model_id)
        _set(job_id, status="queued", error=None, message="resuming")
        task = asyncio.create_task(_do_update(job_id, model_id))
        task.add_done_callback(_release(model_id, job_id))
        _TASKS[job_id] = task
        return job_id

    raise ConfigError(f"job {job_id} has no resumable params")


def start_update(model_id: str) -> str:
    """Schedule an update coroutine and return the job id."""
    valid_model_id(model_id)
    if model_id in _ACTIVE_MODELS:
        raise ConfigError(f"model_id {model_id!r} already has an active job")
    _ACTIVE_MODELS.add(model_id)
    job_id = _new_job("update", model_id, params={"model_id": model_id})
    task = asyncio.create_task(_do_update(job_id, model_id))
    task.add_done_callback(_release(model_id, job_id))
    _TASKS[job_id] = task
    return job_id


def cancel_job(job_id: str) -> bool:
    """Signal a running job to cancel. Returns True if signal was set."""
    job = _JOBS.get(job_id)
    if job is None or job.get("status") not in ("running", "queued"):
        return False
    _CANCEL_FLAGS[job_id] = "cancel"
    task = _TASKS.get(job_id)
    if task and not task.done():
        task.cancel()
    return True


def pause_job(job_id: str) -> bool:
    """Signal a running job to pause (keeps partial files for resume). Returns True if signal was set."""
    job = _JOBS.get(job_id)
    if job is None or job.get("status") not in ("running", "queued"):
        return False
    _CANCEL_FLAGS[job_id] = "pause"
    return True


def remove_model(model_id: str) -> None:
    """Synchronous remove: backup config, drop block, restart, delete files+registry."""
    valid_model_id(model_id)
    model_dir = safe_child_dir(settings.models_dir, model_id)
    bak_path = swap_config.backup()
    swap_config.remove_model(model_id)
    try:
        system.restart_swap()
    except SwapctlError as exc:
        if bak_path is not None:
            try:
                swap_config.restore_backup()
            except SwapctlError:
                pass
        raise exc
    if model_dir.exists():
        shutil.rmtree(model_dir, ignore_errors=True)
    registry.delete(model_id)


__all__ = [
    "start_install",
    "start_update",
    "get_status",
    "list_jobs",
    "remove_model",
]
