"""Periodic auto-update check for installed models with repo_id."""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger("swapctl.auto_update")

_SETTINGS_FILE = Path("/var/lib/swapctl/auto_update.json")
_DEFAULTS = {"enabled": False, "interval_hours": 24, "last_check": None, "last_run_results": []}
_TASK: asyncio.Task | None = None
_WAKE_EVENT: asyncio.Event | None = None


def get_settings() -> dict:
    try:
        data = json.loads(_SETTINGS_FILE.read_text())
        out = dict(_DEFAULTS)
        out.update({k: v for k, v in data.items() if k in _DEFAULTS})
        return out
    except (OSError, json.JSONDecodeError):
        return dict(_DEFAULTS)


def set_settings(enabled: bool, interval_hours: int) -> dict:
    current = get_settings()
    current["enabled"] = bool(enabled)
    current["interval_hours"] = max(1, int(interval_hours))
    _SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    _SETTINGS_FILE.write_text(json.dumps(current, indent=2))
    if _WAKE_EVENT is not None:
        _WAKE_EVENT.set()
    return current


def _record_run(results: list[dict]) -> None:
    current = get_settings()
    current["last_check"] = datetime.now(timezone.utc).isoformat()
    current["last_run_results"] = results[-20:]
    _SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    _SETTINGS_FILE.write_text(json.dumps(current, indent=2))


def check_updates() -> list[dict]:
    """Scan registry, return list of models with available updates."""
    from . import hf_client, registry
    out: list[dict] = []
    for rec in registry.list_all():
        if not rec.repo_id:
            continue
        try:
            remote = hf_client.repo_last_modified_cached(rec.repo_id, force=True)
        except Exception as exc:
            log.warning("HF check failed for %s: %s", rec.model_id, exc)
            continue
        if remote is None:
            continue
        remote_iso = remote.isoformat()
        if rec.hf_last_modified != remote_iso:
            out.append({"model_id": rec.model_id, "repo_id": rec.repo_id,
                        "current": rec.hf_last_modified, "remote": remote_iso})
    return out


async def run_check_and_queue() -> list[dict]:
    """Find models with updates and queue start_update for each."""
    from . import downloader
    updates = check_updates()
    results: list[dict] = []
    for u in updates:
        try:
            job_id = downloader.start_update(u["model_id"])
            results.append({"model_id": u["model_id"], "status": "queued", "job_id": job_id})
            log.info("auto-update queued for %s (job %s)", u["model_id"], job_id)
        except Exception as exc:
            results.append({"model_id": u["model_id"], "status": "error", "error": str(exc)})
            log.warning("auto-update failed to queue %s: %s", u["model_id"], exc)
    _record_run(results)
    return results


async def _scheduler_loop() -> None:
    global _WAKE_EVENT
    _WAKE_EVENT = asyncio.Event()
    log.info("auto-update scheduler started")
    while True:
        s = get_settings()
        if not s["enabled"]:
            try:
                await asyncio.wait_for(_WAKE_EVENT.wait(), timeout=60.0)
            except asyncio.TimeoutError:
                pass
            _WAKE_EVENT.clear()
            continue
        interval_s = max(60, s["interval_hours"] * 3600)
        try:
            await run_check_and_queue()
        except Exception:
            log.exception("auto-update check failed")
        try:
            await asyncio.wait_for(_WAKE_EVENT.wait(), timeout=interval_s)
        except asyncio.TimeoutError:
            pass
        _WAKE_EVENT.clear()


def start_scheduler() -> None:
    global _TASK
    if _TASK is not None and not _TASK.done():
        return
    _TASK = asyncio.create_task(_scheduler_loop())


def stop_scheduler() -> None:
    global _TASK
    if _TASK is not None and not _TASK.done():
        _TASK.cancel()
    _TASK = None
