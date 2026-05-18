"""FLM (FastFlowLM) NPU model management."""
from __future__ import annotations

import asyncio
import json
import logging
import shutil
import signal
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

log = logging.getLogger("swapctl.npu")

_PULL_JOBS: dict[str, dict[str, Any]] = {}
_PULL_TASKS: dict[str, asyncio.Task] = {}

_SERVE_PROC: subprocess.Popen | None = None
_SERVE_MODEL: str | None = None
_SERVE_PORT: int | None = None

DEFAULT_SERVE_PORT = 8199
_TEST_RESULTS_FILE = Path("/var/lib/swapctl/npu_test_results.json")

_test_lock = threading.Lock()
_test_running: str | None = None


def available() -> bool:
    return shutil.which("flm") is not None


def list_models() -> list[dict]:
    if not available():
        return []
    try:
        result = subprocess.run(
            ["flm", "list", "--json"],
            check=False, timeout=15, capture_output=True, text=True,
        )
        if result.returncode != 0:
            return []
        data = json.loads(result.stdout)
        return data.get("models", [])
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError, json.JSONDecodeError):
        return []


def installed_models() -> list[str]:
    return [m["model"] for m in list_models() if m.get("installed")]


def remove_model(model: str) -> str | None:
    if not available():
        return "FLM not installed"
    try:
        result = subprocess.run(
            ["flm", "remove", model],
            check=False, timeout=30, capture_output=True, text=True,
        )
        if result.returncode != 0:
            return result.stderr.strip() or result.stdout.strip() or "remove failed"
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        return str(exc)
    _sync_to_swap()
    return None


def pull_status(job_id: str) -> dict | None:
    return _PULL_JOBS.get(job_id)


def pull_jobs() -> list[dict]:
    return list(_PULL_JOBS.values())


def start_pull(model: str) -> str:
    for jid, job in _PULL_JOBS.items():
        if job["model"] == model and job["status"] in ("running", "queued"):
            return jid

    job_id = f"npu-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{model.replace(':', '-')}"
    _PULL_JOBS[job_id] = {
        "job_id": job_id,
        "model": model,
        "status": "queued",
        "output": "",
        "error": None,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    task = asyncio.create_task(_do_pull(job_id, model))
    _PULL_TASKS[job_id] = task
    return job_id


async def _do_pull(job_id: str, model: str) -> None:
    _PULL_JOBS[job_id]["status"] = "running"
    try:
        proc = await asyncio.create_subprocess_exec(
            "flm", "pull", model,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        output_lines: list[str] = []
        assert proc.stdout is not None
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            decoded = line.decode("utf-8", errors="replace").strip()
            if decoded:
                output_lines.append(decoded)
                _PULL_JOBS[job_id]["output"] = "\n".join(output_lines[-20:])
        await proc.wait()
        if proc.returncode == 0:
            _PULL_JOBS[job_id]["status"] = "done"
            _sync_to_swap()
        else:
            _PULL_JOBS[job_id]["status"] = "error"
            _PULL_JOBS[job_id]["error"] = output_lines[-1] if output_lines else "pull failed"
    except Exception as exc:
        _PULL_JOBS[job_id]["status"] = "error"
        _PULL_JOBS[job_id]["error"] = str(exc)


def cancel_pull(job_id: str) -> bool:
    task = _PULL_TASKS.get(job_id)
    if task and not task.done():
        task.cancel()
        _PULL_JOBS[job_id]["status"] = "cancelled"
        return True
    return False


def _sync_to_swap() -> None:
    try:
        from . import swap_config, system
        swap_config.sync_npu_models()
        system.restart_swap()
    except Exception:
        log.exception("failed to sync NPU models to llama-swap")


def clear_finished_jobs() -> int:
    to_drop = [jid for jid, j in _PULL_JOBS.items() if j["status"] in ("done", "error", "cancelled")]
    for jid in to_drop:
        _PULL_JOBS.pop(jid, None)
        _PULL_TASKS.pop(jid, None)
    return len(to_drop)


def serve_status() -> dict | None:
    global _SERVE_PROC, _SERVE_MODEL, _SERVE_PORT
    if _SERVE_PROC is None:
        return None
    if _SERVE_PROC.poll() is not None:
        _SERVE_PROC = None
        _SERVE_MODEL = None
        _SERVE_PORT = None
        return None
    return {"model": _SERVE_MODEL, "port": _SERVE_PORT, "pid": _SERVE_PROC.pid}


def start_serve(model: str, port: int = DEFAULT_SERVE_PORT) -> str | None:
    global _SERVE_PROC, _SERVE_MODEL, _SERVE_PORT
    if not available():
        return "FLM not installed"
    status = serve_status()
    if status is not None:
        if status["model"] == model and status["port"] == port:
            return None
        stop_serve()
    try:
        _SERVE_PROC = subprocess.Popen(
            ["flm", "serve", model, "--port", str(port), "--host", "0.0.0.0"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        _SERVE_MODEL = model
        _SERVE_PORT = port
    except (FileNotFoundError, OSError) as exc:
        return str(exc)
    return None


def stop_serve() -> bool:
    global _SERVE_PROC, _SERVE_MODEL, _SERVE_PORT
    if _SERVE_PROC is None:
        return False
    try:
        _SERVE_PROC.send_signal(signal.SIGTERM)
        _SERVE_PROC.wait(timeout=10)
    except (subprocess.TimeoutExpired, OSError):
        try:
            _SERVE_PROC.kill()
        except OSError:
            pass
    _SERVE_PROC = None
    _SERVE_MODEL = None
    _SERVE_PORT = None
    return True


# ── Speed testing ──────────────────────────────────────────────────────


def _load_test_results() -> dict[str, dict]:
    if not _TEST_RESULTS_FILE.exists():
        return {}
    try:
        return json.loads(_TEST_RESULTS_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _save_test_results(data: dict[str, dict]) -> None:
    _TEST_RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    _TEST_RESULTS_FILE.write_text(json.dumps(data, indent=2))


def get_test_results() -> dict[str, dict]:
    return _load_test_results()


def get_test_status() -> dict | None:
    with _test_lock:
        if _test_running is None:
            return None
        return {"model": _test_running}


def run_speed_test(model: str) -> str | None:
    """Start NPU speed test in background thread. Returns error or None."""
    with _test_lock:
        global _test_running
        if _test_running is not None:
            return f"Test already running for {_test_running}"
        _test_running = model

    t = threading.Thread(target=_speed_test_worker, args=(model,), daemon=True)
    t.start()
    return None


def _wait_for_serve(port: int, timeout: float = 30.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = httpx.get(f"http://127.0.0.1:{port}/v1/models", timeout=2.0)
            if resp.status_code == 200:
                return True
        except httpx.HTTPError:
            pass
        time.sleep(1.0)
    return False


def _speed_test_worker(model: str) -> None:
    global _test_running

    try:
        from .config import settings
        endpoint = settings.swap_endpoint.rstrip("/")
        swap_model = f"npu/{model}"

        # Warm up — triggers llama-swap to start the FLM process
        try:
            httpx.post(
                f"{endpoint}/v1/chat/completions",
                json={"model": swap_model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1},
                timeout=120.0,
            )
        except Exception:
            pass

        prompts = [
            ("chat", "What is the capital of France? One sentence.", 50),
            ("code", "Write a Python fibonacci function. Just the code.", 150),
            ("reasoning", "A farmer has 17 sheep. All but 9 die. How many are left?", 100),
            ("long", "Explain recursion in programming with 3 Python examples.", 500),
        ]

        decode_speeds: list[float] = []
        prefill_speeds: list[float] = []
        tests: dict[str, dict] = {}

        for name, prompt, max_tokens in prompts:
            try:
                resp = httpx.post(
                    f"{endpoint}/v1/chat/completions",
                    json={
                        "model": swap_model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": max_tokens,
                        "temperature": 0,
                    },
                    timeout=120.0,
                )
                resp.raise_for_status()
                data = resp.json()
                usage = data.get("usage", {})
                ds = usage.get("decoding_speed_tps", 0)
                ps = usage.get("prefill_speed_tps", 0)
                content = ""
                try:
                    content = data["choices"][0]["message"]["content"]
                except (KeyError, IndexError):
                    pass
                if ds > 0:
                    decode_speeds.append(ds)
                if ps > 0:
                    prefill_speeds.append(ps)
                tests[name] = {
                    "decode": round(ds, 1),
                    "prefill": round(ps, 1),
                    "response": content[:200],
                }
            except Exception as exc:
                tests[name] = {"error": str(exc)}

        result = {
            "decode_speed": round(sum(decode_speeds) / len(decode_speeds), 1) if decode_speeds else 0,
            "prefill_speed": round(sum(prefill_speeds) / len(prefill_speeds), 1) if prefill_speeds else 0,
            "tested_at": datetime.now(timezone.utc).isoformat(),
            "tests": tests,
        }

        all_results = _load_test_results()
        all_results[model] = result
        _save_test_results(all_results)
        log.info("NPU test done: %s — %.1f t/s decode", model, result["decode_speed"])

    except Exception:
        log.exception("NPU speed test failed for %s", model)
    finally:
        with _test_lock:
            _test_running = None
