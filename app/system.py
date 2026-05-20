"""System-level helpers: systemctl, disk, GPU, llama-swap status."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import httpx

from . import SwapctlError
from .config import settings


def restart_swap() -> None:
    """Restart the llama-swap systemd service. Raises on failure."""
    try:
        subprocess.run(
            ["systemctl", "restart", settings.swap_service],
            check=True,
            timeout=30,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else ""
        raise SwapctlError(f"systemctl restart failed: {stderr or exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise SwapctlError(f"systemctl restart timed out: {exc}") from exc
    except FileNotFoundError as exc:
        raise SwapctlError("systemctl not available on this host") from exc


def is_swap_active() -> bool:
    """Return True iff `systemctl is-active` reports active."""
    try:
        result = subprocess.run(
            ["systemctl", "is-active", settings.swap_service],
            check=False,
            timeout=10,
            capture_output=True,
            text=True,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False
    return result.stdout.strip() == "active"


def wait_swap_healthy(timeout_s: float = 15.0, interval_s: float = 0.5) -> bool:
    """Poll is_swap_active until True or timeout. Also checks /v1/models reachable."""
    import time
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if is_swap_active():
            try:
                resp = httpx.get(
                    settings.swap_endpoint.rstrip("/") + "/v1/models",
                    timeout=2.0,
                )
                if resp.status_code == 200:
                    return True
            except httpx.HTTPError:
                pass
        time.sleep(interval_s)
    return False


def disk_free_gb(path: Path) -> tuple[int, int]:
    """Return (free_gb, total_gb) for the filesystem containing path."""
    usage = shutil.disk_usage(str(path))
    gb = 1024 ** 3
    return usage.free // gb, usage.total // gb


def _read_amdgpu_sysfs() -> tuple[int, int] | None:
    """Read GTT pool from amdgpu sysfs. Strix Halo iGPU loads weights into GTT, not VRAM —
    rocm-smi reports the small UMA framebuffer instead, which is misleading. Returns
    (used_mb, total_mb) summed over GTT (primary) and VRAM (the small UMA buffer).
    """
    base = Path("/sys/class/drm")
    if not base.exists():
        return None
    used_b = 0
    total_b = 0
    found = False
    for card in sorted(base.glob("card*")):
        dev = card / "device"
        gtt_total = dev / "mem_info_gtt_total"
        gtt_used = dev / "mem_info_gtt_used"
        vram_total = dev / "mem_info_vram_total"
        vram_used = dev / "mem_info_vram_used"
        if not gtt_total.exists() and not vram_total.exists():
            continue
        try:
            if gtt_total.exists():
                total_b += int(gtt_total.read_text().strip())
                used_b += int(gtt_used.read_text().strip())
                found = True
            if vram_total.exists():
                total_b += int(vram_total.read_text().strip())
                used_b += int(vram_used.read_text().strip())
                found = True
        except (OSError, ValueError):
            continue
    if not found:
        return None
    mb = 1024 * 1024
    return used_b // mb, total_b // mb


_VRAM_JSON_USED_KEYS = ("VRAM Total Used Memory (B)", "VRAM Total Used Memory", "vram_used")
_VRAM_JSON_TOTAL_KEYS = ("VRAM Total Memory (B)", "VRAM Total Memory", "vram_total")


def _parse_rocm_json(text: str) -> tuple[int, int] | None:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    used_total = 0
    total_total = 0
    found = False
    for card in data.values():
        if not isinstance(card, dict):
            continue
        used = next((card[k] for k in _VRAM_JSON_USED_KEYS if k in card), None)
        total = next((card[k] for k in _VRAM_JSON_TOTAL_KEYS if k in card), None)
        if used is None or total is None:
            continue
        try:
            used_total += int(used)
            total_total += int(total)
            found = True
        except (TypeError, ValueError):
            continue
    if not found:
        return None
    mb = 1024 * 1024
    return used_total // mb, total_total // mb


_VRAM_TEXT_USED_RE = re.compile(r"VRAM Total Used Memory.*?:\s*(\d+)", re.IGNORECASE)
_VRAM_TEXT_TOTAL_RE = re.compile(r"VRAM Total Memory.*?:\s*(\d+)", re.IGNORECASE)


def _parse_rocm_text(text: str) -> tuple[int, int] | None:
    used_m = _VRAM_TEXT_USED_RE.search(text)
    total_m = _VRAM_TEXT_TOTAL_RE.search(text)
    if not used_m or not total_m:
        return None
    try:
        used = int(used_m.group(1))
        total = int(total_m.group(1))
    except ValueError:
        return None
    mb = 1024 * 1024
    return used // mb, total // mb


def _rocm_smi_memory() -> tuple[int, int] | None:
    if shutil.which("rocm-smi") is None:
        return None
    try:
        result = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram", "--json"],
            check=False,
            timeout=10,
            capture_output=True,
            text=True,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None
    if result.returncode == 0 and result.stdout.strip():
        parsed = _parse_rocm_json(result.stdout)
        if parsed is not None:
            return parsed
    try:
        fallback = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram"],
            check=False,
            timeout=10,
            capture_output=True,
            text=True,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None
    if fallback.returncode != 0 or not fallback.stdout.strip():
        return None
    return _parse_rocm_text(fallback.stdout)


def gpu_memory_mb() -> tuple[int, int] | None:
    """Return (used_mb, total_mb). Uses amdgpu sysfs (GTT-aware) first, rocm-smi fallback."""
    sysfs = _read_amdgpu_sysfs()
    if sysfs is not None:
        return sysfs
    return _rocm_smi_memory()


def ram_mb() -> tuple[int, int] | None:
    """Return (used_mb, total_mb) from /proc/meminfo."""
    try:
        text = Path("/proc/meminfo").read_text()
    except OSError:
        return None
    vals: dict[str, int] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].rstrip(":") in ("MemTotal", "MemAvailable"):
            try:
                vals[parts[0].rstrip(":")] = int(parts[1])
            except ValueError:
                pass
    total_kb = vals.get("MemTotal")
    avail_kb = vals.get("MemAvailable")
    if total_kb is None or avail_kb is None:
        return None
    return ((total_kb - avail_kb) // 1024, total_kb // 1024)


def cpu_usage_percent() -> float | None:
    """Return current CPU usage % via /proc/stat (short sample)."""
    import time

    def _read_stat() -> tuple[int, int] | None:
        try:
            line = Path("/proc/stat").read_text().splitlines()[0]
        except (OSError, IndexError):
            return None
        parts = line.split()
        if parts[0] != "cpu" or len(parts) < 5:
            return None
        nums = [int(x) for x in parts[1:]]
        idle = nums[3]
        total = sum(nums)
        return (idle, total)

    s1 = _read_stat()
    if s1 is None:
        return None
    time.sleep(0.3)
    s2 = _read_stat()
    if s2 is None:
        return None
    idle_d = s2[0] - s1[0]
    total_d = s2[1] - s1[1]
    if total_d <= 0:
        return 0.0
    return round((1 - idle_d / total_d) * 100, 1)


def gpu_busy_percent() -> int | None:
    """Return iGPU utilization % from amdgpu sysfs."""
    for card in sorted(Path("/sys/class/drm").glob("card*")):
        p = card / "device" / "gpu_busy_percent"
        if p.exists():
            try:
                return int(p.read_text().strip())
            except (OSError, ValueError):
                continue
    return None


def sensors() -> dict:
    """Read temps, power, clocks from hwmon sysfs. Returns flat dict."""
    out: dict = {}
    hwmon = Path("/sys/class/hwmon")
    if not hwmon.exists():
        return out
    for d in sorted(hwmon.iterdir()):
        try:
            name = (d / "name").read_text().strip()
        except OSError:
            continue
        if name == "k10temp":
            try:
                out["cpu_temp"] = round(int((d / "temp1_input").read_text().strip()) / 1000, 1)
            except (OSError, ValueError):
                pass
        elif name == "amdgpu":
            try:
                out["gpu_temp"] = round(int((d / "temp1_input").read_text().strip()) / 1000, 1)
            except (OSError, ValueError):
                pass
            try:
                out["gpu_power_w"] = round(int((d / "power1_average").read_text().strip()) / 1_000_000, 1)
            except (OSError, ValueError):
                pass
            try:
                out["gpu_clock_mhz"] = int((d / "freq1_input").read_text().strip()) // 1_000_000
            except (OSError, ValueError):
                pass
        elif name == "nvme":
            peak = out.get("nvme_temp") or 0
            for i in range(1, 6):
                f = d / f"temp{i}_input"
                if not f.exists():
                    break
                try:
                    t = round(int(f.read_text().strip()) / 1000, 1)
                    if t > peak:
                        peak = t
                except (OSError, ValueError):
                    continue
            if peak:
                out["nvme_temp"] = peak
    cpu_freqs: list[int] = []
    for p in Path("/sys/devices/system/cpu/").glob("cpu[0-9]*/cpufreq/scaling_cur_freq"):
        try:
            cpu_freqs.append(int(p.read_text().strip()) // 1000)
        except (OSError, ValueError):
            continue
    if cpu_freqs:
        out["cpu_clock_avg_mhz"] = sum(cpu_freqs) // len(cpu_freqs)
        out["cpu_clock_max_mhz"] = max(cpu_freqs)
    return out


def npu_active_contexts() -> int | None:
    """Return number of active NPU HW contexts via xrt-smi."""
    if not Path("/dev/accel0").exists():
        return None
    try:
        result = subprocess.run(
            ["xrt-smi", "examine", "-r", "aie-partitions", "--batch"],
            check=False, timeout=5, capture_output=True, text=True,
        )
        if result.returncode != 0:
            return None
        return result.stdout.count("|Active")
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None


def swap_configured_models() -> list[str]:
    """All models declared in llama-swap config (i.e., listed in /v1/models)."""
    url = settings.swap_endpoint.rstrip("/") + "/v1/models"
    try:
        resp = httpx.get(url, timeout=3.0)
        resp.raise_for_status()
        data = resp.json()
    except (httpx.HTTPError, ValueError):
        return []
    items = data.get("data") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return []
    out: list[str] = []
    for item in items:
        if isinstance(item, dict) and isinstance(item.get("id"), str):
            out.append(item["id"])
    return out


def swap_running_models() -> list[str]:
    """Models currently in-memory (state=ready) per llama-swap /running endpoint."""
    url = settings.swap_endpoint.rstrip("/") + "/running"
    try:
        resp = httpx.get(url, timeout=3.0)
        resp.raise_for_status()
        data = resp.json()
    except (httpx.HTTPError, ValueError):
        return []
    items = data.get("running") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return []
    out: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        mid = item.get("model")
        state = item.get("state")
        if isinstance(mid, str) and (state in (None, "ready")):
            out.append(mid)
    return out


def swap_loaded_models() -> list[str]:
    """Backward-compat alias kept for any older callers; prefer swap_running_models."""
    return swap_running_models()


def dir_size_bytes(path: Path) -> int:
    """Return total bytes of all files under path (best-effort)."""
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


def npu_status() -> dict | None:
    """Return NPU device info via xrt-smi and FLM, or None if unavailable."""
    accel = Path("/dev/accel0")
    if not accel.exists():
        return None
    info: dict = {
        "device": None,
        "firmware": None,
        "driver": None,
        "xrt_version": None,
        "columns": None,
        "power_mode": None,
        "flm_version": None,
        "flm_models": [],
    }
    if shutil.which("xrt-smi") is not None:
        try:
            result = subprocess.run(
                ["xrt-smi", "examine"],
                check=False, timeout=10, capture_output=True, text=True,
            )
            for line in result.stdout.splitlines():
                line = line.strip()
                if "NPU Firmware Version" in line:
                    info["firmware"] = line.split(":")[-1].strip()
                elif "amdxdna Version" in line:
                    info["driver"] = line.split(":")[-1].strip()
                elif line.startswith("Version") and info["xrt_version"] is None:
                    info["xrt_version"] = line.split(":")[-1].strip()
                elif "RyzenAI" in line and "]" in line:
                    info["device"] = line.split("]")[-1].strip().strip("|").strip()
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            pass
        try:
            result2 = subprocess.run(
                ["xrt-smi", "examine", "-r", "platform"],
                check=False, timeout=10, capture_output=True, text=True,
            )
            for line in result2.stdout.splitlines():
                line = line.strip()
                if "Total Columns" in line:
                    info["columns"] = line.split(":")[-1].strip()
                elif "Power Mode" in line:
                    info["power_mode"] = line.split(":")[-1].strip()
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            pass
    if shutil.which("flm") is not None:
        try:
            ver = subprocess.run(
                ["flm", "version"], check=False, timeout=5,
                capture_output=True, text=True,
            )
            for line in ver.stdout.splitlines():
                if line.strip().startswith("FLM"):
                    info["flm_version"] = line.strip()
                    break
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            pass
        try:
            lst = subprocess.run(
                ["flm", "list", "--json"],
                check=False, timeout=10, capture_output=True, text=True,
            )
            if lst.returncode == 0 and lst.stdout.strip():
                data = json.loads(lst.stdout)
                for m in data.get("models", []):
                    info["flm_models"].append({
                        "name": m.get("model", m.get("name", "?")),
                        "family": m.get("details", {}).get("family", ""),
                        "params": m.get("details", {}).get("parameter_size", ""),
                        "quant": m.get("details", {}).get("quantization_level", ""),
                        "size_gb": m.get("footprint", 0),
                        "ctx": m.get("default_context_length", 0),
                        "labels": m.get("label", []),
                        "installed": m.get("installed", False),
                    })
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError, json.JSONDecodeError):
            pass
    if not info["device"]:
        info["device"] = "AMD XDNA NPU"
    return info


__all__ = [
    "restart_swap",
    "is_swap_active",
    "disk_free_gb",
    "gpu_memory_mb",
    "swap_configured_models",
    "swap_running_models",
    "swap_loaded_models",
    "dir_size_bytes",
    "npu_status",
]
