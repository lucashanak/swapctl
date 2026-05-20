"""Power management via sysfs: GPU DPM, GPU clocks, CPU governor, CPU boost.

Socket TDP (STAPM/PPT) lives on the bare-metal PVE host and is reached over
SSH via the `pve2-host` alias, which invokes `tdpctl` (ryzenadj wrapper).
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path

log = logging.getLogger("swapctl.power")

_TDP_SSH_HOST = "pve2-host"
_TDP_SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=4"]
_TDP_KEYS = ("stapm", "fast", "slow", "apu")

_GPU_DPM_PATH = Path("/sys/class/drm/card0/device/power_dpm_force_performance_level")
_GPU_OD_PATH = Path("/sys/class/drm/card0/device/pp_od_clk_voltage")
_CPU_BOOST_PATH = Path("/sys/devices/system/cpu/cpufreq/boost")

DPM_LEVELS = ["auto", "low", "manual", "profile_standard",
              "profile_min_sclk", "profile_min_mclk"]
GOVERNORS = ["performance", "powersave"]


def _read(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def _write(path: Path, value: str) -> str | None:
    try:
        path.write_text(value)
        return None
    except OSError as exc:
        return str(exc)


def get_gpu_dpm() -> str | None:
    return _read(_GPU_DPM_PATH)


def set_gpu_dpm(level: str) -> str | None:
    if level not in DPM_LEVELS:
        return f"invalid level: {level}"
    return _write(_GPU_DPM_PATH, level)


def get_gpu_clocks() -> dict | None:
    text = _read(_GPU_OD_PATH)
    if not text:
        return None
    out: dict = {"current_min": None, "current_max": None, "range_min": None, "range_max": None}
    section = None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("OD_SCLK"):
            section = "sclk"
        elif line.startswith("OD_RANGE"):
            section = "range"
        elif section == "sclk" and ":" in line:
            idx, mhz = line.split(":", 1)
            mhz_val = int(mhz.strip().replace("Mhz", "").replace("MHz", ""))
            idx = int(idx.strip())
            if idx == 0:
                out["current_min"] = mhz_val
            else:
                out["current_max"] = mhz_val
        elif section == "range" and "SCLK" in line:
            parts = line.split()
            for p in parts:
                p = p.replace("Mhz", "").replace("MHz", "")
                try:
                    val = int(p)
                    if out["range_min"] is None:
                        out["range_min"] = val
                    else:
                        out["range_max"] = val
                except ValueError:
                    continue
    return out


def set_gpu_clocks(min_mhz: int, max_mhz: int) -> str | None:
    clocks = get_gpu_clocks()
    if not clocks:
        return "cannot read current clocks"
    rmin = clocks.get("range_min", 600)
    rmax = clocks.get("range_max", 2900)
    if min_mhz < rmin or min_mhz > rmax:
        return f"min_mhz {min_mhz} out of range [{rmin}-{rmax}]"
    if max_mhz < rmin or max_mhz > rmax:
        return f"max_mhz {max_mhz} out of range [{rmin}-{rmax}]"
    if min_mhz > max_mhz:
        return "min_mhz cannot exceed max_mhz"
    err = _write(_GPU_OD_PATH, f"s 0 {min_mhz}")
    if err:
        return err
    err = _write(_GPU_OD_PATH, f"s 1 {max_mhz}")
    if err:
        return err
    return _write(_GPU_OD_PATH, "c")


def reset_gpu_clocks() -> str | None:
    return _write(_GPU_OD_PATH, "r")


def get_cpu_governor() -> str | None:
    p = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    return _read(p)


def _cpufreqctl(*args: str) -> dict:
    if not shutil.which("ssh"):
        return {"ok": False, "error": "ssh client not installed"}
    try:
        p = subprocess.run(
            ["ssh", *_TDP_SSH_OPTS, _TDP_SSH_HOST, "cpufreqctl", *args],
            capture_output=True, text=True, timeout=6,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"ssh {_TDP_SSH_HOST} timeout"}
    except OSError as exc:
        return {"ok": False, "error": str(exc)}
    try:
        return json.loads(p.stdout)
    except json.JSONDecodeError:
        return {"ok": False, "error": f"bad json from cpufreqctl: {p.stdout[:120]}"}


def set_cpu_governor(gov: str) -> str | None:
    if gov not in GOVERNORS:
        return f"invalid governor: {gov}"
    result = _cpufreqctl("set-governor", gov)
    return None if result.get("ok") else (result.get("error") or str(result.get("errors") or "cpufreqctl failed"))


def get_cpu_boost() -> bool | None:
    val = _read(_CPU_BOOST_PATH)
    if val is None:
        return None
    return val == "1"


def set_cpu_boost(enabled: bool) -> str | None:
    result = _cpufreqctl("set-boost", "1" if enabled else "0")
    return None if result.get("ok") else (result.get("error") or "cpufreqctl failed")


def set_cpu_max_freq(mhz: int) -> str | None:
    """Set scaling_max_freq across all host cpufreq policies via cpufreqctl SSH.

    LXC AppArmor blocks /sys/devices/system/cpu/cpufreq writes from inside
    the container, so we hop to the bare-metal pve2 host.
    """
    result = _cpufreqctl("set-max-mhz", str(mhz))
    return None if result.get("ok") else (result.get("error") or str(result.get("errors") or "cpufreqctl failed"))


def get_cpu_freq() -> dict | None:
    p = Path("/sys/devices/system/cpu/cpu0/cpufreq")
    hw_max_file = p / "amd_pstate_max_freq"
    if not hw_max_file.exists():
        hw_max_file = p / "cpuinfo_max_freq"
    try:
        return {
            "min_mhz": int(_read(p / "scaling_min_freq") or 0) // 1000,
            "max_mhz": int(_read(p / "scaling_max_freq") or 0) // 1000,
            "hw_min_mhz": int(_read(p / "cpuinfo_min_freq") or 0) // 1000,
            "hw_max_mhz": int(_read(hw_max_file) or 0) // 1000,
        }
    except (ValueError, TypeError):
        return None


def _tdpctl(*args: str, timeout: float = 6.0) -> dict:
    if not shutil.which("ssh"):
        return {"available": False, "error": "ssh client not installed"}
    try:
        p = subprocess.run(
            ["ssh", *_TDP_SSH_OPTS, _TDP_SSH_HOST, "tdpctl", *args],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"available": False, "error": f"ssh {_TDP_SSH_HOST} timeout"}
    except OSError as exc:
        return {"available": False, "error": str(exc)}
    if p.returncode != 0 and not p.stdout.strip():
        return {"available": False, "error": (p.stderr or "ssh failed").strip()}
    try:
        return json.loads(p.stdout)
    except json.JSONDecodeError:
        return {"available": False, "error": f"bad json from tdpctl: {p.stdout[:120]}"}


def get_socket_tdp() -> dict:
    """Read current STAPM/PPT limits from the pve2 host via tdpctl over SSH."""
    return _tdpctl("get")


def set_socket_tdp(
    stapm_mw: int, fast_mw: int, slow_mw: int, apu_mw: int,
    stapm_time_s: int | None = None,
) -> dict:
    """Apply STAPM/PPT limits on the pve2 host. Values in milliwatts.

    Optional stapm_time_s sets the STAPM filter time constant in seconds.
    Strix Halo doesn't reflect it back in ryzenadj -i, but the SMU
    accepts the write.
    """
    args = ["set", str(stapm_mw), str(fast_mw), str(slow_mw), str(apu_mw)]
    if stapm_time_s is not None:
        args.append(str(int(stapm_time_s)))
    return _tdpctl(*args)


def get_state() -> dict:
    return {
        "gpu_dpm": get_gpu_dpm(),
        "gpu_clocks": get_gpu_clocks(),
        "cpu_governor": get_cpu_governor(),
        "cpu_boost": get_cpu_boost(),
        "cpu_freq": get_cpu_freq(),
        "socket_tdp": get_socket_tdp(),
    }
