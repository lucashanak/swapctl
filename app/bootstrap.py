"""Bootstrap registry from existing /etc/llama-swap/config.yaml on startup.

Models added manually to llama-swap (i.e. not via swapctl install) are missing
from the registry; this scans the config and records what it can derive locally
(model_id, gguf path, mmproj, ctx, file size) so they appear on the Installed
page. repo_id and hf_last_modified stay None — those models support Remove but
not Update (no HF source known).
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

from . import registry, swap_config

_MODEL_FLAG = re.compile(r"--model\s+(\S+)")
_MMPROJ_FLAG = re.compile(r"--mmproj\s+(\S+)")
_CTX_FLAG = re.compile(r"--ctx-size\s+(\d+)")
_REASONING_FLAG = re.compile(r"--reasoning-format\s+(\S+)")
_SPEC_DEFAULT_FLAG = re.compile(r"--spec-default")
_DRAFT_FLAG = re.compile(r"-md\s+(\S+)")
_DRAFT_MAX_FLAG = re.compile(r"--draft-max\s+(\d+)")
_DRAFT_MIN_FLAG = re.compile(r"--draft-min\s+(\d+)")
_QUANT_RE = re.compile(r"(?:UD-)?(?:I?Q\d+(?:_K)?(?:_[A-Z0-9]+)*|F16|F32|BF16)", re.IGNORECASE)


def _extract_quant(filename: str) -> str:
    stem = filename.rsplit(".gguf", 1)[0]
    stem = re.sub(r"-0000\d+-of-0000\d+$", "", stem)
    m = _QUANT_RE.search(stem)
    return m.group(0) if m else ""


def _file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _ctime_iso(path: Path) -> str:
    try:
        ts = datetime.fromtimestamp(path.stat().st_ctime, tz=timezone.utc)
        return ts.isoformat()
    except OSError:
        return datetime.now(timezone.utc).isoformat()


def sync_from_config() -> int:
    """Add registry entries for any config models not yet tracked. Returns count added."""
    try:
        data = swap_config.load()
    except Exception:
        return 0
    models = data.get("models") if isinstance(data, dict) else None
    if not isinstance(models, dict):
        return 0
    known = {r.model_id for r in registry.list_all()}
    added = 0
    for model_id, block in models.items():
        if model_id in known or not isinstance(block, dict):
            continue
        cmd = block.get("cmd", "")
        if not isinstance(cmd, str):
            continue
        m_model = _MODEL_FLAG.search(cmd)
        if not m_model:
            continue
        gguf_path = Path(m_model.group(1))
        m_ctx = _CTX_FLAG.search(cmd)
        ctx_size = int(m_ctx.group(1)) if m_ctx else 0
        m_mm = _MMPROJ_FLAG.search(cmd)
        mmproj = Path(m_mm.group(1)).name if m_mm else None
        m_rf = _REASONING_FLAG.search(cmd)
        reasoning_format = m_rf.group(1) if m_rf else ""
        spec_default = bool(_SPEC_DEFAULT_FLAG.search(cmd))
        m_draft = _DRAFT_FLAG.search(cmd)
        draft_model = Path(m_draft.group(1)).parent.name if m_draft else ""
        m_dmax = _DRAFT_MAX_FLAG.search(cmd)
        draft_max = int(m_dmax.group(1)) if m_dmax else 0
        m_dmin = _DRAFT_MIN_FLAG.search(cmd)
        draft_min = int(m_dmin.group(1)) if m_dmin else 0
        record = registry.ModelRecord(
            model_id=str(model_id),
            repo_id=None,
            filename=gguf_path.name,
            mmproj=mmproj,
            installed_at=_ctime_iso(gguf_path),
            hf_last_modified=None,
            size_bytes=_file_size(gguf_path),
            ctx_size=ctx_size,
            quant=_extract_quant(gguf_path.name),
            reasoning_format=reasoning_format,
            spec_default=spec_default,
            draft_model=draft_model,
            draft_max=draft_max,
            draft_min=draft_min,
        )
        registry.upsert(record)
        added += 1
    _sync_draft_fields(data)
    try:
        from . import swap_config
        swap_config.sync_npu_models()
    except Exception:
        pass
    return added


def _sync_draft_fields(data) -> None:
    """Backfill draft_model/draft_max/draft_min from config cmd for existing records."""
    models = data.get("models") if isinstance(data, dict) else None
    if not isinstance(models, dict):
        return
    for model_id, block in models.items():
        if not isinstance(block, dict):
            continue
        cmd = block.get("cmd", "")
        if not isinstance(cmd, str):
            continue
        m_draft = _DRAFT_FLAG.search(cmd)
        draft_model = Path(m_draft.group(1)).parent.name if m_draft else ""
        m_dmax = _DRAFT_MAX_FLAG.search(cmd)
        draft_max = int(m_dmax.group(1)) if m_dmax else 0
        m_dmin = _DRAFT_MIN_FLAG.search(cmd)
        draft_min = int(m_dmin.group(1)) if m_dmin else 0
        rec = registry.get(model_id)
        if rec is None:
            continue
        if rec.draft_model == draft_model and rec.draft_max == draft_max and rec.draft_min == draft_min:
            continue
        rec.draft_model = draft_model
        rec.draft_max = draft_max
        rec.draft_min = draft_min
        registry.upsert(rec)
