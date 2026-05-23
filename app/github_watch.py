"""Read + dismiss helper for the GitHub watcher state file.

The actual polling lives in /usr/local/bin/llm-github-watch (systemd timer on
the host). This module only reads the resulting JSON and lets the UI dismiss
individual items.
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

STATE_PATH = Path("/var/lib/swapctl/github_watch.json")


def load_state() -> dict:
    if not STATE_PATH.exists():
        return {"items": [], "last_check": None}
    try:
        return json.loads(STATE_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return {"items": [], "last_check": None}


def unread_count() -> int:
    return sum(1 for it in load_state().get("items", []) if it.get("changed"))


def dismiss(key: str) -> bool:
    """Mark item dismissed. Returns True if state changed."""
    data = load_state()
    now = datetime.now(timezone.utc).isoformat()
    hit = False
    for it in data.get("items", []):
        if it.get("key") == key and it.get("changed"):
            it["changed"] = False
            it["dismissed_at"] = now
            hit = True
    if hit:
        tmp = STATE_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
        os.replace(tmp, STATE_PATH)
    return hit
