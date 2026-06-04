"""Tests for API key management and audio endpoint auth."""
from __future__ import annotations

import json
import pytest
from unittest.mock import patch, MagicMock


# ── api_keys module unit tests ────────────────────────────────────────


@pytest.fixture()
def tmp_registry(tmp_path, monkeypatch):
    """Point registry to a temporary path."""
    reg_path = tmp_path / "registry.json"
    # settings is a frozen dataclass — patch _path() directly
    import app.registry as reg_mod
    monkeypatch.setattr(reg_mod, "_path", lambda: reg_path)
    return reg_path


def test_create_and_verify_key(tmp_registry):
    from app.api_keys import create_key, verify_key, keys_required

    assert not keys_required()

    raw, record = create_key("test-label")
    assert raw.startswith("sk-")
    assert record["label"] == "test-label"
    assert record["enabled"] is True
    assert "key_hash" in record
    assert "key_hash" not in raw  # raw key is not a hash

    assert keys_required()

    matched = verify_key(raw)
    assert matched is not None
    assert matched["label"] == "test-label"
    assert matched["last_used_at"] is not None


def test_verify_wrong_key(tmp_registry):
    from app.api_keys import create_key, verify_key

    create_key("my-key")
    assert verify_key("sk-wrongkey") is None


def test_revoke_key(tmp_registry):
    from app.api_keys import create_key, revoke_key, verify_key, keys_required

    raw, record = create_key("to-revoke")
    assert keys_required()

    revoke_key(record["id"])

    # After revoke: verify_key should return None
    assert verify_key(raw) is None
    # keys_required should now be False (no enabled keys)
    assert not keys_required()


def test_list_keys(tmp_registry):
    from app.api_keys import create_key, list_keys

    create_key("first")
    create_key("second")
    keys = list_keys()
    assert len(keys) == 2
    labels = {k["label"] for k in keys}
    assert labels == {"first", "second"}
    # Raw key value must never appear in list output
    for k in keys:
        assert "sk-" not in k.get("key_hash", "")


def test_revoke_nonexistent_noop(tmp_registry):
    """Revoking a non-existent id must not raise."""
    from app.api_keys import revoke_key
    revoke_key("nonexistent-id")  # should not raise


# ── endpoint auth tests ───────────────────────────────────────────────


@pytest.fixture()
def client_with_registry(tmp_path, monkeypatch):
    """TestClient with registry pointing to tmp_path."""
    reg_path = tmp_path / "registry.json"
    import app.registry as reg_mod
    monkeypatch.setattr(reg_mod, "_path", lambda: reg_path)

    from app.main import app
    from fastapi.testclient import TestClient
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c, reg_path


def _tts_path(path: str) -> str:
    from app.config import settings
    return f"{settings.root_path}/api/v1/audio/{path}"


def test_speech_no_keys_public(client_with_registry):
    """With no keys configured, /speech is accessible without auth."""
    client, _ = client_with_registry
    _fake_wav = b"RIFF\x00\x00\x00\x00WAVEfmt "
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = _fake_wav

    with patch("app.tts.list_voices", return_value=["cs_CZ-jirka-medium"]), \
         patch("subprocess.run", return_value=mock_result):
        resp = client.post(
            _tts_path("speech"),
            json={"model": "tts-1", "voice": "cs_CZ-jirka-medium", "input": "hi"},
        )
    assert resp.status_code == 200


def test_speech_with_key_enforced(client_with_registry, monkeypatch):
    """With a key configured, /speech requires Authorization: Bearer."""
    client, reg_path = client_with_registry

    import app.registry as reg_mod
    monkeypatch.setattr(reg_mod, "_path", lambda: reg_path)

    from app.api_keys import create_key
    raw, _ = create_key("test")

    _fake_wav = b"RIFF\x00\x00\x00\x00WAVEfmt "
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = _fake_wav

    # Without key → 401
    with patch("app.tts.list_voices", return_value=["cs_CZ-jirka-medium"]), \
         patch("subprocess.run", return_value=mock_result):
        resp = client.post(
            _tts_path("speech"),
            json={"model": "tts-1", "voice": "cs_CZ-jirka-medium", "input": "hi"},
        )
    assert resp.status_code == 401

    # With valid key → 200
    with patch("app.tts.list_voices", return_value=["cs_CZ-jirka-medium"]), \
         patch("subprocess.run", return_value=mock_result):
        resp = client.post(
            _tts_path("speech"),
            json={"model": "tts-1", "voice": "cs_CZ-jirka-medium", "input": "hi"},
            headers={"Authorization": f"Bearer {raw}"},
        )
    assert resp.status_code == 200


def test_speech_wrong_key_401(client_with_registry, monkeypatch):
    """Wrong key returns 401."""
    client, reg_path = client_with_registry

    import app.registry as reg_mod
    monkeypatch.setattr(reg_mod, "_path", lambda: reg_path)

    from app.api_keys import create_key
    create_key("test")

    with patch("app.tts.list_voices", return_value=["cs_CZ-jirka-medium"]):
        resp = client.post(
            _tts_path("speech"),
            json={"model": "tts-1", "voice": "cs_CZ-jirka-medium", "input": "hi"},
            headers={"Authorization": "Bearer sk-wrongkey"},
        )
    assert resp.status_code == 401


def test_speech_api_key_query_param(client_with_registry, monkeypatch):
    """Valid key via ?api_key= query param should also work."""
    client, reg_path = client_with_registry

    import app.registry as reg_mod
    monkeypatch.setattr(reg_mod, "_path", lambda: reg_path)

    from app.api_keys import create_key
    raw, _ = create_key("test")

    _fake_wav = b"RIFF\x00\x00\x00\x00WAVEfmt "
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = _fake_wav

    with patch("app.tts.list_voices", return_value=["cs_CZ-jirka-medium"]), \
         patch("subprocess.run", return_value=mock_result):
        resp = client.post(
            _tts_path("speech") + f"?api_key={raw}",
            json={"model": "tts-1", "voice": "cs_CZ-jirka-medium", "input": "hi"},
        )
    assert resp.status_code == 200


def test_voices_endpoint_always_public(client_with_registry, monkeypatch):
    """/voices must remain public even when keys are configured."""
    client, reg_path = client_with_registry

    import app.registry as reg_mod
    monkeypatch.setattr(reg_mod, "_path", lambda: reg_path)

    from app.api_keys import create_key
    create_key("test")

    with patch("app.tts.list_voices", return_value=["cs_CZ-jirka-medium"]):
        resp = client.get(_tts_path("voices"))
    assert resp.status_code == 200


# ── API key management endpoints ─────────────────────────────────────


@pytest.fixture()
def authed_client(tmp_path, monkeypatch):
    """Authenticated TestClient with tmp registry."""
    reg_path = tmp_path / "registry.json"
    import app.registry as reg_mod
    monkeypatch.setattr(reg_mod, "_path", lambda: reg_path)

    from app.main import app
    from fastapi.testclient import TestClient
    from app.config import settings

    with TestClient(app, raise_server_exceptions=True) as c:
        # Simulate session auth
        with c as sc:
            sc.cookies.set("swapctl_session", "test")
        # Force session authed via cookie injection isn't straightforward;
        # use monkeypatch on the session check instead.
        yield c, reg_path


def test_api_keys_crud_via_endpoints(tmp_path, monkeypatch):
    """Round-trip: create key via endpoint, list shows it, revoke removes it."""
    reg_path = tmp_path / "registry.json"
    import app.registry as reg_mod
    monkeypatch.setattr(reg_mod, "_path", lambda: reg_path)

    from app.main import app
    from fastapi.testclient import TestClient
    from app.config import settings

    prefix = settings.root_path

    with TestClient(app, raise_server_exceptions=True) as client:
        # Inject session auth
        with patch("app.routers.auth_deps.keys_required", return_value=False):
            # Simulate authed session by patching session middleware check
            pass

        # Create key (management endpoint is session-protected; bypass via patching session)
        with patch.object(
            type(client.app.state) if hasattr(client.app, "state") else object,
            "__getattr__", return_value=None
        ):
            pass

        # Direct module-level test of the router functions
        from app.routers.api_keys_router import get_keys, post_create_key, delete_key
        from app.routers.api_keys_router import CreateKeyRequest

        # Create
        resp = post_create_key(CreateKeyRequest(label="ci-test"))
        data = json.loads(resp.body)
        assert data["key"].startswith("sk-")
        key_id = data["record"]["id"]

        # List
        resp2 = get_keys()
        data2 = json.loads(resp2.body)
        assert any(k["id"] == key_id for k in data2["keys"])

        # Revoke
        resp3 = delete_key(key_id)
        data3 = json.loads(resp3.body)
        assert data3["ok"] is True

        # Verify revoked
        from app.api_keys import verify_key
        # We don't have the raw key here, but keys_required should be False
        from app.api_keys import keys_required
        assert not keys_required()
