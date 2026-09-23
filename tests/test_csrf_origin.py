"""Cross-site writes must be rejected, non-browser clients must not be.

The session cookie is SameSite=None so the page can be framed, which means
SameSite no longer protects state-changing requests. `_origin_allowed` took
over that job, and it is the only thing standing between a foreign page and
a logged-in browser hitting POST /manage/api/models/{id}/delete.
"""
from __future__ import annotations

import pytest

from app.main import _origin_allowed, _origin_host


@pytest.mark.parametrize(
    "origin,expected",
    [
        ("https://ai.hanaktech.org", "ai.hanaktech.org"),
        ("http://127.0.0.1:8090/manage/", "127.0.0.1:8090"),
        ("HTTPS://AI.HANAKTECH.ORG", "ai.hanaktech.org"),
        ("  https://x.test  ", "x.test"),
        ("null", ""),
        ("", ""),
    ],
)
def test_origin_host_extraction(origin, expected):
    assert _origin_host(origin) == expected


def test_same_origin_write_passes():
    assert _origin_allowed("https://ai.hanaktech.org", "ai.hanaktech.org", "", "")


def test_foreign_origin_write_is_rejected():
    assert not _origin_allowed("https://evil.example", "ai.hanaktech.org", "", "")


def test_missing_origin_passes_because_it_is_not_a_browser():
    """curl and scripts send no Origin — and carry no ambient cookie either."""
    assert _origin_allowed("", "ai.hanaktech.org", "", "")


def test_null_origin_passes_like_a_missing_one():
    assert _origin_allowed("null", "ai.hanaktech.org", "", "")


def test_proxy_rewritten_host_is_honoured():
    """Behind the reverse proxy, Host is the internal name and the browser
    sends the public one; X-Forwarded-Host is what reconciles them."""
    assert _origin_allowed(
        "https://ai.hanaktech.org", "127.0.0.1:8090", "ai.hanaktech.org", ""
    )


@pytest.mark.parametrize("extra", [
    "https://extra.test",
    "extra.test",
    "https://other.test, https://extra.test",
    "https://other.test https://extra.test",
])
def test_configured_extra_origins_accepted_in_several_spellings(extra):
    assert _origin_allowed("https://extra.test", "ai.hanaktech.org", "", extra)


def test_extra_origins_do_not_open_everything():
    assert not _origin_allowed(
        "https://evil.example", "ai.hanaktech.org", "", "https://extra.test"
    )


def test_port_is_part_of_the_origin():
    """A different port is a different origin to the browser; treat it so."""
    assert not _origin_allowed(
        "http://127.0.0.1:9999", "127.0.0.1:8090", "", ""
    )
