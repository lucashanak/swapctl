"""The ranking's speed column misread any server that batches.

The IQ suite sends `parallel_slots` questions at once. A llama-server at
`--parallel 1` queues them and answers one at a time at full speed, so its
per-request decode median is its real speed. A batching engine answers them
together, and each request sees only its share of the batch.

Measured 2026-09-24 on the same suite: halogen-flash-next 14.4 t/s per
request against qwen38-flash-next-agention's 38.5 — while halogen generated
131,828 tokens in 2,532 s and flash-next 156,287 in 7,287 s. At the recorded
speed halogen's decode alone would have taken 9,409 s, 3.7x longer than the
whole run: the per-request figure was physically impossible as a throughput.
"""
from __future__ import annotations

import pytest

from app.tester import batched_speed, suite_throughput


def run(tokens_each, n, duration, decode_tps=None):
    r = {"details": [{"tokens": tokens_each} for _ in range(n)], "duration": duration}
    if decode_tps is not None:
        r["decode_tps"] = decode_tps
    return r


def test_throughput_is_tokens_over_wall_clock():
    assert suite_throughput(run(100, 10, 50)) == 20.0


def test_the_measured_halogen_and_flash_next_runs():
    """The real numbers, so the comparison the fix exists for stays pinned."""
    halogen = {"details": [{"tokens": 131828}], "duration": 2532, "decode_tps": 14.4}
    flash = {"details": [{"tokens": 156287}], "duration": 7287, "decode_tps": 38.5}
    assert suite_throughput(halogen) == 52.1
    assert suite_throughput(flash) == 21.4
    assert suite_throughput(halogen) > 2.4 * suite_throughput(flash)


@pytest.mark.parametrize("res", [
    {},
    {"details": [], "duration": 100},
    {"details": [{"tokens": 50}], "duration": 0},
    {"details": [{"tokens": 50}]},
    {"details": [{"tokens": 0}], "duration": 100},
    {"details": [{"tokens": None}], "duration": 100},
])
def test_nothing_to_divide_is_none_not_zero_or_a_crash(res):
    assert suite_throughput(res) is None


def test_batching_is_detected_when_throughput_beats_per_request():
    """Only possible with several requests in flight at once."""
    assert batched_speed(run(885, 149, 2532, decode_tps=14.4))


def test_a_serial_server_is_not_flagged():
    """Throughput below per-request is the normal shape: one answer at a
    time, and the wall clock also pays for prefill."""
    assert not batched_speed(run(1049, 149, 7287, decode_tps=38.5))


def test_no_per_request_figure_means_no_claim():
    """Older results predate decode_tps; there is nothing to call distorted."""
    assert not batched_speed(run(885, 149, 2532))
