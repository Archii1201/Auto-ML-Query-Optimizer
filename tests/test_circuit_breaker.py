"""
Unit tests for the Phase 4A circuit breaker state machine.
Uses an injectable clock so we can test time-based transitions
deterministically (no sleeps).
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from services.ml_service.circuit_breaker import CircuitBreaker, State


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def make(**kw):
    clock = FakeClock()
    cb = CircuitBreaker(failure_threshold=3, window_s=30, reset_timeout_s=60,
                        now=clock, **kw)
    return cb, clock


def test_starts_closed_and_allows():
    cb, _ = make()
    assert cb.state is State.CLOSED
    assert cb.allow() is True


def test_opens_after_threshold():
    cb, _ = make()
    for _ in range(3):
        cb.record_failure()
    assert cb.state is State.OPEN
    assert cb.allow() is False           # short-circuited
    assert cb.short_circuited_total == 1
    assert cb.opened_total == 1


def test_failures_outside_window_do_not_open():
    cb, clock = make()
    cb.record_failure()
    cb.record_failure()
    clock.advance(31)                    # first two fall out of the 30s window
    cb.record_failure()
    assert cb.state is State.CLOSED


def test_half_open_after_reset_timeout():
    cb, clock = make()
    for _ in range(3):
        cb.record_failure()
    assert cb.allow() is False
    clock.advance(60)
    assert cb.allow() is True            # transitions to HALF_OPEN, permits trial
    assert cb.state is State.HALF_OPEN


def test_half_open_success_closes():
    cb, clock = make()
    for _ in range(3):
        cb.record_failure()
    clock.advance(60)
    cb.allow()                           # -> HALF_OPEN
    cb.record_success()
    assert cb.state is State.CLOSED
    assert cb.allow() is True


def test_half_open_failure_reopens():
    cb, clock = make()
    for _ in range(3):
        cb.record_failure()
    clock.advance(60)
    cb.allow()                           # -> HALF_OPEN
    cb.record_failure()                  # trial fails
    assert cb.state is State.OPEN
    assert cb.allow() is False           # still open until next reset window


def test_success_resets_failure_count():
    cb, _ = make()
    cb.record_failure()
    cb.record_failure()
    cb.record_success()                  # clears the rolling failures
    cb.record_failure()
    assert cb.state is State.CLOSED
