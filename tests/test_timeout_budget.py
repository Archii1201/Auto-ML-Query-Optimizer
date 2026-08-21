"""
Unit tests for the Phase 4A per-request time budget.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from services.ml_service.timeout_budget import TimeoutBudget


def test_remaining_starts_near_total():
    b = TimeoutBudget.start(1000)
    assert 990 <= b.remaining_ms() <= 1000
    assert not b.expired()


def test_expires_when_total_zero():
    b = TimeoutBudget.start(0)
    assert b.expired()
    assert b.remaining_ms() == 0.0


def test_acquire_timeout_is_capped():
    b = TimeoutBudget.start(10_000)
    # remaining is ~10s but cap is 2s -> should return the cap
    assert b.acquire_timeout_s(2.0) == 2.0


def test_acquire_timeout_bounded_by_remaining():
    b = TimeoutBudget.start(500)
    # remaining ~0.5s, cap 2s -> bounded by remaining
    assert b.acquire_timeout_s(2.0) <= 0.5


def test_statement_timeout_subtracts_reserve_and_floors():
    b = TimeoutBudget.start(1000, reserve_ms=250)
    st = b.statement_timeout_ms()
    # ~1000 - 250 = ~750, never below the 100ms floor
    assert 600 <= st <= 750


def test_statement_timeout_never_zero_when_drained():
    b = TimeoutBudget.start(1)
    time.sleep(0.01)
    assert b.statement_timeout_ms(minimum_ms=100.0) == 100
