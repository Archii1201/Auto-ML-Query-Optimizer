"""
Unit tests for Phase 5F — post-promotion watchdog + auto-rollback.
Pure decision + orchestrator with faked fetch/rollback/reload.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from services.automl_service.watchdog import (
    HealthSnapshot,
    WatchdogDeps,
    WatchdogPolicy,
    evaluate,
    run_watchdog,
)

BASE = HealthSnapshot(error_rate=0.01, p95_latency_ms=200.0, pred_actual_ratio=1.05)


def test_healthy_no_rollback():
    cur = HealthSnapshot(error_rate=0.015, p95_latency_ms=210.0, pred_actual_ratio=1.1)
    d = evaluate(BASE, cur)
    assert d.rollback is False
    assert all(d.checks.values())


def test_error_rate_spike_rolls_back():
    cur = HealthSnapshot(error_rate=0.20, p95_latency_ms=200.0, pred_actual_ratio=1.0)
    d = evaluate(BASE, cur)
    assert d.rollback is True and d.checks["error_rate"] is False


def test_error_rate_increase_over_baseline_rolls_back():
    cur = HealthSnapshot(error_rate=0.05, p95_latency_ms=200.0, pred_actual_ratio=1.0)
    d = evaluate(BASE, cur, WatchdogPolicy(max_error_rate=0.10,
                                           max_error_rate_increase=0.02))
    assert d.rollback is True and d.checks["error_rate"] is False


def test_latency_regression_rolls_back():
    cur = HealthSnapshot(error_rate=0.01, p95_latency_ms=400.0, pred_actual_ratio=1.0)
    d = evaluate(BASE, cur, WatchdogPolicy(max_latency_regression_frac=0.5))
    assert d.rollback is True and d.checks["latency"] is False


def test_calibration_breach_rolls_back():
    cur = HealthSnapshot(error_rate=0.01, p95_latency_ms=200.0, pred_actual_ratio=3.0)
    d = evaluate(BASE, cur)
    assert d.rollback is True and d.checks["calibration"] is False


def test_zero_baseline_latency_skips_latency_check():
    base = HealthSnapshot(error_rate=0.0, p95_latency_ms=0.0, pred_actual_ratio=1.0)
    cur = HealthSnapshot(error_rate=0.0, p95_latency_ms=999.0, pred_actual_ratio=1.0)
    d = evaluate(base, cur)
    assert d.checks["latency"] is True     # no baseline to regress against


# ----- orchestrator -------------------------------------------------------
def test_run_watchdog_performs_rollback():
    called = {"rollback": None, "reload": 0}
    deps = WatchdogDeps(
        fetch=lambda: HealthSnapshot(error_rate=0.5, p95_latency_ms=200, pred_actual_ratio=1.0),
        rollback=lambda v: called.__setitem__("rollback", v),
        reload=lambda: called.__setitem__("reload", called["reload"] + 1),
    )
    d = run_watchdog(BASE, previous_version="prevabc", deps=deps)
    assert d.rollback is True
    assert called["rollback"] == "prevabc"
    assert called["reload"] == 1


def test_run_watchdog_healthy_no_action():
    called = {"rollback": None}
    deps = WatchdogDeps(
        fetch=lambda: HealthSnapshot(error_rate=0.01, p95_latency_ms=205, pred_actual_ratio=1.0),
        rollback=lambda v: called.__setitem__("rollback", v),
        reload=lambda: None,
    )
    d = run_watchdog(BASE, previous_version="prevabc", deps=deps)
    assert d.rollback is False
    assert called["rollback"] is None


def test_run_watchdog_unhealthy_but_no_previous_version():
    deps = WatchdogDeps(
        fetch=lambda: HealthSnapshot(error_rate=0.9, p95_latency_ms=200, pred_actual_ratio=1.0),
        rollback=lambda v: (_ for _ in ()).throw(AssertionError("must not roll back")),
        reload=lambda: None,
    )
    d = run_watchdog(BASE, previous_version=None, deps=deps)
    assert d.rollback is True
    assert any("no previous version" in r for r in d.reasons)
