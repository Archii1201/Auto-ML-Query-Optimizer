"""
watchdog.py
===========
Phase 5F — post-promotion monitoring and auto-rollback.

The promotion gate (5C) is an *offline* judgement on held-out data. It can
still be wrong online: a promoted model might misbehave on live traffic in a
way OOF didn't reveal (distribution shift, a bad feature at serving time, an
infra interaction). The watchdog is the online safety net:

    capture BASELINE metrics (just before promote)
                │  promote + monitor window
                ▼
    read CURRENT metrics
                │
    evaluate(baseline, current, policy)
        breach ─► registry.promote(previous_version)  (instant rollback)
                  + reload replicas + state.mark_rollback

Metrics come from the ml-service Prometheus endpoint the service already
exports (error rate, p95 latency, pred/actual calibration). The decision is
pure and unit-tested; the orchestrator takes injectable fetch/rollback/reload
steps so tests never touch the network.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import math


@dataclass
class HealthSnapshot:
    error_rate:        float = 0.0     # execution failures / executions
    p95_latency_ms:    float = 0.0     # inference latency p95
    pred_actual_ratio: float = 1.0     # calibration (1.0 = perfect)


@dataclass(frozen=True)
class WatchdogPolicy:
    # Absolute ceiling on live error rate after a promotion.
    max_error_rate: float = 0.10
    # Allowed error-rate increase vs. the pre-promote baseline.
    max_error_rate_increase: float = 0.02
    # Allowed p95 latency regression as a fraction of baseline (0.5 = +50%).
    max_latency_regression_frac: float = 0.5
    # Hard calibration band; ratio outside this after promote ⇒ rollback.
    ratio_low: float = 0.5
    ratio_high: float = 2.0


@dataclass
class RollbackDecision:
    rollback: bool
    reasons:  list[str] = field(default_factory=list)
    checks:   dict[str, bool] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"rollback": self.rollback, "checks": self.checks, "reasons": self.reasons}


def _finite(x: float) -> bool:
    return x is not None and not math.isinf(x) and not math.isnan(x)


def evaluate(baseline: HealthSnapshot, current: HealthSnapshot,
             policy: WatchdogPolicy = WatchdogPolicy()) -> RollbackDecision:
    """A check that FAILS (False) means that dimension is unhealthy."""
    d = RollbackDecision(rollback=False)

    # 1) error rate — absolute ceiling AND regression vs baseline.
    err_ok = (
        _finite(current.error_rate)
        and current.error_rate <= policy.max_error_rate
        and current.error_rate <= baseline.error_rate + policy.max_error_rate_increase
    )
    d.checks["error_rate"] = err_ok
    if not err_ok:
        d.reasons.append(
            f"error_rate {baseline.error_rate:.3f}→{current.error_rate:.3f} "
            f"(cap {policy.max_error_rate:.3f}, +{policy.max_error_rate_increase:.3f})"
        )

    # 2) latency regression vs baseline.
    limit = baseline.p95_latency_ms * (1.0 + policy.max_latency_regression_frac)
    lat_ok = (baseline.p95_latency_ms <= 0) or (current.p95_latency_ms <= limit)
    d.checks["latency"] = lat_ok
    if not lat_ok:
        d.reasons.append(
            f"p95_latency {baseline.p95_latency_ms:.0f}→{current.p95_latency_ms:.0f}ms "
            f"(limit {limit:.0f}ms)"
        )

    # 3) calibration inside the hard band.
    cal_ok = (
        _finite(current.pred_actual_ratio)
        and policy.ratio_low <= current.pred_actual_ratio <= policy.ratio_high
    )
    d.checks["calibration"] = cal_ok
    if not cal_ok:
        d.reasons.append(
            f"pred_actual_ratio {current.pred_actual_ratio:.3f} "
            f"outside [{policy.ratio_low}, {policy.ratio_high}]"
        )

    d.rollback = not all(d.checks.values())
    if not d.rollback:
        d.reasons.append("healthy: no rollback")
    return d


# ---------------------------------------------------------------------------
# Orchestrator (injectable steps).
# ---------------------------------------------------------------------------
FetchFn = Callable[[], HealthSnapshot]
RollbackFn = Callable[[str], None]   # promote(previous_version)
ReloadFn = Callable[[], None]


@dataclass
class WatchdogDeps:
    fetch:    FetchFn
    rollback: RollbackFn
    reload:   ReloadFn
    policy:   WatchdogPolicy = field(default_factory=WatchdogPolicy)


def run_watchdog(baseline: HealthSnapshot, previous_version: str | None,
                 deps: WatchdogDeps) -> RollbackDecision:
    """
    Read current health and roll back to previous_version if unhealthy.
    Returns the decision (with rollback=True iff a rollback was performed).
    """
    current = deps.fetch()
    decision = evaluate(baseline, current, deps.policy)
    if decision.rollback and previous_version:
        deps.rollback(previous_version)
        try:
            deps.reload()
        except Exception as exc:  # noqa: BLE001 — rollback already applied
            decision.reasons.append(f"reload after rollback failed: {exc}")
    elif decision.rollback and not previous_version:
        decision.reasons.append("rollback wanted but no previous version to restore")
    return decision
