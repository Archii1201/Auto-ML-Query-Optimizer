"""
worker.py
=========
Phase 5E — the long-running AutoML retraining worker.

One process ties the whole loop together:

    every POLL_INTERVAL:
        gather signals (watermark delta, minutes-since, drift ratio)   [5D]
        should_retrain? ─ no ─► sleep
                        └ yes ─► single-flight lock                     [5E]
                                   retrain (merge+gate+train+register)  [5A/5B]
                                   promote gate (evaluate + decide)     [5C]
                                     pass ─► registry.promote + reload  [5C]
                                   update RetrainState                  [5D]

Design: the *cycle body* is `run_cycle(state, deps)`, a pure orchestrator
that takes injectable callables. The real `WorkerDeps.default()` wires the
concrete modules; unit tests pass fakes and assert the control flow without
running a trainer, touching Postgres, or scraping metrics. Fail-open: any
exception in a cycle is logged and the loop continues — a retraining bug must
never crash the worker, let alone the serving path (separate process).
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from services.automl_service.filelock import FileLock, LockBusy
from services.automl_service.state import RetrainState
from services.automl_service.triggers import (
    TriggerConfig,
    TriggerDecision,
    TriggerSignals,
    should_retrain,
)

POLL_INTERVAL_S = float(os.environ.get("AUTOML_POLL_INTERVAL_S", "300"))
RETRAIN_PROFILE = os.environ.get("AUTOML_PROFILE", "fast")
LOCK_PATH = PROJECT_ROOT / "data" / "processed" / "retrain.lock"
STATE_PATH = PROJECT_ROOT / "data" / "processed" / "retrain_state.json"


@dataclass
class CycleResult:
    trigger:   TriggerDecision
    retrained: bool = False
    promoted:  bool = False
    version:   str | None = None
    note:      str = ""

    def as_dict(self) -> dict:
        return {
            "trigger": self.trigger.as_dict(),
            "retrained": self.retrained,
            "promoted": self.promoted,
            "version": self.version,
            "note": self.note,
        }


# Injectable steps ---------------------------------------------------------
SignalsFn = Callable[[RetrainState], TriggerSignals]
RetrainFn = Callable[[str], "RetrainOutcome"]
PromoteFn = Callable[[str], "PromoteOutcome"]
ReloadFn = Callable[[], None]


@dataclass
class RetrainOutcome:
    ok: bool
    version: str | None = None
    features_rows: int = 0
    error: str | None = None


@dataclass
class PromoteOutcome:
    promoted: bool
    version: str | None = None
    reason: str = ""


@dataclass
class WorkerDeps:
    signals: SignalsFn
    retrain: RetrainFn
    promote: PromoteFn
    reload: ReloadFn
    config: TriggerConfig = field(default_factory=TriggerConfig)
    profile: str = RETRAIN_PROFILE

    @staticmethod
    def default() -> "WorkerDeps":
        return WorkerDeps(
            signals=_default_signals,
            retrain=_default_retrain,
            promote=_default_promote,
            reload=_default_reload,
        )


# ---------------------------------------------------------------------------
# The cycle body — pure orchestration over injected steps.
# ---------------------------------------------------------------------------
def run_cycle(state: RetrainState, deps: WorkerDeps,
              *, state_path: Path = STATE_PATH,
              lock_path: Path = LOCK_PATH) -> CycleResult:
    signals = deps.signals(state)
    decision = should_retrain(signals, deps.config)
    result = CycleResult(trigger=decision)
    if not decision.should_retrain:
        return result

    lock = FileLock(lock_path)
    try:
        lock.acquire()
    except LockBusy:
        result.note = "another retrain in progress; skipping"
        return result

    try:
        outcome = deps.retrain(deps.profile)
        result.retrained = outcome.ok
        result.version = outcome.version
        if not outcome.ok:
            result.note = f"retrain failed: {outcome.error}"
            return result

        state.mark_retrain(reason=decision.reason,
                           features_rows=outcome.features_rows,
                           candidate_version=outcome.version or "")

        promo = deps.promote(outcome.version)
        result.promoted = promo.promoted
        result.note = promo.reason
        if promo.promoted:
            state.mark_promote(promo.version or outcome.version or "")
            try:
                deps.reload()
            except Exception as exc:  # noqa: BLE001 — model promoted regardless
                result.note += f"; reload failed: {exc}"
        state.save(state_path)
        return result
    finally:
        lock.release()


# ---------------------------------------------------------------------------
# Default concrete steps (wire the real 5A–5C modules).
# ---------------------------------------------------------------------------
def _default_signals(state: RetrainState) -> TriggerSignals:
    from services.automl_service.drift import parse_prom_gauge
    from services.automl_service.watermark import Watermark

    wm = Watermark.load()
    new_rows = max(0, wm.features_rows - state.features_rows_at_retrain)

    ratio = None
    metrics_url = os.environ.get("AUTOML_METRICS_URL")
    if metrics_url:
        try:
            import urllib.request
            with urllib.request.urlopen(metrics_url, timeout=5) as r:  # noqa: S310
                ratio = parse_prom_gauge(r.read().decode("utf-8"), "pred_actual_ratio")
        except Exception:  # noqa: BLE001 — drift is best-effort
            ratio = None

    return TriggerSignals(
        new_rows=new_rows,
        minutes_since_retrain=state.minutes_since_retrain(),
        pred_actual_ratio=ratio,
    )


def _default_retrain(profile_name: str) -> RetrainOutcome:
    from services.automl_service.config import get_profile
    from services.automl_service.trainer import run_retrain

    res = run_retrain(get_profile(profile_name), do_merge=True, gate=True)
    rows = 0
    try:
        from services.automl_service.watermark import Watermark
        rows = Watermark.load().features_rows
    except Exception:  # noqa: BLE001
        pass
    return RetrainOutcome(ok=res.ok, version=res.version,
                          features_rows=rows, error=res.error)


def _default_promote(candidate_version: str) -> PromoteOutcome:
    import pandas as pd

    from services.automl_service.promotion import (
        PromotionPolicy, decide, evaluate_candidate,
    )
    from services.ml_service.model_registry import REGISTRY

    regime = os.environ.get("AUTOML_REGIME", "plan_time")
    features_csv = PROJECT_ROOT / "data" / "processed" / "features.csv"
    df = pd.read_csv(features_csv)

    incumbent = REGISTRY.current_version(regime)
    if incumbent == candidate_version:
        incumbent = None
    cand, inc = evaluate_candidate(regime, candidate_version, incumbent, df, REGISTRY)
    decision = decide(cand, inc, PromotionPolicy())
    if decision.promote:
        REGISTRY.promote(regime, candidate_version)
        return PromoteOutcome(True, candidate_version, "; ".join(decision.reasons))
    return PromoteOutcome(False, candidate_version, "; ".join(decision.reasons))


def _default_reload() -> None:
    # Comma-separated so a multi-replica deployment reloads EVERY replica
    # (each ml-service caches its own Predictor; the registry pointer alone
    # doesn't refresh an already-running process).
    urls = os.environ.get("AUTOML_RELOAD_URL", "")
    token = os.environ.get("ML_ADMIN_TOKEN", "")
    if not urls.strip():
        return
    import urllib.request
    for base in [u.strip() for u in urls.split(",") if u.strip()]:
        req = urllib.request.Request(
            base.rstrip("/") + "/admin/reload-models",
            method="POST", headers={"X-Admin-Token": token},
        )
        urllib.request.urlopen(req, timeout=30).close()  # noqa: S310


# ---------------------------------------------------------------------------
# Loop entry point.
# ---------------------------------------------------------------------------
def main() -> int:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger("automl_worker")

    deps = WorkerDeps.default()
    log.info("automl worker up: profile=%s poll=%ss", deps.profile, POLL_INTERVAL_S)

    while True:
        try:
            state = RetrainState.load(STATE_PATH)
            result = run_cycle(state, deps)
            log.info("cycle: %s", result.as_dict())
        except Exception as exc:  # noqa: BLE001 — never die on a bad cycle
            log.exception("cycle error: %s", exc)
        time.sleep(POLL_INTERVAL_S)


if __name__ == "__main__":
    raise SystemExit(main())
