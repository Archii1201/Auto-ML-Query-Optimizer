"""
Unit tests for Phase 5E — the worker cycle orchestration and single-flight
lock. All steps are faked; no trainer, registry, Postgres, or network.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from services.automl_service.filelock import FileLock, LockBusy
from services.automl_service.state import RetrainState
from services.automl_service.triggers import TriggerConfig, TriggerSignals
from services.automl_service.worker import (
    PromoteOutcome,
    RetrainOutcome,
    WorkerDeps,
    run_cycle,
)


def _deps(*, signals, retrain=None, promote=None, reload=None, config=None):
    return WorkerDeps(
        signals=lambda st: signals,
        retrain=retrain or (lambda p: RetrainOutcome(ok=True, version="v1", features_rows=10)),
        promote=promote or (lambda v: PromoteOutcome(promoted=True, version=v, reason="ok")),
        reload=reload or (lambda: None),
        config=config or TriggerConfig(cooldown_minutes=360),
    )


def test_no_trigger_does_nothing(tmp_path):
    st = RetrainState()
    st.last_retrain_utc = RetrainState().last_retrain_utc  # empty
    # in cooldown: pretend retrained 10m ago
    from services.automl_service.state import now_utc
    from datetime import timedelta
    st.last_retrain_utc = (now_utc() - timedelta(minutes=10)).isoformat()

    deps = _deps(signals=TriggerSignals(new_rows=10_000, minutes_since_retrain=10))
    res = run_cycle(st, deps, state_path=tmp_path / "s.json", lock_path=tmp_path / "l.lock")
    assert res.trigger.should_retrain is False
    assert res.retrained is False


def test_full_retrain_and_promote(tmp_path):
    reloaded = {"n": 0}
    deps = _deps(
        signals=TriggerSignals(minutes_since_retrain=None),  # bootstrap
        reload=lambda: reloaded.__setitem__("n", reloaded["n"] + 1),
    )
    st = RetrainState()
    res = run_cycle(st, deps, state_path=tmp_path / "s.json", lock_path=tmp_path / "l.lock")
    assert res.retrained is True
    assert res.promoted is True
    assert res.version == "v1"
    assert reloaded["n"] == 1
    assert st.retrains_total == 1 and st.promotions_total == 1


def test_retrain_ok_but_gate_rejects(tmp_path):
    reloaded = {"n": 0}
    deps = _deps(
        signals=TriggerSignals(minutes_since_retrain=None),
        promote=lambda v: PromoteOutcome(promoted=False, version=v, reason="rejected"),
        reload=lambda: reloaded.__setitem__("n", reloaded["n"] + 1),
    )
    st = RetrainState()
    res = run_cycle(st, deps, state_path=tmp_path / "s.json", lock_path=tmp_path / "l.lock")
    assert res.retrained is True
    assert res.promoted is False
    assert reloaded["n"] == 0            # no reload when not promoted
    assert st.retrains_total == 1 and st.promotions_total == 0


def test_retrain_failure_no_promote(tmp_path):
    deps = _deps(
        signals=TriggerSignals(minutes_since_retrain=None),
        retrain=lambda p: RetrainOutcome(ok=False, error="boom"),
        promote=lambda v: pytest.fail("promote must not run when retrain fails"),
    )
    st = RetrainState()
    res = run_cycle(st, deps, state_path=tmp_path / "s.json", lock_path=tmp_path / "l.lock")
    assert res.retrained is False
    assert "boom" in res.note


def test_lock_busy_skips_cycle(tmp_path):
    lock_path = tmp_path / "l.lock"
    held = FileLock(lock_path).acquire()          # simulate another holder
    try:
        deps = _deps(
            signals=TriggerSignals(minutes_since_retrain=None),
            retrain=lambda p: pytest.fail("must not retrain while locked"),
        )
        res = run_cycle(RetrainState(), deps,
                        state_path=tmp_path / "s.json", lock_path=lock_path)
        assert res.retrained is False
        assert "in progress" in res.note
    finally:
        held.release()


# ----- filelock ------------------------------------------------------------
def test_filelock_exclusive(tmp_path):
    p = tmp_path / "x.lock"
    a = FileLock(p).acquire()
    with pytest.raises(LockBusy):
        FileLock(p).acquire()
    a.release()
    FileLock(p).acquire().release()   # free again


def test_filelock_reclaims_stale(tmp_path):
    p = tmp_path / "x.lock"
    FileLock(p).acquire()             # leave it held
    time.sleep(0.05)
    # stale_after_s tiny → the next acquire reclaims it
    got = FileLock(p, stale_after_s=0.01).acquire()
    assert p.exists()
    got.release()


def test_filelock_context_manager(tmp_path):
    p = tmp_path / "x.lock"
    with FileLock(p):
        assert p.exists()
    assert not p.exists()
