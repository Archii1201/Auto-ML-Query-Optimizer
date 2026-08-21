"""
Unit tests for Phase 5B — retrain profiles + trainer orchestration.

We inject fake train/register/merge steps so no real trainer runs and no
Postgres/registry is touched. This exercises the control flow: gate abort,
train failure, missing artifact, and the happy path.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from services.automl_service import trainer as trainer_mod
from services.automl_service.config import get_profile, RetrainProfile


# ----- config / profiles --------------------------------------------------
def test_fast_profile_skips_tuning():
    p = get_profile("fast")
    assert p.skip_tuning is True
    assert "--skip-tuning" in p.train_argv()


def test_full_profile_tunes():
    p = get_profile("full")
    assert p.skip_tuning is False
    argv = p.train_argv()
    assert "--n-trials" in argv and "--skip-tuning" not in argv


def test_unknown_profile_raises():
    with pytest.raises(ValueError):
        get_profile("turbo")


def test_train_argv_includes_regime():
    p = RetrainProfile(name="x", skip_tuning=True, regimes=("plan_time",))
    argv = p.train_argv()
    assert "--regimes" in argv and "plan_time" in argv


# ----- trainer control flow ----------------------------------------------
def _no_merge_result(monkeypatch, artifact_exists: bool):
    """Patch artifact_for to a temp path we control existence of."""
    return artifact_exists


def test_happy_path_registers_candidate(monkeypatch, tmp_path):
    artifact = tmp_path / "automl_best.joblib"
    artifact.write_bytes(b"fake-model")

    monkeypatch.setattr(trainer_mod, "artifact_for", lambda regime: artifact)

    registered = {}

    def fake_train(profile):
        return 0

    def fake_register(regime, path):
        registered["regime"] = regime
        registered["path"] = path
        return "deadbeefcafe0001"

    res = trainer_mod.run_retrain(
        get_profile("fast"),
        do_merge=False,
        gate=False,
        train_fn=fake_train,
        register_fn=fake_register,
    )
    assert res.ok is True
    assert res.version == "deadbeefcafe0001"
    assert registered["regime"] == "plan_time"


def test_train_failure_aborts(monkeypatch, tmp_path):
    monkeypatch.setattr(trainer_mod, "artifact_for", lambda regime: tmp_path / "x.joblib")
    res = trainer_mod.run_retrain(
        get_profile("fast"),
        do_merge=False, gate=False,
        train_fn=lambda p: 1,
        register_fn=lambda r, p: "nope",
    )
    assert res.ok is False
    assert "exited 1" in (res.error or "")


def test_missing_artifact_aborts(monkeypatch, tmp_path):
    missing = tmp_path / "does_not_exist.joblib"
    monkeypatch.setattr(trainer_mod, "artifact_for", lambda regime: missing)
    res = trainer_mod.run_retrain(
        get_profile("fast"),
        do_merge=False, gate=False,
        train_fn=lambda p: 0,
        register_fn=lambda r, p: "nope",
    )
    assert res.ok is False
    assert "artifact missing" in (res.error or "")


def test_gate_failure_aborts_before_training(monkeypatch, tmp_path):
    called = {"train": False}
    monkeypatch.setattr(trainer_mod, "merge_feedback",
                        lambda **kw: type("R", (), {"new_rows": 3})())
    monkeypatch.setattr(trainer_mod, "run_validation_gate",
                        lambda **kw: (False, "boom"))

    def fake_train(p):
        called["train"] = True
        return 0

    res = trainer_mod.run_retrain(
        get_profile("fast"),
        do_merge=True, gate=True,
        train_fn=fake_train,
        register_fn=lambda r, p: "x",
    )
    assert res.ok is False
    assert called["train"] is False
    assert "gate failed" in (res.error or "")
