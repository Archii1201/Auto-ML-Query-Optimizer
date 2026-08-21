"""
Unit tests for Phase 5C — the promotion gate (pure decision logic) and the
registry promote/rollback roundtrip.

The heavy OOF measurement (evaluate_candidate) needs the dataset + sklearn;
here we test the *policy*, which is the safety-critical part, with synthetic
metric bundles.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from services.automl_service.promotion import (
    GateDecision,
    PromotionPolicy,
    decide,
)
from services.ml_service.model_registry import ModelRegistry


def _metrics(plan_pick, q, regret, lo95=None):
    d = {"plan_pick": plan_pick, "q_err_median": q, "regret_mean_ms": regret}
    if lo95 is not None:
        d["delta_plan_pick_lo95"] = lo95
    return d


INC = _metrics(0.60, 1.25, 400.0)


def test_clear_improvement_promotes():
    cand = _metrics(0.66, 1.20, 350.0, lo95=0.01)
    d = decide(cand, INC)
    assert d.promote is True
    assert all(d.checks.values())


def test_noise_level_gain_blocked_by_ci():
    # point gain positive, but paired CI lower bound dips below tolerance.
    cand = _metrics(0.61, 1.25, 400.0, lo95=-0.05)
    d = decide(cand, INC)
    assert d.promote is False
    assert d.checks["delta_ci_lower"] is False


def test_qerror_regression_blocks():
    cand = _metrics(0.66, 1.40, 350.0, lo95=0.02)  # q-error worsened a lot
    d = decide(cand, INC, PromotionPolicy(max_qerror_regression=0.05))
    assert d.promote is False
    assert d.checks["qerror"] is False


def test_regret_regression_blocks():
    cand = _metrics(0.66, 1.20, 700.0, lo95=0.02)  # +300ms regret
    d = decide(cand, INC, PromotionPolicy(max_regret_regression_ms=250.0))
    assert d.promote is False
    assert d.checks["regret"] is False


def test_no_point_gain_blocks():
    cand = _metrics(0.60, 1.20, 350.0, lo95=0.0)  # equal plan-pick, need >0 gain
    d = decide(cand, INC, PromotionPolicy(min_plan_pick_gain=0.005))
    assert d.promote is False
    assert d.checks["plan_pick_gain"] is False


def test_nan_metrics_rejected():
    cand = _metrics(float("nan"), 1.2, 350.0, lo95=0.02)
    d = decide(cand, INC)
    assert d.promote is False
    assert d.checks["metrics_valid"] is False


def test_implausible_plan_pick_rejected():
    cand = _metrics(1.5, 1.2, 350.0, lo95=0.02)  # >1.0 impossible
    d = decide(cand, INC)
    assert d.promote is False


def test_decision_serializes():
    cand = _metrics(0.66, 1.2, 350.0, lo95=0.02)
    d = decide(cand, INC)
    js = d.as_dict()
    assert set(js) == {"promote", "checks", "reasons"}
    assert isinstance(d, GateDecision)


# ----- registry promote / rollback roundtrip ------------------------------
def _fake_artifact(tmp_path: Path, name: str, payload: bytes) -> Path:
    p = tmp_path / name
    p.write_bytes(payload)
    return p


def test_registry_promote_and_rollback(tmp_path):
    reg = ModelRegistry(registry_dir=tmp_path / "registry")
    a1 = _fake_artifact(tmp_path, "m1.joblib", b"model-one")
    a2 = _fake_artifact(tmp_path, "m2.joblib", b"model-two")

    v1 = reg.register("plan_time", a1, promote=True)
    assert reg.current_version("plan_time") == v1

    v2 = reg.register("plan_time", a2, promote=False)  # candidate, not current
    assert reg.current_version("plan_time") == v1

    reg.promote("plan_time", v2)                        # promote candidate
    assert reg.current_version("plan_time") == v2

    reg.promote("plan_time", v1)                        # rollback (5F path)
    assert reg.current_version("plan_time") == v1
