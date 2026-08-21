"""
promotion.py
============
Phase 5C — the promotion gate.

A candidate model (registered by 5B) may only become `current` if it is
*measurably at least as good* as the incumbent on the honest, out-of-fold
plan-pick metric — with statistical significance, not noise. This module:

    1. evaluate_candidate()  — heavy: recompute OOF metrics for candidate
       AND incumbent on the SAME GroupKFold splits, plus a paired bootstrap
       CI of the plan-pick delta. (Reuses the Phase 3G/3H evaluator so the
       numbers are identical to the offline baseline.)

    2. decide()              — pure: apply the PromotionPolicy to the two
       metric bundles and return a GateDecision with per-rule reasons.
       Unit-tested with synthetic metrics — no models, no dataset.

Why separate: the gate *policy* is the risky, business-critical logic and
must be trivially testable; the *measurement* is expensive and needs the
dataset. Keeping them apart means the policy is covered by fast unit tests.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Policy + decision (PURE — unit-tested)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PromotionPolicy:
    # Candidate must gain at least this much plan-pick (point estimate).
    min_plan_pick_gain: float = 0.0
    # Paired 95% CI lower bound of Δplan-pick must be ≥ this (guards noise
    # AND caps how much of a regression we tolerate). -0.02 = "at worst 2pp".
    min_delta_lo95: float = -0.02
    # q-error (median) may not worsen by more than this.
    max_qerror_regression: float = 0.05
    # Mean plan-pick regret (ms) may not worsen by more than this.
    max_regret_regression_ms: float = 250.0


@dataclass
class GateDecision:
    promote: bool
    reasons: list[str] = field(default_factory=list)
    checks:  dict[str, bool] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"promote": self.promote, "checks": self.checks, "reasons": self.reasons}


def _finite(*xs: float) -> bool:
    return all(x is not None and np.isfinite(x) for x in xs)


def decide(cand: dict, inc: dict, policy: PromotionPolicy = PromotionPolicy()) -> GateDecision:
    """
    cand / inc metric dicts must contain: plan_pick, q_err_median,
    regret_mean_ms. cand must also contain delta_plan_pick_lo95 (paired CI
    lower bound vs incumbent).
    """
    d = GateDecision(promote=False)

    # 0) sanity / leakage guard — metrics must be finite and plausible.
    plausible = (
        _finite(cand.get("plan_pick"), inc.get("plan_pick"),
                cand.get("q_err_median"), inc.get("q_err_median"),
                cand.get("regret_mean_ms"), inc.get("regret_mean_ms"),
                cand.get("delta_plan_pick_lo95"))
        and 0.0 <= cand["plan_pick"] <= 1.0
    )
    d.checks["metrics_valid"] = plausible
    if not plausible:
        d.reasons.append("metrics missing/implausible (NaN, inf, or plan_pick∉[0,1])")
        return d

    # 1) point improvement
    gain = cand["plan_pick"] - inc["plan_pick"]
    ok_gain = gain >= policy.min_plan_pick_gain
    d.checks["plan_pick_gain"] = ok_gain
    d.reasons.append(
        f"plan_pick {inc['plan_pick']:.4f}→{cand['plan_pick']:.4f} "
        f"(Δ={gain:+.4f}, need ≥{policy.min_plan_pick_gain:+.4f}) "
        f"{'OK' if ok_gain else 'FAIL'}"
    )

    # 2) significance / non-regression (paired CI lower bound)
    lo = cand["delta_plan_pick_lo95"]
    ok_ci = lo >= policy.min_delta_lo95
    d.checks["delta_ci_lower"] = ok_ci
    d.reasons.append(
        f"Δplan_pick lo95={lo:+.4f} (need ≥{policy.min_delta_lo95:+.4f}) "
        f"{'OK' if ok_ci else 'FAIL'}"
    )

    # 3) q-error must not regress
    qreg = cand["q_err_median"] - inc["q_err_median"]
    ok_q = qreg <= policy.max_qerror_regression
    d.checks["qerror"] = ok_q
    d.reasons.append(
        f"q_err_median {inc['q_err_median']:.4f}→{cand['q_err_median']:.4f} "
        f"(Δ={qreg:+.4f}, allow ≤{policy.max_qerror_regression:+.4f}) "
        f"{'OK' if ok_q else 'FAIL'}"
    )

    # 4) regret must not regress
    rreg = cand["regret_mean_ms"] - inc["regret_mean_ms"]
    ok_r = rreg <= policy.max_regret_regression_ms
    d.checks["regret"] = ok_r
    d.reasons.append(
        f"regret_mean_ms {inc['regret_mean_ms']:.1f}→{cand['regret_mean_ms']:.1f} "
        f"(Δ={rreg:+.1f}, allow ≤{policy.max_regret_regression_ms:+.1f}) "
        f"{'OK' if ok_r else 'FAIL'}"
    )

    d.promote = all(d.checks.values())
    return d


# ---------------------------------------------------------------------------
# Measurement (HEAVY — needs the dataset + sklearn)
# ---------------------------------------------------------------------------
def _load_estimator(regime: str, version: str, registry):
    import joblib
    path = registry.resolve_artifact(regime, version)
    return joblib.load(path)["model"]


def _paired_delta_lo95(df, cand_est, inc_est, *, n_boot: int = 2000, seed: int = 42) -> float:
    """
    Paired bootstrap 95% CI lower bound of (cand hit-rate − inc hit-rate),
    resampling query-groups. Both estimators share the same GroupKFold OOF
    splits so the comparison is apples-to-apples.
    """
    import pandas as pd
    from phase3a.feature_selection import build_feature_matrix
    from phase3b.plan_pick import evaluate_plan_pick
    from scripts.evaluate_baseline import oof_predictions

    fm = build_feature_matrix(df, regime="plan_time", drop_zero_variance=True)
    X = fm.X.reset_index(drop=True)
    y = fm.y.reset_index(drop=True).to_numpy()
    groups = fm.groups.reset_index(drop=True)
    base = df.reset_index(drop=True)

    def per_group_hits(est):
        oof = oof_predictions(est, X, y, groups)
        preds = pd.DataFrame({
            "query_id": base["query_id"].astype(str),
            "variant":  base["variant"].astype(str),
            "y_true":   y,
            "y_pred":   oof,
        })
        rep = evaluate_plan_pick(preds)
        return rep.per_group.set_index("query_id")["hit"]

    ch = per_group_hits(cand_est).rename("c")
    ih = per_group_hits(inc_est).rename("i")
    joined = pd.concat([ch, ih], axis=1).dropna()
    delta = (joined["c"] - joined["i"]).to_numpy(dtype=float)
    if len(delta) == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    n = len(delta)
    means = [delta[rng.integers(0, n, n)].mean() for _ in range(n_boot)]
    return float(np.percentile(means, 2.5))


def evaluate_candidate(regime: str, cand_version: str, inc_version: str | None, df, registry):
    """
    Return (cand_metrics, inc_metrics) suitable for decide().
    inc_version=None means "no incumbent" → incumbent metrics are all-zero
    so any finite candidate clears the gate on first promotion.
    """
    from scripts.evaluate_baseline import evaluate_model

    cand_est = _load_estimator(regime, cand_version, registry)
    cand = evaluate_model(df, "candidate", cand_est)

    if inc_version is None:
        inc = {"plan_pick": 0.0, "q_err_median": float("inf"),
               "regret_mean_ms": float("inf")}
        cand["delta_plan_pick_lo95"] = cand["plan_pick"]  # trivially positive
        # inf incumbent q/regret makes those checks pass on first promotion.
        inc["q_err_median"] = cand["q_err_median"] + 1e9
        inc["regret_mean_ms"] = cand["regret_mean_ms"] + 1e9
        return cand, inc

    inc_est = _load_estimator(regime, inc_version, registry)
    inc = evaluate_model(df, "incumbent", inc_est)
    cand["delta_plan_pick_lo95"] = _paired_delta_lo95(df, cand_est, inc_est)
    return cand, inc
