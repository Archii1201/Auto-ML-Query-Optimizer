"""
evaluate_baseline.py
====================
Phase 3G + 3H — Honest evaluation on the corrected, regenerated dataset.

This produces the *new baseline* metric panel using GroupKFold(5)
out-of-fold predictions (the only honest number for unseen queries),
across every available model family, and reports:

    plan-pick accuracy            (the deployment metric)
    median q-error / p95 q-error
    MAE / RMSE (ms)
    Spearman rho
    regret mean / p95 (ms)

Phase 3H additions (scientific rigor):
    * 95% bootstrap CI for plan-pick (resampling query-groups), so we
      can say whether a change is real or inside the noise floor.
    * Comparison against the FROZEN old baseline (pre-fix) so there is
      no goalpost-moving:
          OLD_PROD_PLAN_PICK = 0.576
          OLD_ABLATION_RANGE = 0.438 .. 0.504
      We report whether the new CI excludes the old number (=> the
      change is statistically meaningful).

Run (after extract_features + validate_dataset pass):
    python scripts/evaluate_baseline.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from phase3a.feature_selection import build_feature_matrix  # noqa: E402
from phase3b.plan_pick import evaluate_plan_pick  # noqa: E402
from scripts.experiment_ablation import make_models  # noqa: E402

warnings.filterwarnings("ignore")

FEATURES_CSV = PROJECT_ROOT / "data" / "processed" / "features.csv"
OUT_CSV = PROJECT_ROOT / "reports" / "phase3b" / "baseline_eval.csv"
N_SPLITS = 5
SEED = 42
N_BOOT = 2000

# --- FROZEN reference (pre-registered so we cannot move the goalposts) ---
OLD_PROD_PLAN_PICK = 0.576
OLD_ABLATION_BEST = 0.504
OLD_ABLATION_WORST = 0.438


def oof_predictions(model, X, y, groups):
    oof = np.zeros(len(y), dtype=float)
    y_log = np.log1p(np.clip(y, 0, None))
    for tr, te in GroupKFold(n_splits=N_SPLITS).split(X, y_log, groups):
        m = clone(model)
        m.fit(X.iloc[tr], y_log[tr])
        oof[te] = np.expm1(m.predict(X.iloc[te]))
    return oof


def q_errors(y_true, y_pred):
    yt = np.clip(y_true, 1.0, None)
    yp = np.clip(y_pred, 1.0, None)
    return np.maximum(yp / yt, yt / yp)


def spearman(y_true, y_pred) -> float:
    try:
        from scipy.stats import spearmanr
        return float(spearmanr(y_true, y_pred).correlation)
    except Exception:  # noqa: BLE001
        a = pd.Series(y_true).rank().to_numpy()
        b = pd.Series(y_pred).rank().to_numpy()
        return float(np.corrcoef(a, b)[0, 1])


def bootstrap_ci(per_group_hits: np.ndarray, n_boot: int = N_BOOT):
    """95% CI of the mean hit-rate, resampling groups with replacement."""
    if len(per_group_hits) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(SEED)
    n = len(per_group_hits)
    means = [per_group_hits[rng.integers(0, n, n)].mean() for _ in range(n_boot)]
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def evaluate_model(df, model_name, model) -> dict:
    fm = build_feature_matrix(df, regime="plan_time", drop_zero_variance=True)
    X = fm.X.reset_index(drop=True)
    y = fm.y.reset_index(drop=True).to_numpy()
    groups = fm.groups.reset_index(drop=True)

    oof = oof_predictions(model, X, y, groups)

    preds = pd.DataFrame({
        "query_id": df.reset_index(drop=True)["query_id"].astype(str),
        "variant":  df.reset_index(drop=True)["variant"].astype(str),
        "y_true":   y,
        "y_pred":   oof,
    })
    rep = evaluate_plan_pick(preds)
    qe = q_errors(y, oof)
    lo, hi = bootstrap_ci(rep.per_group["hit"].to_numpy())

    return {
        "model":          model_name,
        "n_features":     X.shape[1],
        "groups":         rep.groups_eval,
        "plan_pick":      round(rep.accuracy, 4),
        "plan_pick_lo95": round(lo, 4),
        "plan_pick_hi95": round(hi, 4),
        "q_err_median":   round(float(np.median(qe)), 4),
        "q_err_p95":      round(float(np.percentile(qe, 95)), 4),
        "mae_ms":         round(float(np.mean(np.abs(y - oof))), 1),
        "rmse_ms":        round(float(np.sqrt(np.mean((y - oof) ** 2))), 1),
        "spearman":       round(spearman(y, oof), 4),
        "regret_mean_ms": round(rep.regret_ms_mean, 1),
        "regret_p95":     round(rep.regret_ratio_p95, 4),
    }


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--id-regex", default=None,
                    help="keep only query_id matching this regex "
                         r"(e.g. '^q\d\d' for TPC-H only)")
    args = ap.parse_args()

    if not FEATURES_CSV.exists():
        print(f"[!] {FEATURES_CSV} not found — run extract_features first.",
              file=sys.stderr)
        return 1
    df = pd.read_csv(FEATURES_CSV)
    if args.id_regex:
        before = len(df)
        df = df[df["query_id"].astype(str).str.match(args.id_regex)].reset_index(drop=True)
        print(f"[i] filtered query_id ~ /{args.id_regex}/: {before} -> {len(df)} rows")
    print(f"[i] features.csv: {len(df)} rows, {df['query_id'].nunique()} groups\n")

    models = make_models()
    rows = [evaluate_model(df, name, m) for name, m in models.items()]
    res = pd.DataFrame(rows).sort_values("plan_pick", ascending=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print(res.to_string(index=False))

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT_CSV, index=False)

    best = res.iloc[0]
    print("\n" + "=" * 70)
    print("SCIENTIFIC ANALYSIS (Phase 3H)")
    print("=" * 70)
    print(f"Best model           : {best['model']}")
    print(f"Plan-pick (OOF)      : {best['plan_pick']:.3f}  "
          f"95% CI [{best['plan_pick_lo95']:.3f}, {best['plan_pick_hi95']:.3f}]")
    print(f"Frozen old baseline  : prod={OLD_PROD_PLAN_PICK:.3f}  "
          f"ablation={OLD_ABLATION_WORST:.3f}..{OLD_ABLATION_BEST:.3f}")

    lo, hi = best["plan_pick_lo95"], best["plan_pick_hi95"]
    if lo > OLD_PROD_PLAN_PICK:
        verdict = "IMPROVED and significant (CI excludes old prod baseline)"
    elif hi < OLD_PROD_PLAN_PICK:
        verdict = "REGRESSED and significant (CI below old prod baseline)"
    else:
        verdict = ("INSIDE the noise floor (CI spans old baseline) — "
                   "no statistically significant change yet")
    print(f"Verdict              : {verdict}")
    print(f"\n[OK] wrote {OUT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
