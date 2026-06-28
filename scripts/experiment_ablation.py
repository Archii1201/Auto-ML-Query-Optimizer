"""
experiment_ablation.py
=======================
Phase 3E root-cause validation. Answers, with evidence:

  * Did the Phase 3E features (knob + plan_rows) actually hurt plan-pick?
  * Which feature subset is best?
  * Which model family is best for the DEPLOYMENT metric (plan-pick)?

Method
------
For each (model, feature-subset) we run GroupKFold(5) cross-validation,
collect *out-of-fold* predictions, then compute plan-pick accuracy with
phase3b.plan_pick.evaluate_plan_pick. OOF (not in-sample) is the honest
number — it is what the deployed model will actually achieve on unseen
queries.

We train on log1p(target) and invert, exactly like the production
pipeline, so the numbers are comparable to reports/phase3b/.

Run:
    python scripts/experiment_ablation.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.model_selection import GroupKFold

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from phase3a.feature_selection import build_feature_matrix  # noqa: E402
from phase3b.plan_pick import evaluate_plan_pick  # noqa: E402

warnings.filterwarnings("ignore")

FEATURES_CSV = PROJECT_ROOT / "data" / "processed" / "features.csv"
N_SPLITS = 5
SEED = 42

KNOB_COLS = ["enable_hashjoin", "enable_mergejoin", "enable_nestloop"]
PLANROWS_COLS = [
    "plan_rows_max_node", "plan_rows_min_nonzero_node",
    "plan_rows_std_to_mean", "plan_rows_log_range",
]
PHASE3E_COLS = KNOB_COLS + PLANROWS_COLS


# ---------------------------------------------------------------------------
def make_models() -> dict:
    """Sensible fixed params (no Optuna) so every cell is directly comparable."""
    models = {
        "extra_trees": ExtraTreesRegressor(
            n_estimators=400, max_features=0.6, min_samples_leaf=2,
            random_state=SEED, n_jobs=-1,
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=400, max_features=0.6, min_samples_leaf=2,
            random_state=SEED, n_jobs=-1,
        ),
    }
    try:
        from xgboost import XGBRegressor
        models["xgboost"] = XGBRegressor(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=SEED,
            n_jobs=-1, verbosity=0,
        )
    except Exception:
        pass
    try:
        from lightgbm import LGBMRegressor
        models["lightgbm"] = LGBMRegressor(
            n_estimators=400, max_depth=-1, learning_rate=0.05,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            random_state=SEED, n_jobs=-1, verbose=-1,
        )
    except Exception:
        pass
    return models


def oof_predictions(model, X: pd.DataFrame, y: np.ndarray,
                    groups: pd.Series) -> np.ndarray:
    """GroupKFold OOF predictions in *original* (ms) space (log-target inside)."""
    oof = np.zeros(len(y), dtype=float)
    gkf = GroupKFold(n_splits=N_SPLITS)
    y_log = np.log1p(np.clip(y, 0, None))
    for tr, te in gkf.split(X, y_log, groups):
        from sklearn.base import clone
        m = clone(model)
        m.fit(X.iloc[tr], y_log[tr])
        pred_log = m.predict(X.iloc[te])
        oof[te] = np.expm1(pred_log)
    return oof


def q_error_median(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = np.clip(y_true, 1.0, None)
    yp = np.clip(y_pred, 1.0, None)
    q = np.maximum(yp / yt, yt / yp)
    return float(np.median(q))


def run_cell(df: pd.DataFrame, model_name: str, model,
             drop_extra: list[str], label: str) -> dict:
    fm = build_feature_matrix(df, regime="plan_time", drop_zero_variance=True)
    X = fm.X.reset_index(drop=True)
    keep = [c for c in X.columns if c not in drop_extra]
    X = X[keep]
    y = fm.y.reset_index(drop=True).to_numpy()
    groups = fm.groups.reset_index(drop=True)

    oof = oof_predictions(model, X, y, groups)

    meta = df.reset_index(drop=True)
    preds = pd.DataFrame({
        "query_id": meta["query_id"].astype(str),
        "variant":  meta["variant"].astype(str),
        "y_true":   y,
        "y_pred":   oof,
    })
    rep = evaluate_plan_pick(preds)
    return {
        "subset":      label,
        "model":       model_name,
        "n_features":  X.shape[1],
        "plan_pick":   round(rep.accuracy, 4),
        "groups":      rep.groups_eval,
        "regret_ms":   round(rep.regret_ms_mean, 1),
        "q_err_med":   round(q_error_median(y, oof), 4),
    }


def main() -> int:
    df = pd.read_csv(FEATURES_CSV)
    print(f"[i] features.csv: {len(df)} rows, {df['query_id'].nunique()} groups")

    subsets = {
        "B3_all_3E (current)": [],                      # keep everything
        "B0_baseline_44":      PHASE3E_COLS,            # drop all 3E
        "B1_knob_only":        PLANROWS_COLS,           # drop plan_rows, keep knob
        "B2_planrows_only":    KNOB_COLS,               # drop knob, keep plan_rows
    }

    models = make_models()
    print(f"[i] models: {list(models.keys())}")
    print(f"[i] GroupKFold({N_SPLITS}) OOF plan-pick\n")

    results = []
    for sub_label, drop_extra in subsets.items():
        for mname, model in models.items():
            row = run_cell(df, mname, model, drop_extra, sub_label)
            results.append(row)
            print(f"  {sub_label:<22} {mname:<14} "
                  f"feat={row['n_features']:>2}  "
                  f"plan_pick={row['plan_pick']:.3f}  "
                  f"q_err={row['q_err_med']:.3f}  "
                  f"regret={row['regret_ms']:.0f}ms")

    res = pd.DataFrame(results)
    out = PROJECT_ROOT / "reports" / "phase3b" / "ablation_3e.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out, index=False)

    print("\n" + "=" * 70)
    print("BEST plan-pick per feature subset:")
    for sub_label in subsets:
        sub = res[res["subset"] == sub_label].sort_values("plan_pick", ascending=False)
        top = sub.iloc[0]
        print(f"  {sub_label:<22} -> {top['model']:<14} "
              f"plan_pick={top['plan_pick']:.3f}")

    print("\nBEST overall cell:")
    best = res.sort_values("plan_pick", ascending=False).iloc[0]
    print(f"  {best['subset']} / {best['model']} = {best['plan_pick']:.3f} "
          f"(q_err {best['q_err_med']:.3f})")
    print(f"\n[OK] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
