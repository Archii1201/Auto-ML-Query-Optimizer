"""
train_models.py
===============
Phase 3A entry point.

For each of the two regimes (plan-time, post-mortem) and each of the
nine models, this script:

    1. Builds (X, y, groups) from data/processed/features.csv.
    2. Runs 5-fold GroupKFold cross-validation grouped by query_id
       (so a query never appears in train and test simultaneously).
    3. Trains every model on log1p(y) and evaluates on the
       back-transformed prediction.
    4. Computes MAE / RMSE / R^2 / MAPE / q-error (median + p95) /
       Spearman ρ per fold and averages them.
    5. Refits the model on the full dataset and persists it to
       models/phase3a/{regime}/{model}.joblib alongside its feature
       list and target transform.

Outputs:
    reports/phase3a/model_comparison.csv
    reports/phase3a/model_comparison.md
    reports/phase3a/cv_predictions.parquet     (fold-level y_true/y_pred)
    reports/phase3a/feature_importance.csv     (tree models)
    models/phase3a/{regime}/{slug}.joblib
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from phase3a.baseline import LinearCostBaseline, LogLinearCostBaseline  # noqa: E402
from phase3a.evaluation import (  # noqa: E402
    Metrics,
    average_fold_metrics,
    compute_metrics,
)
from phase3a.feature_selection import (  # noqa: E402
    FeatureMatrix,
    build_feature_matrix,
)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
FEATURES_CSV = PROJECT_ROOT / "data" / "processed" / "features.csv"
REPORT_DIR   = PROJECT_ROOT / "reports" / "phase3a"
MODELS_DIR   = PROJECT_ROOT / "models" / "phase3a"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Model zoo
# ---------------------------------------------------------------------------
RANDOM_STATE: int = 42
N_SPLITS:     int = 5


def _xgb() -> Any:
    from xgboost import XGBRegressor
    return XGBRegressor(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        tree_method="hist",
        n_jobs=-1,
        verbosity=0,
    )


def _lgbm() -> Any:
    from lightgbm import LGBMRegressor
    return LGBMRegressor(
        n_estimators=600,
        max_depth=-1,
        num_leaves=31,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
        min_data_in_leaf=2,
        min_split_gain=0.0,
    )


def _catboost() -> Any:
    from catboost import CatBoostRegressor
    return CatBoostRegressor(
        iterations=600,
        depth=6,
        learning_rate=0.05,
        l2_leaf_reg=3.0,
        random_seed=RANDOM_STATE,
        loss_function="RMSE",
        verbose=False,
        allow_writing_files=False,
    )


def linear_pipeline(estimator: Any) -> Pipeline:
    """Standardize features for linear models (gradient boosters don't need it)."""
    return Pipeline([("scaler", StandardScaler()), ("est", estimator)])


def model_zoo() -> dict[str, Callable[[], Any]]:
    """Name → factory. Factories are called once per fold for a fresh estimator."""
    return {
        "linear_regression":  lambda: linear_pipeline(LinearRegression()),
        "ridge":              lambda: linear_pipeline(Ridge(alpha=1.0, random_state=RANDOM_STATE)),
        "lasso":              lambda: linear_pipeline(Lasso(alpha=0.001, max_iter=20000,
                                                            random_state=RANDOM_STATE)),
        "random_forest":      lambda: RandomForestRegressor(
            n_estimators=400, max_depth=None, min_samples_leaf=1,
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "extra_trees":        lambda: ExtraTreesRegressor(
            n_estimators=400, max_depth=None, min_samples_leaf=1,
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "gradient_boosting":  lambda: GradientBoostingRegressor(
            n_estimators=400, max_depth=3, learning_rate=0.05,
            subsample=0.9, random_state=RANDOM_STATE,
        ),
        "xgboost":            _xgb,
        "lightgbm":           _lgbm,
        "catboost":           _catboost,
    }


BASELINES: dict[str, Callable[[], Any]] = {
    "pg_baseline_linear":     LinearCostBaseline,
    "pg_baseline_loglinear":  LogLinearCostBaseline,
}


# ---------------------------------------------------------------------------
# Cross-validation loop
# ---------------------------------------------------------------------------
@dataclass
class CVResult:
    model:       str
    regime:      str
    fold_metrics: list[Metrics] = field(default_factory=list)
    fold_preds:   list[pd.DataFrame] = field(default_factory=list)
    train_seconds: float = 0.0


def run_cv(
    name: str,
    factory: Callable[[], Any],
    fm: FeatureMatrix,
    log_target: bool,
) -> CVResult:
    """5-fold GroupKFold CV. log_target=False for the PG baselines."""
    cv = GroupKFold(n_splits=N_SPLITS)
    result = CVResult(model=name, regime=fm.regime)

    t0 = time.perf_counter()
    for fold_idx, (tr, te) in enumerate(cv.split(fm.X, fm.y, fm.groups)):
        X_tr, X_te = fm.X.iloc[tr], fm.X.iloc[te]
        y_tr, y_te = fm.y.iloc[tr].to_numpy(), fm.y.iloc[te].to_numpy()

        model = factory()
        if log_target:
            model.fit(X_tr, np.log1p(np.maximum(y_tr, 0.0)))
            y_pred = np.expm1(model.predict(X_te))
        else:
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)

        # Clip predictions to a sane physical range so a single
        # blown-up linear-regression fold doesn't destroy the
        # cross-fold mean. Floor at 0.1 ms (sub-ms is meaningless here)
        # and cap at 10x the largest observed runtime in this dataset.
        cap = 10.0 * float(fm.y.max()) if len(fm.y) else 1e7
        y_pred = np.where(np.isfinite(y_pred), y_pred, cap)
        y_pred = np.clip(y_pred, 0.1, cap)

        m = compute_metrics(y_te, y_pred)
        result.fold_metrics.append(m)

        result.fold_preds.append(pd.DataFrame({
            "fold":     fold_idx,
            "model":    name,
            "regime":   fm.regime,
            "query_id": fm.groups.iloc[te].to_numpy(),
            "y_true":   y_te,
            "y_pred":   y_pred,
        }))

    result.train_seconds = time.perf_counter() - t0
    return result


# ---------------------------------------------------------------------------
# Refit-on-full-data + persist
# ---------------------------------------------------------------------------
def fit_and_save_full(
    name: str,
    factory: Callable[[], Any],
    fm: FeatureMatrix,
    log_target: bool,
) -> Path:
    out_dir = MODELS_DIR / fm.regime
    out_dir.mkdir(parents=True, exist_ok=True)
    model = factory()
    if log_target:
        model.fit(fm.X, np.log1p(np.maximum(fm.y.to_numpy(), 0.0)))
    else:
        model.fit(fm.X, fm.y.to_numpy())

    artifact = {
        "model":       model,
        "feature_names": fm.feature_names,
        "regime":      fm.regime,
        "log_target":  log_target,
        "model_name":  name,
        "trained_at":  pd.Timestamp.utcnow().isoformat(),
    }
    out_path = out_dir / f"{name}.joblib"
    joblib.dump(artifact, out_path)
    return out_path


# ---------------------------------------------------------------------------
# Feature importance harvest
# ---------------------------------------------------------------------------
def _extract_importance(model: Any, feature_names: list[str]) -> dict[str, float] | None:
    if isinstance(model, Pipeline):
        model = model.named_steps.get("est", model)

    if hasattr(model, "feature_importances_"):
        vals = np.asarray(model.feature_importances_, dtype=float)
        if vals.shape[0] == len(feature_names):
            return dict(zip(feature_names, vals.tolist()))
    if hasattr(model, "coef_"):
        coef = np.asarray(model.coef_, dtype=float).ravel()
        if coef.shape[0] == len(feature_names):
            return {n: abs(v) for n, v in zip(feature_names, coef.tolist())}
    return None


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main() -> int:
    print(f"[i] Reading features from {FEATURES_CSV.relative_to(PROJECT_ROOT)}")
    df = pd.read_csv(FEATURES_CSV)
    print(f"[i] Loaded {len(df)} rows, {df.shape[1]} columns")
    n_groups = df["query_id"].nunique()
    print(f"[i] Distinct query_ids (CV groups): {n_groups}")
    if n_groups < N_SPLITS:
        print(f"[!] only {n_groups} groups; reducing splits to {n_groups}", file=sys.stderr)

    cmp_rows: list[dict[str, Any]] = []
    pred_chunks: list[pd.DataFrame] = []
    importance_rows: list[dict[str, Any]] = []

    for regime in ("plan_time", "post_mortem"):
        print(f"\n=== Regime: {regime} ===")
        fm = build_feature_matrix(df, regime=regime, drop_zero_variance=True)
        print(f"[i] X shape: {fm.X.shape}, features: {len(fm.feature_names)}")

        for bname, bfactory in BASELINES.items():
            res = run_cv(bname, bfactory, fm, log_target=False)
            agg = average_fold_metrics(res.fold_metrics)
            cmp_rows.append({
                "model":  bname,
                "regime": regime,
                "kind":   "baseline",
                "train_seconds": round(res.train_seconds, 3),
                **agg,
            })
            pred_chunks.extend(res.fold_preds)
            print(
                f"  [B] {bname:<24} R2={agg['r2_mean']:+.3f}  "
                f"MAE={agg['mae_mean']:.1f}ms  qerr_med={agg['q_error_median_mean']:.2f}"
            )

        for mname, mfactory in model_zoo().items():
            res = run_cv(mname, mfactory, fm, log_target=True)
            agg = average_fold_metrics(res.fold_metrics)
            cmp_rows.append({
                "model":  mname,
                "regime": regime,
                "kind":   "ml",
                "train_seconds": round(res.train_seconds, 3),
                **agg,
            })
            pred_chunks.extend(res.fold_preds)
            print(
                f"  [M] {mname:<24} R2={agg['r2_mean']:+.3f}  "
                f"MAE={agg['mae_mean']:.1f}ms  qerr_med={agg['q_error_median_mean']:.2f}"
            )

            saved_path = fit_and_save_full(mname, mfactory, fm, log_target=True)
            artifact   = joblib.load(saved_path)
            importance = _extract_importance(artifact["model"], fm.feature_names)
            if importance is not None:
                for f, v in importance.items():
                    importance_rows.append({
                        "model":   mname,
                        "regime":  regime,
                        "feature": f,
                        "importance": float(v),
                    })

        for bname, bfactory in BASELINES.items():
            fit_and_save_full(bname, bfactory, fm, log_target=False)

    cmp_df = pd.DataFrame(cmp_rows)
    cmp_df = cmp_df.sort_values(["regime", "q_error_median_mean", "rmse_mean"]).reset_index(drop=True)
    cmp_df.to_csv(REPORT_DIR / "model_comparison.csv", index=False)

    pred_df = pd.concat(pred_chunks, ignore_index=True)
    pred_df.to_csv(REPORT_DIR / "cv_predictions.csv", index=False)

    if importance_rows:
        imp_df = pd.DataFrame(importance_rows)
        imp_df = imp_df.sort_values(["regime", "model", "importance"], ascending=[True, True, False])
        imp_df.to_csv(REPORT_DIR / "feature_importance.csv", index=False)

    _write_markdown_summary(cmp_df)

    print("\n[OK] Phase 3A training complete.")
    print(f"[OK] Comparison table : {REPORT_DIR / 'model_comparison.csv'}")
    print(f"[OK] Markdown summary : {REPORT_DIR / 'model_comparison.md'}")
    print(f"[OK] CV predictions   : {REPORT_DIR / 'cv_predictions.csv'}")
    print(f"[OK] Feature imps.    : {REPORT_DIR / 'feature_importance.csv'}")
    print(f"[OK] Saved models     : {MODELS_DIR}")
    return 0


# ---------------------------------------------------------------------------
def _write_markdown_summary(cmp_df: pd.DataFrame) -> None:
    cols = [
        "regime", "kind", "model",
        "r2_mean", "mae_mean", "rmse_mean",
        "mape_pct_mean", "q_error_median_mean", "q_error_p95_mean",
        "spearman_rho_mean", "train_seconds",
    ]
    show = cmp_df[cols].copy()
    rename = {
        "r2_mean": "R²",
        "mae_mean": "MAE (ms)",
        "rmse_mean": "RMSE (ms)",
        "mape_pct_mean": "MAPE (%)",
        "q_error_median_mean": "q-err median",
        "q_error_p95_mean": "q-err p95",
        "spearman_rho_mean": "Spearman ρ",
        "train_seconds": "Train (s)",
    }
    show = show.rename(columns=rename)
    for c in show.select_dtypes(include="float").columns:
        show[c] = show[c].round(3)

    lines = [
        "# Phase 3A — Model Comparison",
        "",
        "5-fold **GroupKFold** cross-validation grouped by `query_id` "
        "(no query appears in both train and test).",
        "",
        "Models are trained on `log1p(execution_time_ms)` and evaluated on the "
        "back-transformed prediction; PostgreSQL baselines are uncalibrated "
        "(linear) and log-linear calibrated.",
        "",
        "Sort: ascending **median q-error**, then ascending RMSE.",
        "",
    ]
    for regime, grp in show.groupby("regime"):
        lines.append(f"## Regime: `{regime}`")
        lines.append("")
        lines.append(grp.to_markdown(index=False))
        lines.append("")
    (REPORT_DIR / "model_comparison.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
