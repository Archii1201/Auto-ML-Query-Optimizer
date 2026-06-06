"""
phase3b/train_models.py
=======================
Phase 3B entry point — AutoML-tuned learned cost model.

Pipeline per regime (`plan_time` / `post_mortem`):

    1. Load features.csv and assemble (X, y, groups).
    2. Run baseline CV  : pg_baseline_linear, pg_baseline_loglinear.
    3. Run default CV   : every model in `model_zoo()` with
                          *out-of-the-box* params (so we can show
                          the lift Optuna provides).
    4. Run TUNED CV     : Optuna-optimised RF / ET / GB / XGB / LGBM,
                          minimising q-error MEDIAN.
    5. Plan-pick scoring: for each model+regime, group predictions
                          by query_id and compute pick accuracy +
                          regret using phase3b.plan_pick.
    6. AutoML selection : pick the single best model per regime by
                          composite score (q-err median, then plan-
                          pick accuracy). Save under
                          models/phase3b/{regime}/automl_best.joblib.
    7. Refit on full data and persist every tuned model.

Outputs:
    reports/phase3b/model_comparison.csv
    reports/phase3b/model_comparison.md
    reports/phase3b/cv_predictions.csv
    reports/phase3b/feature_importance.csv
    reports/phase3b/plan_pick_summary.csv
    reports/phase3b/plan_pick_detail.csv
    reports/phase3b/optuna_history.csv
    reports/phase3b/automl_winner.json
    models/phase3b/{regime}/{model_name}.joblib
    models/phase3b/{regime}/automl_best.joblib
"""

from __future__ import annotations

import argparse
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
from sklearn.linear_model import ElasticNet, Lasso, Ridge
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
from phase3a.feature_selection import FeatureMatrix, build_feature_matrix  # noqa: E402
from phase3b.plan_pick import evaluate_plan_pick  # noqa: E402
from phase3b.tuning import TUNABLE, build_tuned, tune_model  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
FEATURES_CSV = PROJECT_ROOT / "data" / "processed" / "features.csv"
REPORT_DIR   = PROJECT_ROOT / "reports" / "phase3b"
MODELS_DIR   = PROJECT_ROOT / "models" / "phase3b"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE: int = 42
N_SPLITS:     int = 5


# ---------------------------------------------------------------------------
# Default model zoo (Phase 3B drops unstable plain LinearRegression;
# adds ElasticNet to keep a regularized linear option around).
# ---------------------------------------------------------------------------
def _xgb() -> Any:
    from xgboost import XGBRegressor
    return XGBRegressor(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
        random_state=RANDOM_STATE, tree_method="hist",
        n_jobs=-1, verbosity=0,
    )


def _lgbm() -> Any:
    from lightgbm import LGBMRegressor
    return LGBMRegressor(
        n_estimators=600, max_depth=-1, num_leaves=31, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
        random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1,
        min_data_in_leaf=2, min_split_gain=0.0,
    )


def _catboost() -> Any:
    from catboost import CatBoostRegressor
    return CatBoostRegressor(
        iterations=600, depth=6, learning_rate=0.05, l2_leaf_reg=3.0,
        random_seed=RANDOM_STATE, loss_function="RMSE",
        verbose=False, allow_writing_files=False,
    )


def linear_pipeline(estimator: Any) -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("est", estimator)])


def model_zoo() -> dict[str, Callable[[], Any]]:
    return {
        "ridge": lambda: linear_pipeline(Ridge(alpha=1.0, random_state=RANDOM_STATE)),
        "lasso": lambda: linear_pipeline(Lasso(alpha=0.001, max_iter=20000,
                                               random_state=RANDOM_STATE)),
        "elasticnet": lambda: linear_pipeline(ElasticNet(alpha=0.001, l1_ratio=0.5,
                                                        max_iter=20000,
                                                        random_state=RANDOM_STATE)),
        "random_forest": lambda: RandomForestRegressor(
            n_estimators=400, max_depth=None, min_samples_leaf=1,
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "extra_trees": lambda: ExtraTreesRegressor(
            n_estimators=400, max_depth=None, min_samples_leaf=1,
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "gradient_boosting": lambda: GradientBoostingRegressor(
            n_estimators=400, max_depth=3, learning_rate=0.05,
            subsample=0.9, random_state=RANDOM_STATE,
        ),
        "xgboost":   _xgb,
        "lightgbm":  _lgbm,
        "catboost":  _catboost,
    }


BASELINES: dict[str, Callable[[], Any]] = {
    "pg_baseline_linear":    LinearCostBaseline,
    "pg_baseline_loglinear": LogLinearCostBaseline,
}


# ---------------------------------------------------------------------------
# CV
# ---------------------------------------------------------------------------
@dataclass
class CVResult:
    model:        str
    regime:       str
    kind:         str           # "baseline" | "default_ml" | "tuned_ml"
    fold_metrics: list[Metrics] = field(default_factory=list)
    fold_preds:   list[pd.DataFrame] = field(default_factory=list)
    train_seconds: float = 0.0


def run_cv(
    name: str,
    factory: Callable[[], Any],
    fm: FeatureMatrix,
    *,
    kind: str,
    log_target: bool,
    variant_series: pd.Series | None = None,
) -> CVResult:
    """variant_series must align with fm.X by positional index."""
    cv = GroupKFold(n_splits=N_SPLITS)
    out = CVResult(model=name, regime=fm.regime, kind=kind)
    cap = 10.0 * float(fm.y.max()) if len(fm.y) else 1e7

    if variant_series is None:
        variant_series = pd.Series([""] * len(fm.X))

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

        y_pred = np.where(np.isfinite(y_pred), y_pred, cap)
        y_pred = np.clip(y_pred, 0.1, cap)

        out.fold_metrics.append(compute_metrics(y_te, y_pred))
        out.fold_preds.append(pd.DataFrame({
            "fold":     fold_idx,
            "model":    name,
            "regime":   fm.regime,
            "kind":     kind,
            "query_id": fm.groups.iloc[te].to_numpy(),
            "variant":  variant_series.iloc[te].to_numpy(),
            "y_true":   y_te,
            "y_pred":   y_pred,
        }))

    out.train_seconds = time.perf_counter() - t0
    return out


# ---------------------------------------------------------------------------
# Refit + persist
# ---------------------------------------------------------------------------
def fit_and_save_full(
    name: str,
    factory: Callable[[], Any],
    fm: FeatureMatrix,
    log_target: bool,
    extra_meta: dict[str, Any] | None = None,
) -> Path:
    out_dir = MODELS_DIR / fm.regime
    out_dir.mkdir(parents=True, exist_ok=True)
    model = factory()
    if log_target:
        model.fit(fm.X, np.log1p(np.maximum(fm.y.to_numpy(), 0.0)))
    else:
        model.fit(fm.X, fm.y.to_numpy())
    artifact = {
        "model":         model,
        "feature_names": fm.feature_names,
        "regime":        fm.regime,
        "log_target":    log_target,
        "model_name":    name,
        "trained_at":    pd.Timestamp.utcnow().isoformat(),
        **(extra_meta or {}),
    }
    out_path = out_dir / f"{name}.joblib"
    joblib.dump(artifact, out_path)
    return out_path


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
# Plan-pick eval (joins variant col onto fold predictions)
# ---------------------------------------------------------------------------
def _plan_pick_for(preds: pd.DataFrame, df_meta: pd.DataFrame,
                   model_name: str, regime: str) -> tuple[dict, pd.DataFrame]:
    """
    For one (model, regime), aggregate the fold-level predictions back
    onto the original (query_id, variant) keys and run plan_pick eval.
    """
    sub = preds[(preds["model"] == model_name) & (preds["regime"] == regime)].copy()
    if "variant" not in sub.columns or (sub["variant"] == "").all():
        # Fallback: align by index using df_meta
        return ({"model": model_name, "regime": regime,
                 "accuracy": np.nan, "groups_eval": 0,
                 "regret_ms_mean": np.nan, "regret_ratio_p95": np.nan},
                pd.DataFrame())

    rep = evaluate_plan_pick(
        sub.rename(columns={"variant": "variant"}),
        group_col="query_id", truth_col="y_true",
        predict_col="y_pred", variant_col="variant",
    )
    summary = {
        "model":            model_name,
        "regime":           regime,
        "accuracy":         rep.accuracy,
        "groups_eval":      rep.groups_eval,
        "top1_count":       rep.top1_count,
        "regret_ms_mean":   rep.regret_ms_mean,
        "regret_ratio_p95": rep.regret_ratio_p95,
    }
    detail = rep.per_group.assign(model=model_name, regime=regime)
    return summary, detail


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n-trials", type=int, default=25,
                   help="Optuna trials per tunable model (default: 25)")
    p.add_argument("--tune-timeout", type=int, default=120,
                   help="Hard time-cap per Optuna study, seconds (default: 120)")
    p.add_argument("--skip-tuning", action="store_true",
                   help="Skip Optuna stage (fast smoke run)")
    p.add_argument("--regimes", nargs="+",
                   default=["plan_time", "post_mortem"],
                   help="Regimes to train (default: both)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    print(f"[i] Reading features from {FEATURES_CSV.relative_to(PROJECT_ROOT)}")
    df = pd.read_csv(FEATURES_CSV)
    print(f"[i] Loaded {len(df)} rows, {df.shape[1]} columns")
    print(f"[i] Distinct query_ids (CV groups): {df['query_id'].nunique()}")

    cmp_rows: list[dict[str, Any]] = []
    pred_chunks: list[pd.DataFrame] = []
    importance_rows: list[dict[str, Any]] = []
    pp_summary_rows: list[dict[str, Any]] = []
    pp_detail_chunks: list[pd.DataFrame] = []
    optuna_rows: list[dict[str, Any]] = []
    automl_winners: dict[str, dict[str, Any]] = {}

    for regime in args.regimes:
        print(f"\n=== Regime: {regime} ===")
        fm = build_feature_matrix(df, regime=regime, drop_zero_variance=True)
        variant_series = (
            df.loc[fm.X.index, "variant"].astype(str).reset_index(drop=True)
            if "variant" in df.columns
            else pd.Series([""] * len(fm.X))
        )
        fm = FeatureMatrix(
            X=fm.X.reset_index(drop=True),
            y=fm.y.reset_index(drop=True),
            groups=fm.groups.reset_index(drop=True),
            feature_names=fm.feature_names,
            regime=fm.regime,
        )
        print(f"[i] X shape: {fm.X.shape}, features: {len(fm.feature_names)}")

        # 1. PG baselines
        for bname, bfactory in BASELINES.items():
            res = run_cv(bname, bfactory, fm, kind="baseline", log_target=False,
                         variant_series=variant_series)
            agg = average_fold_metrics(res.fold_metrics)
            cmp_rows.append({"model": bname, "regime": regime, "kind": "baseline",
                             "train_seconds": round(res.train_seconds, 3), **agg})
            pred_chunks.extend(res.fold_preds)
            print(f"  [B] {bname:<26} qerr_med={agg['q_error_median_mean']:.2f}  "
                  f"R2={agg['r2_mean']:+.3f}")

        # 2. Default ML
        default_zoo = model_zoo()
        for mname, mfactory in default_zoo.items():
            res = run_cv(mname, mfactory, fm, kind="default_ml", log_target=True,
                         variant_series=variant_series)
            agg = average_fold_metrics(res.fold_metrics)
            cmp_rows.append({"model": mname, "regime": regime, "kind": "default_ml",
                             "train_seconds": round(res.train_seconds, 3), **agg})
            pred_chunks.extend(res.fold_preds)
            saved = fit_and_save_full(mname, mfactory, fm, log_target=True)
            artifact = joblib.load(saved)
            imp = _extract_importance(artifact["model"], fm.feature_names)
            if imp is not None:
                for f, v in imp.items():
                    importance_rows.append({"model": mname, "regime": regime,
                                            "feature": f, "importance": float(v)})
            print(f"  [M] {mname:<26} qerr_med={agg['q_error_median_mean']:.2f}  "
                  f"R2={agg['r2_mean']:+.3f}")

        # 3. Tuned ML (Optuna)
        tuned_results: dict[str, dict[str, Any]] = {}
        if not args.skip_tuning:
            print(f"  [Optuna] tuning {len(TUNABLE)} models, "
                  f"{args.n_trials} trials each, timeout={args.tune_timeout}s")
            for tname in TUNABLE:
                t0 = time.perf_counter()
                tr = tune_model(tname, fm,
                                n_trials=args.n_trials,
                                timeout_sec=args.tune_timeout)
                tuned_results[tname] = {"best_params": tr.best_params,
                                        "best_qerror": tr.best_qerror}
                print(f"    [T] {tname:<24} best qerr_med={tr.best_qerror:.3f} "
                      f"({tr.n_trials} trials, {time.perf_counter() - t0:.1f}s)")
                for trial_idx, qerr in enumerate(tr.history):
                    optuna_rows.append({"regime": regime, "model": tname,
                                        "trial": trial_idx, "qerror_median": qerr})

                # Re-run CV with the best params so we have full metrics + preds.
                tuned_factory = lambda p=tr.best_params, n=tname: build_tuned(n, p)
                tname_full = f"{tname}_tuned"
                res = run_cv(tname_full, tuned_factory, fm,
                             kind="tuned_ml", log_target=True,
                             variant_series=variant_series)
                agg = average_fold_metrics(res.fold_metrics)
                cmp_rows.append({"model": tname_full, "regime": regime,
                                 "kind": "tuned_ml",
                                 "train_seconds": round(res.train_seconds, 3),
                                 **agg})
                pred_chunks.extend(res.fold_preds)
                fit_and_save_full(tname_full, tuned_factory, fm,
                                  log_target=True,
                                  extra_meta={"best_params": tr.best_params,
                                              "tuner": "optuna",
                                              "n_trials": tr.n_trials})

        # 4. Save baselines + bookkeeping
        for bname, bfactory in BASELINES.items():
            fit_and_save_full(bname, bfactory, fm, log_target=False)

    # ----- Plan-pick eval over ALL fold predictions -----
    print("\n[i] Evaluating plan-pick accuracy ...")
    pred_df = pd.concat(pred_chunks, ignore_index=True)
    for (mname, regime), _ in pred_df.groupby(["model", "regime"]):
        summary, detail = _plan_pick_for(pred_df, df, mname, regime)
        pp_summary_rows.append(summary)
        if not detail.empty:
            pp_detail_chunks.append(detail)

    pp_summary = pd.DataFrame(pp_summary_rows)
    if pp_detail_chunks:
        pp_detail = pd.concat(pp_detail_chunks, ignore_index=True)
    else:
        pp_detail = pd.DataFrame()

    # ----- Build comparison table joined with plan-pick -----
    cmp_df = pd.DataFrame(cmp_rows).merge(
        pp_summary[["model", "regime", "accuracy", "regret_ms_mean", "regret_ratio_p95"]],
        on=["model", "regime"], how="left",
    )
    cmp_df = cmp_df.rename(columns={"accuracy": "plan_pick_acc"})
    cmp_df = cmp_df.sort_values(
        ["regime", "q_error_median_mean", "rmse_mean"]
    ).reset_index(drop=True)

    cmp_df.to_csv(REPORT_DIR / "model_comparison.csv", index=False)
    pred_df.to_csv(REPORT_DIR / "cv_predictions.csv", index=False)
    pp_summary.to_csv(REPORT_DIR / "plan_pick_summary.csv", index=False)
    if not pp_detail.empty:
        pp_detail.to_csv(REPORT_DIR / "plan_pick_detail.csv", index=False)

    if importance_rows:
        imp_df = pd.DataFrame(importance_rows).sort_values(
            ["regime", "model", "importance"], ascending=[True, True, False]
        )
        imp_df.to_csv(REPORT_DIR / "feature_importance.csv", index=False)

    if optuna_rows:
        pd.DataFrame(optuna_rows).to_csv(REPORT_DIR / "optuna_history.csv", index=False)

    # ----- AutoML winner per regime -----
    print("\n[i] Selecting AutoML winners ...")
    for regime in args.regimes:
        # Only ML models, not baselines.
        cand = cmp_df[(cmp_df["regime"] == regime) & (cmp_df["kind"].isin(["default_ml", "tuned_ml"]))].copy()
        if cand.empty:
            continue
        # Composite: minimise q-err median, break ties by maximising plan-pick acc.
        cand["__score__"] = cand["q_error_median_mean"] - 0.1 * cand["plan_pick_acc"].fillna(0.0)
        winner = cand.sort_values("__score__").iloc[0]
        winner_dict = {
            "regime":  regime,
            "model":   str(winner["model"]),
            "kind":    str(winner["kind"]),
            "q_error_median": float(winner["q_error_median_mean"]),
            "rmse_ms":        float(winner["rmse_mean"]),
            "r2":             float(winner["r2_mean"]),
            "plan_pick_acc":  float(winner["plan_pick_acc"]) if pd.notna(winner["plan_pick_acc"]) else None,
        }
        automl_winners[regime] = winner_dict
        # Symlink-style copy.
        src = MODELS_DIR / regime / f"{winner['model']}.joblib"
        if src.exists():
            dst = MODELS_DIR / regime / "automl_best.joblib"
            artifact = joblib.load(src)
            artifact["automl_winner"] = winner_dict
            joblib.dump(artifact, dst)
            print(f"  [W] {regime:<14} -> {winner['model']}  "
                  f"(qerr_med={winner['q_error_median_mean']:.2f}, "
                  f"plan-pick={winner_dict['plan_pick_acc'] or 'n/a'})")
        else:
            print(f"[!] winner artifact missing for {regime}/{winner['model']}",
                  file=sys.stderr)

    (REPORT_DIR / "automl_winner.json").write_text(
        json.dumps(automl_winners, indent=2), encoding="utf-8")

    _write_markdown_summary(cmp_df, pp_summary, automl_winners)

    print("\n[OK] Phase 3B training complete.")
    print(f"[OK] Comparison table : {REPORT_DIR / 'model_comparison.csv'}")
    print(f"[OK] Plan-pick summary: {REPORT_DIR / 'plan_pick_summary.csv'}")
    print(f"[OK] AutoML winner    : {REPORT_DIR / 'automl_winner.json'}")
    print(f"[OK] Saved models     : {MODELS_DIR}")
    return 0


def _write_markdown_summary(
    cmp_df: pd.DataFrame,
    pp_summary: pd.DataFrame,
    winners: dict[str, dict[str, Any]],
) -> None:
    cols = [
        "regime", "kind", "model",
        "r2_mean", "mae_mean", "rmse_mean",
        "q_error_median_mean", "q_error_p95_mean",
        "spearman_rho_mean", "plan_pick_acc",
        "regret_ms_mean", "train_seconds",
    ]
    show = cmp_df[cols].copy()
    rename = {
        "r2_mean": "R²",
        "mae_mean": "MAE (ms)",
        "rmse_mean": "RMSE (ms)",
        "q_error_median_mean": "q-err median",
        "q_error_p95_mean":    "q-err p95",
        "spearman_rho_mean":   "Spearman ρ",
        "plan_pick_acc":       "plan-pick acc",
        "regret_ms_mean":      "regret (ms)",
        "train_seconds":       "Train (s)",
    }
    show = show.rename(columns=rename)
    for c in show.select_dtypes(include="float").columns:
        show[c] = show[c].round(3)

    lines = [
        "# Phase 3B — AutoML-Tuned Cost Model Leaderboard",
        "",
        "5-fold **GroupKFold** CV grouped by `query_id`. Models trained on "
        "`log1p(execution_time_ms)`, scored on the back-transformed prediction.",
        "",
        "**Sort key**: ascending median q-error, then ascending RMSE.",
        "",
        "Columns:",
        "- `kind`: `baseline` (calibrated PG cost), `default_ml` (3A defaults), "
        "`tuned_ml` (Optuna-tuned).",
        "- `plan-pick acc`: fraction of `query_id` groups where the model picks "
        "the truly fastest variant.",
        "- `regret (ms)`: average extra runtime paid vs. the oracle.",
        "",
    ]

    if winners:
        lines.append("## AutoML winners")
        lines.append("")
        for regime, w in winners.items():
            lines.append(
                f"- **{regime}**: `{w['model']}` "
                f"(q-err med = {w['q_error_median']:.2f}, "
                f"plan-pick acc = {w['plan_pick_acc']!s})"
            )
        lines.append("")

    for regime, grp in show.groupby("regime"):
        lines.append(f"## Regime: `{regime}`")
        lines.append("")
        lines.append(grp.to_markdown(index=False))
        lines.append("")

    (REPORT_DIR / "model_comparison.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
