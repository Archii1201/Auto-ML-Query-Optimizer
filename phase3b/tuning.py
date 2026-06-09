"""
phase3b/tuning.py
=================
Optuna-driven hyperparameter search.

For each tunable model we define:

    - a search space (Optuna trial → kwargs dict)
    - a builder    (kwargs → fresh estimator instance)

The actual search is wrapped in a single function `tune_model(...)`
that runs N Optuna trials, each scored by the q-error MEDIAN under
the same GroupKFold CV used in Phase 3A. The best params are
returned plus the full study object for inspection.

We minimise q-error median (not RMSE / MAE) because:
    1. It's the metric the leaderboard actually sorts by.
    2. It's symmetric in over/under-estimation, matching what a
       real query optimizer cares about.
    3. With a tiny dataset, RMSE is dominated by 1-2 outliers per
       fold; q-error median is robust to that.

Three things this module deliberately does NOT do:
    - Tune the linear models. They have ~1 hyperparameter (alpha)
      and the gain is marginal vs. the cost of the search budget.
    - Tune CatBoost (its bayesian search is built-in and faster
      with `auto_class_weights` etc. — keep its 3A defaults).
    - Search over feature transforms. The pipeline transforms
      are fixed per Phase 3B's design.
"""

from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.model_selection import GroupKFold

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from phase3a.evaluation import compute_metrics  # noqa: E402
from phase3a.feature_selection import FeatureMatrix  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

RANDOM_STATE: int = 42
N_SPLITS:     int = 5


# ---------------------------------------------------------------------------
# Search spaces
# ---------------------------------------------------------------------------
def _space_random_forest(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators":     trial.suggest_int("n_estimators",     200, 800, step=100),
        "max_depth":        trial.suggest_int("max_depth",          3,  20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf",   1,   8),
        "max_features":     trial.suggest_categorical("max_features", ["sqrt", "log2", 1.0]),
        "random_state":     RANDOM_STATE,
        "n_jobs":           -1,
    }


def _space_extra_trees(trial: optuna.Trial) -> dict[str, Any]:
    p = _space_random_forest(trial)
    return p


def _space_gradient_boosting(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators":  trial.suggest_int("n_estimators",  200, 800, step=100),
        "max_depth":     trial.suggest_int("max_depth",       2,   8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample":     trial.suggest_float("subsample",     0.6,  1.0),
        "random_state":  RANDOM_STATE,
    }


def _space_xgboost(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators":     trial.suggest_int("n_estimators",     200, 800, step=100),
        "max_depth":        trial.suggest_int("max_depth",          3,  10),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample":        trial.suggest_float("subsample",     0.6,   1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_lambda":       trial.suggest_float("reg_lambda",    0.1, 10.0, log=True),
        "random_state":     RANDOM_STATE,
        "tree_method":      "hist",
        "n_jobs":           -1,
        "verbosity":        0,
    }


def _space_lightgbm(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators":     trial.suggest_int("n_estimators",     200, 800, step=100),
        "num_leaves":       trial.suggest_int("num_leaves",         15, 127),
        "max_depth":        trial.suggest_int("max_depth",         -1,  16),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample":        trial.suggest_float("subsample",     0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_lambda":       trial.suggest_float("reg_lambda",    0.1, 10.0, log=True),
        # NOTE: use sklearn-style name `min_child_samples`. The LightGBM
        # alias `min_data_in_leaf` is also accepted but emits a "will be
        # ignored" warning per booster predict() call at inference time.
        "min_child_samples": trial.suggest_int("min_child_samples", 1, 20),
        "random_state":     RANDOM_STATE,
        "n_jobs":           -1,
        "verbosity":        -1,
    }


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def _build_random_forest(p: dict[str, Any]) -> Any:
    return RandomForestRegressor(**p)


def _build_extra_trees(p: dict[str, Any]) -> Any:
    return ExtraTreesRegressor(**p)


def _build_gradient_boosting(p: dict[str, Any]) -> Any:
    return GradientBoostingRegressor(**p)


def _build_xgboost(p: dict[str, Any]) -> Any:
    from xgboost import XGBRegressor
    return XGBRegressor(**p)


def _build_lightgbm(p: dict[str, Any]) -> Any:
    from lightgbm import LGBMRegressor
    return LGBMRegressor(**p)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
TUNABLE: dict[str, tuple[Callable, Callable]] = {
    "random_forest":     (_space_random_forest,     _build_random_forest),
    "extra_trees":       (_space_extra_trees,       _build_extra_trees),
    "gradient_boosting": (_space_gradient_boosting, _build_gradient_boosting),
    "xgboost":           (_space_xgboost,           _build_xgboost),
    "lightgbm":          (_space_lightgbm,          _build_lightgbm),
}


# ---------------------------------------------------------------------------
# Objective + driver
# ---------------------------------------------------------------------------
@dataclass
class TuneResult:
    model_name:   str
    best_params:  dict[str, Any]
    best_qerror:  float
    n_trials:     int
    history:      list[float]   # q-err median per trial


def _cv_qerror_median(
    builder: Callable,
    params:  dict[str, Any],
    fm:      FeatureMatrix,
    log_target: bool = True,
) -> float:
    """One CV pass, returns mean (across folds) of fold q-error medians."""
    cv = GroupKFold(n_splits=N_SPLITS)
    qmeds: list[float] = []
    cap = 10.0 * float(fm.y.max()) if len(fm.y) else 1e7

    for tr, te in cv.split(fm.X, fm.y, fm.groups):
        X_tr, X_te = fm.X.iloc[tr], fm.X.iloc[te]
        y_tr, y_te = fm.y.iloc[tr].to_numpy(), fm.y.iloc[te].to_numpy()

        model = builder(params)
        if log_target:
            model.fit(X_tr, np.log1p(np.maximum(y_tr, 0.0)))
            y_pred = np.expm1(model.predict(X_te))
        else:
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)

        y_pred = np.where(np.isfinite(y_pred), y_pred, cap)
        y_pred = np.clip(y_pred, 0.1, cap)
        qmeds.append(compute_metrics(y_te, y_pred).q_error_median)

    return float(np.mean(qmeds))


def tune_model(
    name:        str,
    fm:          FeatureMatrix,
    n_trials:    int = 25,
    timeout_sec: float | None = None,
    seed:        int = RANDOM_STATE,
) -> TuneResult:
    if name not in TUNABLE:
        raise KeyError(f"unknown tunable model: {name}; have {list(TUNABLE)}")

    space, builder = TUNABLE[name]
    history: list[float] = []

    def objective(trial: optuna.Trial) -> float:
        params = space(trial)
        score = _cv_qerror_median(builder, params, fm)
        history.append(score)
        return score

    sampler = optuna.samplers.TPESampler(seed=seed)
    study   = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, timeout=timeout_sec,
                   show_progress_bar=False)

    return TuneResult(
        model_name=name,
        best_params=study.best_params,
        best_qerror=float(study.best_value),
        n_trials=len(study.trials),
        history=history,
    )


def build_tuned(name: str, params: dict[str, Any]) -> Any:
    """Construct the estimator with the best-known params (used at refit)."""
    if name not in TUNABLE:
        raise KeyError(name)
    _, builder = TUNABLE[name]
    return builder(params)
