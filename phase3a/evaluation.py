"""
evaluation.py
=============
Metric definitions for Phase 3A.

We report the standard regression metrics (MAE, RMSE, R^2, MAPE) plus
two metrics that are specific to the *learned cost model* literature:

  * q-error  — max(pred/actual, actual/pred). The de-facto metric in
               every learned-cost-model paper (Neo, MSCN, Bao, Balsa).
               Reported as median and 95th percentile across the
               evaluation set.

  * Spearman ρ — rank correlation. A query optimizer ultimately picks
                 the cheapest plan among candidates, so what matters
                 most is whether the model *orders* plans correctly,
                 not whether the absolute predictions are calibrated.

A small floor (`MIN_TIME_MS`) is applied wherever we divide by actual
runtime to keep MAPE / q-error finite for queries that finish in
sub-millisecond time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


MIN_TIME_MS: float = 1.0  # floor to keep ratio metrics finite


# ---------------------------------------------------------------------------
# Q-error — the workhorse metric for cost models
# ---------------------------------------------------------------------------
def q_error(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Per-sample q-error: max(pred/actual, actual/pred). Always >= 1.0.
    A perfect prediction has q_error = 1; q_error = 10 means the model
    is off by 10x in either direction.
    """
    yt = np.maximum(np.asarray(y_true,  dtype=float), MIN_TIME_MS)
    yp = np.maximum(np.asarray(y_pred,  dtype=float), MIN_TIME_MS)
    return np.maximum(yp / yt, yt / yp)


def safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAPE with a floor so sub-ms queries don't blow it up."""
    yt = np.maximum(np.asarray(y_true, dtype=float), MIN_TIME_MS)
    yp = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs((yp - yt) / yt)) * 100.0)


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------
@dataclass
class Metrics:
    mae:           float
    rmse:          float
    r2:            float
    mape:          float
    q_error_median: float
    q_error_p95:    float
    spearman:      float
    n:             int

    def to_dict(self) -> dict[str, float]:
        return {
            "mae":            self.mae,
            "rmse":           self.rmse,
            "r2":             self.r2,
            "mape_pct":       self.mape,
            "q_error_median": self.q_error_median,
            "q_error_p95":    self.q_error_p95,
            "spearman_rho":   self.spearman,
            "n":              self.n,
        }


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    """All metrics in a single pass."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)

    yp = np.where(np.isfinite(yp), yp, 0.0)
    yp = np.maximum(yp, 0.0)

    mae   = float(mean_absolute_error(yt, yp))
    rmse  = float(np.sqrt(mean_squared_error(yt, yp)))
    r2    = float(r2_score(yt, yp))

    mape  = safe_mape(yt, yp)
    qe    = q_error(yt, yp)
    qmed  = float(np.median(qe))
    qp95  = float(np.percentile(qe, 95))

    if len(yt) >= 2 and np.std(yt) > 0 and np.std(yp) > 0:
        rho, _ = spearmanr(yt, yp)
        rho = float(rho)
    else:
        rho = float("nan")

    return Metrics(
        mae=mae, rmse=rmse, r2=r2, mape=mape,
        q_error_median=qmed, q_error_p95=qp95,
        spearman=rho, n=len(yt),
    )


# ---------------------------------------------------------------------------
# Convenient cross-validation aggregator
# ---------------------------------------------------------------------------
def average_fold_metrics(per_fold: list[Metrics]) -> dict[str, float]:
    """Mean ± std across folds."""
    if not per_fold:
        return {}
    rows = pd.DataFrame([m.to_dict() for m in per_fold])
    out: dict[str, float] = {}
    for col in rows.columns:
        if col == "n":
            out["n_total"] = int(rows[col].sum())
            continue
        out[f"{col}_mean"] = float(rows[col].mean())
        out[f"{col}_std"]  = float(rows[col].std(ddof=0))
    return out
