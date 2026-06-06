"""
baseline.py
===========
The PostgreSQL "native cost model" baseline.

PG's optimizer reports `Total Cost` in arbitrary planner units (mostly
calibrated to the cost of a sequential page read). To compare it
fairly against ML models that predict milliseconds, we have to
calibrate it on the training data, then apply that calibration to the
held-out test fold.

Two calibrators are provided:

  * `LinearCostBaseline` — fits  time ≈ k × cost  with k from training
    (no intercept, since cost = 0 should mean ~0 ms).

  * `LogLinearCostBaseline` — fits  log(time) ≈ a + b · log(cost) ,
    typically much better when both span orders of magnitude.

Both expose a sklearn-style `.fit(X, y).predict(X)` interface so the
training pipeline can drop them into the same evaluation loop as the
real ML models without special-casing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


COST_COLUMN: str = "estimated_total_cost"


# ---------------------------------------------------------------------------
# Tiny sklearn-compatible regressors (no dependency on BaseEstimator
# to keep the file standalone).
# ---------------------------------------------------------------------------
@dataclass
class LinearCostBaseline:
    """time_ms ≈ k * estimated_total_cost, k fit on training data."""
    k_: float = 1.0

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "LinearCostBaseline":
        cost = _extract_cost(X)
        y    = np.asarray(y, dtype=float)
        denom = float((cost ** 2).sum())
        self.k_ = float((cost * y).sum() / denom) if denom > 0 else 0.0
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        cost = _extract_cost(X)
        return self.k_ * cost


@dataclass
class LogLinearCostBaseline:
    """log1p(time_ms) ≈ a + b * log1p(estimated_total_cost)."""
    a_: float = 0.0
    b_: float = 1.0

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "LogLinearCostBaseline":
        cost = _extract_cost(X)
        y    = np.asarray(y, dtype=float)
        log_cost = np.log1p(np.maximum(cost, 0.0))
        log_y    = np.log1p(np.maximum(y,    0.0))

        if np.std(log_cost) == 0:
            self.a_, self.b_ = float(log_y.mean()), 0.0
            return self

        b, a = np.polyfit(log_cost, log_y, deg=1)
        self.a_, self.b_ = float(a), float(b)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        cost = _extract_cost(X)
        log_cost = np.log1p(np.maximum(cost, 0.0))
        log_pred = self.a_ + self.b_ * log_cost
        return np.expm1(log_pred)


# ---------------------------------------------------------------------------
def _extract_cost(X: pd.DataFrame) -> np.ndarray:
    if COST_COLUMN not in X.columns:
        raise KeyError(f"baseline needs '{COST_COLUMN}' but it's not in X")
    return np.asarray(X[COST_COLUMN], dtype=float)
