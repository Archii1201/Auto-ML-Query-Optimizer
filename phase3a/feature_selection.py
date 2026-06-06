"""
feature_selection.py
====================
The single source of truth for "which columns are features, which are
identifiers, and which are leaky".

Three regimes are supported:

  - PLAN_TIME : columns the PostgreSQL planner has *before* execution.
                This is the realistic, deployable model.
  - POST_MORTEM : every numeric column including post-execution
                  observations. Useful as an upper-bound sanity check —
                  shows the best a model could do with perfect runtime
                  side-information.
  - ID         : metadata never given to the model (provenance only).

Why this matters
----------------
PostgreSQL's EXPLAIN ANALYZE output mixes plan-time *estimates*
(`Total Cost`, `Plan Rows`) with post-execution *observations*
(`Actual Rows`, `Actual Total Time`, `Shared Hit Blocks`, ...). At
training time both look like ordinary CSV columns, but a real cost
model deployed inside a query optimizer can only see the first group.
Letting any post-execution column leak into training will make every
metric look amazing and the resulting model useless in production.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


TARGET_COLUMN: str = "execution_time_ms"
GROUP_COLUMN:  str = "query_id"


# ---------------------------------------------------------------------------
# Identifier / metadata columns. Never used as features OR as the target.
# ---------------------------------------------------------------------------
ID_COLUMNS: tuple[str, ...] = (
    "source_file",
    "query_id",
    "variant",
    "tag",
    "sql_hash",
    "collected_at",
)


# ---------------------------------------------------------------------------
# Post-execution columns — *only* available AFTER the query runs.
# These leak the target. Drop them in the realistic (plan-time) regime.
# ---------------------------------------------------------------------------
LEAKY_COLUMNS: tuple[str, ...] = (
    "actual_rows",
    "actual_total_time_ms",
    "max_actual_loops",
    "total_rows_removed_by_filter",
    "parallel_worker_count",
    "sum_shared_hit_blocks",
    "sum_shared_read_blocks",
    "sum_temp_read_blocks",
    "sum_temp_written_blocks",
    "wall_time_ms",
    "target_execution_time_ms",
)


# ---------------------------------------------------------------------------
# The categorical column we one-hot-encode for linear models.
# Tree boosters can also accept it as a category natively.
# ---------------------------------------------------------------------------
CATEGORICAL_COLUMNS: tuple[str, ...] = (
    "root_node_type",
)


@dataclass(frozen=True)
class FeatureMatrix:
    """A bundle that any trainer can consume directly."""
    X:        pd.DataFrame
    y:        pd.Series
    groups:   pd.Series
    feature_names: list[str]
    regime:   str  # "plan_time" or "post_mortem"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _coerce_numeric_inplace(df: pd.DataFrame, cols: Iterable[str]) -> None:
    """Force numeric dtype + fill NaN with 0 (rare; only for absent EXPLAIN keys)."""
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)


def _one_hot_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode every CATEGORICAL_COLUMNS column that is present."""
    cat_cols = [c for c in CATEGORICAL_COLUMNS if c in df.columns]
    if not cat_cols:
        return df
    return pd.get_dummies(
        df,
        columns=cat_cols,
        prefix=cat_cols,
        prefix_sep="__",
        dummy_na=False,
        dtype=np.uint8,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def build_feature_matrix(
    df: pd.DataFrame,
    regime: str = "plan_time",
    drop_zero_variance: bool = True,
) -> FeatureMatrix:
    """
    Build (X, y, groups) for a given regime.

    Parameters
    ----------
    df : DataFrame loaded from data/processed/features.csv
    regime : "plan_time" | "post_mortem"
        - plan_time   → drop ID + LEAKY columns
        - post_mortem → drop ID columns only; keep leaky ones (sanity ceiling)
    drop_zero_variance : bool
        If True, columns whose value is identical for every row are dropped
        (they carry zero signal and break some scalers).

    Returns
    -------
    FeatureMatrix with X (one-hot encoded), y (target), groups (query_id).
    """
    if regime not in {"plan_time", "post_mortem"}:
        raise ValueError(f"unknown regime: {regime}")

    if TARGET_COLUMN not in df.columns:
        raise KeyError(f"target column '{TARGET_COLUMN}' missing from CSV")
    if GROUP_COLUMN not in df.columns:
        raise KeyError(f"group column '{GROUP_COLUMN}' missing from CSV")

    work = df.copy()

    y      = work[TARGET_COLUMN].astype(float)
    groups = work[GROUP_COLUMN].astype(str)

    drop_cols = list(ID_COLUMNS) + [TARGET_COLUMN]
    if regime == "plan_time":
        drop_cols.extend(LEAKY_COLUMNS)
    else:
        drop_cols.extend(c for c in (TARGET_COLUMN, "target_execution_time_ms")
                         if c in LEAKY_COLUMNS)

    drop_cols = [c for c in drop_cols if c in work.columns]
    work = work.drop(columns=drop_cols, errors="ignore")

    work = _one_hot_categoricals(work)

    numeric_cols = [c for c in work.columns
                    if c not in CATEGORICAL_COLUMNS]
    _coerce_numeric_inplace(work, numeric_cols)

    if drop_zero_variance:
        nunique = work.nunique(dropna=False)
        zero_var = nunique[nunique <= 1].index.tolist()
        if zero_var:
            work = work.drop(columns=zero_var)

    work = work.astype(float)

    return FeatureMatrix(
        X=work,
        y=y,
        groups=groups,
        feature_names=list(work.columns),
        regime=regime,
    )


def describe_regime_split(df: pd.DataFrame) -> pd.DataFrame:
    """Quick human-readable table of which columns go where."""
    cats = []
    for c in df.columns:
        if c in ID_COLUMNS:
            cats.append((c, "identifier"))
        elif c == TARGET_COLUMN:
            cats.append((c, "target"))
        elif c in LEAKY_COLUMNS:
            cats.append((c, "leaky (post-execution)"))
        else:
            cats.append((c, "plan-time feature"))
    return pd.DataFrame(cats, columns=["column", "category"])
