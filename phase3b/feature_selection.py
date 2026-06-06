"""
phase3b/feature_selection.py
============================
Extends Phase 3A's feature_selection with the new Phase 3B columns
(log-transformed cost features and plan-time ratios).

The semantics stay identical:
    - PLAN_TIME regime drops every LEAKY column.
    - POST_MORTEM regime keeps them (sanity ceiling only).

We re-export the same `build_feature_matrix(df, regime)` API as 3A
so phase3b/train_models.py is the only thing that needs to import
from here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from phase3a.feature_selection import (  # noqa: E402,F401
    CATEGORICAL_COLUMNS,
    FeatureMatrix,
    GROUP_COLUMN,
    ID_COLUMNS,
    LEAKY_COLUMNS,
    TARGET_COLUMN,
    build_feature_matrix,
    describe_regime_split,
)

# Phase 3B introduces these strictly-plan-time columns (computed in
# feature_engineering/extract_features.py, never derived from
# post-execution observations). Listing them here documents the
# pipeline contract.
PHASE3B_NEW_COLUMNS: tuple[str, ...] = (
    "log1p_estimated_total_cost",
    "log1p_estimated_startup_cost",
    "log1p_estimated_rows",
    "log1p_max_subtree_cost",
    "log1p_total_nodes",
    "cost_per_estimated_row",
    "startup_to_total_ratio",
    "max_to_root_cost_ratio",
    "est_rows_per_node",
)


def expected_feature_count(df: pd.DataFrame, regime: str) -> int:
    """Sanity helper for tests — total numeric features per regime."""
    fm = build_feature_matrix(df, regime=regime)
    return len(fm.feature_names)
