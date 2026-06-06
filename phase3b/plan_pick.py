"""
phase3b/plan_pick.py
====================
Plan-pick accuracy: the metric that matters most in production.

Setup:
    Each row in features.csv represents *one execution plan* — one
    variant of one query. The `query_id` column groups together the
    4 plans (default + 3 disabled-join-variants) for a single query.

    For every group we ask:

        1. Which variant has the LOWEST actual execution time?  (truth)
        2. Which variant does the model PREDICT to be fastest?   (pick)
        3. Does pick == truth?                                  (hit)

Then:
    plan_pick_accuracy = mean(hit) across groups

We also report:
    * regret_ms_mean   — avg extra time we'd pay vs. the oracle
    * regret_ratio_p95 — 95-th percentile of (picked / oracle - 1)
    * top1_count       — number of correct picks
    * groups_eval      — number of groups with >=2 variants
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PlanPickReport:
    accuracy:          float
    top1_count:        int
    groups_eval:       int
    regret_ms_mean:    float
    regret_ratio_p95:  float
    per_group:         pd.DataFrame   # detail rows (one per group)


def evaluate_plan_pick(
    df_preds: pd.DataFrame,
    *,
    group_col:   str = "query_id",
    truth_col:   str = "y_true",
    predict_col: str = "y_pred",
    variant_col: str = "variant",
) -> PlanPickReport:
    """
    Parameters
    ----------
    df_preds : DataFrame with columns (group_col, variant_col, truth_col, predict_col)
               -- one row per (query, variant).
    """
    needed = {group_col, truth_col, predict_col, variant_col}
    if not needed.issubset(df_preds.columns):
        missing = needed - set(df_preds.columns)
        raise KeyError(f"plan-pick input missing columns: {missing}")

    rows = []
    for gid, g in df_preds.groupby(group_col, sort=False):
        if len(g) < 2:
            continue  # no choice to make
        true_best_idx = g[truth_col].idxmin()
        pick_idx      = g[predict_col].idxmin()

        true_var  = g.loc[true_best_idx, variant_col]
        pick_var  = g.loc[pick_idx,      variant_col]
        true_ms   = float(g.loc[true_best_idx, truth_col])
        pick_ms   = float(g.loc[pick_idx,      truth_col])
        regret_ms = max(pick_ms - true_ms, 0.0)
        ratio     = pick_ms / max(true_ms, 1e-3) - 1.0

        rows.append({
            group_col:        gid,
            "true_best_variant": true_var,
            "picked_variant":    pick_var,
            "true_best_ms":      true_ms,
            "picked_ms":         pick_ms,
            "regret_ms":         regret_ms,
            "regret_ratio":      ratio,
            "hit":               int(true_var == pick_var),
        })

    detail = pd.DataFrame(rows)
    if detail.empty:
        return PlanPickReport(
            accuracy=float("nan"),
            top1_count=0, groups_eval=0,
            regret_ms_mean=float("nan"),
            regret_ratio_p95=float("nan"),
            per_group=detail,
        )

    return PlanPickReport(
        accuracy=         float(detail["hit"].mean()),
        top1_count=       int(detail["hit"].sum()),
        groups_eval=      int(len(detail)),
        regret_ms_mean=   float(detail["regret_ms"].mean()),
        regret_ratio_p95= float(np.percentile(detail["regret_ratio"], 95)),
        per_group=        detail,
    )
