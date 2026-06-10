"""
test_no_target_leakage.py
=========================
Belt-and-braces test: assert that no feature in the *plan_time*
training matrix has a Spearman correlation > 0.95 with the target
(`execution_time_ms`). Spearman picks up monotonic relationships,
which is what we'd see if a "leaky" feature (e.g. `wall_time_ms`
or `actual_total_time_ms`) accidentally survived the filter.

Why a test (and not just code review)? Phase 5 retraining will
rebuild `features.csv` regularly. A future engineer adding a new
feature could silently re-introduce leakage and we'd never know
until model quality went wrong in production. This test catches
that at PR-time.

Run via:
    python -m pytest tests/test_no_target_leakage.py -q
or:
    python tests/test_no_target_leakage.py   # runs as a script too
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from phase3a.feature_selection import (  # noqa: E402
    TARGET_COLUMN,
    build_feature_matrix,
)

LEAKAGE_THRESHOLD = 0.95


def _correlations(regime: str) -> list[tuple[str, float]]:
    csv = PROJECT_ROOT / "data" / "processed" / "features.csv"
    if not csv.exists():
        return []  # nothing to test against yet
    df = pd.read_csv(csv)
    fm = build_feature_matrix(df, regime=regime)
    y = df.loc[fm.X.index, TARGET_COLUMN]

    out: list[tuple[str, float]] = []
    for col in fm.X.columns:
        x = fm.X[col]
        # Skip columns that are constant (no variance => correlation NaN).
        if x.nunique(dropna=True) < 2:
            continue
        try:
            rho, _ = spearmanr(x, y, nan_policy="omit")
        except Exception:
            continue
        if pd.notna(rho):
            out.append((col, float(rho)))
    return out


def test_plan_time_features_have_no_leakage() -> None:
    """The plan_time regime should NOT include any post-execution feature."""
    bad = [
        (col, rho) for col, rho in _correlations("plan_time")
        if abs(rho) > LEAKAGE_THRESHOLD
    ]
    assert not bad, (
        f"target-leakage suspects in plan_time regime "
        f"(|spearman| > {LEAKAGE_THRESHOLD}):\n  "
        + "\n  ".join(f"{c}: {r:+.3f}" for c, r in bad)
    )


def test_post_mortem_regime_includes_actuals() -> None:
    """Sanity: post_mortem regime SHOULD show high correlation features."""
    rows = _correlations("post_mortem")
    if not rows:
        return  # nothing collected yet
    leaders = sorted((abs(r), c, r) for c, r in rows)[-5:]
    # post_mortem can include actual_* features so we expect at least one
    # high-corr feature; if there are *zero*, the regime is broken.
    high = [c for _, c, _ in leaders if abs(_) > 0.7]
    assert high, "post_mortem regime: no feature has |spearman| > 0.7 with target"


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n[plan_time] top correlations with target:")
    rows = sorted(_correlations("plan_time"), key=lambda kv: abs(kv[1]), reverse=True)
    for c, r in rows[:10]:
        flag = "  <<< LEAK!" if abs(r) > LEAKAGE_THRESHOLD else ""
        print(f"  {c:<40} {r:+.3f}{flag}")
    test_plan_time_features_have_no_leakage()
    print("\n[OK] no plan_time feature exceeds the leakage threshold.")

    print("\n[post_mortem] top correlations with target:")
    rows = sorted(_correlations("post_mortem"), key=lambda kv: abs(kv[1]), reverse=True)
    for c, r in rows[:10]:
        print(f"  {c:<40} {r:+.3f}")
    test_post_mortem_regime_includes_actuals()
    print("\n[OK] post_mortem regime has informative actual_* features.")
