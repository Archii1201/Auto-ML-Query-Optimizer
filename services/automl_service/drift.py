"""
drift.py
========
Phase 5D — prediction drift detection.

The cheapest, most honest drift signal we already emit is the
*predicted/actual runtime ratio* (`pred_actual_ratio`). When the live model
is well-calibrated this sits near 1.0. If it drifts persistently high
(over-predicting) or low (under-predicting), the world has moved away from
the training distribution and it's time to retrain — regardless of how much
new data has arrived.

This module is pure: it turns a ratio (and optionally the raw Prometheus
exposition text the ml-service already serves) into a boolean drift verdict.
No network, no scraping client — the worker passes in whatever it read.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class DriftThresholds:
    low:  float = 0.5    # ratio below this ⇒ systematically under-predicting
    high: float = 2.0    # ratio above this ⇒ systematically over-predicting


@dataclass
class DriftVerdict:
    drifted: bool
    ratio:   float | None
    reason:  str


def evaluate_ratio(ratio: float | None,
                   thresholds: DriftThresholds = DriftThresholds()) -> DriftVerdict:
    if ratio is None:
        return DriftVerdict(False, None, "no ratio available")
    if not (ratio == ratio):  # NaN
        return DriftVerdict(False, None, "ratio is NaN")
    if ratio < thresholds.low:
        return DriftVerdict(True, ratio,
                            f"under-predicting: ratio {ratio:.3f} < {thresholds.low}")
    if ratio > thresholds.high:
        return DriftVerdict(True, ratio,
                            f"over-predicting: ratio {ratio:.3f} > {thresholds.high}")
    return DriftVerdict(False, ratio, f"calibrated: ratio {ratio:.3f} in band")


_METRIC_LINE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+([0-9eE.+-]+)\s*$")


def parse_prom_gauge(prom_text: str, metric: str) -> float | None:
    """
    Best-effort extraction of the last value of a gauge from Prometheus
    text exposition. Returns None if the metric isn't present.
    """
    val: float | None = None
    for line in prom_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _METRIC_LINE.match(line)
        if m and m.group(1) == metric:
            try:
                val = float(m.group(2))
            except ValueError:
                continue
    return val
