"""
metrics.py
==========
A tiny, dependency-free metrics registry that the FastAPI app
exposes on `GET /metrics`.

In Phase 4 this gets swapped for `prometheus_client`. For now we
collect the same numbers as plain Python so we can serve them
both as JSON and in the Prometheus text exposition format
without pulling extra deps.

What we track
-------------
counters:
    predictions_total              total /predict + /plan-pick predictions
    plan_picks_total               total /plan-pick or /run-and-learn calls
    plan_pick_oracle_hits_total    times the picked variant *was* the fastest
                                   (only counted when oracle mode is on)
    executions_total               total /execute or /run-and-learn executions
    execution_failures_total       SQL errors / timeouts during execution
    feedback_rows_written_total    rows persisted to data/feedback/

histograms (kept as a bounded ring buffer of recent observations):
    inference_latency_ms           wallclock for predict/plan-pick
    execution_latency_ms           actual SQL execution times
    pred_actual_ratio              predicted_ms / actual_ms per pred-pair
                                   (1.0 = perfect, >1 = over-estimate)
    regret_ms                      extra time paid vs. oracle (plan-pick)
                                   when /run-and-learn oracle mode is on

We expose:
    .json()  →  dict of everything (used by /metrics)
    .prom()  →  Prometheus text exposition (also /metrics?fmt=prom)
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Iterable


# ---------------------------------------------------------------------------
def _percentile(values: Iterable[float], q: float) -> float:
    """Inclusive linear-interpolation percentile (q in [0, 1])."""
    arr = sorted(values)
    if not arr:
        return float("nan")
    if len(arr) == 1:
        return arr[0]
    k = (len(arr) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return arr[int(k)]
    return arr[lo] + (arr[hi] - arr[lo]) * (k - lo)


# ---------------------------------------------------------------------------
@dataclass
class Histogram:
    name:       str
    buf:        deque = field(default_factory=lambda: deque(maxlen=4096))

    def observe(self, value: float) -> None:
        if math.isfinite(value):
            self.buf.append(float(value))

    def summary(self) -> dict[str, float]:
        if not self.buf:
            return {"count": 0, "sum": 0.0, "p50": float("nan"),
                    "p95": float("nan"), "p99": float("nan")}
        return {
            "count": len(self.buf),
            "sum":   float(sum(self.buf)),
            "mean":  float(sum(self.buf) / len(self.buf)),
            "p50":   _percentile(self.buf, 0.50),
            "p95":   _percentile(self.buf, 0.95),
            "p99":   _percentile(self.buf, 0.99),
        }


# ---------------------------------------------------------------------------
class MetricsRegistry:
    """Process-wide metrics. Thread-safe via a single lock."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._counters: dict[str, int]   = {}
        self._hists:    dict[str, Histogram] = {}
        self._started_at: float = time.time()

    # -- counters ----------------------------------------------------------
    def inc(self, name: str, by: int = 1) -> None:
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + by

    def get_counter(self, name: str) -> int:
        with self._lock:
            return int(self._counters.get(name, 0))

    # -- histograms --------------------------------------------------------
    def observe(self, name: str, value: float) -> None:
        with self._lock:
            if name not in self._hists:
                self._hists[name] = Histogram(name=name)
            self._hists[name].observe(value)

    # -- exporters ---------------------------------------------------------
    def json(self) -> dict[str, Any]:
        with self._lock:
            return {
                "uptime_seconds": time.time() - self._started_at,
                "counters":       dict(self._counters),
                "histograms":     {k: v.summary() for k, v in self._hists.items()},
            }

    def prom(self) -> str:
        """Prometheus text exposition (subset of v0.0.4)."""
        lines: list[str] = []
        with self._lock:
            for name, val in sorted(self._counters.items()):
                lines.append(f"# TYPE {name} counter")
                lines.append(f"{name} {val}")
            for name, hist in sorted(self._hists.items()):
                s = hist.summary()
                lines.append(f"# TYPE {name} summary")
                lines.append(f"{name}_count {int(s['count'])}")
                lines.append(f"{name}_sum {s['sum']:.6f}")
                if s["count"] > 0:
                    for q, key in (("0.5", "p50"), ("0.95", "p95"), ("0.99", "p99")):
                        v = s[key]
                        if math.isfinite(v):
                            lines.append(f'{name}{{quantile="{q}"}} {v:.6f}')
        return "\n".join(lines) + "\n"


# A shared registry — service-wide singleton.
REGISTRY = MetricsRegistry()
