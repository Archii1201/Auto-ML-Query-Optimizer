"""
timeout_budget.py
=================
Phase 4A — per-request time budget.

A request to /plan-pick does several blocking things in sequence:
acquire a pooled connection, run N EXPLAINs on PG, run N model
predictions. Without an end-to-end budget, a slow PG plan or a pool
wait can blow the latency SLO unboundedly. The budget gives every
request a single deadline and hands each stage only the time that's
left, so the *whole* request fails fast instead of hanging.

    total budget (e.g. 8000 ms)
      ├── pool acquire        (cap: POOL_ACQUIRE_TIMEOUT_S)
      ├── plan generation     (PG statement_timeout = remaining - reserve)
      └── prediction          (remaining)

`reserve_ms` keeps a little headroom so we can always serialize a
response (and write feedback) before the client's own timeout fires.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class TimeoutBudget:
    total_ms: float
    reserve_ms: float = 250.0
    _start: float = field(default_factory=time.monotonic)

    @classmethod
    def start(cls, total_ms: float, reserve_ms: float = 250.0) -> "TimeoutBudget":
        return cls(total_ms=float(total_ms), reserve_ms=float(reserve_ms))

    def elapsed_ms(self) -> float:
        return (time.monotonic() - self._start) * 1000.0

    def remaining_ms(self) -> float:
        return max(0.0, self.total_ms - self.elapsed_ms())

    def expired(self) -> bool:
        return self.remaining_ms() <= 0.0

    def acquire_timeout_s(self, cap_s: float) -> float:
        """Seconds to wait for a pooled connection (bounded by remaining)."""
        return max(0.0, min(cap_s, self.remaining_ms() / 1000.0))

    def statement_timeout_ms(self, *, minimum_ms: float = 100.0) -> int:
        """
        PG statement_timeout for the plan-generation stage: whatever is
        left minus the response reserve, floored so we never pass 0
        (which PG interprets as 'no timeout').
        """
        budget = self.remaining_ms() - self.reserve_ms
        return int(max(minimum_ms, budget))
