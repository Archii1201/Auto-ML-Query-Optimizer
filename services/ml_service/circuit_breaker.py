"""
circuit_breaker.py
==================
Phase 4A — circuit breaker for the ML prediction path.

Concern from the system-flow doc: "Fault Tolerance — if ML fails,
fall back to the PostgreSQL optimizer." A circuit breaker is the
standard pattern: stop hammering a failing dependency, fail fast, and
periodically probe for recovery.

State machine (DSA: finite state machine)
-----------------------------------------
        success
   ┌───────────────────────────┐
   ▼                           │
 CLOSED ──(failures >= N        │
   │        within window)──► OPEN ──(reset_timeout elapsed)──► HALF_OPEN
   ▲                                                              │  │
   │                                              trial success   │  │ trial failure
   └──────────────────────────────────────────────────────────┘  └──► OPEN

- CLOSED: calls flow normally; we count failures in a rolling window.
- OPEN: calls are short-circuited (caller serves the PG default plan).
  After `reset_timeout_s` we allow a single trial -> HALF_OPEN.
- HALF_OPEN: one trial is permitted; success closes the circuit, any
  failure re-opens it for another `reset_timeout_s`.

Thread-safe: a single lock guards all state, because uvicorn serves our
blocking handlers from a worker thread-pool.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from enum import Enum


class State(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    def __init__(
        self,
        *,
        failure_threshold: int = 5,
        window_s: float = 30.0,
        reset_timeout_s: float = 60.0,
        name: str = "predict",
        now: "callable" = time.monotonic,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.window_s = window_s
        self.reset_timeout_s = reset_timeout_s
        self.name = name
        self._now = now

        self._lock = threading.Lock()
        self._state = State.CLOSED
        self._failures: deque[float] = deque()   # timestamps within window
        self._opened_at: float = 0.0

        # Observability counters.
        self.opened_total = 0
        self.short_circuited_total = 0

    # ------------------------------------------------------------------
    @property
    def state(self) -> State:
        with self._lock:
            return self._state

    def _prune(self, t: float) -> None:
        cutoff = t - self.window_s
        while self._failures and self._failures[0] < cutoff:
            self._failures.popleft()

    def allow(self) -> bool:
        """Return True if a call may proceed (updates state if needed)."""
        with self._lock:
            t = self._now()
            if self._state == State.OPEN:
                if t - self._opened_at >= self.reset_timeout_s:
                    self._state = State.HALF_OPEN  # permit a single trial
                    return True
                self.short_circuited_total += 1
                return False
            # CLOSED or HALF_OPEN both allow the (trial) call
            return True

    def record_success(self) -> None:
        with self._lock:
            self._failures.clear()
            if self._state in (State.HALF_OPEN, State.OPEN):
                self._state = State.CLOSED

    def record_failure(self) -> None:
        with self._lock:
            t = self._now()
            if self._state == State.HALF_OPEN:
                # trial failed -> straight back to OPEN
                self._state = State.OPEN
                self._opened_at = t
                self.opened_total += 1
                return
            self._failures.append(t)
            self._prune(t)
            if len(self._failures) >= self.failure_threshold:
                self._state = State.OPEN
                self._opened_at = t
                self._failures.clear()
                self.opened_total += 1

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "name":               self.name,
                "state":              self._state.value,
                "recent_failures":    len(self._failures),
                "failure_threshold":  self.failure_threshold,
                "opened_total":       self.opened_total,
                "short_circuited_total": self.short_circuited_total,
            }
