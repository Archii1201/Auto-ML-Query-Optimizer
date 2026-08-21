"""
plan_pick.py
============
High-level orchestrator: SQL in, ranked variants out.

Pipeline:
    sql
     │
     ▼
    plan_generator.explain.generate_variants(conn, sql)   # N candidates
     │
     ▼
    inference.Predictor.predict_one(plan_json) for each   # N predictions
     │
     ▼
    heapq.nsmallest(top_k, candidates, key=predicted_ms)  # DSA: heap

Why heapq.nsmallest?

    With only 4 variants today the heap is overkill, but the
    contract is "plan_generator may grow to 32+ variants in the
    future" (we'll add `from_collapse_limit` toggles, GUC tuning,
    etc.). At that point `nsmallest(k, list, key)` runs in
    O(N log k) instead of the naive O(N log N) full sort. We use
    the right algorithm now so we don't have to revisit when N grows.

Caching: `PlanPicker` keeps its own SQL-keyed cache so a repeated
identical query doesn't re-call PG at all.
"""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass
from typing import Any

from services.ml_service.cache import HashedLRUCache, hash_sql
from services.ml_service.inference import Predictor
from services.plan_generator.explain import generate_variants
from services.plan_generator.pg_variants import VARIANTS


@dataclass
class RankedCandidate:
    variant:        str
    knobs:          list[str]
    predicted_ms:   float | None    # None = PG-default fallback (no ML call)
    estimated_cost: float
    plan_json:      list[dict[str, Any]]


@dataclass
class PlanPickResult:
    sql_hash:    str
    winner:      RankedCandidate
    candidates:  list[RankedCandidate]   # already top-k sorted (asc)
    cache_hit:   bool
    elapsed_ms:  float


class PlanPicker:
    """
    Combines a Predictor (already loaded) with a per-request PG connection
    factory. We don't open a PG connection here — the caller (FastAPI
    server or CLI demo) controls the connection lifecycle.
    """

    def __init__(self, predictor: Predictor, sql_cache_capacity: int = 1024) -> None:
        self.predictor: Predictor      = predictor
        self.cache:    HashedLRUCache  = HashedLRUCache(
            capacity=sql_cache_capacity,
            namespace=f"planpick:{predictor.regime}",
        )

    # ------------------------------------------------------------------
    def pick(
        self,
        conn,
        sql: str,
        *,
        top_k: int = 1,
        variants: dict[str, list[str]] | None = None,
        plan_time_timeout_ms: int | None = None,
    ) -> PlanPickResult:
        t0 = time.perf_counter()
        sql_h = hash_sql(sql)

        cached = self.cache.get(sql_h)
        if cached is not None:
            winner, ranked = cached
            return PlanPickResult(
                sql_hash=sql_h,
                winner=winner,
                candidates=ranked[:top_k],
                cache_hit=True,
                elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            )

        gen_kwargs = {}
        if plan_time_timeout_ms is not None:
            gen_kwargs["plan_time_timeout_ms"] = plan_time_timeout_ms
        plans = generate_variants(conn, sql, variants or VARIANTS, **gen_kwargs)
        if not plans:
            raise RuntimeError("plan generator returned zero candidates")

        scored: list[RankedCandidate] = []
        for p in plans:
            # Pass variant so the predictor sees the correct knob-state
            # features. Required since Phase 3E for plan differentiation.
            pred = self.predictor.predict_one(p.plan_json, variant=p.variant)
            scored.append(RankedCandidate(
                variant=p.variant,
                knobs=p.knobs,
                predicted_ms=pred.predicted_ms,
                estimated_cost=p.estimated_cost,
                plan_json=p.plan_json,
            ))

        # Sort by (predicted_ms, estimated_cost). The estimated_cost
        # tiebreaker matters whenever PG ignores a knob (e.g. the plan
        # never used hash-join anyway) so multiple variants produce
        # the *same* plan JSON and hence the same predicted_ms.
        # Without a tiebreaker, the iteration order of VARIANTS leaks
        # through and biases the winner toward `default`.
        ranked = heapq.nsmallest(
            len(scored), scored,
            key=lambda c: (c.predicted_ms, c.estimated_cost),
        )
        winner = ranked[0]
        self.cache.set(sql_h, (winner, ranked))

        return PlanPickResult(
            sql_hash=sql_h,
            winner=winner,
            candidates=ranked[:top_k],
            cache_hit=False,
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
        )

    # ------------------------------------------------------------------
    def pick_default(
        self,
        conn,
        sql: str,
        *,
        plan_time_timeout_ms: int | None = None,
    ) -> PlanPickResult:
        """
        Fault-tolerance fallback (Phase 4A): when the ML prediction path
        is unavailable (circuit breaker OPEN), we do NOT call the model.
        We ask PostgreSQL for its own default plan and return it as the
        winner with `predicted_ms=NaN` (i.e. "PG decided, not the model").
        The system keeps serving correct plans, just without learned
        ranking, exactly as the system-flow doc requires.
        """
        t0 = time.perf_counter()
        sql_h = hash_sql(sql)

        gen_kwargs = {}
        if plan_time_timeout_ms is not None:
            gen_kwargs["plan_time_timeout_ms"] = plan_time_timeout_ms
        plans = generate_variants(conn, sql, {"default": []}, **gen_kwargs)
        if not plans:
            raise RuntimeError("plan generator returned zero candidates")

        p = plans[0]
        cand = RankedCandidate(
            variant=p.variant,
            knobs=p.knobs,
            predicted_ms=None,
            estimated_cost=p.estimated_cost,
            plan_json=p.plan_json,
        )
        return PlanPickResult(
            sql_hash=sql_h,
            winner=cand,
            candidates=[cand],
            cache_hit=False,
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
        )
