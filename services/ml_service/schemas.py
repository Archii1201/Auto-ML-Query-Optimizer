"""
schemas.py
==========
Pydantic v2 request/response contracts for the ML inference service.

Keep these models small and explicit — they're the public API.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Common
# ---------------------------------------------------------------------------
class ServiceInfo(BaseModel):
    """Returned by /healthz and /readyz."""
    status:        str
    model_loaded:  bool
    regime:        str | None     = None
    model_name:    str | None     = None
    feature_count: int | None     = None
    # Values are mixed (ints for hit/miss counts, strings for backend name)
    # since Phase 4B made the cache backend pluggable.
    cache_stats:   dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# /predict — one plan in, one prediction out
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    """
    Body of POST /predict.

    The `plan_json` payload should be exactly the structure returned by:
        EXPLAIN (FORMAT JSON) <sql>
    i.e. a list with one element shaped like
        [{"Plan": {...}, "Planning Time": ..., "Execution Time": ...}]

    `variant` (Phase 3E+) describes which planner knobs were active
    when the plan was generated. Examples: "default" (all on),
    "no_hashjoin", "no_mergejoin", "no_nestloop". The model uses
    these as features so the same plan tree under different knobs
    can produce different predictions.
    """
    plan_json:  list[dict[str, Any]] = Field(..., min_length=1)
    regime:     str = "plan_time"
    variant:    str = "default"


class PredictResponse(BaseModel):
    predicted_ms: float
    regime:       str
    model_name:   str
    cache_hit:    bool      = False
    elapsed_ms:   float


# ---------------------------------------------------------------------------
# /plan-pick — SQL in, winner plan out
# ---------------------------------------------------------------------------
class PlanCandidate(BaseModel):
    variant:        str
    knobs:          list[str]
    # None when the circuit breaker is OPEN and we served PG's default
    # plan without calling the model (Phase 4A fault-tolerance fallback).
    predicted_ms:   float | None
    estimated_cost: float
    plan_json:      list[dict[str, Any]] | None = None


class PlanPickRequest(BaseModel):
    sql:           str = Field(..., min_length=1)
    top_k:         int = Field(1, ge=1, le=10)
    regime:        str = "plan_time"
    include_plan:  bool = False
    """If true, candidate plan JSON is included in the response (verbose)."""


class PlanPickResponse(BaseModel):
    sql_hash:    str
    winner:      PlanCandidate
    candidates:  list[PlanCandidate]
    model_name:  str
    regime:      str
    cache_hit:   bool   = False
    elapsed_ms:  float
