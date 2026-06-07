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
    cache_stats:   dict[str, int] = Field(default_factory=dict)


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
    """
    plan_json:  list[dict[str, Any]] = Field(..., min_length=1)
    regime:     str = "plan_time"


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
    predicted_ms:   float
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
