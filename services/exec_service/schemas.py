"""
schemas.py
==========
Pydantic v2 contracts for the execution endpoints.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# /execute  — explicit: caller specifies which variant to run
# ---------------------------------------------------------------------------
class ExecuteRequest(BaseModel):
    sql:                  str             = Field(..., min_length=1)
    variant:              str             = "default"
    knobs:                list[str] | None = None
    """If `knobs` is None and `variant` is one of the predefined names
    (default / no_hashjoin / no_mergejoin / no_nestloop), the server
    looks them up automatically. Otherwise pass an explicit list of
    `SET enable_*=off` statements."""
    statement_timeout_ms: int  = Field(60_000, ge=100, le=600_000)
    write_feedback:       bool = True
    selected_by:          str  = "user"


class ExecuteResponse(BaseModel):
    variant:        str
    knobs:          list[str]
    wall_time_ms:   float
    timed_out:      bool
    feedback_path:  str | None = None
    request_id:     str
    elapsed_ms:     float


# ---------------------------------------------------------------------------
# /run-and-learn  — the full loop in one call
# ---------------------------------------------------------------------------
class RunAndLearnRequest(BaseModel):
    sql:                  str  = Field(..., min_length=1)
    regime:               str  = "plan_time"
    statement_timeout_ms: int  = Field(60_000, ge=100, le=600_000)
    oracle:               bool = False
    """If true, every candidate variant is also executed and the response
    contains regret vs. the true fastest variant. Useful for evaluating
    online plan-pick accuracy (it doubles the wallclock cost)."""
    write_feedback:       bool = True


class CandidatePrediction(BaseModel):
    variant:        str
    knobs:          list[str]
    # None when served via the PG-default fallback (circuit breaker OPEN).
    predicted_ms:   float | None
    estimated_cost: float


class OracleVariantResult(BaseModel):
    variant:      str
    wall_time_ms: float
    timed_out:    bool


class RunAndLearnResponse(BaseModel):
    sql_hash:        str
    request_id:      str
    regime:          str
    model_name:      str

    candidates:      list[CandidatePrediction]
    picked_variant:  str
    predicted_ms:    float | None

    actual_wall_ms:  float
    timed_out:       bool
    feedback_path:   str | None = None

    # populated only if request.oracle = True
    oracle_variant:  str | None       = None
    oracle_wall_ms:  float | None     = None
    regret_ms:       float | None     = None
    regret_ratio:    float | None     = None
    plan_pick_hit:   bool | None      = None
    truths:          list[OracleVariantResult] | None = None

    elapsed_ms:      float


# ---------------------------------------------------------------------------
# /metrics — process-wide registry snapshot
# ---------------------------------------------------------------------------
class MetricsSnapshot(BaseModel):
    uptime_seconds: float
    counters:       dict[str, int]
    histograms:     dict[str, dict[str, float]]
    feedback:       dict[str, Any]
