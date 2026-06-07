"""
server.py
=========
FastAPI app exposing:

    POST /predict     — single plan -> predicted_ms
    POST /plan-pick   — SQL -> winning variant (PG must be reachable)
    GET  /healthz     — liveness
    GET  /readyz      — readiness (model loaded?)
    GET  /info        — model + cache stats

Run:
    uvicorn services.ml_service.server:app --host 127.0.0.1 --port 8000

Or via the convenience entry-point:
    python -m services.ml_service.server
"""

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import psycopg2
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from config.db_config import DB_CONFIG  # noqa: E402

from services.ml_service.inference import Predictor, get_predictor  # noqa: E402
from services.ml_service.plan_pick import PlanPicker  # noqa: E402
from services.ml_service.schemas import (  # noqa: E402
    PlanCandidate,
    PlanPickRequest,
    PlanPickResponse,
    PredictRequest,
    PredictResponse,
    ServiceInfo,
)


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Eagerly load both regimes' AutoML winners on startup. This way
    the first request is fast and we fail loud if a joblib is missing.
    """
    regimes = (os.environ.get("ML_SERVICE_REGIMES") or "plan_time,post_mortem").split(",")
    loaded: dict[str, Predictor] = {}
    for r in regimes:
        r = r.strip()
        if not r:
            continue
        try:
            loaded[r] = get_predictor(r)
        except FileNotFoundError as exc:
            print(f"[!] Could not load regime '{r}': {exc}", file=sys.stderr)
    if not loaded:
        raise RuntimeError("no regime models could be loaded; aborting startup")

    pickers: dict[str, PlanPicker] = {r: PlanPicker(p) for r, p in loaded.items()}
    app.state.predictors = loaded
    app.state.pickers    = pickers
    print(f"[i] ML service ready; regimes loaded: {list(loaded)}")
    yield


app = FastAPI(
    title="AutoML Learned Query Optimizer — ML Service",
    version="3c.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
def _get_predictor_or_404(regime: str) -> Predictor:
    p: Predictor | None = app.state.predictors.get(regime)
    if p is None:
        raise HTTPException(404, f"regime '{regime}' not loaded; "
                                 f"available: {list(app.state.predictors)}")
    return p


def _get_picker_or_404(regime: str) -> PlanPicker:
    p: PlanPicker | None = app.state.pickers.get(regime)
    if p is None:
        raise HTTPException(404, f"regime '{regime}' not loaded; "
                                 f"available: {list(app.state.pickers)}")
    return p


def _new_pg_connection():
    """
    Open a *new* connection per request. PG connections are not
    thread-safe; FastAPI is async-first but our handlers do
    blocking PG work, so a fresh connection avoids cross-request
    interference. A real deployment would use a pool (psycopg-pool).
    """
    try:
        return psycopg2.connect(**DB_CONFIG)
    except psycopg2.OperationalError as exc:
        raise HTTPException(503, f"PostgreSQL unreachable: {exc}") from exc


# ---------------------------------------------------------------------------
# Health / info
# ---------------------------------------------------------------------------
@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz")
def readyz() -> dict[str, Any]:
    info: dict[str, Any] = {"status": "ok", "regimes": []}
    for r, p in app.state.predictors.items():
        info["regimes"].append({
            "regime":        r,
            "model":         p.model_name,
            "feature_count": len(p.feature_names),
        })
    return info


@app.get("/info", response_model=ServiceInfo)
def info(regime: str = "plan_time") -> ServiceInfo:
    p = _get_predictor_or_404(regime)
    return ServiceInfo(
        status="ok",
        model_loaded=True,
        regime=p.regime,
        model_name=p.model_name,
        feature_count=len(p.feature_names),
        cache_stats=p.cache.stats(),
    )


# ---------------------------------------------------------------------------
# /predict
# ---------------------------------------------------------------------------
@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    p = _get_predictor_or_404(req.regime)
    try:
        result = p.predict_one(req.plan_json)
    except Exception as exc:
        raise HTTPException(400, f"prediction failed: {exc}") from exc
    return PredictResponse(
        predicted_ms=result.predicted_ms,
        regime=result.regime,
        model_name=result.model_name,
        cache_hit=result.cache_hit,
        elapsed_ms=result.elapsed_ms,
    )


# ---------------------------------------------------------------------------
# /plan-pick
# ---------------------------------------------------------------------------
@app.post("/plan-pick", response_model=PlanPickResponse)
def plan_pick(req: PlanPickRequest) -> PlanPickResponse:
    picker = _get_picker_or_404(req.regime)

    conn = _new_pg_connection()
    try:
        result = picker.pick(conn, req.sql, top_k=req.top_k)
    finally:
        conn.close()

    def _candidate(c, include_plan: bool) -> PlanCandidate:
        return PlanCandidate(
            variant=c.variant,
            knobs=c.knobs,
            predicted_ms=c.predicted_ms,
            estimated_cost=c.estimated_cost,
            plan_json=c.plan_json if include_plan else None,
        )

    return PlanPickResponse(
        sql_hash=result.sql_hash,
        winner=_candidate(result.winner, req.include_plan),
        candidates=[_candidate(c, req.include_plan) for c in result.candidates],
        model_name=picker.predictor.model_name,
        regime=req.regime,
        cache_hit=result.cache_hit,
        elapsed_ms=result.elapsed_ms,
    )


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------
def main() -> int:
    import uvicorn
    host = os.environ.get("ML_SERVICE_HOST", "127.0.0.1")
    port = int(os.environ.get("ML_SERVICE_PORT", "8000"))
    uvicorn.run("services.ml_service.server:app", host=host, port=port, reload=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
