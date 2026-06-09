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
import psycopg2.errors
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from config.db_config import DB_CONFIG  # noqa: E402

from services.ml_service.inference import (  # noqa: E402
    InvalidPlanError,
    Predictor,
    get_predictor,
)
from services.ml_service.plan_pick import PlanPicker  # noqa: E402
from services.ml_service.schemas import (  # noqa: E402
    PlanCandidate,
    PlanPickRequest,
    PlanPickResponse,
    PredictRequest,
    PredictResponse,
    ServiceInfo,
)

# ----- Phase 3D: execution loop ------------------------------------------
from services.exec_service.capture import FeedbackWriter  # noqa: E402
from services.exec_service.metrics import REGISTRY  # noqa: E402
from services.exec_service.runner import ExecutionRunner  # noqa: E402
from services.exec_service.schemas import (  # noqa: E402
    CandidatePrediction,
    ExecuteRequest,
    ExecuteResponse,
    MetricsSnapshot,
    OracleVariantResult,
    RunAndLearnRequest,
    RunAndLearnResponse,
)
from services.plan_generator.pg_variants import VARIANTS  # noqa: E402


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------
# Hard cap on POST body size (10 MiB). Plan JSONs for SF1 TPC-H rarely
# exceed 200 KiB, so 10 MiB is generous head-room and still protects
# the server against accidental or malicious giant payloads.
MAX_REQUEST_BODY_BYTES = 10 * 1024 * 1024


def _warmup_predictors(predictors: dict[str, Predictor]) -> None:
    """
    Touch every predictor with a tiny dummy plan so the first real
    request doesn't pay the cold-start tax (joblib unpickle, numpy
    JIT, lightgbm booster init). Failures here are non-fatal.
    """
    dummy = [{
        "Plan": {
            "Node Type": "Seq Scan", "Total Cost": 1.0, "Startup Cost": 0.0,
            "Plan Rows": 1, "Plan Width": 4, "Actual Rows": 1,
        },
        "Planning Time": 0.1, "Execution Time": 0.1,
    }]
    for r, p in predictors.items():
        try:
            p.predict_one(dummy)
            p.cache.clear()  # don't pollute the real cache
            print(f"[i] warm-up OK: {r} ({p.model_name})")
        except Exception as exc:
            print(f"[!] warm-up failed for {r}: {exc}", file=sys.stderr)


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

    # Phase 3D: execution + feedback writer (process-wide singletons)
    app.state.feedback_writer = FeedbackWriter()
    app.state.runner          = ExecutionRunner(writer=app.state.feedback_writer)

    _warmup_predictors(loaded)
    print(f"[i] ML service ready; regimes loaded: {list(loaded)}")
    print(f"[i] Feedback dir: {app.state.feedback_writer.base_dir}")
    yield


app = FastAPI(
    title="AutoML Learned Query Optimizer — ML Service",
    version="3d.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Middleware: bound request body sizes
# ---------------------------------------------------------------------------
@app.middleware("http")
async def limit_request_size(request: Request, call_next):
    cl = request.headers.get("content-length")
    if cl is not None:
        try:
            if int(cl) > MAX_REQUEST_BODY_BYTES:
                return PlainTextResponse(
                    f"request body exceeds {MAX_REQUEST_BODY_BYTES} bytes",
                    status_code=413,
                )
        except ValueError:
            pass
    return await call_next(request)


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
    except InvalidPlanError as exc:
        # 422 = body parsed, but semantically wrong (FastAPI convention).
        raise HTTPException(422, f"invalid plan_json: {exc}") from exc
    except Exception as exc:
        raise HTTPException(500, f"unexpected inference failure: {exc}") from exc

    REGISTRY.inc("predictions_total")
    REGISTRY.inc("predict_cache_hits_total" if result.cache_hit else "predict_cache_misses_total")
    REGISTRY.observe("inference_latency_ms", result.elapsed_ms)
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
    except psycopg2.Error as exc:
        # `generate_variants` re-raises the PG SQL error when *every*
        # variant failed for the same user-error reason. Translate to 422.
        raise HTTPException(
            422,
            f"SQL rejected by PostgreSQL ({getattr(exc, 'pgcode', '?')}): {exc}",
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(422, str(exc)) from exc
    finally:
        conn.close()

    REGISTRY.inc("plan_picks_total")
    REGISTRY.inc("plan_pick_cache_hits_total" if result.cache_hit else "plan_pick_cache_misses_total")
    REGISTRY.observe("inference_latency_ms", result.elapsed_ms)

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
# Phase 3D — execution + feedback
# ---------------------------------------------------------------------------
@app.post("/execute", response_model=ExecuteResponse)
def execute(req: ExecuteRequest) -> ExecuteResponse:
    """Run a *specific* variant against PG and (optionally) write feedback."""
    import time
    import uuid
    runner: ExecutionRunner = app.state.runner

    knobs = req.knobs
    if knobs is None:
        if req.variant not in VARIANTS:
            raise HTTPException(
                400,
                f"unknown variant '{req.variant}'; available: {list(VARIANTS)} "
                f"(or pass `knobs` explicitly)",
            )
        knobs = VARIANTS[req.variant]

    request_id = uuid.uuid4().hex
    t0 = time.perf_counter()
    conn = _new_pg_connection()
    try:
        res = runner.run_single(
            conn,
            sql=req.sql,
            variant=req.variant,
            knobs=knobs,
            statement_timeout_ms=req.statement_timeout_ms,
            selected_by=req.selected_by,
            request_id=request_id,
            write_feedback=req.write_feedback,
        )
    finally:
        conn.close()

    return ExecuteResponse(
        variant=res.variant,
        knobs=res.knobs,
        wall_time_ms=res.wall_time_ms,
        timed_out=res.timed_out,
        feedback_path=str(res.feedback_path) if res.feedback_path else None,
        request_id=request_id,
        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
    )


@app.post("/run-and-learn", response_model=RunAndLearnResponse)
def run_and_learn(req: RunAndLearnRequest) -> RunAndLearnResponse:
    """
    The full closed loop in one call:
        SQL  →  generate variants  →  predict each  →  pick winner
             →  EXECUTE winner   →  capture wall_ms + plan
             →  write feedback row  →  return everything

    Set `oracle=true` to also execute every other variant for regret
    analysis (doubles the cost; turn off in production).
    """
    import time
    import uuid

    picker: PlanPicker      = _get_picker_or_404(req.regime)
    runner: ExecutionRunner = app.state.runner
    request_id = uuid.uuid4().hex

    conn = _new_pg_connection()
    t0 = time.perf_counter()
    try:
        # ---- Step 1: plan-pick -----------------------------------------
        try:
            pick = picker.pick(conn, req.sql, top_k=len(VARIANTS))
        except psycopg2.Error as exc:
            raise HTTPException(
                422,
                f"SQL rejected by PostgreSQL ({getattr(exc, 'pgcode', '?')}): {exc}",
            ) from exc
        except RuntimeError as exc:
            raise HTTPException(422, str(exc)) from exc
        candidates = [CandidatePrediction(
            variant=c.variant, knobs=c.knobs,
            predicted_ms=c.predicted_ms, estimated_cost=c.estimated_cost,
        ) for c in pick.candidates]

        REGISTRY.inc("plan_picks_total")
        REGISTRY.observe("inference_latency_ms", pick.elapsed_ms)

        # ---- Step 2: execute --------------------------------------------
        if req.oracle:
            # run every variant; pick is a "hit" if it equals the oracle
            tuples = [(c.variant, c.knobs, c.predicted_ms) for c in pick.candidates]
            report = runner.run_with_oracle(
                conn, sql=req.sql, candidates=tuples,
                picked_variant=pick.winner.variant,
                statement_timeout_ms=req.statement_timeout_ms,
                model_name=picker.predictor.model_name,
                regime=req.regime, request_id=request_id,
                write_feedback=req.write_feedback,
            )
            picked = report.picked
            truths = [
                OracleVariantResult(
                    variant=v, wall_time_ms=r.wall_time_ms,
                    timed_out=r.timed_out,
                ) for v, r in report.truths.items()
            ]
            return RunAndLearnResponse(
                sql_hash=pick.sql_hash, request_id=request_id,
                regime=req.regime, model_name=picker.predictor.model_name,
                candidates=candidates,
                picked_variant=picked.variant,
                predicted_ms=pick.winner.predicted_ms,
                actual_wall_ms=picked.wall_time_ms,
                timed_out=picked.timed_out,
                feedback_path=str(picked.feedback_path) if picked.feedback_path else None,
                oracle_variant=report.oracle_variant or None,
                oracle_wall_ms=report.oracle_wall_ms,
                regret_ms=report.regret_ms,
                regret_ratio=report.regret_ratio,
                plan_pick_hit=report.plan_pick_hit,
                truths=truths,
                elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            )

        # Non-oracle: execute only the picked variant.
        winner = pick.winner
        res = runner.run_single(
            conn,
            sql=req.sql,
            variant=winner.variant,
            knobs=winner.knobs,
            statement_timeout_ms=req.statement_timeout_ms,
            predicted_ms=winner.predicted_ms,
            model_name=picker.predictor.model_name,
            regime=req.regime,
            selected_by="ml",
            request_id=request_id,
            write_feedback=req.write_feedback,
        )
    finally:
        conn.close()

    return RunAndLearnResponse(
        sql_hash=pick.sql_hash, request_id=request_id,
        regime=req.regime, model_name=picker.predictor.model_name,
        candidates=candidates,
        picked_variant=res.variant,
        predicted_ms=winner.predicted_ms,
        actual_wall_ms=res.wall_time_ms,
        timed_out=res.timed_out,
        feedback_path=str(res.feedback_path) if res.feedback_path else None,
        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
@app.get("/metrics")
def metrics(fmt: str = "json"):
    """`?fmt=prom` returns Prometheus text exposition; default is JSON."""
    if fmt == "prom":
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse(REGISTRY.prom(), media_type="text/plain; version=0.0.4")

    snap = REGISTRY.json()
    snap["feedback"] = app.state.feedback_writer.stats()
    return MetricsSnapshot(**snap)


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
