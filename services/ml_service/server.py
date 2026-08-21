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

import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import psycopg2
import psycopg2.errors
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from config.db_config import DB_CONFIG  # noqa: E402

# ----- Phase 4A: resilience primitives -----------------------------------
from services.ml_service.circuit_breaker import CircuitBreaker  # noqa: E402
from services.ml_service.db_pool import (  # noqa: E402
    PgConnectionPool,
    PoolClosed,
    PoolTimeout,
)
from services.ml_service.obs_logging import configure_logging, log_request  # noqa: E402
from services.ml_service.observability import setup_tracing  # noqa: E402
from services.ml_service.timeout_budget import TimeoutBudget  # noqa: E402

from services.ml_service.inference import (  # noqa: E402
    InvalidPlanError,
    Predictor,
    get_predictor,
    reset_predictors,
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
from services.exec_service.metrics import REGISTRY  # noqa: E402
from services.exec_service.runner import ExecutionRunner  # noqa: E402
from services.feedback_bus.publisher import make_publisher  # noqa: E402
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

# ----- Phase 4A configuration (all overridable via env) ------------------
POOL_MIN            = int(os.environ.get("ML_POOL_MIN", "2"))
POOL_MAX            = int(os.environ.get("ML_POOL_MAX", "10"))
POOL_ACQUIRE_TO_S   = float(os.environ.get("ML_POOL_ACQUIRE_TIMEOUT_S", "2.0"))
REQUEST_BUDGET_MS   = float(os.environ.get("ML_REQUEST_BUDGET_MS", "8000"))
CB_FAILS            = int(os.environ.get("ML_CB_FAILS", "5"))
CB_WINDOW_S         = float(os.environ.get("ML_CB_WINDOW_S", "30"))
CB_RESET_S          = float(os.environ.get("ML_CB_RESET_S", "60"))

# Phase 5C: shared secret protecting the model hot-swap endpoint. Empty by
# default → the endpoint refuses all callers until a token is configured, so
# an unconfigured deployment can never be reloaded by an anonymous request.
ADMIN_TOKEN         = os.environ.get("ML_ADMIN_TOKEN", "")

logger = logging.getLogger("ml_service")


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
            p.predict_one(dummy, variant="default")
            p.cache.clear()  # don't pollute the real cache
            print(f"[i] warm-up OK: {r} ({p.model_name})")
        except Exception as exc:
            print(f"[!] warm-up failed for {r}: {exc}", file=sys.stderr)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Eagerly load both regimes' AutoML winners on startup. This way
    the first request is fast and we fail loud if a joblib is missing.
    Phase 4A also stands up the connection pool + circuit breaker here
    and tears them down on shutdown (graceful drain for rolling deploys).
    """
    configure_logging(os.environ.get("ML_LOG_LEVEL", "INFO"))
    setup_tracing(app)  # Phase 4D: opt-in OTel tracing (no-op unless enabled)

    regimes = (os.environ.get("ML_SERVICE_REGIMES") or "plan_time,post_mortem").split(",")
    loaded: dict[str, Predictor] = {}
    for r in regimes:
        r = r.strip()
        if not r:
            continue
        try:
            loaded[r] = get_predictor(r)
        except FileNotFoundError as exc:
            logger.error("regime load failed", extra={"fields": {"regime": r, "error": str(exc)}})
    if not loaded:
        raise RuntimeError("no regime models could be loaded; aborting startup")

    pickers: dict[str, PlanPicker] = {r: PlanPicker(p) for r, p in loaded.items()}
    app.state.predictors = loaded
    app.state.pickers    = pickers

    # Phase 4A: bounded connection pool + ML circuit breaker.
    app.state.pool = PgConnectionPool(
        DB_CONFIG, minconn=POOL_MIN, maxconn=POOL_MAX,
        acquire_timeout_s=POOL_ACQUIRE_TO_S,
    )
    app.state.cb_predict = CircuitBreaker(
        failure_threshold=CB_FAILS, window_s=CB_WINDOW_S,
        reset_timeout_s=CB_RESET_S, name="predict",
    )

    # Phase 3D + 4C: execution + pluggable feedback publisher (file|kafka).
    app.state.publisher = make_publisher()
    app.state.runner    = ExecutionRunner(publisher=app.state.publisher)

    _warmup_predictors(loaded)
    logger.info("service ready", extra={"fields": {
        "regimes": list(loaded),
        "pool": app.state.pool.stats(),
        "feedback": app.state.publisher.stats(),
    }})
    try:
        yield
    finally:
        # Graceful shutdown: close the pool so in-flight work drains and
        # PG connections are released cleanly during a rolling restart.
        try:
            app.state.pool.closeall()
            logger.info("pool closed on shutdown")
        except Exception as exc:  # noqa: BLE001
            logger.warning("pool close failed", extra={"fields": {"error": str(exc)}})
        try:
            app.state.publisher.close()  # flush Kafka producer if active
        except Exception as exc:  # noqa: BLE001
            logger.warning("publisher close failed", extra={"fields": {"error": str(exc)}})


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


@app.middleware("http")
async def access_log(request: Request, call_next):
    """One structured JSON log line per request (Phase 4A)."""
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
    request.state.request_id = request_id
    request.state.log_fields = {}
    t0 = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["x-request-id"] = request_id
        return response
    finally:
        fields = {
            "request_id":  request_id,
            "method":      request.method,
            "path":        request.url.path,
            "status_code": status,
            "latency_ms":  round((time.perf_counter() - t0) * 1000.0, 2),
            **getattr(request.state, "log_fields", {}),
        }
        level = logging.INFO if status < 500 else logging.ERROR
        log_request(logger, level=level, **fields)


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


def _acquire_pg(budget: TimeoutBudget):
    """
    Borrow a connection from the pool within the request's time budget.
    Pool exhaustion or an unreachable PG become a clean 503 (backpressure)
    instead of crashing PG or hanging the request.
    """
    pool: PgConnectionPool = app.state.pool
    try:
        return pool.acquire(timeout=budget.acquire_timeout_s(POOL_ACQUIRE_TO_S))
    except PoolTimeout as exc:
        REGISTRY.inc("pool_timeouts_total")
        raise HTTPException(503, f"service busy / PostgreSQL unavailable: {exc}") from exc
    except PoolClosed as exc:
        raise HTTPException(503, "service shutting down") from exc


def _release_pg(conn) -> None:
    try:
        app.state.pool.release(conn)
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# Health / info
# ---------------------------------------------------------------------------
@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz")
def readyz() -> JSONResponse:
    """
    Readiness (Phase 4A): we are only ready to serve traffic if BOTH the
    model is loaded AND PostgreSQL is reachable through the pool. Returns
    503 when not ready, so an orchestrator stops routing to this replica.
    (Liveness — /healthz — stays 200 as long as the process is up.)
    """
    model_ok = bool(getattr(app.state, "predictors", None))
    pg_ok = app.state.pool.ping(timeout=1.0) if hasattr(app.state, "pool") else False
    ready = model_ok and pg_ok
    info: dict[str, Any] = {
        "status":   "ok" if ready else "not_ready",
        "model_loaded": model_ok,
        "postgres_reachable": pg_ok,
        "regimes": [
            {"regime": r, "model": p.model_name, "feature_count": len(p.feature_names)}
            for r, p in getattr(app.state, "predictors", {}).items()
        ],
    }
    return JSONResponse(info, status_code=200 if ready else 503)


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
        result = p.predict_one(req.plan_json, variant=req.variant)
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
def plan_pick(req: PlanPickRequest, request: Request) -> PlanPickResponse:
    picker = _get_picker_or_404(req.regime)
    breaker: CircuitBreaker = app.state.cb_predict
    budget = TimeoutBudget.start(REQUEST_BUDGET_MS)

    conn = _acquire_pg(budget)
    fallback_used = False
    try:
        stmt_to = budget.statement_timeout_ms()
        if breaker.allow():
            # Normal path: ML-ranked plan-pick.
            try:
                result = picker.pick(conn, req.sql, top_k=req.top_k,
                                     plan_time_timeout_ms=stmt_to)
                breaker.record_success()
            except psycopg2.Error as exc:
                # DB / SQL error — NOT an ML failure; don't trip the breaker.
                raise HTTPException(
                    422,
                    f"SQL rejected by PostgreSQL ({getattr(exc, 'pgcode', '?')}): {exc}",
                ) from exc
            except RuntimeError as exc:
                raise HTTPException(422, str(exc)) from exc
            except Exception as exc:  # noqa: BLE001 — ML/inference failure
                breaker.record_failure()
                REGISTRY.inc("plan_pick_fallback_total")
                logger.warning("predict failed; falling back to PG default",
                               extra={"fields": {"error": str(exc)}})
                result = picker.pick_default(conn, req.sql, plan_time_timeout_ms=stmt_to)
                fallback_used = True
        else:
            # Circuit OPEN: serve PG's default plan without calling the model.
            REGISTRY.inc("plan_pick_fallback_total")
            result = picker.pick_default(conn, req.sql, plan_time_timeout_ms=stmt_to)
            fallback_used = True
    finally:
        _release_pg(conn)

    REGISTRY.inc("plan_picks_total")
    REGISTRY.inc("plan_pick_cache_hits_total" if result.cache_hit else "plan_pick_cache_misses_total")
    REGISTRY.observe("inference_latency_ms", result.elapsed_ms)

    request.state.log_fields.update({
        "regime":   req.regime,
        "sql_hash": result.sql_hash,
        "fallback": fallback_used,
        "cb_state": breaker.state.value,
        "predicted_ms": (None if fallback_used else round(result.winner.predicted_ms, 3)),
    })

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
def execute(req: ExecuteRequest, request: Request) -> ExecuteResponse:
    """Run a *specific* variant against PG and (optionally) write feedback."""
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

    request_id = getattr(request.state, "request_id", uuid.uuid4().hex)
    t0 = time.perf_counter()
    # Execution can be long (real query run); give it its own budget for
    # acquiring a pooled connection but keep the SQL statement_timeout
    # under the caller's control via req.statement_timeout_ms.
    conn = _acquire_pg(TimeoutBudget.start(REQUEST_BUDGET_MS))
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
        _release_pg(conn)

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
def run_and_learn(req: RunAndLearnRequest, request: Request) -> RunAndLearnResponse:
    """
    The full closed loop in one call:
        SQL  →  generate variants  →  predict each  →  pick winner
             →  EXECUTE winner   →  capture wall_ms + plan
             →  write feedback row  →  return everything

    Set `oracle=true` to also execute every other variant for regret
    analysis (doubles the cost; turn off in production).
    """
    picker: PlanPicker      = _get_picker_or_404(req.regime)
    runner: ExecutionRunner = app.state.runner
    breaker: CircuitBreaker = app.state.cb_predict
    request_id = getattr(request.state, "request_id", uuid.uuid4().hex)

    budget = TimeoutBudget.start(REQUEST_BUDGET_MS)
    conn = _acquire_pg(budget)
    t0 = time.perf_counter()
    fallback_used = False
    pick = None
    try:
        # ---- Step 1: plan-pick (with circuit-breaker fallback) ---------
        stmt_to = budget.statement_timeout_ms()
        try:
            if breaker.allow():
                try:
                    pick = picker.pick(conn, req.sql, top_k=len(VARIANTS),
                                       plan_time_timeout_ms=stmt_to)
                    breaker.record_success()
                except (psycopg2.Error, RuntimeError):
                    raise
                except Exception as exc:  # noqa: BLE001 — ML failure
                    breaker.record_failure()
                    REGISTRY.inc("plan_pick_fallback_total")
                    logger.warning("predict failed; PG default fallback",
                                   extra={"fields": {"error": str(exc)}})
                    pick = picker.pick_default(conn, req.sql, plan_time_timeout_ms=stmt_to)
                    fallback_used = True
            else:
                REGISTRY.inc("plan_pick_fallback_total")
                pick = picker.pick_default(conn, req.sql, plan_time_timeout_ms=stmt_to)
                fallback_used = True
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
                model_version=picker.predictor.model_version,
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
            model_version=picker.predictor.model_version,
            regime=req.regime,
            selected_by="pg_fallback" if fallback_used else "ml",
            request_id=request_id,
            write_feedback=req.write_feedback,
        )
    finally:
        _release_pg(conn)
        request.state.log_fields.update({
            "regime":   req.regime,
            "sql_hash": pick.sql_hash if pick is not None else None,
            "fallback": fallback_used,
            "cb_state": breaker.state.value,
        })

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
        # Phase 4D: prefer native prometheus_client output (real histogram
        # buckets) when available; fall back to the homegrown exposition.
        if REGISTRY.prometheus_enabled:
            return PlainTextResponse(
                REGISTRY.prometheus_text(),
                media_type="text/plain; version=0.0.4",
            )
        return PlainTextResponse(REGISTRY.prom(), media_type="text/plain; version=0.0.4")

    snap = REGISTRY.json()
    snap["feedback"] = app.state.publisher.stats()
    return MetricsSnapshot(**snap)


@app.get("/resilience")
def resilience() -> dict[str, Any]:
    """Pool + breaker (4A) + cache backend (4B) + model registry (4B)."""
    from services.ml_service.model_registry import REGISTRY as MODEL_REGISTRY
    cache_info = {}
    for r, p in app.state.predictors.items():
        cache_info[r] = {
            "predict": p.cache.stats(),
            "plan_pick": app.state.pickers[r].cache.stats(),
        }
    return {
        "pool":            app.state.pool.stats(),
        "circuit_breaker": app.state.cb_predict.snapshot(),
        "budget_ms":       REQUEST_BUDGET_MS,
        "pool_acquire_timeout_s": POOL_ACQUIRE_TO_S,
        "cache":           cache_info,
        "model_registry":  MODEL_REGISTRY.snapshot(),
    }


# ---------------------------------------------------------------------------
# Phase 5C: hot model swap
# ---------------------------------------------------------------------------
@app.post("/admin/reload-models")
def reload_models(x_admin_token: str | None = Header(default=None)) -> dict[str, Any]:
    """
    Rebuild predictors/pickers from the model registry's *current* version,
    so a freshly promoted model (Phase 5C) starts serving without a process
    restart. Auth: X-Admin-Token must match ML_ADMIN_TOKEN.

    This is a graceful swap: we build the new predictors first and only then
    replace app.state, so a failed rebuild leaves the old models serving.
    """
    if not ADMIN_TOKEN or x_admin_token != ADMIN_TOKEN:
        raise HTTPException(403, "forbidden: bad or missing X-Admin-Token")

    before = {
        r: getattr(p, "model_version", None)
        for r, p in getattr(app.state, "predictors", {}).items()
    }
    regimes = list(before) or ["plan_time"]

    reset_predictors()  # drop memoized singletons so we reload from registry
    try:
        loaded: dict[str, Predictor] = {r: get_predictor(r) for r in regimes}
        pickers = {r: PlanPicker(p) for r, p in loaded.items()}
    except Exception as exc:  # noqa: BLE001 — keep old models on failure
        raise HTTPException(500, f"reload failed, kept previous models: {exc}") from exc

    _warmup_predictors(loaded)
    app.state.predictors = loaded
    app.state.pickers = pickers

    after = {r: p.model_version for r, p in loaded.items()}
    changed = {r: {"from": before.get(r), "to": after[r]}
               for r in after if before.get(r) != after[r]}
    log_request(logger, event="reload_models", changed=changed)
    return {"reloaded": True, "before": before, "after": after, "changed": changed}


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
