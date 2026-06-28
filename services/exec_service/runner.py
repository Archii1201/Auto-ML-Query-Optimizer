"""
runner.py
=========
Executes SQL on PostgreSQL under a chosen variant's knobs and
captures both the wallclock time *and* the EXPLAIN ANALYZE plan.

Two paths:

    * `run_single`     — one variant, one execution.
    * `run_with_oracle`— execute every candidate variant so we can
                         compute regret vs. ground truth. Used by the
                         /run-and-learn?oracle=true mode and by the
                         demo. Disabled by default (expensive).

Every successful execution is persisted via FeedbackWriter so the
training set grows.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import psycopg2
import psycopg2.errors

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from services.exec_service.capture import FeedbackWriter  # noqa: E402
from services.exec_service.metrics import REGISTRY  # noqa: E402
from services.plan_generator.explain import execute_with_variant  # noqa: E402


@dataclass
class ExecutionResult:
    variant:        str
    knobs:          list[str]
    wall_time_ms:   float
    plan_json:      list[dict[str, Any]] | None
    timed_out:      bool
    feedback_path:  Path | None = None


@dataclass
class OracleReport:
    picked:           ExecutionResult
    truths:           dict[str, ExecutionResult] = field(default_factory=dict)
    oracle_variant:   str = ""
    oracle_wall_ms:   float = float("nan")
    regret_ms:        float = float("nan")
    regret_ratio:     float = float("nan")
    plan_pick_hit:    bool = False


# ---------------------------------------------------------------------------
class ExecutionRunner:
    """
    Holds a single FeedbackWriter and exposes execution helpers.

    `conn` is passed in per request — we don't manage the PG
    connection lifecycle here (the FastAPI handler does, with a
    fresh connection per request).
    """

    def __init__(self, writer: FeedbackWriter | None = None) -> None:
        self.writer: FeedbackWriter = writer or FeedbackWriter()

    # ------------------------------------------------------------------
    def run_single(
        self,
        conn,
        *,
        sql:                  str,
        variant:              str,
        knobs:                list[str],
        statement_timeout_ms: int = 60_000,
        predicted_ms:         float | None = None,
        model_name:           str | None   = None,
        model_version:        str | None   = None,
        regime:               str | None   = None,
        selected_by:          str = "ml",
        request_id:           str | None = None,
        write_feedback:       bool = True,
        tag:                  str = "online",
        extra:                dict[str, Any] | None = None,
    ) -> ExecutionResult:
        """Run one variant and (optionally) persist the trace."""
        REGISTRY.inc("executions_total")
        t0 = time.perf_counter()
        wall_ms, plan_json = execute_with_variant(
            conn, sql, knobs, statement_timeout_ms=statement_timeout_ms,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        REGISTRY.observe("execution_latency_ms", elapsed_ms)

        timed_out = plan_json is None
        if timed_out:
            REGISTRY.inc("execution_failures_total")

        feedback_path: Path | None = None
        if write_feedback and not timed_out and plan_json is not None:
            feedback_path = self.writer.write(
                sql=sql, variant=variant, knobs=knobs,
                plan_json=plan_json, wall_time_ms=wall_ms,
                predicted_ms=predicted_ms,
                model_name=model_name, model_version=model_version,
                regime=regime, selected_by=selected_by,
                request_id=request_id, tag=tag, extra=extra,
            )
            REGISTRY.inc("feedback_rows_written_total")

            # Predicted/actual ratio (1.0 = perfect).
            if predicted_ms is not None and wall_ms > 0:
                REGISTRY.observe("pred_actual_ratio",
                                 float(predicted_ms) / float(wall_ms))

        return ExecutionResult(
            variant=variant,
            knobs=list(knobs),
            wall_time_ms=float(wall_ms),
            plan_json=plan_json,
            timed_out=timed_out,
            feedback_path=feedback_path,
        )

    # ------------------------------------------------------------------
    def run_with_oracle(
        self,
        conn,
        *,
        sql:                  str,
        candidates:           list[tuple[str, list[str], float]],  # (variant, knobs, predicted_ms)
        picked_variant:       str,
        statement_timeout_ms: int = 60_000,
        model_name:           str | None = None,
        model_version:        str | None = None,
        regime:               str | None = None,
        request_id:           str | None = None,
        write_feedback:       bool = True,
    ) -> OracleReport:
        """
        Execute *every* candidate to determine the true fastest, then
        compute regret. The picked variant's run is the one persisted
        to feedback by default; the others can also be persisted with
        selected_by='oracle' for a richer training set.
        """
        truths: dict[str, ExecutionResult] = {}
        for variant, knobs, pred_ms in candidates:
            res = self.run_single(
                conn, sql=sql, variant=variant, knobs=knobs,
                statement_timeout_ms=statement_timeout_ms,
                predicted_ms=pred_ms,
                model_name=model_name, model_version=model_version,
                regime=regime,
                selected_by=("ml" if variant == picked_variant else "oracle"),
                request_id=request_id,
                write_feedback=write_feedback,
                tag="online_oracle",
            )
            truths[variant] = res

        successful = {v: r for v, r in truths.items() if not r.timed_out}
        if not successful:
            return OracleReport(picked=truths[picked_variant], truths=truths)

        oracle_var, oracle_res = min(
            successful.items(), key=lambda kv: kv[1].wall_time_ms
        )
        picked = truths[picked_variant]
        regret_ms = max(picked.wall_time_ms - oracle_res.wall_time_ms, 0.0)
        regret_ratio = (
            picked.wall_time_ms / max(oracle_res.wall_time_ms, 1e-3) - 1.0
        )
        plan_pick_hit = (picked_variant == oracle_var)

        REGISTRY.inc("plan_picks_total")
        if plan_pick_hit:
            REGISTRY.inc("plan_pick_oracle_hits_total")
        # Regret histogram (ms): how many ms slower than oracle did
        # the picked variant run? 0 == perfect pick.
        REGISTRY.observe("regret_ms", float(regret_ms))
        # Same number expressed as a multiplier (picked / oracle - 1).
        REGISTRY.observe("regret_ratio", float(regret_ratio))

        return OracleReport(
            picked=picked, truths=truths,
            oracle_variant=oracle_var,
            oracle_wall_ms=oracle_res.wall_time_ms,
            regret_ms=regret_ms,
            regret_ratio=regret_ratio,
            plan_pick_hit=plan_pick_hit,
        )
