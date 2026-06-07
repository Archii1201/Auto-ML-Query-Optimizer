"""
explain.py
==========
Helpers around `EXPLAIN (FORMAT JSON)` — the *plan-time* (not
analyze) form, so we can ask PostgreSQL to produce a plan in a
few milliseconds without actually executing the query.

`generate_variants(conn, sql, variants)` walks every entry in
the variants dict, toggles the corresponding planner knobs,
runs EXPLAIN, captures the JSON payload, and returns a list
of `GeneratedPlan` objects.

`execute_with_variant(conn, sql, knobs)` applies the same
knobs and runs the SQL for real — used by the demo to compute
oracle / regret numbers vs. the model's pick.
"""

from __future__ import annotations

import time
from typing import Any

import psycopg2
import psycopg2.errors

from services.plan_generator.pg_variants import VARIANTS, GeneratedPlan

EXPLAIN_PLAN_PREFIX = "EXPLAIN (FORMAT JSON) "
EXPLAIN_ANALYZE_PREFIX = "EXPLAIN (ANALYZE, FORMAT JSON) "


def _set_knobs(cur, knobs: list[str]) -> None:
    cur.execute("RESET ALL;")
    for stmt in knobs:
        cur.execute(stmt + ";")


def _root_total_cost(plan_json: list[dict[str, Any]]) -> float:
    try:
        return float(plan_json[0]["Plan"]["Total Cost"])
    except (KeyError, IndexError, TypeError, ValueError):
        return 0.0


def generate_variants(
    conn,
    sql: str,
    variants: dict[str, list[str]] | None = None,
) -> list[GeneratedPlan]:
    """
    Produce one plan per variant. Variants that fail (e.g. an
    optimizer that can't satisfy the request without nestloops)
    are silently skipped — never raise.
    """
    variants = variants if variants is not None else VARIANTS
    sql = sql.rstrip().rstrip(";")
    out: list[GeneratedPlan] = []

    conn.autocommit = True
    with conn.cursor() as cur:
        for name, knobs in variants.items():
            _set_knobs(cur, knobs)
            try:
                cur.execute(EXPLAIN_PLAN_PREFIX + sql)
                plan_json = cur.fetchone()[0]
            except (psycopg2.errors.QueryCanceled, psycopg2.Error):
                continue
            out.append(GeneratedPlan(
                variant=name,
                knobs=list(knobs),
                plan_json=plan_json,
                estimated_cost=_root_total_cost(plan_json),
            ))
    return out


def execute_with_variant(
    conn,
    sql: str,
    knobs: list[str],
    *,
    statement_timeout_ms: int = 60_000,
) -> tuple[float, list[dict[str, Any]] | None]:
    """
    Run SQL under the given knobs and return (wall_ms, analyze_plan_json).

    Used by the demo to obtain ground truth: how long did the
    model's chosen plan actually take? Capped at 60s by default;
    a timeout returns (timeout_ms, None).
    """
    sql = sql.rstrip().rstrip(";")
    conn.autocommit = True
    with conn.cursor() as cur:
        _set_knobs(cur, knobs)
        cur.execute(f"SET statement_timeout = {statement_timeout_ms};")
        t0 = time.perf_counter()
        try:
            cur.execute(EXPLAIN_ANALYZE_PREFIX + sql)
            plan_json = cur.fetchone()[0]
        except psycopg2.errors.QueryCanceled:
            return (statement_timeout_ms, None)
        wall_ms = (time.perf_counter() - t0) * 1000.0
    return (wall_ms, plan_json)
