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

import os
import time
from typing import Any

import psycopg2
import psycopg2.errors

from services.plan_generator.pg_variants import VARIANTS, GeneratedPlan

EXPLAIN_PLAN_PREFIX = "EXPLAIN (FORMAT JSON) "
EXPLAIN_ANALYZE_PREFIX = "EXPLAIN (ANALYZE, FORMAT JSON) "

# TPC-H/TPC-DS tables live in dedicated schemas (see Phase 3E.1 /
# migrate_to_schemas.py). `RESET ALL` clears search_path, so unqualified
# table names in served SQL would fail to resolve. We restore a sane
# search_path after every RESET. Override via ML_SERVICE_SEARCH_PATH.
SEARCH_PATH = os.environ.get("ML_SERVICE_SEARCH_PATH", "tpch, public")


def _set_knobs(cur, knobs: list[str]) -> None:
    cur.execute("RESET ALL;")
    cur.execute(f"SET search_path = {SEARCH_PATH};")
    for stmt in knobs:
        cur.execute(stmt + ";")


def _root_total_cost(plan_json: list[dict[str, Any]]) -> float:
    try:
        return float(plan_json[0]["Plan"]["Total Cost"])
    except (KeyError, IndexError, TypeError, ValueError):
        return 0.0


# PG error classes that mean "your SQL is broken" — caller wants to see these.
# Everything else (timeouts, planner refusals, feature unsupported) is benign:
# we silently skip the variant and let other variants succeed.
_SQL_USER_ERROR_CLASSES = (
    "42",  # syntax / access rule (SQL state 42xxx)
    "3D",  # invalid catalog (3D000)
    "3F",  # invalid schema name
    "23",  # integrity constraint violation (rare for SELECT)
)


def _is_user_sql_error(exc: psycopg2.Error) -> bool:
    """True iff this is the user's SQL being malformed, not a knob problem."""
    pgcode = getattr(exc, "pgcode", None) or ""
    return any(pgcode.startswith(cls) for cls in _SQL_USER_ERROR_CLASSES)


def generate_variants(
    conn,
    sql: str,
    variants: dict[str, list[str]] | None = None,
    *,
    plan_time_timeout_ms: int = 5_000,
) -> list[GeneratedPlan]:
    """
    Produce one plan per variant. Variants that fail because the
    optimizer can't satisfy the constraints (e.g. requires nestloops)
    are silently skipped. SQL-level errors (syntax, missing table)
    are *raised* so the API can return 4xx instead of an empty list.

    `plan_time_timeout_ms` caps each EXPLAIN to a few seconds so a
    pathological query can't tie up a request thread.
    """
    variants = variants if variants is not None else VARIANTS
    sql = sql.rstrip().rstrip(";")
    out: list[GeneratedPlan] = []
    last_user_err: psycopg2.Error | None = None

    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(f"SET statement_timeout = {plan_time_timeout_ms};")
        for name, knobs in variants.items():
            _set_knobs(cur, knobs)
            cur.execute(f"SET statement_timeout = {plan_time_timeout_ms};")
            try:
                cur.execute(EXPLAIN_PLAN_PREFIX + sql)
                plan_json = cur.fetchone()[0]
            except psycopg2.errors.QueryCanceled:
                continue
            except psycopg2.Error as exc:
                if _is_user_sql_error(exc):
                    last_user_err = exc
                continue
            out.append(GeneratedPlan(
                variant=name,
                knobs=list(knobs),
                plan_json=plan_json,
                estimated_cost=_root_total_cost(plan_json),
            ))

    # If *every* variant failed and the cause was a user-SQL error,
    # propagate it so the caller can return a 422 instead of 500.
    if not out and last_user_err is not None:
        raise last_user_err
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
