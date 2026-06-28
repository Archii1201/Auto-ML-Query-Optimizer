"""
collect_tpch_param_plans.py
===========================
Phase 3B plan collection — parameterized TPC-H.

Iterates over every (query, parameter-set, optimizer-variant) tuple
produced by `db.tpch_param_queries.generate()` and runs
EXPLAIN (ANALYZE, BUFFERS, VERBOSE, FORMAT JSON) under each variant.
Output is schema-compatible with Phase 2A so the existing feature
extractor consumes it unchanged.

Naming:
    data/tpch/plans_param/{query_id}__{variant}__{sql_hash}.json
    data/tpch/plans_param/_index.jsonl

Why a separate output directory?
    Keeps the canonical 87 Phase-2A plans pristine for reproducibility,
    while still letting Phase 2B's extractor glob both directories
    in one pass (`--input data/tpch/plans --input data/tpch/plans_param`).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import psycopg2
import psycopg2.errors

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from config.db_config import DB_CONFIG  # noqa: E402
from collect_data import (  # noqa: E402
    aggregate_label_runs,
    extract_summary,
    short_hash,
)

from db.tpch_param_queries import generate as generate_param_queries  # noqa: E402

PLANS_DIR  = PROJECT_ROOT / "data" / "tpch" / "plans_param"
INDEX_FILE = PLANS_DIR / "_index.jsonl"
PLANS_DIR.mkdir(parents=True, exist_ok=True)

# Optimizer variants — same matrix as Phase 2A.
VARIANTS: dict[str, list[str]] = {
    "default":      [],
    "no_hashjoin":  ["SET enable_hashjoin  = off"],
    "no_mergejoin": ["SET enable_mergejoin = off"],
    "no_nestloop":  ["SET enable_nestloop  = off"],
}

EXPLAIN_PREFIX = "EXPLAIN (ANALYZE, BUFFERS, VERBOSE, FORMAT JSON) "
STATEMENT_TIMEOUT_MS = 5 * 60 * 1000  # 5 minutes per query


def _run_explain(cur, sql: str) -> list | None:
    try:
        cur.execute(EXPLAIN_PREFIX + sql)
        return cur.fetchone()[0]
    except psycopg2.errors.QueryCanceled:
        return None
    except Exception:  # noqa: BLE001
        return None


def collect_one(
    cur,
    query: dict,
    variant: str,
    settings: list[str],
    *,
    label_runs: int = 1,
) -> dict | None:
    sql = query["sql"].rstrip().rstrip(";")
    label = f"{query['id']}/{variant}"

    cur.execute("RESET ALL;")
    # TPC-H tables live in the `tpch` schema (see migrate_to_schemas.py).
    # RESET ALL clears search_path, so re-set it on every query.
    cur.execute("SET search_path = tpch, public;")
    cur.execute(f"SET statement_timeout = {STATEMENT_TIMEOUT_MS};")
    for stmt in settings:
        cur.execute(stmt + ";")

    runs_tag = f" x{label_runs}" if label_runs > 1 else ""
    print(f"[*] {label:<32}{runs_tag}", end=" ", flush=True)

    attempts: list[dict] = []
    for _ in range(max(label_runs, 1)):
        plan_json = _run_explain(cur, sql)
        if plan_json is None:
            print(f"TIMEOUT (>{STATEMENT_TIMEOUT_MS / 1000:.0f}s)")
            return None
        summary = extract_summary(plan_json)
        attempts.append({
            "plan_json":         plan_json,
            "execution_time_ms": summary["execution_time_ms"],
        })

    if label_runs > 1:
        agg = aggregate_label_runs(attempts)
        plan_json = agg["plan_json"]
        summary   = agg["summary"]
        wall_ms   = agg["wall_time_ms"]
        extra = {
            "target_variance_ms": agg["target_variance_ms"],
            "label_runs":         agg["label_runs"],
        }
    else:
        plan_json = attempts[0]["plan_json"]
        summary   = extract_summary(plan_json)
        wall_ms   = round(float(summary["execution_time_ms"] or 0.0), 3)
        extra = {}

    record = {
        "query_id":      query["id"],
        "variant":       variant,
        "variant_knobs": settings,
        "tag":           query["tag"],
        "params":        query.get("params", {}),
        "sql":           sql,
        "sql_hash":      short_hash(sql),
        "collected_at":  datetime.now(timezone.utc).isoformat(),
        "wall_time_ms":  wall_ms,
        "summary":       summary,
        "plan":          plan_json,
        **extra,
    }

    out_path = PLANS_DIR / f"{query['id']}__{variant}__{record['sql_hash']}.json"
    out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"  est={summary['estimated_total_cost']!s:>10}  "
          f"exec={summary['execution_time_ms']!s:>10}ms")
    return record


def append_index(record: dict) -> None:
    line = {
        "query_id":     record["query_id"],
        "variant":      record["variant"],
        "tag":          record["tag"],
        "sql_hash":     record["sql_hash"],
        "collected_at": record["collected_at"],
        **record["summary"],
    }
    with INDEX_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(line) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--label-runs", type=int, default=1,
        help="Execute each (query, variant) N times, drop slowest, "
             "median-label the rest (default: 1).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    queries = generate_param_queries()
    n_total = len(queries) * len(VARIANTS)
    print(f"[i] {len(queries)} parameterized queries  x  "
          f"{len(VARIANTS)} variants  =  {n_total} plans to collect")
    print(f"[i] Output dir: {PLANS_DIR}")
    print(f"[i] Statement timeout: {STATEMENT_TIMEOUT_MS / 1000:.0f}s")
    if args.label_runs > 1:
        print(f"[i] Label runs: {args.label_runs} (drop slowest, median label)")
    print()

    try:
        conn = psycopg2.connect(**DB_CONFIG)
    except psycopg2.OperationalError as exc:
        print(f"[!] Could not connect to PostgreSQL: {exc}", file=sys.stderr)
        return 1
    conn.autocommit = True

    ok, failed = 0, 0
    try:
        with conn.cursor() as cur:
            for q in queries:
                for vname, settings in VARIANTS.items():
                    rec = collect_one(
                        cur, q, vname, settings, label_runs=args.label_runs,
                    )
                    if rec is None:
                        failed += 1
                        continue
                    append_index(rec)
                    ok += 1
    finally:
        conn.close()

    print(f"\n[OK] Done. {ok}/{n_total} plans captured, {failed} failed/timeout.")
    print(f"[OK] Plans -> {PLANS_DIR}")
    print(f"[OK] Index -> {INDEX_FILE}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
