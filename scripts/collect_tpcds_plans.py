"""
collect_tpcds_plans.py
======================
Phase 3B: collect EXPLAIN (ANALYZE, ...) plans for the curated
TPC-DS workload. Schema-compatible with Phase 2A output:

    data/tpcds/plans/{query_id}__{variant}__{sql_hash}.json
    data/tpcds/plans/_index.jsonl
"""

from __future__ import annotations

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
from collect_data import short_hash, extract_summary  # noqa: E402

from db.tpcds_queries import generate as generate_tpcds_queries  # noqa: E402

PLANS_DIR  = PROJECT_ROOT / "data" / "tpcds" / "plans"
INDEX_FILE = PLANS_DIR / "_index.jsonl"
PLANS_DIR.mkdir(parents=True, exist_ok=True)

VARIANTS: dict[str, list[str]] = {
    "default":      [],
    "no_hashjoin":  ["SET enable_hashjoin  = off"],
    "no_mergejoin": ["SET enable_mergejoin = off"],
    "no_nestloop":  ["SET enable_nestloop  = off"],
}

EXPLAIN_PREFIX       = "EXPLAIN (ANALYZE, BUFFERS, VERBOSE, FORMAT JSON) "
STATEMENT_TIMEOUT_MS = 5 * 60 * 1000


def collect_one(cur, query: dict, variant: str, settings: list[str]) -> dict | None:
    sql = query["sql"].rstrip().rstrip(";")
    label = f"{query['id']}/{variant}"

    cur.execute("RESET ALL;")
    cur.execute(f"SET statement_timeout = {STATEMENT_TIMEOUT_MS};")
    for stmt in settings:
        cur.execute(stmt + ";")

    print(f"[*] {label:<32}", end=" ", flush=True)

    t0 = time.perf_counter()
    try:
        cur.execute(EXPLAIN_PREFIX + sql)
        plan_json = cur.fetchone()[0]
    except psycopg2.errors.QueryCanceled:
        print(f"TIMEOUT (>{STATEMENT_TIMEOUT_MS / 1000:.0f}s)")
        return None
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc.__class__.__name__}: {exc}")
        return None
    wall_ms = (time.perf_counter() - t0) * 1000.0

    summary = extract_summary(plan_json)
    record = {
        "query_id":      query["id"],
        "variant":       variant,
        "variant_knobs": settings,
        "tag":           query["tag"],
        "sql":           sql,
        "sql_hash":      short_hash(sql),
        "collected_at":  datetime.now(timezone.utc).isoformat(),
        "wall_time_ms":  round(wall_ms, 3),
        "summary":       summary,
        "plan":          plan_json,
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


def main() -> int:
    queries = generate_tpcds_queries()
    n_total = len(queries) * len(VARIANTS)
    print(f"[i] {len(queries)} TPC-DS queries  x  {len(VARIANTS)} variants  =  "
          f"{n_total} plans to collect")
    print(f"[i] Output dir: {PLANS_DIR}\n")

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
                    rec = collect_one(cur, q, vname, settings)
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
