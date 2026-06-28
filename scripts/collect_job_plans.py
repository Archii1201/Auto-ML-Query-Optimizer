"""
collect_job_plans.py
====================
Phase 3E — Join Order Benchmark (JOB / IMDB) plan collection.

JOB adds 113 real-world join-heavy queries that complement the
synthetic TPC-H / TPC-DS workloads.  This script mirrors
``collect_tpch_plans.py`` but reads SQL from the JOB workload.

Prerequisites (one-time):
    1. Download IMDB data (~3.5 GB) and load into PostgreSQL.
       See: https://github.com/gregrahn/join-order-benchmark
    2. Place JOB query files under ``db/job_queries/`` (one .sql per
       query, or a single ``job_queries.sql`` with ``-- @QUERY:`` markers).

Usage:
    python scripts/collect_job_plans.py
    python scripts/collect_job_plans.py --label-runs 3

Output:
    data/job/plans/{query_id}__{variant}__{sql_hash}.json
    data/job/plans/_index.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
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

QUERIES_FILE = PROJECT_ROOT / "db" / "job_queries.sql"
PLANS_DIR    = PROJECT_ROOT / "data" / "job" / "plans"
INDEX_FILE   = PLANS_DIR / "_index.jsonl"
PLANS_DIR.mkdir(parents=True, exist_ok=True)

VARIANTS: dict[str, list[str]] = {
    "default":      [],
    "no_hashjoin":  ["SET enable_hashjoin  = off"],
    "no_mergejoin": ["SET enable_mergejoin = off"],
    "no_nestloop":  ["SET enable_nestloop  = off"],
}

EXPLAIN_PREFIX = "EXPLAIN (ANALYZE, BUFFERS, VERBOSE, FORMAT JSON) "
STATEMENT_TIMEOUT_MS = 10 * 60 * 1000  # JOB queries can be slow

HEADER_RE = re.compile(
    r"^--\s*@QUERY:\s*(?P<id>\S+)\s*\|\s*tag:\s*(?P<tag>.+?)\s*$",
    re.MULTILINE,
)


def parse_queries(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"JOB queries file missing: {path}\n"
            "Download JOB SQL from https://github.com/gregrahn/join-order-benchmark "
            "and add db/job_queries.sql with -- @QUERY: markers."
        )
    text = path.read_text(encoding="utf-8")
    headers = list(HEADER_RE.finditer(text))
    if not headers:
        raise ValueError(f"No '-- @QUERY:' markers found in {path}")

    queries: list[dict] = []
    for i, m in enumerate(headers):
        start = m.end()
        end   = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        body  = text[start:end].strip()
        queries.append({
            "id":  m.group("id"),
            "tag": m.group("tag").strip(),
            "sql": body,
        })
    return queries


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
    cur.execute(f"SET statement_timeout = {STATEMENT_TIMEOUT_MS};")
    for stmt in settings:
        cur.execute(stmt + ";")

    print(f"[*] {label:<36}", end=" ", flush=True)

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
    print(f"exec={summary['execution_time_ms']!s:>10}ms")
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
    p.add_argument("--label-runs", type=int, default=1)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    queries = parse_queries(QUERIES_FILE)
    n_total = len(queries) * len(VARIANTS)
    print(f"[i] {len(queries)} JOB queries x {len(VARIANTS)} variants = {n_total}")
    print(f"[i] Output: {PLANS_DIR}\n")

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

    print(f"\n[OK] {ok}/{n_total} plans captured, {failed} failed.")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
