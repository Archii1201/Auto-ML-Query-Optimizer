"""
demo_phase3c.py
===============
End-to-end Phase 3C demo:

    SQL  ──►  PG plan generator  ──►  ML predictor  ──►  pick winner
       └──────────────────►  PG executor (oracle)  ──────────────────┘

For each query:
    * generate 4 variants (default + 3 disabled-join knobs)
    * predict each variant's runtime
    * pick the winner (= argmin predicted)
    * EXECUTE all variants for the truth (uses statement_timeout)
    * report:  predicted vs actual, regret vs default,
               regret vs oracle (best of 4)

Usage:
    python scripts/demo_phase3c.py                 # uses 5 sample queries
    python scripts/demo_phase3c.py --regime post_mortem
    python scripts/demo_phase3c.py --sql "select 1"
    python scripts/demo_phase3c.py --tpch q05      # runs TPC-H Q5 vanilla
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import psycopg2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.db_config import DB_CONFIG  # noqa: E402

from services.ml_service.inference import Predictor  # noqa: E402
from services.ml_service.plan_pick import PlanPicker  # noqa: E402
from services.plan_generator.explain import execute_with_variant  # noqa: E402


SAMPLE_QUERIES: list[tuple[str, str]] = [
    ("tpch_q01_pricing",
     """SELECT l_returnflag, l_linestatus,
              SUM(l_quantity) AS sum_qty,
              SUM(l_extendedprice) AS sum_base_price,
              AVG(l_discount) AS avg_disc, COUNT(*) AS count_order
        FROM lineitem
        WHERE l_shipdate <= DATE '1998-12-01'
        GROUP BY l_returnflag, l_linestatus
        ORDER BY l_returnflag, l_linestatus"""),
    ("tpch_q06",
     """SELECT SUM(l_extendedprice * l_discount) AS revenue
        FROM lineitem
        WHERE l_shipdate >= DATE '1994-01-01'
          AND l_shipdate <  DATE '1995-01-01'
          AND l_discount BETWEEN 0.05 AND 0.07
          AND l_quantity < 24"""),
    ("tpch_q14_promo",
     """SELECT 100.0 * SUM(CASE WHEN p_type LIKE 'PROMO%'
                                 THEN l_extendedprice*(1-l_discount) ELSE 0 END)
                    / SUM(l_extendedprice*(1-l_discount)) AS promo_revenue
        FROM lineitem, part
        WHERE l_partkey = p_partkey
          AND l_shipdate >= DATE '1995-09-01'
          AND l_shipdate <  DATE '1995-10-01'"""),
    ("ds_topk_brand",
     """SELECT i_brand, SUM(ss_ext_sales_price) brand_rev
        FROM store_sales, item
        WHERE ss_item_sk = i_item_sk
        GROUP BY i_brand ORDER BY brand_rev DESC LIMIT 30"""),
    ("ds_yearly_trend",
     """SELECT d_year, SUM(ss_ext_sales_price) yearly
        FROM store_sales, date_dim
        WHERE ss_sold_date_sk = d_date_sk
        GROUP BY d_year ORDER BY d_year"""),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--regime", default="plan_time",
                   choices=["plan_time", "post_mortem"])
    p.add_argument("--sql", default=None,
                   help="run a single SQL string instead of the sample suite")
    p.add_argument("--statement-timeout-ms", type=int, default=60_000)
    return p.parse_args()


def _safe_run(conn, sql: str, knobs: list[str], timeout_ms: int) -> float:
    """Return wallclock ms (or sentinel = timeout_ms on timeout)."""
    wall_ms, _ = execute_with_variant(conn, sql, knobs,
                                      statement_timeout_ms=timeout_ms)
    return wall_ms


def run_one(picker: PlanPicker, conn, label: str, sql: str,
            timeout_ms: int) -> None:
    print(f"\n{'=' * 78}\n[query] {label}")
    print(f"        {' '.join(sql.split())[:140]}")

    t0 = time.perf_counter()
    pick = picker.pick(conn, sql, top_k=4)
    pick_ms = (time.perf_counter() - t0) * 1000.0

    print(f"\n  ML inference: {pick_ms:.1f} ms total ({len(pick.candidates)} candidates ranked)\n")
    print(f"  {'variant':<14} {'pred ms':>10} {'est cost':>12}")
    for c in pick.candidates:
        marker = " <- WINNER" if c.variant == pick.winner.variant else ""
        print(f"  {c.variant:<14} {c.predicted_ms:>10.1f} {c.estimated_cost:>12.1f}{marker}")

    # Ground-truth: run every variant
    print("\n  measuring oracle by executing all variants ...")
    truths: dict[str, float] = {}
    for c in pick.candidates:
        ms = _safe_run(conn, sql, c.knobs, timeout_ms)
        truths[c.variant] = ms

    oracle_var, oracle_ms = min(truths.items(), key=lambda kv: kv[1])
    default_ms            = truths.get("default", float("nan"))
    picked_var            = pick.winner.variant
    picked_ms             = truths[picked_var]

    print(f"\n  {'variant':<14} {'actual ms':>12}")
    for v, ms in truths.items():
        marker = []
        if v == oracle_var:
            marker.append("ORACLE")
        if v == picked_var:
            marker.append("PICKED")
        if v == "default":
            marker.append("PG-default")
        tag = "  <- " + " / ".join(marker) if marker else ""
        print(f"  {v:<14} {ms:>12.1f}{tag}")

    regret_vs_default = picked_ms - default_ms
    regret_vs_oracle  = picked_ms - oracle_ms
    speedup           = (default_ms / picked_ms) if picked_ms > 0 else float("nan")
    print(f"\n  result: picked={picked_var}  picked_ms={picked_ms:.1f}  oracle={oracle_var}  "
          f"oracle_ms={oracle_ms:.1f}")
    print(f"  vs default:  delta={regret_vs_default:+.1f} ms  ({speedup:.2f}x speedup)")
    print(f"  vs oracle:   delta={regret_vs_oracle:+.1f} ms")


def main() -> int:
    args = parse_args()

    print(f"[i] Loading AutoML winner for regime '{args.regime}' ...")
    predictor = Predictor(regime=args.regime)
    picker    = PlanPicker(predictor)
    print(f"    model={predictor.model_name}  features={len(predictor.feature_names)}\n"
          f"    {predictor.metadata}")

    print(f"[i] Connecting to PostgreSQL @ {DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}")
    try:
        conn = psycopg2.connect(**DB_CONFIG)
    except psycopg2.OperationalError as exc:
        print(f"[!] DB unreachable: {exc}", file=sys.stderr)
        return 1
    conn.autocommit = True

    queries: list[tuple[str, str]]
    if args.sql:
        queries = [("user_query", args.sql)]
    else:
        queries = SAMPLE_QUERIES

    try:
        for label, sql in queries:
            try:
                run_one(picker, conn, label, sql,
                        timeout_ms=args.statement_timeout_ms)
            except Exception as exc:  # noqa: BLE001
                print(f"  [!] failed: {exc.__class__.__name__}: {exc}")
    finally:
        conn.close()

    print("\n[OK] demo complete.")
    print(f"    predictor cache: {predictor.cache.stats()}")
    print(f"    picker cache   : {picker.cache.stats()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
