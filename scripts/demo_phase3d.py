"""
demo_phase3d.py
===============
End-to-end Phase 3D demo over HTTP.

Walks the user's full SYSTEM FLOW arc:

    Query ─► Generate Plans ─► ML Prediction ─► Best Plan Selection
          ─► Execution ─► Collect Actual Time ─► Store Data

Run:
    # In one terminal:
    python -m services.ml_service.server

    # In another:
    python scripts/demo_phase3d.py            # default: 5 sample queries
    python scripts/demo_phase3d.py --oracle   # also runs every variant for regret
    python scripts/demo_phase3d.py --sql "SELECT ..."
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

BASE = "http://127.0.0.1:8000"

SAMPLE_QUERIES: list[tuple[str, str]] = [
    ("tpch_q01_pricing",
     """SELECT l_returnflag, l_linestatus,
              SUM(l_quantity) AS sum_qty, SUM(l_extendedprice) AS sum_base_price,
              AVG(l_discount) AS avg_disc, COUNT(*) AS count_order
        FROM lineitem WHERE l_shipdate <= DATE '1998-12-01'
        GROUP BY l_returnflag, l_linestatus ORDER BY l_returnflag, l_linestatus"""),
    ("tpch_q06",
     """SELECT SUM(l_extendedprice * l_discount) AS revenue FROM lineitem
        WHERE l_shipdate >= DATE '1994-01-01' AND l_shipdate < DATE '1995-01-01'
          AND l_discount BETWEEN 0.05 AND 0.07 AND l_quantity < 24"""),
    ("tpch_q14_promo",
     """SELECT 100.0 * SUM(CASE WHEN p_type LIKE 'PROMO%'
                                 THEN l_extendedprice*(1-l_discount) ELSE 0 END)
                    / SUM(l_extendedprice*(1-l_discount)) AS promo_revenue
        FROM lineitem, part WHERE l_partkey = p_partkey
          AND l_shipdate >= DATE '1995-09-01' AND l_shipdate < DATE '1995-10-01'"""),
    ("ds_topk_brand",
     """SELECT i_brand, SUM(ss_ext_sales_price) brand_rev FROM store_sales, item
        WHERE ss_item_sk = i_item_sk GROUP BY i_brand ORDER BY brand_rev DESC LIMIT 30"""),
    ("ds_yearly_trend",
     """SELECT d_year, SUM(ss_ext_sales_price) yearly FROM store_sales, date_dim
        WHERE ss_sold_date_sk = d_date_sk GROUP BY d_year ORDER BY d_year"""),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--regime",  default="plan_time")
    p.add_argument("--sql",     default=None)
    p.add_argument("--oracle",  action="store_true",
                   help="also execute every other variant for regret analysis")
    p.add_argument("--no-feedback", dest="write_feedback", action="store_false",
                   help="don't persist feedback rows (e.g. for benchmarking)")
    p.add_argument("--statement-timeout-ms", type=int, default=60_000)
    return p.parse_args()


def call_run_and_learn(sql: str, args: argparse.Namespace) -> dict:
    payload = {
        "sql":                  sql,
        "regime":               args.regime,
        "statement_timeout_ms": args.statement_timeout_ms,
        "oracle":               args.oracle,
        "write_feedback":       args.write_feedback,
    }
    r = httpx.post(f"{BASE}/run-and-learn", json=payload,
                   timeout=args.statement_timeout_ms / 1000.0 + 30)
    r.raise_for_status()
    return r.json()


def render(label: str, sql: str, out: dict, oracle_mode: bool) -> None:
    print(f"\n{'=' * 78}\n[query] {label}")
    print(f"        {' '.join(sql.split())[:140]}")

    print(f"\n  picked     : {out['picked_variant']}  (predicted={out['predicted_ms']:.1f} ms)")
    print(f"  actual_ms  : {out['actual_wall_ms']:.1f}")
    print(f"  feedback   : {out['feedback_path'] or 'NOT WRITTEN'}")

    print(f"\n  {'variant':<14} {'pred ms':>10} {'est cost':>12}")
    for c in out["candidates"]:
        marker = " <- WINNER" if c["variant"] == out["picked_variant"] else ""
        print(f"  {c['variant']:<14} {c['predicted_ms']:>10.1f} "
              f"{c['estimated_cost']:>12.1f}{marker}")

    if oracle_mode:
        truths = out.get("truths") or []
        if truths:
            print(f"\n  {'variant':<14} {'actual ms':>12}")
            for t in truths:
                tag = []
                if t["variant"] == out["picked_variant"]:
                    tag.append("PICKED")
                if t["variant"] == out.get("oracle_variant"):
                    tag.append("ORACLE")
                if t["timed_out"]:
                    tag.append("TIMEOUT")
                tag_str = "  <- " + " / ".join(tag) if tag else ""
                print(f"  {t['variant']:<14} {t['wall_time_ms']:>12.1f}{tag_str}")

        if out.get("oracle_variant"):
            hit = "HIT" if out.get("plan_pick_hit") else "MISS"
            print(f"\n  oracle     : {out['oracle_variant']}  "
                  f"({out['oracle_wall_ms']:.1f} ms)")
            print(f"  result     : {hit}  "
                  f"regret={out['regret_ms']:.1f} ms  "
                  f"({(out['regret_ratio'] or 0) * 100:+.1f}%)")

    print(f"  total HTTP elapsed: {out['elapsed_ms']:.1f} ms")


def main() -> int:
    args = parse_args()

    try:
        h = httpx.get(f"{BASE}/healthz", timeout=2.0)
        h.raise_for_status()
    except Exception as exc:
        print(f"[!] cannot reach service at {BASE}: {exc}", file=sys.stderr)
        print("    start it with:  python -m services.ml_service.server",
              file=sys.stderr)
        return 1

    queries = [("user_query", args.sql)] if args.sql else SAMPLE_QUERIES

    failures = 0
    pp_hits  = 0
    pp_total = 0
    total_regret = 0.0
    for label, sql in queries:
        try:
            out = call_run_and_learn(sql, args)
            render(label, sql, out, oracle_mode=args.oracle)
            if args.oracle and out.get("plan_pick_hit") is not None:
                pp_total += 1
                if out["plan_pick_hit"]:
                    pp_hits += 1
                total_regret += float(out.get("regret_ms") or 0)
        except Exception as exc:  # noqa: BLE001
            print(f"  [!] failed: {exc.__class__.__name__}: {exc}")
            failures += 1

    print("\n" + "=" * 78)
    if args.oracle and pp_total:
        acc = pp_hits / pp_total
        print(f"online plan-pick accuracy: {pp_hits}/{pp_total} = {acc:.1%}")
        print(f"online avg regret_ms     : {total_regret / pp_total:.1f}")

    m = httpx.get(f"{BASE}/metrics").json()
    print(f"\n[metrics] feedback rows: {m['feedback']['files_on_disk']} on disk "
          f"(this session: {m['feedback']['session_writes']})")
    print(f"[metrics] counters: {json.dumps(m['counters'])}")

    return 0 if failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
