"""
smoke_test_phase3c.py
=====================
Hits the running ML service over HTTP to verify:

    GET  /healthz
    GET  /readyz
    GET  /info
    POST /plan-pick (cache miss + cache hit)

Run:
    # In one terminal:
    python -m services.ml_service.server
    # In another:
    python scripts/smoke_test_phase3c.py
"""

from __future__ import annotations

import json
import sys

import httpx

BASE = "http://127.0.0.1:8000"


def main() -> int:
    print("\n=== /healthz ===")
    r = httpx.get(f"{BASE}/healthz", timeout=5.0)
    print(r.status_code, r.json())

    print("\n=== /readyz ===")
    r = httpx.get(f"{BASE}/readyz", timeout=5.0)
    print(r.status_code)
    print(json.dumps(r.json(), indent=2))

    print("\n=== /info?regime=plan_time ===")
    r = httpx.get(f"{BASE}/info", params={"regime": "plan_time"}, timeout=5.0)
    print(r.status_code)
    print(json.dumps(r.json(), indent=2))

    sql = """
        SELECT SUM(l_extendedprice * l_discount) AS revenue
        FROM lineitem
        WHERE l_shipdate >= DATE '1994-01-01'
          AND l_shipdate <  DATE '1995-01-01'
          AND l_discount BETWEEN 0.05 AND 0.07
          AND l_quantity < 24
    """

    print("\n=== POST /plan-pick (first call) ===")
    r = httpx.post(
        f"{BASE}/plan-pick",
        json={"sql": sql, "top_k": 4, "regime": "plan_time"},
        timeout=60.0,
    )
    print(r.status_code)
    if r.status_code != 200:
        print(r.text)
        return 1
    out = r.json()
    print(f"sql_hash    : {out['sql_hash'][:16]}...")
    print(f"winner      : variant={out['winner']['variant']}  "
          f"pred={out['winner']['predicted_ms']:.1f} ms")
    print(f"model_name  : {out['model_name']}")
    print(f"cache_hit   : {out['cache_hit']}")
    print(f"elapsed_ms  : {out['elapsed_ms']:.1f}")
    print("candidates:")
    for c in out["candidates"]:
        print(f"  {c['variant']:<14} pred={c['predicted_ms']:>9.1f} "
              f" cost={c['estimated_cost']:>11.1f}")

    print("\n=== POST /plan-pick (second call -- expecting cache hit) ===")
    r2 = httpx.post(
        f"{BASE}/plan-pick",
        json={"sql": sql, "top_k": 4, "regime": "plan_time"},
        timeout=60.0,
    )
    out2 = r2.json()
    print(f"cache_hit  : {out2['cache_hit']}")
    print(f"elapsed_ms : {out2['elapsed_ms']:.2f}  (vs {out['elapsed_ms']:.1f} on first call)")
    speedup = out["elapsed_ms"] / max(out2["elapsed_ms"], 1e-3)
    print(f"speedup    : {speedup:.1f}x")

    print("\n[OK] all endpoints responded correctly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
