"""
chaos.py
========
Phase 4F — chaos test driver.

Validates the Phase 4A/4C fault-tolerance work *under load*: while Locust
hammers the gateway, this script kills a dependency for a while, then
brings it back, polling the service the whole time to record how it
degrades and recovers.

Expected, graceful behaviour:
    - kill **postgres** -> /plan-pick returns 503 (pool can't get a conn),
      /readyz flips to 503; on restart the pool self-heals -> 200 again.
    - kill **redis**    -> requests keep succeeding (cache is optional;
      the backend logs warnings and behaves like a cold cache).
    - kill **kafka**    -> /run-and-learn keeps succeeding; the publisher
      counts produce errors but never fails the request.

Usage:
    python loadtest/chaos.py --scenario postgres --down-seconds 20
    python loadtest/chaos.py --scenario all

Run a Locust load in another terminal first for a meaningful test.
"""

from __future__ import annotations

import argparse
import subprocess
import time
import urllib.request

DEFAULT_GATEWAY = "http://localhost"


def _compose(*args: str) -> None:
    cmd = ["docker", "compose", *args]
    print(f"[chaos] $ {' '.join(cmd)}")
    subprocess.run(cmd, check=False)


def probe(gateway: str) -> str:
    """Return a compact status string for /readyz."""
    try:
        with urllib.request.urlopen(f"{gateway}/readyz", timeout=3) as r:
            return f"readyz={r.status}"
    except Exception as exc:  # noqa: BLE001
        return f"readyz=ERR({type(exc).__name__})"


def probe_resilience(gateway: str) -> str:
    try:
        with urllib.request.urlopen(f"{gateway}/resilience", timeout=3) as r:
            import json
            data = json.loads(r.read().decode())
            cb = data.get("circuit_breaker", {}).get("state")
            pool = data.get("pool", {})
            return f"cb={cb} pool_timeouts={pool.get('timeouts_total')}"
    except Exception as exc:  # noqa: BLE001
        return f"resilience=ERR({type(exc).__name__})"


def watch(gateway: str, seconds: int, label: str) -> None:
    end = time.time() + seconds
    while time.time() < end:
        print(f"[chaos] {label:10s} {probe(gateway)}  {probe_resilience(gateway)}")
        time.sleep(2)


def run_scenario(svc: str, gateway: str, down_seconds: int) -> None:
    print(f"\n[chaos] === scenario: kill {svc} for {down_seconds}s ===")
    watch(gateway, 4, "baseline")
    _compose("stop", svc)
    watch(gateway, down_seconds, f"{svc}-down")
    _compose("start", svc)
    watch(gateway, 20, f"{svc}-recover")
    print(f"[chaos] === {svc} scenario complete ===")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", choices=["postgres", "redis", "kafka", "all"],
                   default="postgres")
    p.add_argument("--gateway", default=DEFAULT_GATEWAY)
    p.add_argument("--down-seconds", type=int, default=20)
    args = p.parse_args()

    targets = (["postgres", "redis", "kafka"] if args.scenario == "all"
               else [args.scenario])
    for svc in targets:
        run_scenario(svc, args.gateway, args.down_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
