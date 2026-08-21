"""
locustfile.py
=============
Phase 4F — load test for the learned query optimizer gateway.

Run (UI):    locust -f loadtest/locustfile.py --host http://localhost
Run (head):  locust -f loadtest/locustfile.py --host http://localhost \
                    --headless -u 100 -r 20 -t 5m

It drives the two endpoints that matter in production:
    /plan-pick      (read-only: generate variants + predict + pick)
    /run-and-learn  (full loop: also executes the winner + writes feedback)

SLOs (asserted in the performance report, tracked live by Locust):
    - /plan-pick     p95 latency  < 500 ms
    - error rate                  < 1 %
    - plan-pick accuracy (oracle) tracked via /metrics, not here

We weight /plan-pick heavily (it's the hot path) and call /run-and-learn
occasionally (it actually runs SQL, so it's expensive).
"""

from __future__ import annotations

import random

from locust import HttpUser, between, task

# A small bank of TPC-H-shaped queries. Unqualified names resolve via the
# service's ML_SERVICE_SEARCH_PATH (default: tpch, public).
QUERIES = [
    "SELECT count(*) FROM lineitem",
    "SELECT count(*) FROM orders",
    "SELECT l_returnflag, count(*) FROM lineitem GROUP BY l_returnflag",
    "SELECT o_orderstatus, count(*) FROM orders GROUP BY o_orderstatus",
    """SELECT c_custkey, count(o_orderkey)
       FROM customer LEFT JOIN orders ON c_custkey = o_custkey
       GROUP BY c_custkey LIMIT 100""",
    """SELECT l_orderkey, sum(l_extendedprice)
       FROM lineitem GROUP BY l_orderkey ORDER BY 2 DESC LIMIT 20""",
]


class PlanPickUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task(9)
    def plan_pick(self):
        sql = random.choice(QUERIES)
        with self.client.post(
            "/plan-pick",
            json={"sql": sql, "regime": "plan_time", "top_k": 4},
            name="/plan-pick",
            catch_response=True,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"status {resp.status_code}")
            elif "winner" not in (resp.json() or {}):
                resp.failure("no winner in response")
            else:
                resp.success()

    @task(1)
    def run_and_learn(self):
        sql = random.choice(QUERIES)
        with self.client.post(
            "/run-and-learn",
            json={"sql": sql, "regime": "plan_time",
                  "statement_timeout_ms": 30000, "oracle": False,
                  "write_feedback": True},
            name="/run-and-learn",
            catch_response=True,
        ) as resp:
            # Execution can legitimately be slow; only non-200 is a failure.
            if resp.status_code != 200:
                resp.failure(f"status {resp.status_code}")
            else:
                resp.success()


class HealthUser(HttpUser):
    """Tiny background load on the gateway/readiness paths."""
    weight = 1
    wait_time = between(1, 3)

    @task
    def healthz(self):
        self.client.get("/healthz", name="/healthz")

    @task
    def readyz(self):
        with self.client.get("/readyz", name="/readyz",
                             catch_response=True) as r:
            # 503 is expected while a replica warms up; don't count as error.
            if r.status_code in (200, 503):
                r.success()
            else:
                r.failure(f"status {r.status_code}")
