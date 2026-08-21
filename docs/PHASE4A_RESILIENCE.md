# Phase 4A — Resilience

**Goal:** make the ML service survive load spikes and dependency failures
without taking PostgreSQL down with it, and make every request
observable. Ends with a system that *degrades gracefully* instead of
crashing.

This phase adds five things:

1. Bounded **connection pool**
2. **Circuit breaker** around the ML prediction path
3. Per-request **timeout budget**
4. **Structured JSON logging** (one line per request)
5. **Liveness/readiness** split + **graceful shutdown**
6. **GitHub Actions** CI (lint + fast unit tests)

All of it is local-only — no new containers. Containers arrive in 4B+.

---

## 1. Connection pool — `services/ml_service/db_pool.py`

### What was wrong
The old code did `psycopg2.connect()` **per request** and closed it in a
`finally`. Under a burst of *N* concurrent requests that opens *N*
connections. A large enough burst exhausts PostgreSQL's
`max_connections`, after which **PG refuses everyone** — not just us. One
traffic spike can take down the whole database.

### What we do
A bounded pool caps live connections at `maxconn` (default 10). Excess
requests **wait** for a free connection up to a timeout, then get a clean
`503` (backpressure) rather than melting PG.

### Design (DSA: bounded blocking queue)
- A `queue.Queue(maxsize=maxconn)` holds idle connections.
- `acquire()` reuses an idle connection, or opens a new one if we're under
  the cap, or blocks up to the budget for one to free up.
- Connections are **validated with `SELECT 1` on acquire**; a dead one
  (PG restarted, network blip) is discarded and replaced transparently —
  this is what makes "kill PG, bring it back" recover on its own.
- `release()` rolls back any open transaction and returns the connection.

### Why not alternatives
- **`psycopg2.pool.ThreadedConnectionPool`** — exists, but its `getconn()`
  raises immediately when exhausted; it has **no acquire-with-timeout**,
  which is exactly the backpressure primitive we need. Our queue gives a
  blocking `get(timeout=…)` for free.
- **`psycopg-pool` (psycopg3)** — excellent, but switching drivers is a
  large change across every collector and the runner for the same
  guarantee. Deferred; not worth the blast radius now.
- **No pool (status quo)** — the failure mode above.

---

## 2. Circuit breaker — `services/ml_service/circuit_breaker.py`

### What it protects
The system-flow doc requires: *"if ML fails, fall back to the PostgreSQL
optimizer."* The breaker wraps the **prediction** step only.

### State machine (DSA: finite state machine)
```
 CLOSED ──(≥N failures in window)──► OPEN ──(reset timeout)──► HALF_OPEN
   ▲                                                            │   │
   │  success                                    trial success  │   │ trial failure
   └────────────────────────────────────────────────────────┘   └─► OPEN
```
- **CLOSED**: normal; failures counted in a rolling 30 s window.
- **OPEN**: predictions are **short-circuited** — `/plan-pick` serves
  `PlanPicker.pick_default()`, i.e. PostgreSQL's own default plan with
  `predicted_ms = null`. The system stays correct, just unranked.
- **HALF_OPEN**: after 60 s, one trial is allowed; success closes the
  circuit, failure re-opens it.

### Important: DB errors don't trip the breaker
A bad SQL / PG error is a **client** problem, not an ML outage — those map
to `422` and are *not* counted as breaker failures. Only genuine
inference failures open the circuit. This avoids false trips.

### Why a breaker (vs. plain try/except fallback)
A bare try/except would call the broken model on **every** request,
paying its latency/cost each time. The breaker *stops calling* a failing
dependency for a cool-down, then probes for recovery — fail fast, recover
automatically.

---

## 3. Timeout budget — `services/ml_service/timeout_budget.py`

A `/plan-pick` does several blocking stages in sequence (pool acquire → N
EXPLAINs → N predictions). Without one deadline, a slow PG plan or a pool
wait can blow the latency SLO unboundedly.

`TimeoutBudget` starts a single deadline (default `ML_REQUEST_BUDGET_MS =
8000`) and hands each stage only the time that's left:
- pool acquire wait is bounded by remaining (capped at
  `ML_POOL_ACQUIRE_TIMEOUT_S`),
- the PG `statement_timeout` for plan generation = remaining − a small
  response reserve,

so the **whole request** fails fast instead of hanging.

---

## 4. Structured JSON logging — `services/ml_service/obs_logging.py`

`print()` is unindexable. We emit one JSON object per log line so any
collector (Loki/ELK/CloudWatch) can filter on fields. A middleware writes
**one access line per request**:

```json
{"ts":"…","level":"INFO","service":"ml_service","msg":"request",
 "request_id":"…","method":"POST","path":"/plan-pick","status_code":200,
 "latency_ms":248.9,"regime":"plan_time","sql_hash":"…","fallback":false,
 "cb_state":"closed","predicted_ms":1970.572}
```

`request_id` is generated (or taken from an inbound `x-request-id`) and
echoed back as a response header — the seed for distributed tracing in 4D.

---

## 5. Health probes + graceful shutdown

- `GET /healthz` — **liveness**: 200 as long as the process is up.
- `GET /readyz` — **readiness**: 200 only if the model is loaded **and**
  PostgreSQL is reachable through the pool; otherwise **503**, so an
  orchestrator stops routing to a replica that can't serve.
- On shutdown the lifespan **drains and closes the pool**, so rolling
  restarts don't leak PG connections.

This is the standard Kubernetes liveness-vs-readiness convention and is a
prerequisite for the orchestration work in 4E.

---

## 6. CI — `.github/workflows/ci.yml`

On every push/PR:
- **Lint (hard gate):** `ruff` for real bugs — syntax errors + undefined
  names (`E9,F63,F7,F82`). Full style report runs but is advisory.
- **Fast unit tests:** the resilience suite
  (`test_circuit_breaker`, `test_timeout_budget`, `test_db_pool`).

These need **no database and no heavy ML stack**, so CI is fast and
deterministic. Data-dependent tests (e.g. `test_no_target_leakage.py`,
which reads `features.csv`) are intentionally excluded from CI.

---

## Configuration (env vars)

| Variable | Default | Meaning |
|---|---|---|
| `ML_POOL_MIN` | 2 | connections pre-opened at boot |
| `ML_POOL_MAX` | 10 | hard cap on live connections |
| `ML_POOL_ACQUIRE_TIMEOUT_S` | 2.0 | max wait for a pooled connection |
| `ML_REQUEST_BUDGET_MS` | 8000 | end-to-end per-request deadline |
| `ML_CB_FAILS` | 5 | failures to open the breaker |
| `ML_CB_WINDOW_S` | 30 | rolling failure window |
| `ML_CB_RESET_S` | 60 | cool-down before a HALF_OPEN trial |
| `ML_SERVICE_SEARCH_PATH` | `tpch, public` | search_path restored after `RESET ALL` |
| `ML_LOG_LEVEL` | INFO | log level |

> **Note (carried over from Phase 3E.1):** `EXPLAIN` generation runs
> `RESET ALL`, which clears `search_path`. Since TPC-H/TPC-DS tables now
> live in dedicated schemas, the serving path restores `search_path` after
> every reset (`ML_SERVICE_SEARCH_PATH`) so unqualified table names in
> served SQL still resolve.

---

## New / changed observability endpoints

- `GET /resilience` → live pool stats + circuit-breaker snapshot + budget.
- `GET /readyz` → now returns 503 when PG is unreachable.
- New counters in `/metrics`: `pool_timeouts_total`,
  `plan_pick_fallback_total`.

---

## Acceptance tests (manual)

1. **100 concurrent `/plan-pick`** → all succeed; `/resilience` shows pool
   `created ≤ maxconn`, some `acquired_total`, ideally `timeouts_total = 0`.
2. **Kill PostgreSQL mid-load** → requests return **503** cleanly (no
   stack-trace 500s); `/readyz` flips to 503. Bring PG back → pool
   self-heals, `/readyz` returns 200.
3. **Break the model** (force ≥5 prediction failures) → breaker opens;
   `/plan-pick` keeps returning **200** with `winner = default`,
   `predicted_ms = null`, `fallback = true`; after the reset window it
   probes and closes again.

Automated unit coverage for the state machine, budget math, and pool
behaviour lives in `tests/test_circuit_breaker.py`,
`tests/test_timeout_budget.py`, `tests/test_db_pool.py` (19 tests).
