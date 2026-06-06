# AutoML-Powered Learned Query Optimizer — Project Roadmap

> Mapping the user's "SYSTEM FLOW" to concrete, buildable phases.

The user's vision in one line:

> **A scalable ML-based system that learns from query execution
> data to predict and optimize database query plans dynamically
> using AutoML techniques.**

with the end-to-end loop:

```
Query → Generate Plans → Feature Extraction → ML Prediction →
        ↑                                          ↓
        └─── Retrain Model ← Store Data ← Collect Actual Time ← Execute
```

This document shows where we are, what each future phase delivers,
and how each phase maps to a piece of the System Flow.

---

## Where we are now (Phase 1 → 3B complete)

| Phase | Output | System Flow piece |
| --- | --- | --- |
| **1. Data foundation** | PG schema + EXPLAIN-ANALYZE collector | "Execute → Collect Actual Time → Store Data" (offline) |
| **2A. TPC-H workload** | 22 queries × 4 variants on PG SF=1 | "Generate Plans" (offline) |
| **2B. Feature extraction** | DFS plan-tree → 50-col CSV | "Feature Extraction" |
| **3A. Baseline ML** | 9 models, GroupKFold CV, q-err, calibrated PG baseline | "ML Prediction" v1 |
| **3B. AutoML cost model** | Param TPC-H + TPC-DS, Optuna, plan-pick accuracy, AutoML winner | "ML Prediction" v2 |

Everything above is **offline batch training**. The next phases
are about **wiring the model into a live system** — that's where
the user's System Flow architecture comes to life.

---

## Phase 3C — Inference service & plan-pick API  *(2-3 days)*

**Goal:** Wrap the AutoML winner in a Python service that any
database client can call to ask "given these N candidate plans,
which one should I run?"

### 3C deliverables

```
services/
  ml_service/
    __init__.py
    server.py            # FastAPI app, single-process
    inference.py         # load joblib + transform + predict
    plan_pick.py         # plan-rank endpoint
    schemas.py           # pydantic request/response models
    health.py            # /healthz, /readyz
  plan_generator/
    pg_variants.py       # produces N plan variants by toggling
                         # enable_hashjoin / enable_mergejoin /
                         # enable_nestloop / from_collapse_limit
    explain_helper.py    # runs EXPLAIN (no ANALYZE) → JSON
docs/
  PHASE3C_INFERENCE.md
```

### 3C endpoints

```
POST /predict
  body: {"plan_json": <PG plan>}
  resp: {"predicted_ms": 142.3, "model": "lightgbm_tuned"}

POST /plan-pick
  body: {"sql": "SELECT ..."}
  resp: {
    "candidates": [
      {"variant": "default",     "plan_json": {...}, "predicted_ms": 88.4},
      {"variant": "no_hashjoin", "plan_json": {...}, "predicted_ms": 142.0},
      {"variant": "no_nestloop", "plan_json": {...}, "predicted_ms": 91.0}
    ],
    "winner": {"variant": "default", "predicted_ms": 88.4},
    "model":  "lightgbm_tuned",
    "elapsed_ms": 6.1
  }
```

### 3C maps to the System Flow

```
Query → Generate Plans → Feature Extraction → ML Prediction → Best Plan Selection
  ↑           ↑                  ↑                  ↑                  ↑
  POST /plan-pick   pg_variants.py   feature_utils.py   inference.py   plan_pick.py
```

### 3C DSA usage (what to call out in interviews)

- **Trees** — plan tree + DFS for feature extraction (already done).
- **Heap / priority queue** — `plan_pick` builds a `min-heap` of
  predicted times to return the top-K candidates in O(log n)
  rather than O(n log n).
- **Hashing** — request-level cache keyed on
  `sha256(canonical_sql)` → cached `winner.plan_json` for repeat
  queries.

---

## Phase 3D — Execution service + feedback log *(1-2 days)*

**Goal:** Actually *run* the chosen plan and capture its real
runtime back into the dataset.

### 3D deliverables

```
services/
  exec_service/
    runner.py            # accepts winner + sql, runs on PG with
                         # the right SET enable_* flags
    capture.py           # EXPLAIN ANALYZE → store as new training
                         # row in data/feedback/{sql_hash}.json
    metrics.py           # prom-style /metrics: pred_vs_actual,
                         # plan_pick_hit_rate, p50/p95 latency
```

### 3D maps to the System Flow

```
Best Plan Selection → Execution → Collect Actual Time → Store Data
                       ↑              ↑                    ↑
                    runner.py      capture.py         data/feedback/
```

### 3D produces the feedback dataset

Every executed query becomes a labelled training row. Over time
this is what makes the system *self-improving* — see Phase 4.

---

## Phase 4 — System hardening: API gateway, caching, queueing  *(3-5 days)*

**Goal:** Turn the two services into a proper distributed system
matching the user's "FAANG-level architecture" diagram.

```
Client → API Gateway → Query Service → ┌→ Plan Generator ─┐
                                       │                  ↓
                                       │              ML Service
                                       │                  │
                                       └─→ Execution Service
                                                          │
                                                          ↓
                                                   Logging (Kafka)
                                                          │
                                                          ↓
                                              AutoML Training (Phase 5)
```

### 4 deliverables

| Component | Tech | Why |
| --- | --- | --- |
| **API Gateway** | FastAPI + uvicorn | Single SQL ingress, auth, rate-limit |
| **Query Service** | same FastAPI app, separate router | SQL parsing, sanitisation, request-id propagation |
| **Plan cache (Redis)** | redis-py | `sha256(sql) → winner.plan_json`. TTL 5 min. |
| **Prediction cache (Redis)** | redis-py | `sha256(plan_json) → predicted_ms`. TTL 1 hour. |
| **Async log queue (Kafka)** | kafka-python or redpanda | Every executed query streams to a Kafka topic for the trainer |
| **Fault tolerance** | health-checks + circuit breaker | If ML service unhealthy, fall back to PG's optimizer. |
| **Observability** | OpenTelemetry → Jaeger + Prometheus | Trace each query across all services. |

### 4 docker-compose preview

```yaml
services:
  postgres:        image: postgres:16
  redis:           image: redis:7-alpine
  kafka:           image: redpandadata/redpanda:latest
  ml-service:      build: services/ml_service
  exec-service:    build: services/exec_service
  query-service:   build: services/query_service
  api-gateway:     build: services/api_gateway
  prometheus:      image: prom/prometheus:latest
  jaeger:          image: jaegertracing/all-in-one
```

---

## Phase 5 — AutoML retraining loop *(2-3 days)*

**Goal:** Close the user's feedback loop — the system gets better
with every query it serves.

### 5 deliverables

```
services/
  automl_service/
    consumer.py          # Kafka consumer → appends to features.csv
    trainer.py           # nightly job: run phase3b/train_models
    promotion.py         # if new model beats current on holdout,
                         # atomically swap automl_best.joblib
    drift.py             # detect distribution shift → trigger
                         # an out-of-cycle retrain
```

### 5 retraining policy

```
trigger_retrain when:
    new_rows_since_last_retrain > 500
    OR
    rolling_q_error_p95 > 1.5 * baseline_q_error_p95   # drift
    OR
    cron("0 2 * * *")                                   # nightly
```

### 5 promotion gate

```
candidate must beat current on:
    plan_pick_accuracy >= current.plan_pick_accuracy
    AND
    q_error_median <= 1.05 * current.q_error_median
    AND
    inference_p99_ms <= 50    # SLA
```

If all three hold → atomic swap. Otherwise log + alert.

### 5 maps to the System Flow

```
... Store Data → Retrain Model → Improve System
       ↑              ↑                ↑
   Kafka topic    trainer.py     promotion.py
```

This is where the *AutoML* in the project name finally pays off:
the trainer re-runs the entire Phase 3B pipeline (Optuna + AutoML
selector) on the growing dataset, with no human in the loop.

---

## Phase 6 — Research & evaluation polish *(2-3 days, optional)*

**Goal:** Build the experimental tables and figures you'd actually
put in a paper / portfolio.

### 6 deliverables

| Artifact | Purpose |
| --- | --- |
| `experiments/regret_curves.py` | Plot cumulative regret over a streaming workload |
| `experiments/online_vs_offline.py` | Show how the model gets better as more data lands |
| `experiments/cross_workload.py` | Train on TPC-H, test on TPC-DS, and vice versa — proves generalisation |
| `experiments/cardinality_error.py` | Phase 3C deep-dive: add fold-aware historical card-error feature, measure lift |
| `experiments/lambdarank_ablation.py` | Optional: try LambdaRank, compare with pointwise + plan-pick |
| `paper/` | LaTeX skeleton you can fill in: problem, related work, methodology, results, conclusion |

---

## Phase 7 — DSA showcase polish *(1-2 days, optional but high-impact)*

The user's spec calls out specific DSA components. Phase 3B
already uses Trees and Hashing; Phase 3C adds Heap. To round out
the DSA story:

| DSA piece | Where it lives | Demo notebook |
| --- | --- | --- |
| **Trees** | `feature_engineering/feature_utils.py` (DFS) | `notebooks/dsa_01_plan_trees.ipynb` |
| **Heap** | `services/ml_service/plan_pick.py` (top-K) | `notebooks/dsa_02_topk_heap.ipynb` |
| **Graphs** | new `analysis/join_graph.py` — build the join graph for each query, show stats (cycles, max degree, diameter) | `notebooks/dsa_03_join_graphs.ipynb` |
| **Hashing** | `services/ml_service/inference.py` (request cache) | `notebooks/dsa_04_caches.ipynb` |
| **Dynamic programming (theory)** | `docs/THEORY_DP_VS_ML.md` — explanation of System R / Selinger DP and why ML augments rather than replaces it | n/a |

---

## Suggested order of execution

```
Phase 3B  ─── DONE
Phase 3C  ──── ML inference service (1 weekend)
Phase 3D  ──── Execution service + feedback (next weekend)
Phase 4   ──── Distributed plumbing (longest, ~5 days spread out)
Phase 5   ──── Retraining loop (~3 days)
Phase 6/7 ──── Polish: paper / DSA notebooks (parallel with everything else)
```

If you want to **demo something impressive in a week**, do
Phase 3C + 3D — that gives you a working `curl localhost:8000/plan-pick`
demo where the ML model picks faster plans than PG defaults and
you can see real wallclock savings.

If you want to **publish a paper**, prioritise Phase 6 cross-workload
generalisation experiments after 3C/3D.

If you want to **show FAANG system design**, prioritise Phase 4
docker-compose + Kafka + Redis + tracing.

---

## What changes in the existing code as we go

The good news: **nothing in Phase 1-3B has to be rewritten**. All
future phases sit on top of:

- `data/processed/features.csv`        ← grows over time, same schema
- `models/phase3b/.../automl_best.joblib` ← swap in newer versions
- `feature_engineering/`               ← reused as inference-time transform
- `phase3a/evaluation.py`              ← reused for online metrics
- `phase3b/plan_pick.py`               ← reused for both eval and online

Phase 5 might add new feature columns (e.g. historical cardinality
error). The pipeline already tolerates this because
`feature_selection.py` discovers columns from the CSV header
rather than hard-coding them.

---

## TL;DR for the user

> Yes — **everything you described is exactly what we're building.**
> Phase 1-3B = the offline ML brain (training + AutoML selector).
> Phase 3C+ = the online services that make it a system.
> Phase 5 = the AutoML feedback loop that makes it self-improving.
> Phase 7 = the DSA notebooks that make it portfolio-worthy.
