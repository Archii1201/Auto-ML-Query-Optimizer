# Phase 3C — ML Inference Service & Plan-Pick API

> Goal: turn the offline AutoML winner into a live, callable system.
> Send a SQL query, get back the predicted-best execution plan.
> This is where Phases 1-3B's offline pipeline becomes a real
> service that any database client could use.

---

## 1. What changed

Phase 3B left us with a serialised model under
`models/phase3b/plan_time/automl_best.joblib`. Phase 3C wraps
that artifact in:

```
            ┌─────────────────────────────────────────────────┐
   SQL ───► │  POST /plan-pick                                │
            │     │                                           │
            │     ├─► PG: EXPLAIN (FORMAT JSON) under         │
            │     │      4 variants of enable_*=on/off        │
            │     │                                           │
            │     ├─► Predictor.predict_one() per plan        │
            │     │                                           │
            │     └─► heapq.nsmallest(top_k) by predicted_ms  │
            │                                                 │
            │  → returns winner + ranked candidates           │
            └─────────────────────────────────────────────────┘
```

The API is **stateless**, **thread-safe**, and **cached at two
layers** (per-SQL and per-plan).

---

## 2. Directory additions

```
services/
  __init__.py
  ml_service/
    __init__.py
    server.py            # FastAPI app, lifespan model loader
    inference.py         # Predictor: load joblib, transform, predict
    plan_pick.py         # PlanPicker: orchestrates SQL → variants → predictions
    schemas.py           # pydantic v2 request/response models
    cache.py             # HashedLRUCache + sha256 helpers
  plan_generator/
    __init__.py
    pg_variants.py       # the 4 optimizer-knob variants
    explain.py           # EXPLAIN (FORMAT JSON) + execute helpers
scripts/
  demo_phase3c.py        # full SQL → predict → execute → regret demo
  smoke_test_phase3c.py  # HTTP smoke test against a running server
docs/
  PHASE3C_INFERENCE.md   # ← this file
```

No existing module was modified — Phase 3C is purely additive.

---

## 3. End-to-end usage

### 3.1 Run the demo (offline, no server needed)

```bash
python scripts/demo_phase3c.py
```

For every sample query the demo:

1. Generates 4 plan variants via PG knob toggles
2. Asks the AutoML winner to predict each variant's runtime
3. Picks the variant with the lowest predicted time
4. **Actually runs all 4 variants** to get ground truth
5. Reports `picked_ms` vs `oracle_ms` (regret) and vs `default_ms` (speedup)

Sample output (one query):

```
[query] tpch_q01_pricing
  ML inference: 2133.6 ms total (4 candidates ranked)

  variant           pred ms     est cost
  default            3979.8     182358.2 <- WINNER
  no_hashjoin        3979.8     182358.2
  no_mergejoin       3979.8     182358.2
  no_nestloop        3979.8     182358.2

  variant           actual ms
  default              1284.1  <- ORACLE / PICKED / PG-default
  no_hashjoin          1597.2
  no_mergejoin         1672.7
  no_nestloop          1722.1

  result: picked=default  picked_ms=1284.1  oracle=default  oracle_ms=1284.1
  vs default:  delta=+0.0 ms  (1.00x speedup)
  vs oracle:   delta=+0.0 ms
```

### 3.2 Run as an HTTP service

```bash
# Terminal 1
python -m services.ml_service.server
# → uvicorn on http://127.0.0.1:8000

# Terminal 2
python scripts/smoke_test_phase3c.py
# Hits /healthz, /readyz, /info, /plan-pick (twice → cache hit)
```

### 3.3 Talk to it from any HTTP client

```bash
curl -X POST http://127.0.0.1:8000/plan-pick \
     -H 'content-type: application/json' \
     -d '{"sql": "SELECT COUNT(*) FROM lineitem", "top_k": 4, "regime": "plan_time"}'
```

---

## 4. API contract

### 4.1 `GET /healthz` — liveness

```json
{"status": "ok"}
```

### 4.2 `GET /readyz` — readiness

```json
{
  "status": "ok",
  "regimes": [
    {"regime": "plan_time",   "model": "lightgbm_tuned",         "feature_count": 44},
    {"regime": "post_mortem", "model": "gradient_boosting_tuned","feature_count": 54}
  ]
}
```

### 4.3 `GET /info?regime=plan_time` — diagnostics

```json
{
  "status":        "ok",
  "model_loaded":  true,
  "regime":        "plan_time",
  "model_name":    "lightgbm_tuned",
  "feature_count": 44,
  "cache_stats":   {"hits": 12, "misses": 8, "sets": 8, "size": 8, "capacity": 4096}
}
```

### 4.4 `POST /predict` — one plan in, one prediction out

```json
// request
{
  "plan_json": [{"Plan": {...}, "Planning Time": 0.5}],
  "regime":    "plan_time"
}

// response
{
  "predicted_ms": 142.3,
  "regime":       "plan_time",
  "model_name":   "lightgbm_tuned",
  "cache_hit":    false,
  "elapsed_ms":   3.7
}
```

### 4.5 `POST /plan-pick` — SQL → ranked variants

```json
// request
{
  "sql":          "SELECT COUNT(*) FROM lineitem",
  "top_k":        4,
  "regime":       "plan_time",
  "include_plan": false
}

// response
{
  "sql_hash":   "c9038451f1e3dd15...",
  "winner":     {"variant": "default", "predicted_ms": 88.4, "estimated_cost": 12345.6, ...},
  "candidates": [...],
  "model_name": "lightgbm_tuned",
  "regime":     "plan_time",
  "cache_hit":  false,
  "elapsed_ms": 2183.2
}
```

The cache hit on the second call serves in **~0.08 ms** — a 27000× speedup over the 2.1 s cold call.

---

## 5. Module deep dive

### 5.1 `inference.py` — the prediction core

`Predictor.predict_one(plan_json)` is the function the rest of the
service is built around. Internally:

1. **Hash** the plan JSON → check cache.
2. **Wrap** the plan in a record envelope identical to what the
   offline collectors produce.
3. **Call** the *same* `extract_features_from_record()` used in
   training. This guarantees feature parity — bug-for-bug.
4. **Align** the resulting feature dict to `artifact["feature_names"]`:
   - drop ID/leaky/target columns (defensive — extractor returns them)
   - one-hot-encode `root_node_type` against the columns the model
     already knows (so unseen node types produce all-zero columns)
   - reindex to the exact training column order, fill NaN with 0
5. **Predict**. If `log_target=True`, run `np.expm1` to invert.
6. **Clip** to `[0.1 ms, 10 000 000 ms]` so a single garbage
   prediction can't crash downstream consumers.
7. **Cache** under the plan hash and return.

### 5.2 `plan_pick.py` — DSA: heap-based top-K

```python
ranked = heapq.nsmallest(
    len(scored), scored,
    key=lambda c: (c.predicted_ms, c.estimated_cost),
)
```

- **Why nsmallest, not sorted()?** With our current 4 variants both
  are O(N log N), but the contract is "we'll add more variants over
  time". `heapq.nsmallest(k, ...)` runs in O(N log k), so we're set.
- **Why the cost tiebreaker?** When PG ignores a knob (e.g. it
  wasn't going to use a hash-join anyway), multiple variants
  produce *the same plan JSON* → identical features → identical
  predicted_ms. Without a tiebreaker the iteration order of
  `VARIANTS` leaked through and biased winners toward `default`.
  Lower estimated_cost is a sensible secondary signal.

### 5.3 `cache.py` — DSA: hashing

Two LRU caches are used:

| Cache | Key | Value | TTL | Invariant |
| --- | --- | --- | --- | --- |
| `Predictor.cache` | `sha256(plan_json)` | predicted_ms | none | same plan always has same prediction (model is read-only) |
| `PlanPicker.cache` | `sha256(canonical_sql)` | (winner, ranked) | none | same SQL → same plan generation result |

`canonical_sql` collapses whitespace + lowercases + strips trailing
semicolons. **Not SQL-aware**: aliases, column reorders, etc. all
miss. We'd rather have a few cache misses than ever return a wrong
plan.

`cachetools.LRUCache` provides O(1) get/set with bounded size.
A real deployment would back this with Redis using the same get/
set interface (Phase 4).

### 5.4 `plan_generator/explain.py`

Two functions:

```python
generate_variants(conn, sql)            # EXPLAIN (FORMAT JSON), no execute
execute_with_variant(conn, sql, knobs)  # EXPLAIN (ANALYZE, FORMAT JSON)
```

The first runs in **~5 ms per variant** because we never execute
the SQL — we only ask the planner to produce a plan. The second
is what the demo uses to compute oracle / regret numbers.

Failures are silently skipped (some variants are infeasible —
e.g. a query with cross joins can't run with `nestloop=off`).
A query that fails on every variant raises `RuntimeError`.

### 5.5 `server.py`

Standard FastAPI:

- **lifespan**: eagerly load both regimes' winners on boot. If a
  joblib is missing, abort startup loudly.
- **dependency**: each `/plan-pick` request opens a fresh
  `psycopg2.connect()` (PG connections are not thread-safe).
  Phase 4 will move this to `psycopg-pool`.
- **error handling**: missing regime → 404. PG unreachable → 503.
  Bad payload → 400. Everything else bubbles to FastAPI's default
  500.

Environment variables:

| Var | Default | What |
| --- | --- | --- |
| `ML_SERVICE_HOST` | `127.0.0.1` | uvicorn bind |
| `ML_SERVICE_PORT` | `8000` | uvicorn port |
| `ML_SERVICE_REGIMES` | `plan_time,post_mortem` | which regimes to load on startup |

---

## 6. Performance characteristics

Measured on the demo's 5 queries on a Windows laptop (no GPU,
SF=1 TPC-H + SF=0.1 TPC-DS in PG):

| Component | Cold | Warm (cache hit) |
| --- | --- | --- |
| Single `predict_one()` (LightGBM) | ~50–250 ms | <1 ms |
| `generate_variants` (4 EXPLAINs) | ~20–100 ms | n/a |
| Full `/plan-pick` over HTTP | ~150–2200 ms | ~0.1 ms |

The first prediction per process is dominated by LightGBM's
JIT-style first-call warmup. After that the model itself is
~3–10 ms; the rest is the PG round-trips for the 4 variants.

**Cache hit ratio in the demo**: predictor cache hit 12/20 = 60%
(because 4 variants often produce identical plans → identical
plan hashes → cache hits within the same SQL).

---

## 7. DSA showcase summary

| Where | DSA | Why this implementation |
| --- | --- | --- |
| `feature_engineering/feature_utils.dfs_iter` | recursive DFS over plan tree | plan trees are ≤ ~100 nodes — recursion is the most readable; depth tracking is one extra integer |
| `services/ml_service/plan_pick.py` | heap (`heapq.nsmallest`) | O(N log k) top-k; 4 variants today, designed for 32+ tomorrow |
| `services/ml_service/cache.py` | hashing (sha256) + LRU | constant-time lookup, bounded memory; same interface as Redis |
| `services/ml_service/inference.py` | hash-map alignment | feature dict → ordered ndarray via dict lookup, O(F) once |

Phase 7 (optional) will add Graphs (join-graph analysis) and
Dynamic-Programming theory comparison vs. ML, completing the
DSA story.

---

## 8. Limitations & where Phase 3D / 4 step in

1. **Predictions tie when plans tie.** When PG produces the
   *same* plan under two different knob settings, the predicted
   time is identical. The estimated-cost tiebreaker handles this;
   a richer fix would be to add knob-state features so the model
   knows it's running under e.g. `enable_hashjoin=off`.

2. **No connection pooling.** Each `/plan-pick` opens a fresh PG
   connection. Phase 4 wraps this in `psycopg-pool` for ~10×
   throughput.

3. **No fallback path.** If the model fails (corrupt joblib,
   feature mismatch), we 500. Phase 4 adds a circuit-breaker that
   falls back to PG-default plan when ML is unhealthy — the
   "FAULT TOLERANCE" arrow in your System Flow.

4. **No execution loop.** The service today predicts and recommends
   but **never actually runs the chosen plan in production**. That's
   Phase 3D: the Execution Service that runs the picked variant,
   times it, and writes the result back as a new training row.
   That closes the user's "Execute → Collect Time → Store Data" arc.

5. **No streaming feedback to retraining.** The captured runtimes
   accumulate as files. Phase 5 adds a Kafka topic so the AutoML
   trainer consumes them in real time and retrains nightly.

---

## 9. Reproducibility checklist

```bash
pip install -r requirements.txt

# Once-only data setup (already done if you ran 3B)
psql -f db/schema.sql
python scripts/setup_tpch.py
python scripts/setup_tpcds.py --sf 0.1
python scripts/collect_tpch_plans.py
python scripts/collect_tpch_param_plans.py
python scripts/collect_tpcds_plans.py
python -m feature_engineering.extract_features
python -m phase3b.train_models

# Phase 3C
python scripts/demo_phase3c.py                  # offline demo
python -m services.ml_service.server            # HTTP server
python scripts/smoke_test_phase3c.py            # HTTP smoke test
```

---

## 10. What's next

> **Phase 3D — Execution Service & feedback log.** Turn the
> recommended plan into actually-executed work and ship the
> measured runtime back to `data/feedback/` so the dataset grows
> with every served query.

After 3D the loop is **closed end-to-end** and we can move on to
Phase 4 (distributed plumbing) and Phase 5 (online retraining).

See `docs/PROJECT_ROADMAP.md` for the full plan.
