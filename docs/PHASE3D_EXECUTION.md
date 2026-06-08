# Phase 3D — Execution Service & Feedback Log

> Goal: close the offline / online loop. Phase 3C *recommended* a
> plan; Phase 3D actually **runs it**, captures the real wall-time
> + EXPLAIN ANALYZE plan, and writes a new training row to disk.
> The dataset now grows automatically with every served query.

This is the *Execute → Collect Actual Time → Store Data* arc of
the user's SYSTEM FLOW.

---

## 1. What changed

After Phase 3C the service could only predict. Phase 3D adds:

```
            POST /run-and-learn
                  │
            ┌─────┼──────────────────────────────────────────────┐
            ▼     │                                              │
   plan-pick (3C) │                                              │
            │     ▼                                              │
            │   Execute picked variant on PG                     │
            │     │                                              │
            │     ▼                                              │
            │   Capture (wall_ms, EXPLAIN ANALYZE)               │
            │     │                                              │
            │     ▼                                              │
            │   FeedbackWriter → data/feedback/fb_*.json         │
            │     │                                              │
            └─────┴────────────► metrics + return JSON to caller
```

Plus three new pieces of public surface:

| Endpoint | What it does |
| --- | --- |
| `POST /execute`        | Run a *specific* variant, optionally write feedback |
| `POST /run-and-learn`  | The full loop in one call (predict → execute → store) |
| `GET  /metrics`        | JSON or Prometheus-text counter/histogram snapshot |

Plus one critical bridge script:

| Script | What it does |
| --- | --- |
| `scripts/feedback_to_features.py` | Promotes `data/feedback/*.json` rows into `data/processed/features.csv` |

---

## 2. Directory additions

```
services/
  exec_service/
    __init__.py
    metrics.py          # MetricsRegistry + Histogram (no deps)
    capture.py          # FeedbackWriter: atomic JSON + _index.jsonl
    runner.py           # ExecutionRunner: run + capture + persist
    schemas.py          # /execute and /run-and-learn pydantic models
data/
  feedback/             # new dir for online feedback rows
    fb_<ts>_<rid>_<sql_hash>_<variant>.json   # one record per execution
    _index.jsonl                              # one summary per record
scripts/
  feedback_to_features.py
  demo_phase3d.py
docs/
  PHASE3D_EXECUTION.md  # ← this file
```

The Phase 3C `services/ml_service/server.py` is the only existing
file modified — to register the new endpoints. Everything else
is purely additive.

---

## 3. End-to-end usage

### 3.1 Single-call closed loop (recommended)

```bash
# Terminal 1
python -m services.ml_service.server

# Terminal 2 — full SYSTEM FLOW in one HTTP call
curl -X POST http://127.0.0.1:8000/run-and-learn \
     -H 'content-type: application/json' \
     -d '{
       "sql":    "SELECT COUNT(*) FROM lineitem WHERE l_quantity < 24",
       "regime": "plan_time"
     }'
```

Response (abridged):

```json
{
  "sql_hash":       "5d22ee19...",
  "request_id":     "40057de7...",
  "regime":         "plan_time",
  "model_name":     "lightgbm_tuned",
  "candidates":     [{"variant": "default", "predicted_ms": 88.4, ...}, ...],
  "picked_variant": "default",
  "predicted_ms":   88.4,
  "actual_wall_ms": 76.2,
  "timed_out":      false,
  "feedback_path":  "data/feedback/fb_20260607T115753_40057de7_5d22ee19_default.json",
  "elapsed_ms":     112.5
}
```

A new training row now exists at the path returned in `feedback_path`.

### 3.2 Oracle mode — for evaluation only

Pass `"oracle": true` and the service also runs *every other
variant* so we can measure regret vs. the true fastest plan:

```json
{
  "picked_variant":   "default",
  "actual_wall_ms":   257.6,
  "oracle_variant":   "no_nestloop",
  "oracle_wall_ms":   161.4,
  "regret_ms":        96.2,
  "regret_ratio":     0.596,
  "plan_pick_hit":    false,
  "truths": [
    {"variant": "default",      "wall_time_ms": 257.6, "timed_out": false},
    {"variant": "no_hashjoin",  "wall_time_ms": 302.5, "timed_out": false},
    ...
  ]
}
```

Disable in production — it doubles the wallclock cost.

### 3.3 Demo CLI

```bash
python scripts/demo_phase3d.py            # 5 sample queries, no oracle
python scripts/demo_phase3d.py --oracle   # also computes regret
```

### 3.4 Promote feedback into the training set

```bash
python scripts/feedback_to_features.py            # dry run — shows delta
python scripts/feedback_to_features.py --apply    # actually rewrite features.csv
```

---

## 4. Module deep dive

### 4.1 `capture.py` — schema-compatible feedback log

Every row is **byte-for-byte compatible** with what
`scripts/collect_data.py` and `scripts/collect_tpch_plans.py`
write — same top-level shape:

```json
{
  "query_id":     "online_<sql_hash>",
  "variant":      "default",
  "tag":          "online",
  "sql":          "...",
  "sql_hash":     "...",
  "collected_at": "2026-06-07T11:54:55.498277+00:00",
  "wall_time_ms": 2554.893,
  "summary":      {...},
  "plan":         [...],

  "online": {                    ← Phase 3D online-only metadata
    "request_id":  "...",
    "predicted_ms": 3979.79,
    "model_name":  "lightgbm_tuned",
    "regime":      "plan_time",
    "selected_by": "ml",         ← ml | oracle | user | default
    "knobs":       []
  }
}
```

Why does that matter? Because `feature_engineering/extract_features.py`
ignores the `online` block and processes every other field exactly
as if it were an offline collector output — so we can run feature
extraction over `data/raw + data/tpch/plans + data/tpcds/plans +
data/feedback` and get a unified CSV.

The `online` block lets us compute *online* model quality:

```python
df["pred_actual_ratio"] = df.predicted_ms / df.actual_wall_ms
df["q_error_online"]    = (df.predicted_ms / df.actual_wall_ms).clip(lower=1).max(...)
```

**Atomic writes.** We write to `*.json.tmp`, then `os.rename()` it
to the final filename. A power loss mid-write leaves the staging
file untouched and never confuses the feature extractor.

### 4.2 `runner.py` — execution + oracle

`ExecutionRunner.run_single` is the workhorse. It's a thin wrapper
around `plan_generator.explain.execute_with_variant` plus:

- statement-timeout enforcement (default 60 s),
- failure counting (`execution_failures_total`),
- pred-vs-actual ratio recording,
- `FeedbackWriter` invocation.

`ExecutionRunner.run_with_oracle` runs every candidate to compute
ground truth — used by the oracle endpoint and the demo. Each run
becomes its own feedback row tagged `selected_by="oracle"` so the
training set gets richer than what a single chosen variant would
produce.

### 4.3 `metrics.py` — dependency-free Prometheus exposition

| Counter | Incremented when |
| --- | --- |
| `predictions_total` | every `/predict` call |
| `plan_picks_total` | every `/plan-pick` or `/run-and-learn` |
| `plan_pick_oracle_hits_total` | oracle-mode pick equals true fastest |
| `executions_total` | every `/execute` or `/run-and-learn` SQL run |
| `execution_failures_total` | timeout / SQL error during execution |
| `feedback_rows_written_total` | each feedback file persisted |

| Histogram | Observed at |
| --- | --- |
| `inference_latency_ms` | predict + plan-pick endpoints |
| `execution_latency_ms` | every `/execute` |
| `pred_actual_ratio` | each closed-loop call (predicted / actual) |

`GET /metrics` returns a JSON snapshot by default; `GET /metrics?fmt=prom`
returns the Prometheus v0.0.4 text format ready to scrape.

In Phase 4 we'll swap this for `prometheus_client` proper. The
shape stays identical so dashboards built now keep working.

### 4.4 `feedback_to_features.py` — the retraining bridge

Same feature pipeline (no copies, no drift). Joins on
`(sql_hash, variant, collected_at)` and dedupes — running the
script twice on the same feedback dir is a no-op the second time.

Phase 5 will replace the manual invocation with a Kafka consumer
+ scheduler. The promotion logic itself doesn't need to change.

---

## 5. Real demo numbers

Running `python scripts/demo_phase3d.py --oracle` on this dataset:

```
[query] tpch_q01_pricing
  picked     : default  (predicted=3979.8 ms)  actual=2554.9 ms
  oracle     : no_nestloop  (1376.2 ms)
  result     : MISS  regret=1178.7 ms  (+85.6%)

[query] tpch_q06
  picked     : default  (predicted=3010.4 ms)  actual=1748.3 ms
  oracle     : no_mergejoin  (1040.6 ms)
  result     : MISS  regret=707.7 ms  (+68.0%)

[query] tpch_q14_promo
  picked     : default  (predicted=2799.7 ms)  actual=949.3 ms
  oracle     : no_nestloop  (734.1 ms)
  result     : MISS  regret=215.2 ms  (+29.3%)

[query] ds_topk_brand
  picked     : default  (predicted=831.6 ms)   actual=257.6 ms
  oracle     : no_nestloop  (161.4 ms)
  result     : MISS  regret=96.2 ms  (+59.6%)

[query] ds_yearly_trend
  picked     : default  (predicted=696.5 ms)   actual=209.7 ms
  oracle     : no_mergejoin  (195.2 ms)
  result     : MISS  regret=14.5 ms  (+7.4%)

online plan-pick accuracy: 0/5  =  0.0%
online avg regret_ms     : 442.4
[metrics] feedback rows: 20 on disk
```

Honest readout: 0/5 picks correct on this five-query sample. That's
much worse than the offline 51% — the model ties on multiple
variants when PG produces identical plan JSONs, and our
estimated-cost tiebreaker happens to favor `default` here.

That's exactly what the feedback log is for: those 20 new rows now
sit in `data/feedback/`, ready to be promoted via
`feedback_to_features.py --apply` and consumed by the next training
run. **The system corrects itself.**

A second insight from `/metrics`:

```
pred_actual_ratio  median = 3.21   p95 = 5.14
```

The model **over-predicts by ~3×** systematically. With a richer
training set the retraining loop will calibrate this away. This
is the kind of online drift signal Phase 5 will trigger off.

---

## 6. Performance characteristics

Measured on the demo (oracle mode, 5 queries, Windows laptop):

| Phase of `/run-and-learn` | Cold | Warm |
| --- | --- | --- |
| plan-pick (4 variants × predict) | 50–2200 ms | <1 ms (cache) |
| execute picked variant | 200 ms – 3 s | n/a (no cache) |
| feedback write | <2 ms | n/a |
| total HTTP elapsed | 0.9 – 8.9 s | 0.2 – 3 s |

`execute` dominates total time, as expected. Inference is
negligible after warmup.

---

## 7. Pitfalls & gotchas

1. **Feedback grows unboundedly.** Today nothing prunes
   `data/feedback/`. Phase 5 adds a TTL-based cleaner; for now
   `_index.jsonl` makes it cheap to walk. Production deployments
   should mount this on an object store.

2. **Schema overlap with TPC-H/TPC-DS.** Both benchmarks have a
   table called `customer`. Whichever schema was applied last
   wins. The demo deliberately uses queries that work against
   the current state of the DB. Phase 4 will namespace them.

3. **No connection pooling yet.** Each `/run-and-learn` opens a
   fresh PG connection — fine at low QPS, expensive at scale.
   Phase 4 swaps in `psycopg-pool`.

4. **Default-bias from tied predictions.** When PG produces the
   *same* plan JSON for multiple variants, predicted times tie,
   and the heap tiebreaker picks the lowest estimated_cost.
   That's frequently `default`. Two real fixes:
     - add a knob-state feature so the model knows the planner
       was constrained,
     - sample a more diverse training set.
   Both belong in Phase 3C.5 / 5.

5. **Feedback rows include the `online` block.** If you copy them
   into `data/raw/` and re-extract, the extractor will silently
   ignore the `online` block but still produce a row. That's the
   designed contract: feedback is *additive* training data.

6. **Oracle mode doubles cost.** Use only for evaluation, never
   in user-facing serving paths.

---

## 8. What this unlocks (and what's next)

We now have:

- a callable system that **predicts**, **executes**, **measures**,
  and **logs** in one HTTP round trip;
- a feedback log that grows the training set automatically;
- a one-command bridge from feedback → CSV → retraining;
- live process metrics in two formats.

> **Phase 4 — Distributed plumbing.** API gateway, Redis-backed
> caches (drop-in replacement for `HashedLRUCache`), Kafka stream
> for `data/feedback/*.json` records, OpenTelemetry tracing,
> circuit-breaker fallback to PG default. Turns this single-process
> service into the FAANG-style microservice fleet from your spec.
>
> **Phase 5 — Online retraining loop.** Consumes the Kafka topic,
> appends to `features.csv`, runs `phase3b/train_models.py`
> nightly + on drift, and atomically swaps `automl_best.joblib`
> only if a promotion gate passes. The system finally becomes
> *self-improving* — the central claim of the project title.

See `docs/PROJECT_ROADMAP.md` for the multi-phase view.

---

## 9. Reproducibility checklist

```bash
# Already done if you ran 3B + 3C
pip install -r requirements.txt
psql -f db/schema.sql
python scripts/setup_tpch.py
python scripts/setup_tpcds.py --sf 0.1
python scripts/collect_tpch_plans.py
python scripts/collect_tpch_param_plans.py
python scripts/collect_tpcds_plans.py
python -m feature_engineering.extract_features
python -m phase3b.train_models

# Phase 3D
python -m services.ml_service.server                       # terminal 1
python scripts/demo_phase3d.py --oracle                    # terminal 2
python scripts/feedback_to_features.py --apply             # promote rows
python -m phase3b.train_models --skip-tuning               # quick re-train
```
