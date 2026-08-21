# Phase 5 — AutoML Retraining Loop (Roadmap)

**Goal:** close the system-flow feedback loop so the service **gets measurably
better from its own execution traces** — without a human running scripts by
hand.

```
Query → Plans → Predict → Pick → Execute → Feedback
                                              │
                                              ▼
                         ┌── validate ── train ── evaluate ── promote ──┐
                         │         (gated, automated)                 │
                         └──────────────────► Model Registry ─────────► ML Service
```

Phase 3 built an honest offline brain. Phase 4 made it operable (pool,
breaker, Kafka, Redis, registry, compose). **Phase 5 is the missing step 9:**
*Retrain Model → Improve System*.

Each subphase is **standalone shippable** — after every one the system still
serves traffic. Every automated action passes the **same scientific gates**
from Phase 3E–3G (validation before train, OOF before promote).

> Index: [PHASE5_OVERVIEW.md](PHASE5_OVERVIEW.md) · Prerequisites:
> [PHASE4_OVERVIEW.md](PHASE4_OVERVIEW.md) · Evaluation methodology:
> [PHASE3G_EVALUATION.md](PHASE3G_EVALUATION.md) · Dataset gates:
> [PHASE3E4_DATASET_VALIDATION.md](PHASE3E4_DATASET_VALIDATION.md)

---

## What Phase 5 is (and is not)

| In scope | Out of scope (later) |
|---|---|
| Automate feedback → `features.csv` merge | LambdaRank / new objectives (Phase 3I experiment) |
| Scheduled + triggered retraining | Full MLOps platform (MLflow server, feature store) |
| Registry-based promote / rollback | Multi-region model serving |
| Drift signals from `/metrics` + feedback volume | Real-time weight updates (online learning) |
| Hot-swap or rolling reload of promoted model | Cross-workload paper experiments (Phase 6) |

**Why not online learning?** Query latency labels are noisy, retrains are
expensive, and you need reproducible artifacts for research. Batch retrain +
registry promotion gives auditability; online SGD does not.

**Why not “retrain every night unconditionally”?** Phase 3H showed ±8 pp CI
on plan-pick with the current dataset size. Promotion without an OOF gate
would ship regressions. **Train often; promote rarely** — only when the
candidate passes a pre-registered gate.

---

## Prerequisites (already built in Phase 4)

| Seam | Location | Phase 5 uses it for |
|---|---|---|
| Feedback records on disk | `data/feedback/fb_*.json` | training rows |
| Kafka → disk consumer | `services/feedback_bus/consumer.py` | durable ingest (optional) |
| Feedback → features bridge | `scripts/feedback_to_features.py` | merge into `features.csv` |
| Training pipeline | `phase3b/train_models.py` | candidate artifact |
| Honest evaluation | `scripts/evaluate_baseline.py` | promotion gate |
| Dataset validation | `scripts/validate_dataset.py` | hard pre-train gate |
| Model registry | `services/ml_service/model_registry.py` | version, promote, rollback |
| Serving loads registry | `Predictor` → `resolve_artifact()` | hot-swap target |
| Online metadata in feedback | `online.model_version`, `predicted_ms` | attribution + drift |
| Metrics | `/metrics`, Grafana dashboards | drift triggers, post-promote watch |

---

## Design rules (carry forward from Phase 4)

1. **Fail-open serving.** Retrain/promote failures must never take down
   `/plan-pick`. Worst case: log + alert; keep serving the current promoted
   model.
2. **Gated pipeline.** No training on unvalidated data; no promotion without
   OOF beating the incumbent (with CI rules from Phase 3H).
3. **Immutable artifacts.** Retrain writes a **new** registry version; it
   never overwrites the incumbent until `promote()` succeeds.
4. **Idempotent merge.** Running the feedback merge twice must not duplicate
   rows (`sql_hash + variant + collected_at` key — already in
   `feedback_to_features.py`).
5. **Infra-free tests.** Unit tests mock disk/Kafka; CI does not need PG or
   a GPU cluster.

---

## Subphase map

| Sub | Theme | Shippable end state | Est. effort |
|---|---|---|---|
| **5A** | Feedback ingest + dataset merge | Automated, idempotent merge + validation gate | 1–2 days |
| **5B** | Retrain job | One command produces a **candidate** in the registry | 2–3 days |
| **5C** | Promotion gate + model swap | Candidate promoted only if OOF gate passes; service reloads | 2–3 days |
| **5D** | Drift + triggers | Retrain runs on schedule/threshold/drift — not by hand | 1–2 days |
| **5E** | Orchestration | `automl_service` container + compose profile + runbook | 1–2 days |
| **5F** | Online monitoring + rollback | Post-promote watch; auto-rollback on SLO breach | 1–2 days |

**Suggested order:** 5A → 5B → 5C → 5D → 5E → 5F (same as Phase 4: each
step ends with a smoke test).

---

## 5A — Feedback ingest & dataset merge

### Problem
Feedback lands in `data/feedback/` (file publisher or Kafka consumer), but
`features.csv` only grows when someone manually runs
`feedback_to_features.py --apply`. Phase 5 starts here.

### Deliverables
```
services/automl_service/
  merge.py              # library wrapper around feedback_to_features logic
  watermark.py          # persist last_merged feedback file / timestamp
scripts/
  merge_feedback.py     # CLI entry (also called by scheduler)
```

### Pipeline
```
data/feedback/fb_*.json
        │
        ▼
  extract_features (same path as offline)
        │
        ▼
  dedupe vs features.csv  (KEY_COLS: sql_hash, variant, collected_at)
        │
        ▼
  validate_dataset.py --features   ◄── HARD GATE (exit 1 → abort retrain)
        │
        ▼
  features.csv updated + merge report written
```

### Gate (5A acceptance)
```bash
python scripts/merge_feedback.py --apply
python scripts/validate_dataset.py --features
# Second run must report 0 new rows (idempotent)
python scripts/merge_feedback.py --apply
```

### Why this design
- **Reuse** `extract_features_from_record` — no second feature math (Phase 3D
  lesson: inference/training parity).
- **Validate before train** — repeats Phase 3E.4; prevents the `customer`
  table class of silent corruption from reaching a retrain.
- **Watermark file** (`data/processed/merge_watermark.json`) — cheap
  incremental merge; scheduler skips work when no new feedback.

### Why not alternatives
- **Append raw JSON to a data lake only** — still need feature extraction;
  duplicating logic creates drift.
- **Stream directly into CSV from Kafka** — couples consumer to schema;
  disk JSON + batch merge matches existing collectors.

---

## 5B — Retrain job (candidate artifact)

### Problem
`phase3b/train_models.py` is a full AutoML sweep (hours with Optuna). Phase 5
needs a **retrain profile**: fast path for frequent runs, full path for
nightly.

### Deliverables
```
services/automl_service/
  trainer.py            # orchestrates train + eval + register (no promote)
  config.py             # retrain profiles: fast | full
scripts/
  retrain.py            # CLI: merge → validate → train → register → report
reports/phase5/
  retrain_{run_id}.json # metrics, git hash, row counts, candidate version
```

### Retrain profiles

| Profile | When | `train_models.py` | Optuna | Expected runtime |
|---|---|---|---|---|
| **fast** | threshold trigger, >N new rows | `--skip-tuning` or fixed winner family | off | ~5–15 min |
| **full** | nightly cron | default AutoML + Optuna | on | ~45–90 min |

Both profiles must:
1. Train both regimes (`plan_time`, `post_mortem`) or the configured subset.
2. **`registry.register(regime, artifact, promote=False)`** — candidate only.
3. Run **`evaluate_baseline.py`** on the post-merge dataset → OOF metrics for
   the candidate (or compare candidate predictions file if we snapshot joblib
   first — see 5C).

### Gate (5B acceptance)
```bash
python scripts/retrain.py --profile fast --dry-run   # prints plan, no writes
python scripts/retrain.py --profile fast
python -m services.ml_service.model_registry list --regime plan_time
# → new version exists, is_current=false
```

### Why not alternatives
- **Train in the ML service process** — blocks inference; violates latency SLO.
- **Separate Jupyter/manual retrain** — not closed-loop; already rejected by
  project goals.

---

## 5C — Promotion gate & model swap

### Problem
A new joblib is not automatically better. Phase 3H requires OOF comparison
with CI against a **frozen incumbent**. Promotion must be atomic and
reversible.

### Deliverables
```
services/automl_service/
  promotion.py          # gate logic + registry.promote + reload signal
services/ml_service/
  admin.py              # POST /admin/reload-models (auth token)
                        # or SIGHUP handler in server lifespan
```

### Promotion policy (pre-registered)

Candidate **C** promotes over incumbent **I** only if **all** hold on the
**same** post-merge dataset with **GroupKFold OOF** (Phase 3G):

| Criterion | Rule | Rationale |
|---|---|---|
| Primary | `plan_pick_acc_OOF(C) ≥ plan_pick_acc_OOF(I)` | deployment metric |
| Significance | CI lower bound of Δplan_pick ≥ **−2 pp** | noise floor from 3H |
| Secondary | `q_error_median_OOF(C) ≤ 1.05 × q_error_median_OOF(I)` | don't wreck calibration |
| Safety | `inference_p99_ms(C) ≤ 50 ms` on warm dummy plan | serving SLO |
| Leakage | `tests/test_no_target_leakage.py` passes | safety net |

If any check fails → **keep I**, log `promotion_rejected` event, alert
(optionally keep C registered for manual review).

### Swap mechanics
```
promotion.py
    │
    ├─ registry.promote(regime, candidate_version)
    │
    ├─ clear predict/planpick caches (Redis + local)
    │
    └─ POST /admin/reload-models  (each replica)
           or rolling restart via compose
```

`Predictor` already resolves via registry; reload = drop in-memory singletons
and warm-up again (Phase 4A warm-up path).

### Gate (5C acceptance)
- Promote a **known-good** candidate in staging → `/info` shows new
  `model_version`; `/plan-pick` still 200.
- Promote a **deliberately bad** candidate (mock metrics) → gate rejects;
  incumbent unchanged.
- Rollback: `registry.promote(regime, previous_version)` + reload → serving
  restored < 60 s.

### Why registry promote (not overwrite `automl_best.joblib`)
- **Rollback** in one command.
- **Audit trail** — which version served which feedback rows (`online.model_version`).
- Matches Phase 4B design.

---

## 5D — Drift detection & retrain triggers

### Problem
Retraining on a fixed cron wastes compute; never retraining wastes data.
Need explicit, observable triggers.

### Deliverables
```
services/automl_service/
  triggers.py           # evaluates whether a retrain should start
  drift.py              # rolling metrics vs baseline
data/processed/
  retrain_state.json    # last_run, rows_at_last_run, incumbent metrics
```

### Trigger policy (default)

| Trigger | Condition | Profile |
|---|---|---|
| **volume** | `new_feedback_rows_since_last_retrain ≥ 500` | fast |
| **schedule** | cron `0 2 * * *` (02:00 UTC daily) | full |
| **drift** | rolling `pred_actual_ratio` p95 > **2.0** for 1 h (from Prometheus) | fast |
| **manual** | `python scripts/retrain.py --force` | configurable |

**Cooldown:** minimum 6 h between fast retrains; full nightly always allowed.

### Signals we already have
- `pred_actual_ratio` histogram (Phase 3D/4D)
- `plan_pick_oracle_hits_total / plan_picks_total` (eval / oracle mode only —
  not production traffic unless you enable shadow oracle sampling)
- Feedback row count via `merge_watermark` / `_index.jsonl`

### Gate (5D acceptance)
- Simulate 500 new feedback fixtures → volume trigger fires once.
- Second call within cooldown → skipped, logged `retrain_skipped_cooldown`.
- Drift trigger unit-tested with injected metric snapshots (no Prometheus in CI).

### Why not alternatives
- **Retrain on every feedback row** — label noise + cost; Phase 3E median
  labels exist precisely to reduce this.
- **Complex ML drift (Evidently, NannyML)** — valuable later; start with
  pred/actual ratio + row count (interpretable, already instrumented).

---

## 5E — Orchestration & one-command retrain

### Problem
5A–5D are scripts; production needs a long-running **worker** or cron sidecar
that runs the pipeline unattended.

### Deliverables
```
services/automl_service/
  __init__.py
  worker.py             # loop: check triggers → merge → retrain → promote
  Dockerfile            # slim; no need for full ML stack if train is subprocess
docker-compose.yml      # profile: retrain
  automl-worker         # depends on postgres (optional), reads feedback volume
docs/
  PHASE5E_ORCHESTRATION.md   # (written at implementation time)
```

### Compose profile `retrain`
```yaml
# conceptual
automl-worker:
  profiles: [retrain, all]
  build: services/automl_service
  volumes:
    - feedbackdata:/app/data/feedback
    - processed:/app/data/processed
    - models:/app/models
  environment:
    RETRAIN_PROFILE_SCHEDULE: full
    RETRAIN_MIN_NEW_ROWS: "500"
    PROMOTION_ENABLED: "true"
    ML_SERVICE_RELOAD_URL: http://ml-service-1:8000/admin/reload-models
```

### Worker loop (single-process, file lock)
```
while True:
    if triggers.should_retrain():
        with filelock("/tmp/retrain.lock"):
            merge_feedback()
            if validate():          # hard gate
                candidate = retrain()
                if promote_gate(candidate):
                    swap_models()
    sleep(POLL_INTERVAL)
```

**Why file lock, not Kafka for jobs?** One retrain at a time is sufficient;
avoid distributed scheduler complexity until needed.

### Gate (5E acceptance)
- `docker compose --profile retrain up automl-worker` → worker logs idle.
- Drop 10 feedback JSONs + lower threshold in dev → full pipeline runs once,
  report written under `reports/phase5/`.

---

## 5F — Online monitoring & rollback

### Problem
OOF gate reduces but does not eliminate **online** regression (distribution
shift, workload mix). Need post-promote watch and automatic rollback.

### Deliverables
```
services/automl_service/
  watchdog.py           # post-promote window: watch Prometheus counters
prometheus/alerts.yml   # optional: alert on fallback rate, pred/actual p95
Grafana dashboard       # "Model promotions" panel (version, promote time)
```

### Watch window (default: 24 h after promote)

| Signal | Rollback if |
|---|---|
| `plan_pick_fallback_total` rate | > 5× baseline for 15 min |
| `pred_actual_ratio` p95 | > 2.5× pre-promote baseline |
| Error rate 5xx | > 1% on `/plan-pick` |

Rollback action:
```
registry.promote(regime, previous_version)
reload_models()
emit promotion_rollback event
```

### Gate (5F acceptance)
- Integration test with mocked metrics → rollback path calls `promote(previous)`.
- Grafana dashboard shows current version + promotion history (from registry
  snapshot or structured logs).

### Optional (5F+) — shadow evaluation
Run candidate predictions **without** serving them (`selected_by=shadow` in
feedback metadata). Doubles inference cost — **off by default**; document as
advanced mode for research.

---

## End-to-end pipeline (all subphases complete)

```
                    ┌─────────────────────────────────────────┐
  feedback JSON ───►│ 5A merge + validate_dataset (GATE)      │
                    └──────────────────┬──────────────────────┘
                                       ▼
                    ┌─────────────────────────────────────────┐
                    │ 5B retrain → registry.register        │
                    │         (candidate, promote=false)      │
                    └──────────────────┬──────────────────────┘
                                       ▼
                    ┌─────────────────────────────────────────┐
                    │ 5C evaluate_baseline OOF vs incumbent   │
                    │     pass? → promote + reload + clear cache│
                    └──────────────────┬──────────────────────┘
                                       ▼
                    ┌─────────────────────────────────────────┐
                    │ 5F watchdog (24h) → rollback if needed  │
                    └─────────────────────────────────────────┘

  5D triggers when to enter the pipeline · 5E worker runs it unattended
```

---

## Configuration (planned env vars)

| Variable | Default | Purpose |
|---|---|---|
| `RETRAIN_ENABLED` | `true` | master switch |
| `RETRAIN_PROFILE_SCHEDULE` | `full` | cron job profile |
| `RETRAIN_PROFILE_THRESHOLD` | `fast` | volume/drift profile |
| `RETRAIN_MIN_NEW_ROWS` | `500` | volume trigger |
| `RETRAIN_COOLDOWN_HOURS` | `6` | min gap between fast runs |
| `RETRAIN_CRON` | `0 2 * * *` | nightly schedule |
| `PROMOTION_ENABLED` | `true` | auto-promote if gate passes |
| `PROMOTION_MIN_PLAN_PICK_DELTA` | `-0.02` | CI lower bound (pp) |
| `DRIFT_PRED_ACTUAL_P95` | `2.0` | drift threshold |
| `WATCHDOG_WINDOW_HOURS` | `24` | post-promote monitor |
| `ML_SERVICE_RELOAD_URL` | — | reload endpoint per replica |
| `ADMIN_TOKEN` | — | auth for `/admin/*` |

Add to `.env.example` during 5E.

---

## Testing strategy

| Layer | What | Infra |
|---|---|---|
| Unit | merge idempotency, trigger cooldown, promotion gate math | none |
| Unit | registry register→promote→rollback sequence | tmp dir |
| Integration | retrain `--dry-run` produces report | local joblib fixtures |
| CI | merge + gate + promotion with mocked train | GitHub Actions |
| Manual | compose `retrain` profile end-to-end | docker compose |

**Exclude from CI:** full Optuna retrain (too slow); run weekly or on-demand.

---

## Acceptance checklist (Phase 5 complete)

- [ ] Feedback merge runs unattended and is idempotent.
- [ ] `validate_dataset.py --features` blocks retrain on bad data.
- [ ] Retrain produces a **candidate** registry version without promoting.
- [ ] Promotion requires OOF gate; bad candidates never go live.
- [ ] Successful promote updates serving `model_version` after reload.
- [ ] Rollback restores previous version in < 60 s.
- [ ] Triggers: volume, cron, drift (at least two tested).
- [ ] `reports/phase5/` audit trail for every run.
- [ ] Docs: PHASE5A–5F detail pages (written as each subphase lands).
- [ ] System-flow diagram: step 9 "Retrain Model" marked **DONE**.

---

## Relationship to deferred work

| Item | Where it goes |
|---|---|
| LambdaRank (Phase 3I) | Controlled experiment **after** 5C — same promotion gate, different trainer branch |
| Quantile regression | Phase 6 / research |
| More data (JOB, TPC-DS scale) | Improves gate power; parallel to Phase 5 |
| Feature store (Feast, etc.) | Phase 6+ if feedback volume explodes |

---

## Commands cheat sheet (target state after Phase 5)

```bash
# Manual one-shot (what the worker runs internally)
python scripts/merge_feedback.py --apply
python scripts/validate_dataset.py --features
python scripts/retrain.py --profile full
python scripts/promote_model.py --regime plan_time --evaluate  # 5C gate + swap

# Registry inspection
python -m services.ml_service.model_registry list --regime plan_time
python -m services.ml_service.model_registry snapshot

# Unattended
docker compose --profile retrain up -d automl-worker

# Force run (bypass cooldown, not promotion gate)
python scripts/retrain.py --force --profile fast --no-promote
```

---

## Execution plan (when you say "start Phase 5")

```
Week 1   5A merge automation + validation gate + tests
         5B retrain.py wrapper + registry register (no promote)

Week 2   5C promotion gate + /admin/reload-models + cache clear
         5D triggers + retrain_state.json

Week 3   5E automl-worker + compose profile + .env.example
         5F watchdog + Grafana promotion panel + rollback test
```

Start with **5A** — it delivers immediate value (feedback actually enters
`features.csv`) even before unattended retrain exists.

---

## TL;DR

Phase 5 turns the feedback log from a **manual research asset** into a
**gated, automated improvement loop**: merge → validate → train → evaluate →
promote → watch → rollback. It reuses every honest gate from Phase 3 and
every seam from Phase 4. **Train often; promote only when OOF proves it.**

Next step when you're ready: **"start phase 5a"**.
