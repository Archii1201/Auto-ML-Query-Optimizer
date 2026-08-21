# AutoML-Powered Learned Query Optimizer

> Replace PostgreSQL's static, 1990s cost model with a model that **learns
> which execution plan is fastest from real execution history** — and keeps
> improving itself, safely, in production.

Traditional databases pick query plans (hash join vs merge join vs nested
loop, which scan, which order) using fixed cost formulas. On complex joins
those estimates are frequently wrong, so the planner picks a slow plan —
sometimes **10–100× slower** than the best one. This project learns a better
plan selector from measured runtimes, serves it behind a production-grade API,
and closes the loop with an **automated, statistically-gated retraining
pipeline**.

**The learned model picks the truly-fastest plan ~2.5× more often than
PostgreSQL's own optimizer** (plan-pick accuracy **0.205 → ~0.50**), with
median prediction error **~1.25 q-error** and **~7–10× lower plan regret**.

📄 New here for an interview or overview? Read
[`docs/PROJECT_STATEMENT_AND_STAR.md`](docs/PROJECT_STATEMENT_AND_STAR.md)
and [`docs/INTERVIEW_PREP_QA.md`](docs/INTERVIEW_PREP_QA.md).

---

## How it works

```
SQL query
   │
   ▼
Generate candidate plans   ── ask PostgreSQL for the plan under 4 join-knob
                              settings (default / no_hashjoin / no_mergejoin /
                              no_nestloop) via EXPLAIN — no execution yet
   │
   ▼
Feature extraction         ── flatten each EXPLAIN JSON tree into ~50 numeric
                              features (costs, cardinality mis-estimates,
                              tree shape, scan/join counts, knob state)
   │
   ▼
ML prediction              ── a tuned tree-ensemble predicts each plan's
                              runtime (ms)
   │
   ▼
Pick the fastest plan      ── lowest predicted runtime wins
   │
   ▼
Execute on PostgreSQL      ── run it, measure the TRUE runtime
   │
   ▼
Store feedback             ── (features → actual runtime) to disk / Kafka
   │
   ▼
Retrain loop (Phase 5)     ── merge → retrain → statistically gate → promote →
                              hot-swap live → watch & auto-rollback
```

---

## Results at a glance

Out-of-fold (GroupKFold by query), honest numbers from this repo:

| Metric | PostgreSQL cost model | Learned `plan_time` (deployable) | Learned `post_mortem` (upper bound) |
|---|---|---|---|
| **Plan-pick accuracy** | 0.205 | **~0.48–0.51** | **0.962** |
| **Median q-error** | 1.65–1.79 | **~1.24–1.39** | 1.005 |
| **p95 q-error** | 8.3–8.9 | **~3.2–3.7** | ~1.1 |
| **R² (log runtime)** | ~0.05 | **~0.51–0.53** | ~0.87 |
| **Mean regret (ms)** | 4366 | **~412–586** | ~3 |

- **`plan_time`** = deployable model (only pre-execution planner estimates) —
  this is the product.
- **`post_mortem`** = uses actual execution stats; not deployable for
  *selecting* a plan, but its 0.96 proves the feature set is rich enough to
  solve the problem. The remaining `plan_time` gap is the planner's estimate
  error — exactly what a learned optimizer exists to fix.
- **AutoML winners:** `plan_time` → tuned tree ensemble (LightGBM / ExtraTrees
  family; deployed artifact `extra_trees_tuned`); `post_mortem` →
  `gradient_boosting_tuned`.

Full analysis: [`reports/phase3b/REPORT.md`](reports/phase3b/REPORT.md).

---

## Architecture

```
                         ┌───────────────┐
   client ──HTTP──► nginx │ load balancer │──► ml-service × 2 (FastAPI)
                         └───────────────┘         │
                                                   ├─ Plan generator (EXPLAIN variants)
                                                   ├─ Feature extractor  (shared w/ offline)
                                                   ├─ Predictor + model registry (versioned)
                                                   ├─ Circuit breaker → PG-default fallback
                                                   ├─ Bounded PG connection pool
                                                   └─ Redis cache (plan → prediction)
                                                          │
   PostgreSQL ◄── execute plan ──────────────────────────┘
        │ feedback (features → actual ms)
        ▼
   Kafka feedback bus ──► consumer ──► data/feedback/*.json
        │
        ▼
   AutoML worker (Phase 5): merge → retrain → gate → promote → watchdog/rollback
        │
   Observability: Prometheus + Grafana + OpenTelemetry/Tempo
```

Every component is **fail-open**: if the model or a dependency fails, the
service degrades to PostgreSQL's default plan rather than erroring, and a
retraining failure can never touch the live serving path.

---

## Repository layout

```
auto-ml-query-optimizer/
├── services/
│   ├── plan_generator/     # EXPLAIN + join-knob plan variants
│   ├── ml_service/         # FastAPI serving, inference, cache, registry, resilience
│   ├── exec_service/       # execute plans, capture feedback, metrics
│   ├── feedback_bus/       # pluggable feedback publisher/consumer (file | Kafka)
│   └── automl_service/     # Phase 5: merge, trainer, promotion, triggers, worker, watchdog
├── feature_engineering/    # EXPLAIN-JSON → ~50 numeric features (single source of truth)
├── phase3a/ phase3b/       # model training, AutoML selection, plan-pick evaluation
├── scripts/                # data collection, feature build, validation, retrain, promote
├── db/                     # TPC-H / TPC-DS schema, parameterized queries
├── deploy/                 # nginx, prometheus, grafana, tempo, otel-collector configs
├── loadtest/               # Locust load test + chaos script
├── docs/                   # per-phase design docs, roadmap, interview prep
├── models/                 # trained artifacts + versioned registry
├── data/                   # raw plans, processed features.csv, feedback
├── docker-compose.yml      # one-command bring-up (profiles: core/streaming/observability/retrain/all)
└── requirements.txt
```

---

## Quickstart

### Prerequisites
- Python **3.11+**
- PostgreSQL **13+** (local or reachable) — *or* just use Docker (below)
- Docker + Docker Compose (for the full system)

### Option A — Full system with Docker (recommended)

```bash
cp .env.example .env          # then edit ML_ADMIN_TOKEN and any secrets
docker compose up                                   # core: pg, redis, 2× ml-service, nginx
docker compose --profile all up                     # + kafka, observability, retrain worker
```

- API (via nginx): `http://localhost/`
- Grafana: `http://localhost:3000`  ·  Prometheus: `http://localhost:9090`
- Kafka UI: `http://localhost:8080`

### Option B — Local Python (development)

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt

# 1) load the benchmark schema + data
python scripts/setup_tpch.py

# 2) run the ML service
set ML_ADMIN_TOKEN=changeme         # PowerShell: $env:ML_ADMIN_TOKEN="changeme"
python -m services.ml_service.server    # http://127.0.0.1:8000
```

Smoke test:

```bash
curl http://localhost:8000/healthz
curl http://localhost:8000/readyz
curl http://localhost:8000/resilience     # pool, breaker, cache, model registry
```

Predict + execute + learn (writes a feedback record):

```bash
curl -X POST http://localhost:8000/run-and-learn \
  -H "Content-Type: application/json" \
  -d '{"regime":"plan_time","sql":"SELECT * FROM tpch.orders LIMIT 100","oracle":true}'
```

> On PowerShell use `Invoke-RestMethod` with a `ConvertTo-Json` body.
> Pass `"oracle": true` to also execute every variant and record real
> regret / plan-pick-hit labels (higher-quality feedback, ~2× the cost).

---

## Train the model (offline)

```bash
# collect plans (parameterized TPC-H, 3 runs each → median labels)
python scripts/collect_tpch_param_plans.py --label-runs 3

# build the feature matrix, then GATE it
python feature_engineering/extract_features.py --output data/processed/features.csv
python scripts/validate_dataset.py --features        # exits non-zero on any hard failure

# AutoML: train 9 model families, Optuna-tune the trees, select the winner
python phase3b/train_models.py                        # --skip-tuning for a fast pass

# honest evaluation (out-of-fold GroupKFold + bootstrap CI)
python scripts/evaluate_baseline.py
```

Register the trained model in the versioned registry so the service can pin /
roll back versions:

```bash
python -m services.ml_service.model_registry register \
  --regime plan_time --path models/phase3b/plan_time/automl_best.joblib --promote
```

---

## The self-improving loop (Phase 5)

Run one cycle by hand:

```bash
python scripts/merge_feedback.py --apply --gate      # 5A: feedback → features.csv, gated
python scripts/retrain.py --profile fast             # 5B: train + register a CANDIDATE
python scripts/promote_model.py --candidate latest   # 5C: OOF gate (dry-run)
#   promote + zero-downtime hot-swap if it passes:
python scripts/promote_model.py --candidate latest --apply \
  --reload-url http://localhost:8000 --admin-token $ML_ADMIN_TOKEN
```

…or let the worker run it continuously (triggers, promotion, watchdog/rollback):

```bash
docker compose --profile retrain up automl-worker
```

**Safety by design:** a candidate is promoted **only if it beats the incumbent
out-of-fold** with a paired 95% CI lower bound ≥ −2pp (plus q-error and regret
guards). Promotion is an atomic pointer flip in a content-addressed registry;
a watchdog auto-rolls-back if the new model degrades live error rate, latency,
or calibration. The gate correctly **rejects noise-level "improvements."**

---

## Testing

```bash
pytest -q                    # full suite
ruff check services tests scripts
```

CI (GitHub Actions) runs lint + the infra-free unit tests (resilience,
triggers, drift, watchdog, worker cycle, file lock) on every push.

---

## Documentation

| Area | Doc |
|---|---|
| Project statement + STAR + tech rationale | [PROJECT_STATEMENT_AND_STAR.md](docs/PROJECT_STATEMENT_AND_STAR.md) |
| Interview Q&A + ML deep-dive + results | [INTERVIEW_PREP_QA.md](docs/INTERVIEW_PREP_QA.md) |
| Full roadmap (Phases 1–5) | [PROJECT_ROADMAP.md](docs/PROJECT_ROADMAP.md) |
| Phase 4 (production hardening) | [PHASE4_OVERVIEW.md](docs/PHASE4_OVERVIEW.md) |
| Phase 5 (retraining loop) | [PHASE5_OVERVIEW.md](docs/PHASE5_OVERVIEW.md) · 5A–5F docs |
| Methodology / evaluation | [METHODOLOGY.md](docs/METHODOLOGY.md) · [PHASE3G_EVALUATION.md](docs/PHASE3G_EVALUATION.md) |

Per-phase design docs (the "what / why / why-not" for every component) live in
[`docs/`](docs/).

---

## Status

| Phase | Scope | Status |
|---|---|---|
| 1–2 | Data collection, TPC-H/TPC-DS, feature extraction | ✅ |
| 3A–3H | Model training, AutoML selection, honest evaluation | ✅ |
| 4A–4F | Resilience, cache, streaming, observability, orchestration, load testing | ✅ |
| 5A–5F | Feedback merge, retrain, promotion gate, triggers, worker, watchdog | ✅ |
| 3I (optional) | LambdaRank / pairwise ranking objective | ⬜ future |

---

## License

Research / educational project. See individual dependencies for their licenses.
