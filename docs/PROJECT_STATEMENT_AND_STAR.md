# AutoML-Powered Learned Query Optimizer — Project Statement, STAR & Tech Rationale

> A single reference you can read before an interview: what the project is,
> why it matters, the exact numbers, a spoken FAANG-level STAR narrative, a
> 4-line resume version, and the "what we used and why" table.

---

## 1. The problem (in plain terms)

Every relational database (PostgreSQL, MySQL, …) must answer one question
before running your SQL: **"which physical execution plan is fastest?"** A
query like a 3-table join can be executed in dozens of ways — hash join vs
merge join vs nested loop, different join orders, different scan methods.
Picking the wrong one can make a query **10–100× slower**.

PostgreSQL decides this with a **static, hand-tuned cost model** written in
the 1990s. It estimates a plan's cost from table statistics using fixed
formulas. The problem: those estimates are frequently **wrong** — especially
on complex joins where row-count estimates compound and drift far from
reality. The optimizer then confidently picks a slow plan.

**Core idea of this project:** replace that guesswork with a model that
**learns from real execution history**. Instead of trusting a static
formula, we *measure* how long plans actually take, train an ML model to
**predict a plan's runtime from its shape**, and use those predictions to
**pick the fastest plan** — then feed the new measurements back to keep the
model improving. In short: **a self-improving, learned replacement for the
database's cost-based plan selector.**

---

## 2. What the system actually does (the flow)

```
SQL query
   │
   ▼
Generate candidate plans   ── ask PostgreSQL for the plan under 4 join-knob
                              settings: default, no_hashjoin, no_mergejoin,
                              no_nestloop  (EXPLAIN, no execution yet)
   │
   ▼
Feature extraction         ── flatten each EXPLAIN JSON tree into ~50 numeric
                              features (costs, cardinality mis-estimates,
                              tree shape, scan/join counts, knob state)
   │
   ▼
ML prediction              ── a trained tree-ensemble predicts each plan's
                              runtime in milliseconds
   │
   ▼
Pick the best plan         ── choose the variant with the lowest predicted ms
   │
   ▼
Execute on PostgreSQL      ── run it, measure the TRUE runtime
   │
   ▼
Store feedback             ── write (features → actual runtime) to disk/Kafka
   │
   ▼
Retrain loop (Phase 5)     ── merge feedback → retrain → statistically gate →
                              promote a better model → hot-swap live → watch &
                              auto-rollback if it regresses
```

The last stage is what makes it **AutoML**: the system gets better from its
own traffic, automatically, but only ships a new model when it *provably*
beats the current one.

---

## 3. The two prediction regimes (important nuance)

We train two models with the same pipeline but different information:

| Regime | Features available | Deployable? | Plan-pick accuracy |
|---|---|---|---|
| **`plan_time`** | Only the planner's **pre-execution estimates** (EXPLAIN without ANALYZE) | ✅ Yes — this is what you can use to *choose* a plan before running it | **~0.48–0.51** |
| **`post_mortem`** | Includes **actual** execution stats (EXPLAIN ANALYZE) | ❌ No — needs to run the query to know | **~0.96** |

`plan_time` is the real product (you must decide *before* executing).
`post_mortem` is a **scientific upper bound**: it near-perfectly picks the
right plan (0.96), which proves our feature set is rich enough to solve the
problem — the remaining gap is purely the planner's estimate error, exactly
the thing a learned model exists to fix.

---

## 4. The headline results (real numbers from this repo)

| Metric | PostgreSQL's own cost model | Our learned model (`plan_time`) | Meaning |
|---|---|---|---|
| **Plan-pick accuracy** | **0.205** | **~0.48–0.51** | We pick the truly-fastest plan **~2.3–2.5× more often** than Postgres |
| **Median q-error** | 1.65–1.79 | **~1.24–1.39** | Typical prediction is within ~24–39% of actual runtime |
| **p95 q-error** | 8.3–8.9 | **~3.2–3.7** | Even the tail predictions are far tighter |
| **R² (log runtime)** | ~0.05 | **~0.51–0.53** | Explains real variance the static model can't |
| **Mean regret** | 4366 ms | **~412–586 ms** | Extra time paid vs the perfect oracle is ~7–10× smaller |

- **AutoML winner (`plan_time`):** a **tuned tree ensemble** (LightGBM /
  ExtraTrees family; the currently-deployed model is `extra_trees_tuned`),
  selected automatically over 9 model families by a composite of q-error +
  plan-pick accuracy.
- **`post_mortem` winner:** `gradient_boosting_tuned`, plan-pick **0.962**,
  median q-error **1.005** — essentially oracle-level.
- **Closed-loop proof:** the Phase 5 promotion gate correctly **REJECTED** a
  candidate that looked +2.6pp better but whose paired 95% CI lower bound was
  −6.6pp (below the −2pp tolerance) — i.e. it refused to ship a
  noise-level "improvement." **The safety system works.**

> Honest framing for interviews: the absolute plan-pick (~0.5) is modest
> because `plan_time` only sees the planner's (often wrong) estimates — the
> hard version of the problem. The *scientific wins* are (a) more than
> doubling Postgres's own selection accuracy, (b) proving via `post_mortem`
> that the features are sufficient, and (c) building a production-grade
> self-improving loop with statistically-gated promotion and auto-rollback.

---

## 5. STAR narrative — spoken, FAANG interview version

Use this when an interviewer says *"walk me through a project."* It's written
to be **said out loud** in ~2–3 minutes.

**Situation.**
"Databases like PostgreSQL choose how to execute a query using a static cost
model from the 1990s. On complex joins its runtime estimates are often wrong,
so it picks slow plans — sometimes 10 to 100 times slower than the best one.
I wanted to see if a model that learns from real execution history could pick
better plans than the database's own optimizer."

**Task.**
"My goal was to build an end-to-end learned query optimizer: generate
candidate plans for a query, predict each one's runtime with an ML model,
pick the fastest, execute it, and then feed the measured runtime back to
continuously retrain the model — all wrapped in a production-grade service
that's resilient, observable, and safe to auto-update."

**Action.**
"I built it in phases. First, data: I ran parameterized TPC-H and TPC-DS
workloads under four join-knob settings and captured every plan's EXPLAIN
JSON plus its true runtime. Then I engineered about fifty features from each
plan tree — costs, cardinality mis-estimates, tree shape, and crucially the
knob state so the model can tell variants apart. I trained nine model
families, tuned the tree ensembles with Optuna, and used an AutoML selector
that picks the winner by a composite of q-error and plan-pick accuracy. The
key evaluation decision was **GroupKFold by query with a paired bootstrap
confidence interval** — because the metric that matters is ranking plans
*within* a query, and I had to avoid leaking the same query across train and
test. Then I hardened it for production: a FastAPI service with a bounded
connection pool, a circuit breaker that falls back to Postgres's default plan
if the model fails, Redis caching, a Kafka feedback bus, and Prometheus /
Grafana / OpenTelemetry for observability — all one-command in Docker Compose
behind an nginx load balancer across two replicas. Finally I built the AutoML
retraining loop: it merges new feedback behind a validation gate, retrains a
candidate, and only promotes it if a **statistical gate** proves it beats the
current model out-of-fold; promotion is an atomic pointer flip in a versioned
model registry, hot-swapped with no restart, and a watchdog auto-rolls-back
if the new model misbehaves on live traffic."

**Result.**
"The learned model picks the truly-fastest plan about two-and-a-half times as
often as PostgreSQL's own cost model — plan-pick accuracy went from 0.205 to
around 0.5 — and median prediction error dropped to about 1.25 q-error, with
mean regret roughly 7 to 10 times smaller. The post-mortem model hit 0.96,
proving the features are sufficient. And the safety system is real: when I
fed it a small feedback batch, the promotion gate correctly *rejected* a
candidate whose apparent 2.6-point gain was inside the noise floor — it
refused to ship an unproven model. So I ended up with a self-improving
optimizer that measurably beats the database's static planner and is safe
enough to update itself in production."

---

## 6. STAR for your résumé — exactly 4 bullet points

> Copy-paste ready. Each line is Action-led with a quantified result.

**AutoML-Powered Learned Query Optimizer** — *Python, scikit-learn, LightGBM, FastAPI, PostgreSQL, Kafka, Redis, Docker, Prometheus/Grafana*

- **Built an end-to-end learned query optimizer** that predicts SQL plan
  runtimes from ~50 EXPLAIN-derived features and selects the fastest plan,
  **more than doubling plan-pick accuracy vs PostgreSQL's native cost model
  (0.205 → ~0.50)** and cutting mean plan regret ~7–10×.
- **Engineered an AutoML pipeline** (9 model families, Optuna tuning,
  GroupKFold + paired-bootstrap evaluation) that auto-selects the best model
  by q-error and plan-pick accuracy, achieving **median q-error ~1.25** and a
  near-oracle 0.96 on the post-mortem regime.
- **Productionized a resilient microservice** — FastAPI with connection
  pooling, circuit-breaker fallback, Redis cache, Kafka feedback streaming,
  and Prometheus/Grafana/OpenTelemetry observability — deployed via Docker
  Compose behind an nginx-load-balanced 2-replica cluster.
- **Designed a self-improving retraining loop** with a **statistically-gated
  promotion** (paired 95% CI), atomic model-registry versioning, zero-downtime
  hot-swap, and automatic rollback — which correctly rejected noise-level
  candidates and safely promoted proven ones.

---

## 7. Everything we used, and *why* (technology & model rationale)

### 7.1 Data & database
| Tech | Why we used it | Why not the alternative |
|---|---|---|
| **PostgreSQL** | The DBMS we optimize; source of EXPLAIN plans + ground-truth runtimes. Open, scriptable, industry-standard planner. | MySQL's optimizer is less transparent; we needed rich `EXPLAIN (FORMAT JSON)`. |
| **TPC-H + TPC-DS workloads** | Standard analytical benchmarks with genuinely hard joins — the exact cases the planner gets wrong. Parameterized for volume. | A single toy schema wouldn't stress the optimizer or generalize. |
| **psycopg2 + bounded connection pool** | Reliable Postgres driver; the pool caps concurrent connections so a traffic burst can't exhaust `max_connections`. | Raw `connect()` per request exhausts the DB under load. |

### 7.2 Machine learning
| Tech | Why we used it |
|---|---|
| **scikit-learn** | Uniform API for the model zoo, `Pipeline`, and `GroupKFold`. The backbone of training + evaluation. |
| **Tree ensembles — ExtraTrees, RandomForest, GradientBoosting, XGBoost, LightGBM, CatBoost** | Our data is **tabular, ~50 heterogeneous numeric features with strong non-linear interactions** (e.g. "this plan shape is fast *only if* hashjoin is on"), and a few-thousand-row dataset. Gradient-boosted / bagged trees are the proven best-in-class here: they capture interactions, need no feature scaling, are robust to outliers and skew, and expose feature importances. |
| **Optuna** | Bayesian hyperparameter search to tune the tree models efficiently (better than grid search at the same budget). |
| **Log-transformed target** | Runtimes span orders of magnitude; training on `log(runtime)` optimizes **relative** error (q-error), which is what plan-ranking cares about, and stops giant queries from dominating the loss. |
| **AutoML selector** | Automatically picks the deployable model per regime by a **composite of median q-error + plan-pick accuracy**, so selection is objective, not a hunch. |
| **GroupKFold (by query) + paired bootstrap CI** | The honest evaluation core. Grouping by query prevents leaking a query across train/test; the paired bootstrap tells us whether an improvement is **real or noise** — this is what powers the promotion gate. |
| **joblib** | Fast, standard serialization of fitted models into artifacts. |

### 7.3 Serving & resilience (Phase 3C + 4A)
| Tech | Why |
|---|---|
| **FastAPI + uvicorn + pydantic** | Async, high-throughput serving with typed request/response schemas and auto OpenAPI docs. |
| **Circuit breaker** | If the model path fails repeatedly, "open" the circuit and **fall back to PostgreSQL's default plan** — the system degrades gracefully instead of erroring. |
| **Timeout budget + structured JSON logging + liveness/readiness probes** | Bounded per-request latency, machine-parseable logs, and Kubernetes-style health for safe rollouts. |

### 7.4 Caching, streaming, registry (Phase 4B/4C)
| Tech | Why |
|---|---|
| **Redis + cachetools (Strategy pattern)** | Cache `plan → prediction` to cut latency. Pluggable: local LRU for dev, Redis so multiple replicas share a cache and survive restarts. |
| **Kafka (confluent-kafka) + consumer** | Durable, real-time feedback bus decoupling the write path from training; enables the retraining loop and multiple consumers. Pluggable with a file publisher for dev. |
| **Custom model registry (content-addressed, SHA-256)** | Versioned, immutable model artifacts with `promote`/`rollback` as an atomic pointer flip. Lets us attribute predictions to a version and roll back instantly — without MLflow's extra infra. |

### 7.5 Observability & orchestration (Phase 4D/4E)
| Tech | Why |
|---|---|
| **Prometheus + prometheus-client** | Industry-standard metrics with real histograms (latency, calibration, regret). |
| **Grafana** | Pre-built dashboards for service health and plan-pick quality — one screen to see the system. |
| **OpenTelemetry + Tempo + OTel Collector** | Distributed tracing: one trace ID flows plan-gen → predict → execute → feedback. |
| **Docker + docker-compose + nginx** | One-command bring-up of ~9 services; nginx is the API gateway / load balancer across two ML replicas (horizontal scaling). |
| **Locust + chaos script** | Load and fault-injection testing to prove the resilience and SLOs hold. |

### 7.6 The retraining loop (Phase 5)
| Piece | Why |
|---|---|
| **Merge + validation gate** | Fold live feedback into the training set idempotently, and refuse to train on a dataset that fails validation. |
| **Trigger engine (volume / schedule / drift)** | Decide *when* to retrain from data volume, a heartbeat, or the live predicted/actual calibration drifting — all behind a cooldown so it never thrashes. |
| **Statistical promotion gate** | Only ship a candidate if it beats the incumbent out-of-fold with a paired CI lower bound above tolerance, plus q-error and regret guards. |
| **Watchdog + auto-rollback** | After promotion, monitor live error rate / latency / calibration and roll back instantly if the new model misbehaves. |

---

## 8. One-line pitch

> "I replaced PostgreSQL's static cost-based plan selector with a learned
> model that predicts plan runtimes from execution history — more than
> doubling how often we pick the fastest plan — and wrapped it in a
> production-grade, self-improving service that safely retrains and promotes
> itself behind a statistical gate, with automatic rollback."
