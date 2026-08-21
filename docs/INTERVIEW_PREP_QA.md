# Interview Prep — Q&A, ML Model Deep-Dive, and Results

> Written as if a **senior Google interviewer with 20+ years in databases and
> ML** is grilling you. Each answer is phrased so you can **say it out loud**.
> Read the STAR/overview first: [PROJECT_STATEMENT_AND_STAR.md](PROJECT_STATEMENT_AND_STAR.md).

Sections:
1. Project & problem framing
2. Data & feature engineering
3. ML modeling & model choice (deep-dive)
4. Evaluation & the science
5. The retraining loop (MLOps)
6. Systems / production design
7. Curveballs & "what would you do differently"
8. The ML models we used — why each
9. Final results & why they're useful

---

## 1. Project & problem framing

**Q1. In one minute — what did you build and why?**
"PostgreSQL chooses query execution plans with a static cost model, and on
complex joins its runtime estimates are often wrong, so it picks slow plans.
I built a learned query optimizer: for a given SQL query it generates several
candidate plans, an ML model predicts each plan's runtime, we pick the
fastest, execute it, measure the true runtime, and feed that back to retrain
the model. It's wrapped in a production-grade service that safely retrains and
promotes itself. The learned model picks the fastest plan about 2.5× more
often than Postgres's own optimizer."

**Q2. Why is choosing a plan hard? Why doesn't the database already do this well?**
"Because plan cost depends on **cardinality** — how many rows flow between
operators — and cardinality estimation is famously hard. Errors compound
multiplicatively up a join tree, so a small mistake at a leaf becomes a huge
mistake at the root. Postgres uses fixed formulas over table statistics; it
has no memory of what actually happened last time. A learned model *does*
have that memory — it can recognize 'this plan shape with these estimates
tends to actually take X ms.'"

**Q3. Why predict runtime instead of just improving the cost estimate?**
"Because the deliverable is a **ranking decision**, not a physics-accurate
cost. I don't need the exact milliseconds; I need to know which of the four
candidate plans is fastest. Predicting runtime and comparing predictions is a
direct, learnable proxy for that ranking, and it lets me measure success with
a concrete business metric — plan-pick accuracy."

**Q4. How do you generate the candidate plans?**
"I toggle PostgreSQL's join-method knobs: the default plan, then plans with
`enable_hashjoin`, `enable_mergejoin`, and `enable_nestloop` disabled in turn.
That gives four legitimate alternative plans per query from the real planner —
I'm not inventing plans, I'm asking Postgres for different valid ones and
letting the model choose among them."

---

## 2. Data & feature engineering

**Q5. Where did the training data come from?**
"I ran parameterized **TPC-H** and curated **TPC-DS** workloads — standard
analytical benchmarks with genuinely hard joins. For each query and each of
the four knob variants, I captured the `EXPLAIN (FORMAT JSON)` plan and the
true execution time. To fight label noise I ran each configuration multiple
times and took the median (`--label-runs`)."

**Q6. What features did you extract, and from what?**
"About fifty numeric features from each plan's EXPLAIN JSON tree, in a few
families: **cost features** (estimated startup/total cost and their
log-transforms — these dominate importance), **structural features** (tree
depth, total nodes, number of joins, scan-type counts), **cardinality
features** (estimated rows, and mis-estimate ratios — actual vs estimated at
nodes), **ratio features** (cost per row, startup-to-total ratio), and
crucially **knob-state features** (`enable_hashjoin/mergejoin/nestloop` as
booleans)."

**Q7. Why were the knob-state features so important?**
"Because sometimes the *plan tree itself is identical* across two knob
settings — e.g. disabling a join type the planner wasn't using anyway. Without
knob features, the model produces identical predictions for those variants and
can only tie-break on a weak cost signal. Adding the boolean knob state lets
the model learn 'this shape is fast with hashjoin on but slow when forced onto
mergejoin,' even when the tree looks the same. It was one of the higher-impact
feature additions."

**Q8. How did you guarantee the online features match the offline ones?**
"The single biggest risk in a learned optimizer is **online/offline feature
skew** — training on one representation and serving on another. I avoided it
structurally: the live service and the offline pipeline call the **exact same
feature-extraction function**. In Phase 3D I verified the online and offline
feature vectors were bit-identical. The Phase 5 merge also reuses that same
extractor, so skew is impossible by construction."

**Q9. How do you prevent target leakage?**
"Two ways. First, for the deployable `plan_time` regime I only use features
available **before execution** — planner estimates, never actual runtimes.
Second, I added an automated leakage check that asserts no feature correlates
above ~0.95 with the target, so a future engineer can't silently add a leaky
column. The promotion gate also rejects any candidate whose metrics are
implausible or non-finite as a backstop."

---

## 3. ML modeling & model choice (deep-dive)

**Q10. Why tree ensembles? Justify it against linear models and neural nets.**
"Three properties of the data drive it: it's **tabular**, it has **strong
non-linear interactions** (knob × plan-shape), and it's **moderate-sized —
a few thousand rows**. Gradient-boosted and bagged tree ensembles are the
proven best fit: they model interactions automatically, need no feature
scaling, are robust to skewed/outlier runtimes, and give interpretable feature
importances. Linear models (Ridge, Lasso, ElasticNet) underfit badly here —
they scored **negative R²** because the relationship is highly non-linear.
Neural nets would overfit a few-thousand-row tabular set and buy nothing over
GBMs while adding tuning and infra complexity. This mirrors the broad
empirical result that GBDTs still beat deep nets on tabular data."

**Q11. You trained nine models — how did you pick the deployed one?**
"An **AutoML selector** scores every model per regime on a **composite of
median q-error and plan-pick accuracy** — not raw MAE — because those are what
production cares about. For `plan_time` the tuned tree ensemble family won
(LightGBM / ExtraTrees; the currently deployed artifact is `extra_trees_tuned`).
For `post_mortem`, `gradient_boosting_tuned` won at 0.96 plan-pick. Selection
is objective and reproducible, not my personal preference."

**Q12. Why log-transform the target?**
"Runtimes span several orders of magnitude — some queries are milliseconds,
some are seconds. If I regress on raw ms, the loss is dominated by the biggest
queries and I optimize absolute error. Training on `log(runtime)` optimizes
**relative** error, which is exactly q-error — the metric that matters for
ranking — and it stabilizes variance."

**Q13. What's the difference between the `plan_time` and `post_mortem` models?**
"Same pipeline, different information. `plan_time` only sees the planner's
pre-execution estimates — that's the deployable model, because you must choose
a plan *before* running it, and it scores ~0.5 plan-pick. `post_mortem` also
sees actual execution stats — it's not deployable for selection because it
needs to run the query, but it hits **0.96**, which proves the feature set is
rich enough to solve the problem. The gap between them is precisely the
planner's estimate error — the thing the learned optimizer exists to close."

**Q14. LightGBM vs XGBoost vs ExtraTrees — why did the winner win?**
"They're close, and the AutoML selector decides empirically per run.
LightGBM is fast and strong on tabular data with histogram-based splitting;
ExtraTrees adds extra split randomization which **reduces variance on noisy
labels** — valuable because our runtime labels are inherently noisy from cache
state and OS jitter. Whichever posts the best composite q-error + plan-pick on
GroupKFold gets shipped; I don't hard-code a favorite."

---

## 4. Evaluation & the science

**Q15. What's 'plan-pick accuracy' and why is it your headline metric?**
"For each query — a group of 4 variants — does the model's lowest-predicted-
runtime plan equal the actually-fastest plan? Averaged over queries, that's
plan-pick accuracy. It's the headline because it's **exactly the production
decision**. RMSE or MAE can look fine while the model still ranks plans wrong;
plan-pick measures the thing that changes query latency."

**Q16. Why GroupKFold instead of ordinary k-fold?**
"Because the four variants of one query are highly correlated. If some
variants of query Q are in train and others in test, the model has effectively
seen Q — that's leakage and it inflates the score. GroupKFold keeps **all
variants of a query in the same fold**, so I only ever evaluate on *unseen
queries*, which is the honest measure of generalization."

**Q17. You got a promotion REJECT with the candidate looking better. Explain.**
"The candidate's point estimate was +2.6pp plan-pick and its q-error and
regret both improved. But the gate rejected it because the **paired bootstrap
95% CI lower bound of the plan-pick delta was −6.6pp**, below my −2pp
tolerance. In words: when I resample the query groups, the 'improvement' is so
noisy I can't rule out the candidate being meaningfully *worse*. So the gate
correctly refused to ship an unproven model. That's the system working — it's
the discipline that stops you from chasing noise. To pass, I need a tighter CI
(more query groups, more runs per query) or a genuinely stronger candidate
(the `full` Optuna profile)."

**Q18. Why a *paired* bootstrap, not two independent confidence intervals?**
"Because both models are evaluated on the **same** query groups, so their
errors are correlated. A paired bootstrap over groups gives a much tighter,
correct interval for 'did it actually get better on the same queries?' Two
independent CIs would be needlessly wide and could hide a real, consistent
per-query win — or, worse, let a noisy one through."

**Q19. How do you know the whole thing isn't overfitting?**
"Everything reported is **out-of-fold** — predictions on queries the model
never trained on, via GroupKFold. I also froze the old baseline numbers ahead
of time so I couldn't move the goalposts, and I report a bootstrap CI so I only
claim a win when it's statistically separable from the old baseline."

---

## 5. The retraining loop (MLOps)

**Q20. Walk me through how the system improves itself.**
"Five stages, each fail-open. **Merge**: fold new feedback into the training
CSV, idempotently, behind a validation gate. **Retrain**: train a candidate
and register it in the model registry — but never auto-promote. **Gate**:
evaluate candidate vs incumbent out-of-fold with the paired CI plus q-error
and regret guards. **Promote**: if it passes, flip the registry pointer and
hot-swap the live replicas with no restart. **Watchdog**: monitor the new
model on live traffic and auto-roll-back if error rate, latency, or
calibration degrade. A background worker runs this on a schedule with a
single-flight lock so two retrains never overlap."

**Q21. When does it decide to retrain? Won't it thrash?**
"Three OR-ed triggers behind one cooldown. **Volume**: enough new feedback
rows since the last retrain. **Schedule**: a daily heartbeat so a quiet system
still refreshes. **Drift**: the live predicted/actual calibration ratio
leaving a healthy band. The cooldown is a hard guarantee that we never retrain
in a tight loop no matter how loudly the rules fire."

**Q22. Promotion and rollback — how are they safe/atomic?**
"The registry is **content-addressed**: each model is an immutable artifact
keyed by the SHA-256 of its bytes, and 'current' is just a pointer. Promotion
is flipping that pointer; rollback is flipping it back to the previous
version. No file is ever overwritten, so rollback is instant, needs no
retrain, and can't lose the old model. The live swap is graceful — I build the
new predictors first and only replace the serving ones if that succeeds, so a
bad load leaves the old model serving."

**Q23. How does the watchdog decide to roll back without an oracle?**
"On live traffic I can't run every variant to know true plan-pick, so I roll
back on **operational** signals I *can* read cheaply and in real time: error
rate, p95 latency, and the predicted/actual calibration ratio. I compare
against a pre-promote baseline (to catch regressions) with an absolute ceiling
as a backstop. If any breach, I flip back to the previous version and reload."

---

## 6. Systems / production design

**Q24. How does the service stay up if PostgreSQL or the model fails?**
"Layered resilience. A **bounded connection pool** so a burst can't exhaust
Postgres. A **circuit breaker** around the model path: after repeated
failures it opens and serves Postgres's default plan — the classic 'fall back
to the database optimizer' — then probes to recover. A **per-request timeout
budget** so latency is bounded. And **readiness vs liveness** probes so an
orchestrator only routes traffic when the model is loaded and Postgres is
reachable."

**Q25. How does it scale horizontally?**
"Two ML-service replicas behind nginx with least-connections load balancing.
State that must be shared — the cache and the model registry — lives in Redis
and a shared volume, so replicas agree. After a promotion the worker reloads
**every** replica, because each caches its own predictor in-process."

**Q26. Why Kafka for feedback instead of writing to disk?**
"Decoupling and durability. The serving path just produces a feedback event;
a separate consumer persists it and the trainer consumes independently. That
lets the write path stay fast, survive consumer restarts via offset commits,
and support multiple consumers later. It's pluggable — a file publisher is the
default for local dev so tests need no Kafka."

**Q27. How do you observe it in production?**
"Prometheus metrics with real histograms (latency, calibration, regret),
Grafana dashboards for service health and plan-pick quality, and OpenTelemetry
traces so one trace ID follows a request through plan-gen → predict → execute
→ feedback. There's a dedicated dashboard showing exactly the signals the
watchdog rolls back on."

---

## 7. Curveballs & "what would you do differently"

**Q28. Your plan-pick is only ~0.5. Isn't that weak?**
"On the surface, but two things. One, it's **more than double** PostgreSQL's
own 0.205 on the same task — I beat the baseline that ships in every database.
Two, ~0.5 is the *hard* regime where the model only sees the planner's wrong
estimates; my `post_mortem` model hits 0.96, which proves the ceiling is high
and the remaining gap is the estimate error itself. The honest next lever is
more data and a ranking objective, not more features — I proved single-feature
tweaks were inside the noise floor."

**Q29. If you had two more weeks, what would move the number most?**
"A **LambdaRank / pairwise objective**. Right now I regress pointwise on
log-runtime, but I only care about the *rank order within a query*. Optimizing
rank directly with LightGBM's `lambdarank` is the one lever whose expected gain
is above the noise floor. After that, more query diversity — adding the Join
Order Benchmark — and multi-run median labels to cut noise. I'd run each as a
**controlled experiment through the same promotion gate**, so I only keep a
change if it's statistically real."

**Q30. What was the hardest bug or subtlest issue?**
"A schema-collision one: TPC-H and TPC-DS both have a `customer` table, and a
loader mix-up meant some customer-join queries were scanning the wrong,
smaller table — silently corrupting labels. I isolated the benchmarks into
separate Postgres schemas, fixed the `search_path`, and added a **validation
gate** that asserts customer-join queries scan a >10k-row customer table. It's
why the dataset now has a hard, automated trust check before any training."

**Q31. How do you attribute a bad prediction to a specific model?**
"Every feedback record stores the **model version** — the SHA-256 of the exact
joblib that produced the prediction — and the registry maps that hash back to
the artifact. So the retraining loop can attribute prediction quality to a
specific version, which is what makes gated promotion and targeted rollback
possible."

---

## 8. The ML models we used — and why each

| Model | What it is | Why it's in the zoo / when it wins |
|---|---|---|
| **ExtraTrees (Extremely Randomized Trees)** | Bagged trees with randomized split thresholds | **Currently deployed for `plan_time`.** The extra randomness lowers variance — valuable against noisy runtime labels — while still capturing interactions. |
| **LightGBM (tuned)** | Histogram-based gradient boosting | Top `plan_time` scorer in the AutoML bake-off (0.508 plan-pick, q-err 1.39). Fast, strong on tabular data, handles many features well. |
| **Gradient Boosting (tuned)** | Classic sequential boosting | **Winner for `post_mortem`** (0.962 plan-pick, q-err 1.005) — near-oracle when actual stats are available. |
| **RandomForest** | Bagged decision trees | Strong, stable baseline; robust to overfitting; great feature importances. |
| **XGBoost** | Regularized gradient boosting | Powerful boosting alternative; regularization helps on noisy targets. |
| **CatBoost** | Ordered boosting | Handles feature interactions well; competitive default performance. |
| **Ridge / Lasso / ElasticNet (linear)** | Regularized linear regression | Included as **honest baselines** — they underfit (negative R²), which *proves* the problem is non-linear and justifies the tree ensembles. |
| **PG cost-model baselines (linear / log-linear)** | Regression on Postgres's own cost | The **bar to beat** — represents the database's static optimizer (0.205 plan-pick). |

**Why tree ensembles overall (say this):** "Tabular data, non-linear
interactions, moderate size, noisy labels — that's the textbook home turf of
gradient-boosted and bagged trees. Linear models underfit and neural nets
overfit; GBMs give the best accuracy per unit of complexity and come with
feature importances I can actually reason about."

---

## 9. Final results & why they're useful

### The numbers (out-of-fold, honest)
| Metric | PG cost model | Learned `plan_time` | Learned `post_mortem` |
|---|---|---|---|
| Plan-pick accuracy | 0.205 | **~0.48–0.51** | **0.962** |
| Median q-error | 1.65–1.79 | **~1.24–1.39** | **1.005** |
| p95 q-error | 8.3–8.9 | **~3.2–3.7** | ~1.1 |
| R² (log runtime) | ~0.05 | **~0.51–0.53** | **~0.87** |
| Mean regret (ms) | 4366 | **~412–586** | ~3 |

### Why these results are useful (say this)
- **We beat the database's own optimizer at its core job.** Picking the
  fastest plan ~2.5× more often than PostgreSQL directly means **lower query
  latency** on exactly the complex-join workloads where the static planner is
  weakest.
- **Predictions are well-calibrated** (median q-error ~1.25 → within ~25% of
  reality), so downstream decisions are trustworthy, and **regret shrank
  ~7–10×** — the extra time we pay versus a perfect oracle is small.
- **The post-mortem 0.96 proves the approach is sound** — the features carry
  enough signal to nail plan selection; the deployable gap is the planner's
  estimate error, which is a data/objective problem, not a dead end.
- **It's production-real, not a notebook.** Resilient (circuit-breaker
  fallback), scalable (load-balanced replicas), observable (metrics + traces),
  and **self-improving with guardrails** — it retrains from its own traffic
  and only ships a new model when a paired statistical test proves it's
  better, auto-rolling-back if not. That combination — measurable ML wins
  *plus* safe MLOps automation — is the point.

### The closed-loop proof point (great story to end on)
"The most convincing result isn't a single accuracy number — it's that when I
fed the loop a small feedback batch, it retrained a candidate, evaluated it
out-of-fold, and the promotion gate **correctly rejected** it: the apparent
+2.6-point gain had a 95% CI lower bound of −6.6 points, inside the noise
floor. A naive system would have shipped a possible regression. Mine refused —
which is exactly the behavior you want from something allowed to update itself
in production."
