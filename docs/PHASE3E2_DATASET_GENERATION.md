# Phase 3E.2 — Dataset Generation Strategy

> What data we generate, why this exact shape, and why every knob is set
> the way it is. Covers `scripts/collect_tpch_param_plans.py`,
> `db/tpch_param_queries.py`, and the multi-run median labeling in
> `scripts/collect_data.py`.

---

## 0. Mental model

```
   22 TPC-H query templates
        × ~10 parameter bindings each      → ~220 distinct queries
        × 4 join-knob variants             → ~880 execution plans
        × 3 timed runs (drop slowest)      → 1 robust median label / plan
                                            ─────────────────────────────
                                            880 labeled training rows
                                            (each = one plan, one runtime)
```

Each **row** the model sees is *one execution plan with one measured
runtime*. Each **group** (one `query_id` like `q03_p4`) holds the 4
join-knob variants the model must rank. Plan-pick accuracy is measured
per group; everything about the dataset shape exists to make that
ranking learnable and trustworthy.

---

## 1. Why parameterized queries (not the raw 22)

The vanilla TPC-H suite is 22 queries. Train on 22 groups and the model
**memorizes constants** ("query q03 is always fastest with hashjoin")
instead of learning *why*. It also gives GroupKFold almost nothing to
hold out.

`db/tpch_param_queries.py` expands each template with realistic
parameter bindings (date ranges, mktsegments, regions, quantities,
nations, …) — e.g. `q03_p0…q03_p9` differ only in `SEGMENT`/`DATE`.

- **Why this helps:** same plan *shape*, different *selectivities* →
  different optimal join strategies. The model is forced to read the
  plan/cardinality features, not the query id.
- **Why ~10 per template:** enough to populate GroupKFold folds and
  shrink the confidence interval (see 3H), without exploding heavy
  queries (q07/q18) into hours of collection.
- **Why not thousands:** diminishing returns vs. linear collection cost;
  heavy queries dominate wall-clock. We grow breadth (TPC-DS, JOB) before
  depth.

---

## 2. Why 4 join-knob variants per query

```
default        — planner's free choice (baseline plan)
no_hashjoin    — SET enable_hashjoin  = off
no_mergejoin   — SET enable_mergejoin = off
no_nestloop    — SET enable_nestloop  = off
```

**This is what creates the decision the learned optimizer exists to
make.** Without alternatives there is nothing to "pick". By forcing the
planner away from each join method we materialize genuinely different
plans for the *same* query, then measure which is actually fastest. The
model's job: predict that ranking.

- **Why knob-toggling and not random plans:** toggles produce *valid,
  executable, planner-blessed* plans (correct results), unlike hand-built
  plan trees. They're cheap and reproducible.
- **Why these three knobs:** join method is the single biggest lever on
  TPC-H runtime. (Future variants could toggle `from_collapse_limit`,
  parallelism, etc.; out of scope for this baseline.)
- **Knob state is also a feature.** Phase 3E added
  `enable_hashjoin/mergejoin/nestloop` as boolean features so the model
  can distinguish variants even when PG produces an identical plan tree.

---

## 3. Why multi-run **median** labels (`--label-runs 3`, drop slowest)

The label is `execution_time_ms`. Run the *same* plan twice and it can
differ 20–30% (OS scheduling, buffer-cache warmth, autovacuum, power
states). If we treat each noisy run as truth, the model learns the
noise — and plan-pick flips on tiny, meaningless differences.

`aggregate_label_runs()` runs each (query, variant) **3 times**, drops
the **slowest**, and takes the **median** of the rest. We also persist
`target_variance_ms` (diagnostic) and `label_runs`.

### Why this exact recipe — vs. alternatives

| Strategy | Verdict |
|---|---|
| **Single run** | ❌ cheapest but noisiest; the dominant error source per our root-cause analysis |
| **Mean of N** | ❌ a single cold-cache outlier drags the mean; not robust |
| **Min of N** | ❌ best-case only; ignores realistic variance, over-optimistic |
| **Median, drop slowest (chosen)** | ✅ robust to the *one* slow outlier (cold cache / autovacuum tick) that dominates in practice, while staying cheap at N=3 |
| **N=5+ median** | ⏳ even tighter, but 1.7× more collection time; revisit only if label variance is still the bottleneck after 3H |

> Rule of thumb: the **first** run of a plan is often a cold-cache
> outlier. "Drop slowest, median the rest" specifically neutralizes that
> without throwing away signal.

---

## 4. Why TPC-H SF1 first, TPC-DS next, JOB later

- **TPC-H SF1** — small enough to collect many plans fast, big enough
  (6M-row `lineitem`) that join strategy genuinely matters. The standard
  starting point and comparable to published work.
- **TPC-DS** — wider schema, more join shapes; tests generalization
  beyond TPC-H idioms. Already loaded (now in its own `tpcds` schema).
- **JOB (Join Order Benchmark)** — 113 *real* IMDB queries with
  genuinely hard join orders; the exact situation a learned optimizer
  should beat the PG planner on. Highest-value diversity, but a separate
  dataset download + collector — scheduled **after** we have a clean,
  validated TPC-H+TPC-DS baseline (so its impact is measurable, not
  confounded with the schema fix).

We grow **workload diversity** before model complexity because more
representative data shrinks the confidence interval and makes every
later experiment (LambdaRank, etc.) a *measurable* one.

---

## 5. Operational details that matter

- **`SET search_path = tpch, public`** after each `RESET ALL` (see
  Phase 3E.1) — unqualified table names resolve to the TPC-H schema.
- **`statement_timeout = 300s`** — caps pathological plans (e.g.
  `no_hashjoin` forcing a nested loop over millions of rows) so one bad
  variant can't stall the run forever. A timed-out variant is recorded
  as a failure, not a fake label.
- **`EXPLAIN (ANALYZE, BUFFERS, VERBOSE, FORMAT JSON)`** — ANALYZE gives
  *actual* rows/times (our labels + cardinality-error features), BUFFERS
  gives I/O features, JSON is machine-parseable by the feature layer.
- **Output:** one JSON per `{query_id}__{variant}__{sql_hash}.json` plus
  an appended `_index.jsonl`.

> **Data-hygiene caveat (handled before extraction):** `_index.jsonl` is
> append-only and the corpus may contain stale pre-fix plans. Phase 3E.2
> therefore ends with a rebuild/dedup step so `features.csv` is built
> from corrected plan files **only** — otherwise the "new baseline" is
> silently contaminated.

---

## 6. How to run

```bash
python scripts/collect_tpch_param_plans.py --label-runs 3
# → data/tpch/plans_param/*.json  (+ _index.jsonl)
```

Next: **Phase 3E.3** (feature extraction) and **3E.4** (validation gate)
before any training.
