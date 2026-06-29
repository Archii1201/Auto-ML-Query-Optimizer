# Methodology — The Gated Scientific Pipeline (Phase 3E.1 → 3I)

> The spine of the project's research process: a **gated** pipeline where
> each stage must pass before the next runs, so every reported number is
> reproducible and trustworthy. This is the index; each stage links to a
> deep-dive doc.

---

## Why a gated pipeline

The Phase 3E.1 incident (a wrong `customer` table silently feeding the
model — see `PHASE3E1_BENCHMARK_INTEGRITY.md`) proved that *quiet* data
problems are the dangerous ones. The cure is to make data quality and
evaluation honesty **explicit, automated gates** rather than assumptions.

> Principle: **never let an unverified artifact flow downstream.** A gate
> that fails loudly is cheaper than a model that's confidently wrong.

---

## The pipeline

```
  3E.1  Benchmark Integrity      tpch.* / tpcds.* schemas, search_path
         │                       → migrate_to_schemas.py            [DONE]
         ▼
  3E.2  Dataset Generation       22×~10 params ×4 variants ×3 runs (median)
         │                       → collect_tpch_param_plans.py      [RUNNING]
         │                       + data-hygiene: rebuild features / dedup index
         ▼
  3E.3  Feature Extraction       plan-tree DFS → features.csv
         │                       → extract_features.py
         ▼
  3E.4  Dataset Validation  ◄── HARD GATE (exit 1 on failure)
         │                       → validate_dataset.py
         ▼
  3F    Model Training           fit deployable artifact (+ Optuna)
         │                       → train_models.py
         ▼
  3G    Honest Evaluation        GroupKFold OOF, full metric panel
         │                       → evaluate_baseline.py
         ▼
  3H    Scientific Analysis      bootstrap CI + significance vs frozen baseline
         │                       (same script, analysis section)
         ▼
  3I    Improvements             controlled experiments: more data / better
                                 labels / LambdaRank — each measured the same way
```

Each arrow is a gate: 3F does not run until 3E.4 says
`DATASET TRUSTWORTHY`; 3I does not start until 3H gives a CI-bounded
baseline.

---

## Stage-by-stage deep dives

| Stage | Doc | One-line purpose |
|---|---|---|
| 3E.1 | [`PHASE3E1_BENCHMARK_INTEGRITY.md`](PHASE3E1_BENCHMARK_INTEGRITY.md) | isolate TPC-H/TPC-DS into schemas so table names can't collide |
| 3E.2 | [`PHASE3E2_DATASET_GENERATION.md`](PHASE3E2_DATASET_GENERATION.md) | params × knob-variants × median labels — and *why* each |
| 3E.3 | [`PHASE2B_FEATURES.md`](PHASE2B_FEATURES.md) | plan-tree → fixed-length feature vector (existing) |
| 3E.4 | [`PHASE3E4_DATASET_VALIDATION.md`](PHASE3E4_DATASET_VALIDATION.md) | hard gate that proves the dataset is trustworthy |
| 3F | [`PHASE3B_AUTOML.md`](PHASE3B_AUTOML.md) | AutoML model comparison + winner selection (existing) |
| 3G/3H | [`PHASE3G_EVALUATION.md`](PHASE3G_EVALUATION.md) | honest OOF metrics + confidence intervals + significance |

---

## The non-negotiable rules (what keeps it honest)

1. **GroupKFold, never plain KFold** — a query's variants must not leak
   across train/test.
2. **Report OOF, never in-sample** — in-sample numbers are vanity and
   caused the original "degradation" confusion.
3. **Select on plan-pick** (the deployment metric), diagnose with
   q-error/regret — not the other way around.
4. **Every headline number carries a 95% CI** and a verdict against a
   **pre-registered** frozen baseline (`0.576` prod; `0.438–0.504`
   ablation).
5. **No accuracy claim before 3G.** Until then the only statement is:
   *"benchmark corrected, dataset regenerated, model to be re-evaluated
   on a reproducible dataset."*

---

## Commands (the whole pipeline)

```bash
# 3E.1 (one-time / from scratch)
python scripts/migrate_to_schemas.py --sf 1

# 3E.2
python scripts/collect_tpch_param_plans.py --label-runs 3

# 3E.4 gate (plans)  →  3E.3  →  3E.4 gate (features)
python scripts/validate_dataset.py            && \
python feature_engineering/extract_features.py && \
python scripts/validate_dataset.py --features  && \
python phase3b/train_models.py                 && \
python scripts/evaluate_baseline.py
```

If any stage exits non-zero, the chain stops — by design.
