# Phase 3G + 3H — Honest Evaluation & Scientific Analysis

> How we measure the model *honestly*, why each metric exists, why
> plan-pick is the objective we select on, and how we decide whether a
> change is **real** or just noise. Documents `scripts/evaluate_baseline.py`
> and `scripts/experiment_ablation.py`.

---

## 0. Mental model

```
features.csv
   │
   ▼  GroupKFold(5)  — never let a query's variants leak across folds
out-of-fold (OOF) predictions  ← the ONLY number that reflects unseen queries
   │
   ├─► plan-pick accuracy   (deployment metric — selection objective)
   ├─► q-error median / p95 (calibration of the runtime prediction)
   ├─► MAE / RMSE           (raw regression error, ms)
   ├─► Spearman ρ           (rank quality)
   └─► regret mean / p95    (business cost of a wrong pick)
   │
   ▼  bootstrap over query-groups → 95% CI
   ▼  compare to FROZEN old baseline → improved / regressed / noise
```

---

## 1. Why GroupKFold, and why OOF (this caused the original confusion)

The reported "degradation" of the Phase 3E model was largely an
**evaluation-methodology artifact**: optimistic in-sample numbers were
compared against honest out-of-fold numbers. We never repeat that.

- **GroupKFold by `query_id`:** all 4 variants of a query stay in the
  *same* fold. Otherwise the model sees `q03_p4/default` in train and
  `q03_p4/no_hashjoin` in test — near-identical rows — and **leaks**,
  inflating accuracy. Plain `KFold` would do exactly this.
- **Out-of-fold predictions:** we score each row only when it was in the
  held-out fold. This is what the deployed model actually faces on a
  query it has never seen. In-sample scores are vanity.
- **Why not a single train/test split:** with ~220 groups, one split is
  high-variance and wastes data. 5-fold uses every group as test once.

---

## 2. The metric panel — what each one tells you (and hides)

| Metric | Question it answers | Blind spot |
|---|---|---|
| **Plan-pick accuracy** | did we pick the truly fastest variant per query? | binary per group; ignores *how much* slower a wrong pick is |
| **Regret (ms / p95)** | when we pick wrong, how much time do we lose? | complements plan-pick — covers its blind spot |
| **Median q-error** | typical multiplicative prediction error | hides the tail |
| **p95 q-error** | worst-case calibration | sensitive to a few hard plans |
| **MAE / RMSE (ms)** | absolute error in real units | dominated by slow queries; RMSE punishes big misses |
| **Spearman ρ** | are predicted runtimes *rank-correlated* with truth? | rank-only; ignores magnitude |

We report **all** of them because no single number is sufficient: a
model can have great MAE but bad plan-pick (it's accurate on easy
queries, wrong on the close calls that matter), or great plan-pick but
poor q-error (it ranks right but mis-predicts magnitudes).

---

## 3. Why plan-pick is the **selection** objective

The product question is *"which plan should I run?"* — a **ranking within
a query group**, not an absolute runtime. So:

- **Select the deployment model on plan-pick accuracy** (tie-break on
  q-error). This is already wired into `train_models.py`'s AutoML winner
  logic for the `plan_time` regime.
- **Why not lowest MAE/RMSE:** absolute error optimizes magnitude, not
  order. A model can shave MAE by nailing the slow queries while flipping
  the cheap, close races — hurting plan-pick.
- **Why not lowest q-error alone:** q-error is calibration; great
  calibration with wrong *order* still picks the wrong plan. (The
  original model-selection mistake was optimizing q-error.)
- **Why keep q-error/regret at all:** they're the diagnostics that
  explain *why* plan-pick moves and protect against pathological picks.

---

## 4. Phase 3H — is the change real? (confidence & significance)

Plan-pick accuracy is a proportion over ~220 query-groups, so a raw
"57% vs 55%" comparison is meaningless without a confidence interval.

- **Bootstrap CI:** resample the per-group hit/miss vector with
  replacement 2,000× and take the 2.5/97.5 percentiles of the mean. This
  gives a 95% CI on plan-pick **without distributional assumptions** —
  appropriate for a small, grouped dataset.
- **Pre-registered (frozen) baseline:** we hard-code the *old* numbers in
  the script —
  `OLD_PROD_PLAN_PICK = 0.576`, `OLD_ABLATION = 0.438…0.504` — **before**
  seeing the new result, so we can't move the goalposts.
- **Verdict rule:**
  - new CI **entirely above** 0.576 → improved & significant
  - new CI **entirely below** → regressed & significant
  - CI **spans** 0.576 → inside the noise floor (no significant change)

> The prior ablation showed feature tweaks lived inside a **±8.5 pp** CI.
> That's precisely why Phase 3E.2 grows the dataset: a tighter CI is what
> makes the *next* real lever (LambdaRank) produce a *measurable* win.

### Alternatives considered
- ❌ **t-test on per-fold accuracy:** only 5 folds → tiny n, assumes
  normality; bootstrap over groups uses all the information.
- ❌ **No CI, just point estimates:** how the project got confused in the
  first place. Never again.

---

## 5. 3F vs 3G — keep the roles distinct

| Phase | Script | Purpose | Data used |
|---|---|---|---|
| **3F training** | `train_models.py` | fit the **deployable** artifact + Optuna tuning | full data (final fit) |
| **3G evaluation** | `evaluate_baseline.py` | **honest OOF** comparison for **model selection** | GroupKFold OOF |

Never select the deployment model on in-sample numbers. 3F produces the
artifact; 3G/3H decide whether it's allowed to be the new baseline.

### Model roster note
The OOF harness (`make_models`) currently covers `extra_trees`,
`random_forest`, `xgboost`, `lightgbm`. CatBoost / Ridge / Lasso / Linear
can be added as honest baselines, but on a log-target with
tree-friendly, interaction-heavy features the tree ensembles are expected
to win — linear models serve as a floor, not a contender.

---

## 6. What we will NOT claim

Until 3G completes on the corrected, validated dataset, the only honest
statement is:

> "The benchmark has been corrected, the dataset regenerated, and the
> model will be re-evaluated on a reproducible dataset."

A specific accuracy number is reported **only** with its 95% CI and its
verdict against the frozen baseline.

---

## 7. How to run

```bash
python scripts/experiment_ablation.py    # feature-subset × model ablation (OOF plan-pick)
python scripts/evaluate_baseline.py       # full metric panel + CI + significance verdict
# → reports/phase3b/ablation_3e.csv , reports/phase3b/baseline_eval.csv
```

---

## 8. Only after this: improvements (controlled experiments)

With a trustworthy baseline + tight CI, each improvement becomes a
*controlled experiment* measured the same way:

- **More data** (TPC-DS depth, JOB) → expect tighter CI / better
  generalization.
- **Better labels** (`--label-runs 5`) → expect lower q-error variance.
- **LambdaRank** (listwise ranking) → the one lever with expected gain
  *above* the noise floor; evaluated head-to-head vs. pointwise
  regression on the identical GroupKFold split.
