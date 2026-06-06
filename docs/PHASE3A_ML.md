# Phase 3A — Learned Cost Model Deep Dive

> Advanced reference for the first ML layer of the
> **AutoML-Powered Learned Query Optimizer**. Read alongside
> `docs/PHASE2B_FEATURES.md` (which defines the input CSV) and
> `docs/PHASE2A_TPCH.md` (which defines the workload). Phase 3A is
> where we finally answer the project's headline question:
> *can a learned model predict query execution time better than
> PostgreSQL's own cost formula?*

---

## 0. Mental model

```
data/processed/features.csv     reports/phase3a/model_comparison.csv
       (87 rows, 50 cols)              (ranked leaderboard)
              |                              ^
              v                              |
   ┌─────────────────────────┐               |
   │  feature_selection.py   │               |
   │  - drop ID columns      │               |
   │  - split SAFE vs LEAKY  │               |
   │  - one-hot categoricals │               |
   └────────────┬────────────┘               |
                v                              |
   ┌─────────────────────────┐               |
   │   train_models.py       │               |
   │  GroupKFold by query_id │               |
   │  log1p(target) regress  ├───────────────┘
   │  9 models x 2 regimes   │
   └────────────┬────────────┘
                v
   ┌─────────────────────────┐
   │     reports.py          │   plots/, error_analysis.md
   └─────────────────────────┘
```

Phase 3A trains a *bench* of nine ML regressors against a calibrated
PostgreSQL baseline, on **two feature regimes**:

- **`plan_time`** — only features the planner has *before* execution.
  This is the realistic, deployable model.
- **`post_mortem`** — every feature including post-execution
  observations. This is the upper-bound sanity ceiling.

By measuring both we get a hard answer to "how much accuracy do we
sacrifice by being honest about deployment constraints?"

---

## 1. What Phase 3A adds to the repo

```
auto-ml-query-optimizer/
├── phase3a/
│   ├── __init__.py
│   ├── feature_selection.py    # SAFE vs LEAKY vs ID column lists
│   ├── evaluation.py           # MAE, RMSE, R2, MAPE, q-error, Spearman
│   ├── baseline.py             # calibrated PostgreSQL cost baseline
│   ├── train_models.py         # 9 models x 2 regimes x GroupKFold
│   └── reports.py              # plots + error analysis
├── notebooks/
│   └── phase3a_eda.ipynb       # interactive EDA
├── reports/phase3a/            # all generated outputs
│   ├── model_comparison.csv
│   ├── model_comparison.md
│   ├── cv_predictions.csv
│   ├── feature_importance.csv
│   ├── error_analysis.md
│   └── plots/  (44 PNGs)
└── models/phase3a/
    ├── plan_time/   (11 .joblib files)
    └── post_mortem/ (11 .joblib files)
```

Hard rule we kept: **no Phase 1 / 2A / 2B file is edited.** We only
*read* `data/processed/features.csv` — the contract Phase 2B promised
us at the end of `PHASE2B_FEATURES.md` Section 11.

---

## 2. End-to-end usage

```powershell
# 1. Confirm the feature CSV exists
python feature_engineering/extract_features.py

# 2. Train all 9 models x 2 regimes (~2 minutes on a laptop)
python phase3a/train_models.py

# 3. Render plots and the error-analysis markdown
python phase3a/reports.py

# 4. (optional) Open the EDA notebook for interactive exploration
jupyter notebook notebooks/phase3a_eda.ipynb
```

Outputs land under `reports/phase3a/` and `models/phase3a/`. Both
folders are auto-created if missing. The pipeline is fully
deterministic given the same input CSV (`random_state=42` everywhere).

---

## 3. The four design decisions that drive everything

These weren't optional — each one fixes a real, measurable bug that
would otherwise make the metrics meaningless.

### 3.1 Target leakage → SAFE vs LEAKY columns

The features.csv contains **post-execution** measurements that are
nearly identical to the target:

| Column                         | Why it leaks                                  |
|--------------------------------|-----------------------------------------------|
| `actual_total_time_ms`         | literally the per-node version of the target |
| `actual_rows`                  | only known after running                      |
| `max_actual_loops`             | observed during execution                     |
| `total_rows_removed_by_filter` | counted while the filter ran                  |
| `parallel_worker_count`        | how many workers PG actually launched         |
| `sum_shared_*` / `sum_temp_*`  | buffer counters from the run                  |
| `wall_time_ms`                 | client-side wall clock                        |
| `target_execution_time_ms`     | duplicate of the target                       |

A real query optimizer can't see any of these at plan time. We
declare them as `LEAKY_COLUMNS` in `feature_selection.py` and drop
them in the realistic `plan_time` regime. We *keep* them in
`post_mortem` only as an upper-bound sanity check.

### 3.2 GroupKFold by `query_id`

Plain 5-fold CV puts `q05/default` in train and `q05/no_hashjoin` in
test — same query, same data, same join graph. The model learns to
memorise queries, not to predict on unseen ones. We use
`GroupKFold(groups=query_id)` so that all 4 variants of any given
query stay on the *same* side of every fold split. This is exactly
the train/test contract a deployed cost model faces in production.

### 3.3 Log-target regression

`execution_time_ms` spans 0.5 ms to 31,000 ms in this dataset (4.8
orders of magnitude). Linear MSE is dominated by the slowest 5
queries; every model otherwise just learns "predict the median".

We train every model on `log1p(execution_time_ms)` and back-transform
predictions with `expm1` before computing metrics. Two-line change,
massive accuracy gain (post-mortem boosters go from R^2 ~ 0.4 to
~ 0.97).

### 3.4 Q-error as the primary metric

MAE / RMSE are absolute-scale metrics — easy to read but biased
toward large queries. R^2 can go arbitrarily negative on a single
bad fold. The query-optimization community uses
**q-error = max(pred/actual, actual/pred)** because it is:

- Symmetric (over- and under-estimating by 2x are equally bad).
- Multiplicative (a model that's always 10x off has q-error = 10
  regardless of query size).
- Reported as median + p95, so a single blown-up prediction can't
  hide systematic improvement.

We report it alongside the standard metrics. **Q-error median is the
column we sort the leaderboard by.**

---

## 4. Module-by-module deep dive

### 4.1 `phase3a/feature_selection.py`

The single source of truth for which columns are features, leaks, or
identifiers. Everything else in the project asks this module
"what's safe to use?"

```python
ID_COLUMNS = ("source_file", "query_id", "variant", "tag",
              "sql_hash", "collected_at")
LEAKY_COLUMNS = ("actual_rows", "actual_total_time_ms",
                 "max_actual_loops", "total_rows_removed_by_filter",
                 "parallel_worker_count", "sum_shared_hit_blocks",
                 "sum_shared_read_blocks", "sum_temp_read_blocks",
                 "sum_temp_written_blocks", "wall_time_ms",
                 "target_execution_time_ms")
CATEGORICAL_COLUMNS = ("root_node_type",)
```

The public entry point `build_feature_matrix(df, regime)` returns a
`FeatureMatrix` dataclass containing `(X, y, groups, feature_names,
regime)`. It:

1. Drops ID columns + the target.
2. In `plan_time` mode, additionally drops every LEAKY column.
3. One-hot encodes `root_node_type` so linear models can use it.
4. Coerces every column to numeric; fills NaN with 0 (rare — only
   when an EXPLAIN key was missing for a particular plan).
5. Optionally drops zero-variance columns (some operator counts are
   zero for every plan in TPC-H — they carry no signal and can break
   `StandardScaler`).
6. Returns the ready-to-train matrix as `float64`.

The `describe_regime_split(df)` helper produces a human-readable
table classifying every column as `identifier / target /
leaky / plan-time feature`. Used by the EDA notebook.

### 4.2 `phase3a/evaluation.py`

Defines the `Metrics` dataclass and a single `compute_metrics(y_true,
y_pred)` function that computes all six metrics in one pass:

| Metric          | Formula                                          |
|-----------------|--------------------------------------------------|
| `mae`           | mean of \|pred - actual\|                        |
| `rmse`          | sqrt(mean of (pred - actual)^2)                  |
| `r2`            | sklearn r2_score                                 |
| `mape_pct`      | mean of \|pred - actual\| / max(actual, 1ms)     |
| `q_error_median`| median of q_error per row                        |
| `q_error_p95`   | 95th percentile of q_error                       |
| `spearman_rho`  | rank correlation between pred and actual         |
| `n`             | number of rows                                   |

Two defensive details:

- `MIN_TIME_MS = 1.0` — floor used in MAPE and q-error so that
  sub-millisecond queries don't make the ratios infinite.
- `compute_metrics` clips predictions to be finite and non-negative
  *before* computing metrics. This catches a single blown-up linear-
  regression prediction without hiding systematic errors.

`average_fold_metrics(per_fold)` reduces a list of `Metrics` (one per
CV fold) into a `mean / std` dictionary the leaderboard CSV consumes.

### 4.3 `phase3a/baseline.py`

Implements the **PostgreSQL native cost model** as a sklearn-style
regressor so it lands in the same evaluation loop as the ML models.

`Total Cost` from EXPLAIN is in arbitrary planner units (mostly
calibrated to "the cost of a sequential page read"). To compare it
to milliseconds we have to fit a calibration constant on training
data and apply it to test:

```python
class LinearCostBaseline:
    """time_ms ≈ k * estimated_total_cost"""
    def fit(self, X, y):
        cost = X["estimated_total_cost"].values
        self.k_ = (cost * y).sum() / (cost ** 2).sum()
        return self

class LogLinearCostBaseline:
    """log1p(time_ms) ≈ a + b * log1p(estimated_total_cost)"""
    def fit(self, X, y):
        log_cost = np.log1p(X["estimated_total_cost"].values)
        self.b_, self.a_ = np.polyfit(log_cost, np.log1p(y), 1)
        return self
```

Both are deliberately tiny (no scikit-learn dependency, no
intercepts on the linear version) so the baseline has *no* extra
flexibility beyond "scale PG's cost number to milliseconds". This is
the fairest possible comparison — any improvement an ML model shows
is genuinely from the extra features, not from the regressor having
more knobs.

### 4.4 `phase3a/train_models.py`

The driver. For each `(regime, model)` pair:

1. Build the feature matrix for that regime.
2. Run a 5-fold `GroupKFold(query_id)` cross-validation:
   - Fit the model on `log1p(y_train)`.
   - Predict, back-transform via `expm1`.
   - Clip predictions to `[0.1, 10 * max(y_full)]` ms so a single
     unstable fold can't destroy the cross-fold mean.
   - Compute all metrics on the held-out fold.
3. Store per-fold predictions to `cv_predictions.csv` (so reports.py
   can build pred-vs-actual scatter plots later).
4. Refit the model on the **full dataset** and persist the artifact
   to `models/phase3a/{regime}/{name}.joblib` — bundled with its
   feature names and `log_target` flag so a future inference script
   knows how to use it.

The model zoo:

| Model              | Class                              | Pipeline    |
|--------------------|------------------------------------|-------------|
| linear_regression  | sklearn LinearRegression           | + StandardScaler |
| ridge              | sklearn Ridge(alpha=1.0)           | + StandardScaler |
| lasso              | sklearn Lasso(alpha=0.001)         | + StandardScaler |
| random_forest      | sklearn RandomForestRegressor      | raw         |
| extra_trees        | sklearn ExtraTreesRegressor        | raw         |
| gradient_boosting  | sklearn GradientBoostingRegressor  | raw         |
| xgboost            | xgboost XGBRegressor               | raw         |
| lightgbm           | lightgbm LGBMRegressor             | raw         |
| catboost           | catboost CatBoostRegressor         | raw         |

Linear models go through a `StandardScaler` because their loss is
sensitive to feature magnitude (`estimated_total_cost` is in tens
of thousands; counts are 0-15). Tree-based models are scale-
invariant and need no preprocessing.

### 4.5 `phase3a/reports.py`

Pure rendering — reads the artifacts produced by `train_models.py`
and writes:

- `runtime_distribution.png` — linear and log10 histograms of the
  target. Documents the multi-order-of-magnitude span.
- `correlation_heatmap.png` — top-20 `|corr|` features against the
  target, with leaky column labels coloured **red** so it's instantly
  obvious which "strong correlations" are cheating.
- `pred_vs_actual__{regime}__{model}.png` — log-log scatter of
  predicted vs. actual on the held-out CV folds, with the 1x and
  10x error bands drawn for reference. One per (regime, model) =
  22 plots total.
- `feature_importance__{regime}__{model}.png` — top-20 horizontal
  bar charts. One per tree/linear model with extractable importance,
  18 plots total.
- `error_analysis.md` — markdown summary listing the top-3 models per
  regime and the 10 worst predictions in each, so you can see
  *which queries* are hard and *which models* fail on them.

### 4.6 `notebooks/phase3a_eda.ipynb`

Interactive exploration. Six sections:

1. Dataset profile (shape, distinct queries, plans-per-query).
2. Target distribution (linear + log10).
3. Feature regime split (a table showing what's identifier / target
   / leaky / plan-time).
4. Top-20 correlation bar chart with leaky features highlighted.
5. Per-query variant spread (`max/min` runtime ratio per query) —
   shows that `enable_<join>=off` *did* meaningfully change runtime
   for most queries.
6. Plan-shape diversity (how many distinct `root_node_type` values
   each query produced across its 4 variants).

---

## 5. Real results — what actually came out of training

Numbers below are the **5-fold GroupKFold cross-validated means**
from the actual `python phase3a/train_models.py` run on
`data/processed/features.csv` (87 plans, 22 unique queries).

### 5.1 Plan-time regime (the realistic, deployable model)

Sorted ascending by **median q-error** (lower = better):

| Rank | Model                  | Kind     | q-err median | q-err p95 | MAE (ms)  | RMSE (ms) | R²      | Spearman ρ |
|------|------------------------|----------|--------------|-----------|-----------|-----------|---------|------------|
| 1    | **pg_baseline_linear** | baseline | **1.90**     | 9.61      | 4608      | 7680      | -0.40   | +0.39      |
| 2    | pg_baseline_loglinear  | baseline | 1.99         | 8.67      | 4574      | 7369      | -0.27   | +0.39      |
| 3    | catboost               | ml       | 2.14         | 7.27      | 4660      | 7234      | -0.18   | +0.52      |
| 4    | xgboost                | ml       | 2.34         | 8.29      | 5312      | 8062      | -0.64   | +0.39      |
| 5    | extra_trees            | ml       | 2.38         | 9.56      | 4939      | 7799      | -0.42   | +0.50      |
| 6    | random_forest          | ml       | 2.41         | 6.91      | 5064      | 7472      | -0.28   | +0.37      |
| 7    | lightgbm               | ml       | 2.45         | 9.79      | 5309      | 8196      | -0.77   | +0.42      |
| 8    | gradient_boosting      | ml       | 2.56         | 6.82      | 5077      | 7508      | -0.34   | +0.44      |
| 9    | ridge                  | ml       | 3.11         | 41.55     | 5554      | 8298      | -0.69   | -0.25      |
| 10   | lasso                  | ml       | 3.70         | 229.90    | 7069      | 10295     | -2.26   | -0.42      |
| 11   | linear_regression      | ml       | 6.94         | 114.88    | 25484     | 49996     | -91.75  | -0.13      |

### 5.2 Post-mortem regime (sanity ceiling — uses leaky features)

| Rank | Model                  | Kind     | q-err median | q-err p95 | MAE (ms)  | RMSE (ms) | R²      | Spearman ρ |
|------|------------------------|----------|--------------|-----------|-----------|-----------|---------|------------|
| 1    | **gradient_boosting**  | ml       | **1.03**     | 1.28      | 401       | 903       | +0.98   | +0.999     |
| 2    | extra_trees            | ml       | 1.04         | 1.39      | 538       | 1179      | +0.97   | +0.996     |
| 3    | random_forest          | ml       | 1.06         | 1.45      | 605       | 1290      | +0.96   | +0.994     |
| 4    | xgboost                | ml       | 1.08         | 1.37      | 738       | 1350      | +0.96   | +0.996     |
| 5    | lightgbm               | ml       | 1.09         | 1.59      | 660       | 1336      | +0.96   | +0.988     |
| 6    | catboost               | ml       | 1.23         | 2.28      | 1870      | 3417      | +0.75   | +0.975     |
| 7    | pg_baseline_linear     | baseline | 1.90         | 9.61      | 4608      | 7680      | -0.40   | +0.39      |
| 8    | pg_baseline_loglinear  | baseline | 1.99         | 8.67      | 4574      | 7369      | -0.27   | +0.39      |
| 9    | ridge                  | ml       | 2.48         | 317.28    | 6671      | 15130     | -6.21   | +0.38      |
| 10   | lasso                  | ml       | 3.18         | 307.26    | 9628      | 22435     | -15.64  | +0.31      |
| 11   | linear_regression      | ml       | 6.45         | 5404.35   | 11939     | 24294     | -38.85  | +0.13      |

### 5.3 What these numbers actually mean

Three honest takeaways:

**1. PostgreSQL's calibrated cost formula beats every ML model in the
plan-time regime.** The linear baseline lands at q-error median
**1.90** — meaning the typical prediction is off by ~1.9x in either
direction. The best ML model (`catboost`) lands at 2.14, and most
others are worse. This is a *real* finding, not a bug:

- We have only 22 distinct queries. With GroupKFold, 4-5 of them are
  held out per fold — the model has to extrapolate to entirely
  unseen query shapes from ~17 training queries.
- PG's cost formula is decades of tuned domain knowledge. Beating it
  with 17 training examples and zero hyperparameter tuning was never
  realistic.
- This gives us a clear, honest **research target for Phase 3B**:
  collect more workloads (JOB, Stack, custom queries), tune
  hyperparameters, try plan-tree GNNs.

**2. Tree boosters dominate the post-mortem regime — because they
have access to `actual_total_time_ms`.** Gradient Boosting hits
R^2 = 0.98 and q-error = 1.03 (near-perfect predictions). This is
**by design useless for deployment** — at plan time the actual
runtime is exactly what we're trying to predict. Post-mortem
results define the ceiling: "if we knew runtime exactly, what's the
best a model could do given everything else?" The gap between 1.03
(post-mortem) and 2.14 (plan-time) is the **information gap** we
need to close in Phase 3B.

**3. Linear models are unstable on this dataset.** Plain
LinearRegression hit q-error = 6.94 with massive variance across
folds (one fold predicted 30 trillion ms before clipping). With 31
features and ~70 training rows, the design matrix is borderline
ill-conditioned. Ridge / Lasso help but still underperform tree
models. **Lesson: regularised linear is fine, plain OLS isn't.**

---

## 6. Feature importance — what the best plan-time model learned

CatBoost (best ML model in the realistic regime) ranks features by
gain like this (top 10):

| Rank | Feature                  | Importance |
|------|--------------------------|------------|
| 1    | `estimated_total_cost`   | 15.32      |
| 2    | `max_subtree_cost`       | 13.14      |
| 3    | `estimated_startup_cost` | 9.85       |
| 4    | `planning_time_ms`       | 6.84       |
| 5    | `num_scans`              | 5.88       |
| 6    | `bitmap_heap_scan_count` | 5.46       |
| 7    | `estimated_rows`         | 4.78       |
| 8    | `index_scan_count`       | 4.49       |
| 9    | `aggregate_count`        | 3.84       |
| 10   | `bitmap_index_scan_count`| 3.26       |

This ordering is **intuitively correct**: PG's own cost numbers
(`estimated_total_cost`, `max_subtree_cost`, `estimated_startup_cost`)
are the strongest signals; structural counts and operator-shape
features come next; categorical `root_node_type__*` indicators show
up in the tail.

The fact that the model puts so much weight on the very columns
that the PostgreSQL baseline already uses is consistent with the
finding above: with only 22 queries, the ML model is mostly *re-
learning* the planner's cost formula rather than discovering new
signal.

---

## 7. Reproducibility checklist

```powershell
# Full reproduction from scratch
del /S /Q reports\phase3a 2>$null
del /S /Q models\phase3a 2>$null
python phase3a/train_models.py
python phase3a/reports.py
```

Determinism guarantees:

- `RANDOM_STATE = 42` set on every model (where applicable).
- `GroupKFold` is deterministic given the same group order — and
  group order is determined by row order in the CSV.
- No multithreaded shuffling; tree models use `n_jobs=-1` only
  during fit.
- Features.csv is regenerated by Phase 2B in deterministic file
  order (sorted glob).

If you re-run and metrics differ:

1. Did `data/processed/features.csv` change? Inspect with
   `git diff` — Phase 3A is deterministic given its input.
2. Did the package versions move? Pin `requirements.txt` and
   `pip freeze | findstr "scikit xgboost lightgbm catboost"` — minor
   versions of LightGBM in particular have shifted defaults
   between 4.0 and 4.5.
3. Did you add new queries between runs? More groups → different
   fold assignments.

---

## 8. Common pitfalls

- **"My R^2 is amazing!" → check the regime.** If it's `post_mortem`,
  the model is using `actual_total_time_ms` and you've fooled
  yourself. Always quote the `plan_time` row.
- **Negative R^2 on plan-time.** Expected on a 22-query dataset with
  GroupKFold — predicting the mean is often better than extrapolating
  from 17 queries to 5 unseen ones. Q-error and Spearman ρ are more
  informative metrics here.
- **A new model class explodes.** Wrap it in a `Pipeline([("scaler",
  StandardScaler()), ("est", ...)])` and pick a regularised variant.
  The pipeline already handles this for the linear models.
- **`pd.to_markdown` raises ImportError.** Install `tabulate` (it's
  pinned in `requirements.txt`).
- **A folder named `models/` already exists from somewhere else.**
  Phase 3A writes into `models/phase3a/`, namespaced. It won't
  collide with anything else, but check the joblib paths.
- **Plot rendering fails with Agg backend errors.** This usually
  means `matplotlib` was imported before `matplotlib.use("Agg")`.
  `reports.py` calls `use("Agg")` *before* `import matplotlib.pyplot`,
  but if you import it interactively (e.g. notebook) the order
  changes; just restart the kernel.

---

## 9. What Phase 3B will plug into

Phase 3A's output contract:

- **Trained artifacts:** `models/phase3a/{regime}/{model}.joblib`,
  each containing `{model, feature_names, regime, log_target,
  trained_at}` — enough to deserialise and predict in production
  without retraining.
- **CV predictions:** `reports/phase3a/cv_predictions.csv` with
  `(fold, model, regime, query_id, y_true, y_pred)` — directly
  consumable for ensembling, stacking, or rank-loss re-training.
- **Leaderboard:** `reports/phase3a/model_comparison.csv` — a
  versioned baseline that any future model must beat.
- **Feature importance:** `reports/phase3a/feature_importance.csv` —
  a starting point for feature pruning experiments.

Phase 3B will likely add:

- Hyperparameter search via Optuna (the "AutoML" in the project
  name).
- Plan-tree neural networks (Tree-LSTM / GNN over the JSON tree
  rather than the flattened feature vector).
- A **ranking objective** (LambdaRank / pairwise) instead of
  point-wise MSE, since what matters for a query optimizer is
  *plan ranking*, not absolute time prediction.
- More workloads (JOB, Stack, custom queries) so GroupKFold has more
  groups to learn from.

---

## 10. TL;DR

- **9 ML regressors + 2 PG baselines, evaluated under
  GroupKFold-by-query** so no model gets to memorise queries.
- **Two regimes:** `plan_time` (realistic) and `post_mortem`
  (sanity ceiling, leaky-by-design).
- **Q-error median** is the leaderboard sort key.
- **Headline result:** PostgreSQL's calibrated linear cost baseline
  (q-err median **1.90**) beats every ML model in the realistic
  regime. CatBoost is the best ML model at q-err **2.14**. The
  gap to the post-mortem ceiling (q-err **1.03**) is the
  information-and-data deficit we close in Phase 3B.
- **All artifacts** (44 plots, leaderboard, error analysis, 22
  saved model files) are reproducible from a single
  `python phase3a/train_models.py && python phase3a/reports.py`.
