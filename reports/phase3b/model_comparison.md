# Phase 3B — AutoML-Tuned Cost Model Leaderboard

5-fold **GroupKFold** CV grouped by `query_id`. Models trained on `log1p(execution_time_ms)`, scored on the back-transformed prediction.

**Sort key**: ascending median q-error, then ascending RMSE.

Columns:
- `kind`: `baseline` (calibrated PG cost), `default_ml` (3A defaults), `tuned_ml` (Optuna-tuned).
- `plan-pick acc`: fraction of `query_id` groups where the model picks the truly fastest variant.
- `regret (ms)`: average extra runtime paid vs. the oracle.

## AutoML winners

- **plan_time**: `extra_trees_tuned` (q-err med = 1.41, plan-pick acc = 0.48905109489051096)

## Regime: `plan_time`

| regime    | kind       | model                   |       R² |   MAE (ms) |   RMSE (ms) |   q-err median |   q-err p95 |   Spearman ρ |   plan-pick acc |   regret (ms) |   Train (s) |
|:----------|:-----------|:------------------------|---------:|-----------:|------------:|---------------:|------------:|-------------:|----------------:|--------------:|------------:|
| plan_time | tuned_ml   | extra_trees_tuned       |    0.593 |    2826.5  |     9186.99 |          1.406 |       3.824 |        0.881 |           0.489 |       365.125 |      26.215 |
| plan_time | tuned_ml   | random_forest_tuned     |    0.565 |    2806.02 |     9472.96 |          1.434 |       3.508 |        0.884 |           0.511 |       355.756 |       6.34  |
| plan_time | tuned_ml   | xgboost_tuned           |    0.614 |    2736.76 |     8929.65 |          1.438 |       3.924 |        0.889 |           0.518 |       442.027 |      16.421 |
| plan_time | default_ml | xgboost                 |    0.579 |    2899.48 |     9207.61 |          1.441 |       3.993 |        0.879 |           0.423 |       602.283 |       8.25  |
| plan_time | tuned_ml   | gradient_boosting_tuned |    0.538 |    2943.02 |     9671.23 |          1.448 |       3.515 |        0.874 |           0.438 |       559.341 |      34.553 |
| plan_time | default_ml | random_forest           |    0.584 |    2877.69 |     9227.56 |          1.451 |       3.468 |        0.879 |           0.438 |       628.103 |      17.497 |
| plan_time | default_ml | gradient_boosting       |    0.556 |    2848.33 |     9323.46 |          1.455 |       3.396 |        0.882 |           0.46  |       537.935 |      23.842 |
| plan_time | default_ml | catboost                |    0.608 |    2797.95 |     8992.82 |          1.457 |       3.628 |        0.886 |           0.445 |       550.841 |      19.409 |
| plan_time | tuned_ml   | lightgbm_tuned          |    0.572 |    2968.98 |     9303.76 |          1.461 |       3.864 |        0.881 |           0.431 |       640.208 |     150.488 |
| plan_time | default_ml | extra_trees             |    0.549 |    3050.84 |     9477.37 |          1.494 |       4.04  |        0.867 |           0.46  |       643.758 |      10.264 |
| plan_time | default_ml | lightgbm                |    0.519 |    3094.18 |     9554.24 |          1.535 |       3.77  |        0.86  |           0.423 |       739.136 |      43.569 |
| plan_time | default_ml | elasticnet              | -142.68  |    8981.48 |    63327    |          1.63  |       5.642 |        0.767 |           0.365 |      2877.95  |       0.756 |
| plan_time | default_ml | ridge                   | -142.697 |    8999.28 |    63351.3  |          1.631 |       5.54  |        0.764 |           0.401 |      2878.61  |       0.135 |
| plan_time | default_ml | lasso                   | -142.655 |    8931.38 |    63291.7  |          1.634 |       5.62  |        0.766 |           0.343 |      2883.41  |       1.207 |
| plan_time | baseline   | pg_baseline_loglinear   |    0.049 |    4487.22 |    11609.6  |          1.696 |      10.491 |        0.578 |           0.197 |      4223.05  |       0.06  |
| plan_time | baseline   | pg_baseline_linear      |    0.046 |    4629.07 |    10250.6  |          1.882 |       8.876 |        0.578 |           0.197 |      4223.05  |       0.062 |
