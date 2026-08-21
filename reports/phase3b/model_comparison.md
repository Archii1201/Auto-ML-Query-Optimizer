# Phase 3B — AutoML-Tuned Cost Model Leaderboard

5-fold **GroupKFold** CV grouped by `query_id`. Models trained on `log1p(execution_time_ms)`, scored on the back-transformed prediction.

**Sort key**: ascending median q-error, then ascending RMSE.

Columns:
- `kind`: `baseline` (calibrated PG cost), `default_ml` (3A defaults), `tuned_ml` (Optuna-tuned).
- `plan-pick acc`: fraction of `query_id` groups where the model picks the truly fastest variant.
- `regret (ms)`: average extra runtime paid vs. the oracle.

## AutoML winners

- **plan_time**: `xgboost` (q-err med = 1.24, plan-pick acc = 0.4338235294117647)

## Regime: `plan_time`

| regime    | kind       | model                 |    R² |   MAE (ms) |   RMSE (ms) |   q-err median |   q-err p95 |   Spearman ρ |   plan-pick acc |   regret (ms) |   Train (s) |
|:----------|:-----------|:----------------------|------:|-----------:|------------:|---------------:|------------:|-------------:|----------------:|--------------:|------------:|
| plan_time | default_ml | xgboost               | 0.622 |    1505.95 |     6382.2  |          1.242 |       2.8   |        0.955 |           0.434 |       519.194 |      12.566 |
| plan_time | default_ml | gradient_boosting     | 0.627 |    1473.13 |     6347.29 |          1.247 |       2.763 |        0.959 |           0.401 |       624.785 |      11.999 |
| plan_time | default_ml | lightgbm              | 0.61  |    1575.57 |     6493.45 |          1.255 |       3.58  |        0.95  |           0.39  |      1597.95  |      27.032 |
| plan_time | default_ml | catboost              | 0.632 |    1468.22 |     6324.28 |          1.26  |       5.193 |        0.959 |           0.423 |      1704.85  |      22.644 |
| plan_time | default_ml | random_forest         | 0.621 |    1518.68 |     6398.81 |          1.266 |       2.771 |        0.955 |           0.43  |       491.772 |       7.832 |
| plan_time | default_ml | extra_trees           | 0.6   |    1595.33 |     6541.85 |          1.267 |       2.914 |        0.949 |           0.426 |       461.992 |       4.968 |
| plan_time | default_ml | ridge                 | 0.349 |    2516.2  |     8016.38 |          1.496 |       5.144 |        0.883 |           0.338 |      2061.4   |       0.063 |
| plan_time | default_ml | elasticnet            | 0.356 |    2507.79 |     7981.28 |          1.501 |       5.09  |        0.883 |           0.346 |      2038.02  |       0.476 |
| plan_time | default_ml | lasso                 | 0.356 |    2508.84 |     7980.88 |          1.506 |       4.94  |        0.885 |           0.346 |      1994.68  |       0.406 |
| plan_time | baseline   | pg_baseline_loglinear | 0.105 |    3300.1  |     8507.45 |          1.842 |       9.584 |        0.685 |           0.265 |      2266.62  |       0.03  |
| plan_time | baseline   | pg_baseline_linear    | 0.111 |    3683.12 |     7658.96 |          1.909 |       7.65  |        0.685 |           0.265 |      2266.62  |       0.127 |
