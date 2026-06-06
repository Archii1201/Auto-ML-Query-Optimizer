# Phase 3A — Model Comparison

5-fold **GroupKFold** cross-validation grouped by `query_id` (no query appears in both train and test).

Models are trained on `log1p(execution_time_ms)` and evaluated on the back-transformed prediction; PostgreSQL baselines are uncalibrated (linear) and log-linear calibrated.

Sort: ascending **median q-error**, then ascending RMSE.

## Regime: `plan_time`

| regime    | kind     | model                 |      R² |   MAE (ms) |   RMSE (ms) |   MAPE (%) |   q-err median |   q-err p95 |   Spearman ρ |   Train (s) |
|:----------|:---------|:----------------------|--------:|-----------:|------------:|-----------:|---------------:|------------:|-------------:|------------:|
| plan_time | baseline | pg_baseline_linear    |  -0.402 |    4608.01 |     7680.31 |    116.1   |          1.898 |       9.608 |        0.388 |       0.016 |
| plan_time | baseline | pg_baseline_loglinear |  -0.265 |    4574.1  |     7369.33 |    114.659 |          1.985 |       8.667 |        0.388 |       0.016 |
| plan_time | ml       | catboost              |  -0.184 |    4659.52 |     7233.93 |    117.278 |          2.137 |       7.268 |        0.519 |      10.301 |
| plan_time | ml       | xgboost               |  -0.64  |    5311.82 |     8061.52 |    125.678 |          2.344 |       8.291 |        0.392 |       5.75  |
| plan_time | ml       | extra_trees           |  -0.418 |    4938.81 |     7799.48 |    127.685 |          2.376 |       9.563 |        0.502 |       3.05  |
| plan_time | ml       | random_forest         |  -0.283 |    5063.51 |     7472.21 |    116.585 |          2.409 |       6.908 |        0.368 |       3.86  |
| plan_time | ml       | lightgbm              |  -0.773 |    5309.14 |     8195.83 |    132.248 |          2.451 |       9.787 |        0.424 |      15.546 |
| plan_time | ml       | gradient_boosting     |  -0.337 |    5076.85 |     7507.98 |    122.296 |          2.563 |       6.82  |        0.437 |       1.89  |
| plan_time | ml       | ridge                 |  -0.69  |    5554.24 |     8297.52 |    244.35  |          3.112 |      41.546 |       -0.253 |       0.031 |
| plan_time | ml       | lasso                 |  -2.256 |    7069.32 |    10295.3  |    423.204 |          3.702 |     229.9   |       -0.415 |       0.102 |
| plan_time | ml       | linear_regression     | -91.747 |   25484.3  |    49996.1  |    556.485 |          6.939 |     114.881 |       -0.129 |       0.033 |

## Regime: `post_mortem`

| regime      | kind     | model                 |      R² |   MAE (ms) |   RMSE (ms) |   MAPE (%) |   q-err median |   q-err p95 |   Spearman ρ |   Train (s) |
|:------------|:---------|:----------------------|--------:|-----------:|------------:|-----------:|---------------:|------------:|-------------:|------------:|
| post_mortem | ml       | gradient_boosting     |   0.98  |    401.099 |     903.325 |      6.056 |          1.03  |       1.275 |        0.999 |       2.852 |
| post_mortem | ml       | extra_trees           |   0.968 |    538.432 |    1179.47  |      8.861 |          1.039 |       1.391 |        0.996 |       4.038 |
| post_mortem | ml       | random_forest         |   0.96  |    605.058 |    1289.56  |     10.261 |          1.059 |       1.454 |        0.994 |       5.071 |
| post_mortem | ml       | xgboost               |   0.956 |    737.719 |    1349.67  |     10.706 |          1.082 |       1.366 |        0.996 |       5.739 |
| post_mortem | ml       | lightgbm              |   0.956 |    660.038 |    1336.1   |     14.017 |          1.089 |       1.592 |        0.988 |      19.703 |
| post_mortem | ml       | catboost              |   0.754 |   1870.45  |    3417.4   |     31.305 |          1.225 |       2.284 |        0.975 |      13.919 |
| post_mortem | baseline | pg_baseline_linear    |  -0.402 |   4608.01  |    7680.31  |    116.1   |          1.898 |       9.608 |        0.388 |       0.018 |
| post_mortem | baseline | pg_baseline_loglinear |  -0.265 |   4574.1   |    7369.33  |    114.659 |          1.985 |       8.667 |        0.388 |       0.016 |
| post_mortem | ml       | ridge                 |  -6.205 |   6671.08  |   15130     |    159.315 |          2.475 |     317.278 |        0.375 |       0.036 |
| post_mortem | ml       | lasso                 | -15.644 |   9628.03  |   22434.7   |    438.481 |          3.179 |     307.258 |        0.309 |       0.158 |
| post_mortem | ml       | linear_regression     | -38.849 |  11938.7   |   24294.4   |    678.563 |          6.454 |    5404.35  |        0.132 |       0.037 |
