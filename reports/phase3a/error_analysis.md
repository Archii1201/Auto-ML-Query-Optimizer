# Phase 3A — Error Analysis

All metrics below come from the held-out fold of 5-fold **GroupKFold** CV (queries never appear in both train and test).

## Regime: `plan_time`

**Top 3 by median q-error:**

- `pg_baseline_linear` — q-err median **1.90**, R² -0.402, MAE 4608.0 ms
- `pg_baseline_loglinear` — q-err median **1.98**, R² -0.265, MAE 4574.1 ms
- `catboost` — q-err median **2.14**, R² -0.184, MAE 4659.5 ms

**10 worst predictions across all models in this regime:**

| model             | query_id   |   y_true |   y_pred |   q_error |
|:------------------|:-----------|---------:|---------:|----------:|
| lasso             | q09        | 23625.3  |    22.55 |   1047.84 |
| lasso             | q09        | 19454.6  |    22.77 |    854.48 |
| lasso             | q09        |  9471.81 |    24.53 |    386.08 |
| linear_regression | q21        | 13986.1  |    52.28 |    267.51 |
| linear_regression | q21        | 13055.6  |    51.47 |    253.65 |
| linear_regression | q21        | 19448.2  |    99.88 |    194.71 |
| lasso             | q21        | 13986.1  |    76.97 |    181.72 |
| linear_regression | q18        | 14970.4  |    85.09 |    175.93 |
| lasso             | q21        | 13055.6  |    75.32 |    173.33 |
| linear_regression | q16        |  1383.15 |     8.59 |    161.03 |

## Regime: `post_mortem`

**Top 3 by median q-error:**

- `gradient_boosting` — q-err median **1.03**, R² +0.980, MAE 401.1 ms
- `extra_trees` — q-err median **1.04**, R² +0.968, MAE 538.4 ms
- `random_forest` — q-err median **1.06**, R² +0.960, MAE 605.1 ms

**10 worst predictions across all models in this regime:**

| model             | query_id   |   y_true |   y_pred |   q_error |
|:------------------|:-----------|---------:|---------:|----------:|
| linear_regression | q09        | 30860.7  |      0.1 |  30860.7  |
| linear_regression | q09        | 23625.3  |      0.1 |  23625.3  |
| linear_regression | q09        | 19454.6  |      0.1 |  19454.6  |
| linear_regression | q09        |  9471.81 |      0.1 |   9471.81 |
| lasso             | q16        |  1605.61 |      0.1 |   1605.61 |
| linear_regression | q16        |  1605.61 |      0.1 |   1605.61 |
| ridge             | q16        |  1605.61 |      0.1 |   1605.61 |
| lasso             | q16        |  1383.15 |      0.1 |   1383.15 |
| linear_regression | q16        |  1383.15 |      0.1 |   1383.15 |
| ridge             | q16        |  1383.15 |      0.1 |   1383.15 |
