# Phase 3B — AutoML-Tuned Cost Model Leaderboard

5-fold **GroupKFold** CV grouped by `query_id`. Models trained on `log1p(execution_time_ms)`, scored on the back-transformed prediction.

**Sort key**: ascending median q-error, then ascending RMSE.

Columns:
- `kind`: `baseline` (calibrated PG cost), `default_ml` (3A defaults), `tuned_ml` (Optuna-tuned).
- `plan-pick acc`: fraction of `query_id` groups where the model picks the truly fastest variant.
- `regret (ms)`: average extra runtime paid vs. the oracle.

## AutoML winners

- **plan_time**: `lightgbm_tuned` (q-err med = 1.39, plan-pick acc = 0.5075757575757576)
- **post_mortem**: `gradient_boosting_tuned` (q-err med = 1.01, plan-pick acc = 0.9621212121212122)

## Regime: `plan_time`

| regime    | kind       | model                   |       R² |   MAE (ms) |   RMSE (ms) |   q-err median |   q-err p95 |   Spearman ρ |   plan-pick acc |   regret (ms) |   Train (s) |
|:----------|:-----------|:------------------------|---------:|-----------:|------------:|---------------:|------------:|-------------:|----------------:|--------------:|------------:|
| plan_time | tuned_ml   | lightgbm_tuned          |    0.506 |    2940.3  |     9867.03 |          1.394 |       3.257 |        0.894 |           0.508 |       586.62  |      23.226 |
| plan_time | tuned_ml   | gradient_boosting_tuned |    0.5   |    2918.21 |     9993.96 |          1.406 |       3.322 |        0.892 |           0.477 |      2752.64  |      10.891 |
| plan_time | tuned_ml   | extra_trees_tuned       |    0.533 |    2886.9  |     9605.9  |          1.407 |       3.679 |        0.892 |           0.485 |       412.247 |      51.742 |
| plan_time | tuned_ml   | random_forest_tuned     |    0.516 |    2911.5  |     9816.08 |          1.408 |       3.609 |        0.899 |           0.477 |      2650.49  |       6.473 |
| plan_time | default_ml | gradient_boosting       |    0.477 |    3029.98 |    10067.5  |          1.418 |       3.717 |        0.886 |           0.455 |       685.625 |      16.621 |
| plan_time | tuned_ml   | xgboost_tuned           |    0.505 |    2954.13 |     9865.55 |          1.42  |       3.898 |        0.889 |           0.485 |       629.745 |      13.703 |
| plan_time | default_ml | catboost                |    0.484 |    3034.47 |    10083.4  |          1.43  |       3.378 |        0.887 |           0.447 |      2794.92  |      29.684 |
| plan_time | default_ml | random_forest           |    0.501 |    3058.46 |     9941.2  |          1.457 |       3.79  |        0.883 |           0.455 |       745.897 |       6.497 |
| plan_time | default_ml | xgboost                 |    0.477 |    3090.83 |    10091.1  |          1.466 |       3.741 |        0.878 |           0.455 |       699.613 |      11.94  |
| plan_time | default_ml | extra_trees             |    0.488 |    3205.68 |     9988.87 |          1.485 |       3.368 |        0.867 |           0.394 |       686.555 |       9.392 |
| plan_time | default_ml | lightgbm                |    0.449 |    3273.91 |    10314.8  |          1.496 |       3.733 |        0.872 |           0.47  |       781.55  |      33.626 |
| plan_time | default_ml | lasso                   | -180.501 |    9697.92 |    65272.5  |          1.643 |      29.175 |        0.73  |           0.333 |      3138.22  |       0.427 |
| plan_time | baseline   | pg_baseline_loglinear   |    0.055 |    4587.92 |    11561.2  |          1.645 |       8.938 |        0.568 |           0.205 |      4366.26  |       0.067 |
| plan_time | default_ml | elasticnet              | -180.489 |    9681.89 |    65236.5  |          1.659 |      28.68  |        0.728 |           0.356 |      3067.45  |       0.462 |
| plan_time | default_ml | ridge                   | -180.45  |    9646.2  |    65096    |          1.664 |      20.67  |        0.718 |           0.326 |      3217.67  |       0.11  |
| plan_time | baseline   | pg_baseline_linear      |    0.049 |    4666.96 |    10392.4  |          1.786 |       8.346 |        0.568 |           0.205 |      4366.26  |       0.106 |

## Regime: `post_mortem`

| regime      | kind       | model                   |      R² |   MAE (ms) |   RMSE (ms) |   q-err median |   q-err p95 |   Spearman ρ |   plan-pick acc |   regret (ms) |   Train (s) |
|:------------|:-----------|:------------------------|--------:|-----------:|------------:|---------------:|------------:|-------------:|----------------:|--------------:|------------:|
| post_mortem | tuned_ml   | gradient_boosting_tuned |   0.875 |    586.172 |     5100.04 |          1.005 |       1.104 |        1     |           0.962 |         3.304 |      71.977 |
| post_mortem | default_ml | gradient_boosting       |   0.872 |    580.785 |     4955.15 |          1.008 |       1.103 |        1     |           0.947 |         0.74  |      27.937 |
| post_mortem | default_ml | random_forest           |   0.875 |    580.21  |     4974.43 |          1.008 |       1.226 |        1     |           0.977 |         0.166 |      54.253 |
| post_mortem | tuned_ml   | random_forest_tuned     |   0.858 |    653.484 |     5431.73 |          1.008 |       1.54  |        1     |           0.97  |         0.157 |      80.067 |
| post_mortem | tuned_ml   | extra_trees_tuned       |   0.843 |    642.135 |     5482.44 |          1.008 |       1.238 |        0.999 |           0.97  |         0.637 |      82.825 |
| post_mortem | default_ml | extra_trees             |   0.842 |    646.666 |     5494.6  |          1.008 |       1.234 |        0.999 |           0.985 |         0.123 |      52.599 |
| post_mortem | tuned_ml   | xgboost_tuned           |   0.873 |    622.452 |     5060.14 |          1.011 |       1.178 |        1     |           0.962 |         3.069 |      37.377 |
| post_mortem | default_ml | lightgbm                |   0.869 |    689.782 |     5283.42 |          1.011 |       1.156 |        0.999 |           0.955 |        10.991 |     122.135 |
| post_mortem | default_ml | xgboost                 |   0.887 |    580.402 |     4694.84 |          1.013 |       1.215 |        1     |           0.947 |         2.922 |      13.14  |
| post_mortem | tuned_ml   | lightgbm_tuned          |   0.862 |    784.758 |     5463.99 |          1.023 |       1.21  |        0.999 |           0.939 |        11.642 |     135.591 |
| post_mortem | default_ml | catboost                |   0.822 |    911.903 |     6164.64 |          1.042 |       1.444 |        0.997 |           0.947 |        13.3   |      74.329 |
| post_mortem | default_ml | lasso                   | -20.436 |   8310.89  |    66708.2  |          1.353 |       3.828 |        0.926 |           0.689 |       335.716 |       5.343 |
| post_mortem | default_ml | elasticnet              | -20.483 |   8331.97  |    66791.5  |          1.353 |       4.472 |        0.923 |           0.689 |       335.692 |       2.021 |
| post_mortem | default_ml | ridge                   | -20.303 |   8306.5   |    66506    |          1.361 |       7.421 |        0.921 |           0.697 |       335.564 |       0.675 |
| post_mortem | baseline   | pg_baseline_loglinear   |   0.055 |   4587.92  |    11561.2  |          1.645 |       8.938 |        0.568 |           0.205 |      4366.26  |       1.005 |
| post_mortem | baseline   | pg_baseline_linear      |   0.049 |   4666.96  |    10392.4  |          1.786 |       8.346 |        0.568 |           0.205 |      4366.26  |       0.411 |
