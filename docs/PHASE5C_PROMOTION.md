# Phase 5C — Promotion Gate & Model Swap

> A candidate only ships if it is *provably* at least as good as the model
> in production — and shipping it never requires a restart.

## What this subphase delivers

| Artifact | Purpose |
|---|---|
| `services/automl_service/promotion.py` | Pure gate policy + heavy paired-OOF evaluator |
| `scripts/promote_model.py` | CLI: evaluate → gate → `registry.promote` → reload |
| `POST /admin/reload-models` (server.py) | Token-guarded hot model swap |
| `inference.reset_predictors()` | Drops memoized predictors so a reload picks up `current` |
| `tests/test_promotion_gate.py` | Gate-policy + promote/rollback tests |

## The gate

```
evaluate_candidate(regime, cand, incumbent, features.csv)
    │  OOF on the SAME GroupKFold splits for BOTH models
    │  + paired bootstrap 95% CI of Δplan-pick (resample query groups)
    ▼
decide(cand_metrics, inc_metrics, policy)  → GateDecision
    ├─ metrics_valid   : finite, plan_pick∈[0,1]        (leakage/NaN guard)
    ├─ plan_pick_gain  : Δplan_pick ≥ min_plan_pick_gain
    ├─ delta_ci_lower  : Δplan_pick lo95 ≥ min_delta_lo95   (noise guard)
    ├─ qerror          : q_err_median regression ≤ max_qerror_regression
    └─ regret          : regret_mean_ms regression ≤ max_regret_regression_ms
promote = ALL checks pass
```

Only when every check passes does the CLI call `REGISTRY.promote()` and then
`POST /admin/reload-models`.

## Design decisions — what / why / why-not

**Judge on out-of-fold plan-pick, reusing the Phase 3G/3H evaluator.**
Why: plan-pick accuracy on *unseen queries* is the deployment metric; OOF
via GroupKFold is the only honest estimate of it. Reusing
`evaluate_baseline.evaluate_model` means the gate's numbers are identical to
the numbers we report offline — no second, subtly-different implementation
to drift. Why-not train/test on the newest rows only: small held-out sets
give noisy, gameable numbers.

**Paired bootstrap CI of the *delta*, not two independent CIs.**
Why: candidate and incumbent are evaluated on the same query groups, so
their errors are correlated. A paired bootstrap over groups gives a far
tighter, correct CI for "did it actually get better?" — the exact lesson
from the ±8.5pp noise floor we hit in Phase 3H. The gate requires the delta
CI's lower bound ≥ −2pp, which blocks noise-level "wins" from shipping.

**Multi-metric gate (plan-pick + q-error + regret), not plan-pick alone.**
Why: a model can nudge plan-pick up while badly regressing tail latency or
calibration. Gating on q-error and mean regret too means a promotion can
never trade a real latency regression for a cosmetic accuracy bump. Each
metric has an explicit, configurable tolerance in `PromotionPolicy`.

**Pure `decide()` separated from heavy `evaluate_candidate()`.**
Why: the policy is the part that must never have a bug (it decides what
serves production) yet must be testable in milliseconds. Splitting it out
lets the unit tests exercise every reject/accept branch with synthetic
metrics — no dataset, no sklearn, no models.

**Hot swap via `POST /admin/reload-models`, guarded by `ML_ADMIN_TOKEN`.**
Why: promoting must not require a rolling restart (that's downtime and lost
warm caches). The endpoint drops the memoized predictors, rebuilds from the
registry's `current`, warms them, and only *then* swaps `app.state` — so a
failed rebuild leaves the old models serving (fail-safe). Why a token: model
swaps are a privileged operation; the endpoint refuses every caller until a
non-empty `ML_ADMIN_TOKEN` is set, so an unconfigured deployment is closed
by default.

**`registry.promote()` is an atomic pointer flip.**
Why: promotion is just moving `current` to an already-registered, immutable,
content-addressed version. There is no file overwrite, so rollback (5F) is
the same operation pointed at the previous version — instant and safe.

## Usage

```bash
# Dry-run the gate on the newest candidate:
python scripts/promote_model.py --regime plan_time --candidate latest

# Promote if it passes, and hot-swap the live service:
export ML_ADMIN_TOKEN=secret
python scripts/promote_model.py --regime plan_time --candidate latest \
    --apply --reload-url http://localhost:8000 --admin-token secret
```

A full decision report is written to
`models/registry/reports/<candidate_version>.json`.

## Acceptance criteria

- Clear improvement promotes; noise-level gain is blocked by the CI check. ✔ tested
- q-error / regret regressions block promotion even when plan-pick rises. ✔ tested
- NaN / implausible metrics are rejected (leakage guard). ✔ tested
- `registry.promote` flips `current` and rollback restores it. ✔ tested
- Reload endpoint is `403` without the correct `X-Admin-Token`.

## What is intentionally deferred

- Automatic shadow/canary traffic split before full promotion (5F covers
  post-promote monitoring + rollback instead).
- Multi-regime promotion in one call; today the CLI promotes one regime.
