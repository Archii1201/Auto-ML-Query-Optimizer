# Phase 5B — Retrain Job

> Turn "we have new data" into "we have a new candidate model" — repeatably,
> cheaply, and without ever auto-shipping it.

## What this subphase delivers

| Artifact | Purpose |
|---|---|
| `services/automl_service/config.py` | `fast` / `full` retrain profiles (Optuna off/on) |
| `services/automl_service/trainer.py` | Orchestrates merge → train → register-candidate |
| `scripts/retrain.py` | CLI to run one retrain |
| `tests/test_retrain_trainer.py` | Profile + control-flow tests (fully mocked) |

## The pipeline

```
run_retrain(profile)
    │
    ├─ do_merge?  → merge_feedback(apply=True)          [5A]
    │              → run_validation_gate()  ── FAIL ──► abort (no training)
    │
    ├─ train_fn(profile) → phase3b/train_models.py --regimes plan_time
    │                        (--skip-tuning for fast, --n-trials for full)
    │              rc≠0 ──► abort
    │
    ├─ artifact missing ──► abort
    │
    └─ register_fn(regime, artifact) → REGISTRY.register(promote=False)
                                        → RetrainResult(version=…, ok=True)
```

The result is a **candidate**: a new immutable version in the model
registry with `current` untouched. Nothing that serves traffic changes.

## Design decisions — what / why / why-not

**Two profiles, `fast` and `full`, as data not flags.**
Why: the worker (5E) fires on every trigger, so the default path must be
cheap — `fast` skips Optuna and just fits defaults, which is enough to
detect a real regression or win. `full` (Optuna on) is for deliberate,
higher-effort candidates. Encoding this as `RetrainProfile` means the CLI,
worker, and tests can't disagree about what "fast" means. Why-not always
tune: tuning on every feedback batch would make the loop expensive and
flaky for a marginal, noise-level quality delta.

**Reuse `phase3b/train_models.py` via subprocess instead of importing it.**
Why: that script is the *single source of truth* for how a model is trained
(feature matrix, log-target, model zoo, artifact schema). Shelling out means
the automated retrain produces byte-for-byte the same kind of artifact a
human produces, and we never fork training logic. Why-not import `main()`:
process isolation is cleaner (its own argparse, its own crash boundary) and
matches how the offline pipeline is actually run.

**Register as candidate, never promote here.**
Why: separation of concerns is the whole safety story of Phase 5. Training
produces a candidate; a *separate gate* (5C) decides if it's better and only
then flips `current`. Coupling them would make a bad retrain able to ship
itself. `REGISTRY.register(..., promote=False)` enforces this.

**Injectable `train_fn` / `register_fn` / merge steps.**
Why: the orchestration logic (gate-abort, train-failure, missing-artifact,
happy-path) is exactly the part most likely to have bugs, and the part we
*don't* want to need a real trainer + Postgres to test. Dependency injection
lets the unit tests drive every branch in ~5 seconds with fakes.

**Merge is opt-out (`--no-merge`), gate is opt-out (`--no-gate`).**
Why: the safe defaults (merge + gate) are what the automated worker uses. The
opt-outs exist for local experiments ("retrain on the CSV I already have")
without weakening the production path.

## Usage

```bash
python scripts/retrain.py --profile fast              # merge + fast train
python scripts/retrain.py --profile full              # merge + tuned train
python scripts/retrain.py --profile fast --no-merge   # current CSV only
```

Output is a `RetrainResult` JSON: `ok`, `regime`, `profile`, `version`,
`artifact_path`, `merged_rows`, `steps[]`, `error`.

## Acceptance criteria

- `fast` profile emits `--skip-tuning`; `full` emits `--n-trials`. ✔ tested
- Gate failure aborts *before* training runs. ✔ tested
- Non-zero trainer exit aborts and surfaces the code. ✔ tested
- Missing artifact aborts (no phantom registration). ✔ tested
- Happy path registers a candidate and returns its version. ✔ tested

## What is intentionally deferred

- LambdaRank / pairwise objective as an alternate profile (Phase 3I idea).
  The trainer is profile-driven, so adding a `rank` profile later is a
  config change, not a rewrite.
- Distributed / GPU training. `full` runs Optuna locally, which is
  sufficient at this dataset size.
