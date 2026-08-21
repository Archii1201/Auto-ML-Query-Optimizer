# Phase 5D — Drift Detection & Retrain Triggers

> Decide *when* the loop should retrain — driven by data volume, a schedule,
> and live prediction drift — without ever thrashing.

## What this subphase delivers

| Artifact | Purpose |
|---|---|
| `services/automl_service/triggers.py` | `should_retrain(signals, config)` — the decision |
| `services/automl_service/drift.py` | pred/actual ratio drift verdict + Prometheus parse |
| `services/automl_service/state.py` | durable retrain/promote state + counters |
| `tests/test_triggers_drift.py` | pure-logic tests for all three |

## The decision

```
should_retrain(signals, config):
    never retrained?            → YES (bootstrap)
    minutes_since < cooldown?   → NO  (in_cooldown)
    else OR together:
        volume   : new_rows            ≥ min_new_rows
        schedule : minutes_since       ≥ max_interval_minutes
        drift    : pred_actual_ratio   ∉ [low, high]
    any fired → YES, else NO
```

Signals are *gathered* by the worker (5E): `new_rows` from the merge
watermark delta, `minutes_since` from `RetrainState`, `pred_actual_ratio`
scraped from the ml-service `/metrics`. This module only *judges* them.

## Design decisions — what / why / why-not

**Three OR-ed triggers, not one.**
Why: each catches a different failure mode. *Volume* retrains when enough
fresh ground truth exists to plausibly move the model. *Schedule* is a
heartbeat so a low-traffic system still refreshes and we notice silent
breakage. *Drift* reacts fast when the model is visibly miscalibrated even
before much new data lands. Any single rule alone leaves a blind spot.

**A single cooldown in front of all rules.**
Why: retraining is expensive and a promotion churns caches. The cooldown is
the one hard guarantee that we never retrain in a tight loop, no matter how
loudly the rules fire. It's the safety valve; the rules are the accelerator.
Bootstrap (never-retrained) is the only path that skips it, so a fresh
deployment gets its first model promptly.

**Drift from `pred_actual_ratio`, a signal we already emit.**
Why: we don't need a new statistical drift detector or a labelled reference
window — the calibration ratio the service already exports *is* a drift
signal, and it's the one directly tied to the metric users feel (predicted
vs real runtime). Why-not KS/PSI on feature distributions: heavier, needs a
reference snapshot, and doesn't map as cleanly to "the model is wrong now".

**Everything pure; the worker gathers, the module judges.**
Why: timing/scheduling logic is notoriously bug-prone and untestable when
tangled with I/O. By passing plain numbers in, every branch (bootstrap,
cooldown, each rule, no-fire) is covered by fast deterministic tests.

**`RetrainState` separate from the 5A watermark.**
Why: they answer different questions — the watermark tracks *merge*
progress (what data is in the CSV), the state tracks *retrain/promote*
progress (when we last acted, how many times, on which versions). Mixing
them would couple two independently-failing concerns.

## Config knobs (`TriggerConfig`)

| Knob | Default | Meaning |
|---|---|---|
| `min_new_rows` | 200 | volume rule threshold |
| `cooldown_minutes` | 360 | minimum spacing between retrains |
| `max_interval_minutes` | 1440 | heartbeat retrain (daily) |
| `drift.low` / `drift.high` | 0.5 / 2.0 | calibrated ratio band |

## Acceptance criteria

- Bootstrap fires when there is no prior retrain. ✔ tested
- Cooldown blocks every rule, even extreme signals. ✔ tested
- Volume / schedule / drift each fire independently after cooldown. ✔ tested
- `pred_actual_ratio` parsed from Prometheus text (labels → last value). ✔ tested
- State round-trips, counts, self-heals on corruption. ✔ tested

## What is intentionally deferred

- Feature-distribution drift (PSI/KS) as an additional trigger.
- Per-regime independent cooldowns; today one loop drives `plan_time`.
