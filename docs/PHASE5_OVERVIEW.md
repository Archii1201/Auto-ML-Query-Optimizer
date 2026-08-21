# Phase 5 — AutoML Retraining Loop (Overview)

Phase 4 made the system **operable**. Phase 5 makes it **self-improving**:
online feedback flows back into training, and a better model is promoted only
when honest OOF evaluation proves it beats the incumbent.

**Full roadmap:** [PHASE5_RETRAINING.md](PHASE5_RETRAINING.md)

---

## Subphases


| Sub | Theme               | Key deliverables                                 | Status | Doc |
| --- | ------------------- | ------------------------------------------------ | ------ | --- |
| 5A  | Feedback merge      | `merge_feedback.py`, watermark, validation gate  | ✅ done | [PHASE5A_MERGE.md](PHASE5A_MERGE.md) |
| 5B  | Retrain job         | `retrain.py`, candidate → model registry         | ✅ done | [PHASE5B_RETRAIN.md](PHASE5B_RETRAIN.md) |
| 5C  | Promotion + swap    | OOF gate, `promotion.py`, `/admin/reload-models` | ✅ done | [PHASE5C_PROMOTION.md](PHASE5C_PROMOTION.md) |
| 5D  | Drift + triggers    | volume / cron / pred-actual drift                | ✅ done | [PHASE5D_TRIGGERS.md](PHASE5D_TRIGGERS.md) |
| 5E  | Orchestration       | `automl-worker`, compose `retrain` profile       | ✅ done | [PHASE5E_ORCHESTRATION.md](PHASE5E_ORCHESTRATION.md) |
| 5F  | Watchdog + rollback | post-promote monitor, auto-rollback              | ✅ done | [PHASE5F_WATCHDOG.md](PHASE5F_WATCHDOG.md) |


---

## The closed loop

```
Execute → feedback JSON → merge → validate → train → OOF eval → promote → reload
                ▲                                              │
                └──────────── better predictions ──────────────┘
```

---

## Design rules (same spirit as Phase 4)

1. **Fail-open serving** — retrain failures never break `/plan-pick`.
2. **Gated pipeline** — validate before train; OOF before promote (Phase 3G/3H).
3. **Immutable artifacts** — registry versions; promote, never silently overwrite.
4. **Train often; promote rarely** — noise floor requires statistical gates.

---

## Prerequisites (from Phase 4)


| Component                               | Used for                      |
| --------------------------------------- | ----------------------------- |
| `data/feedback/` + Kafka consumer       | training rows                 |
| `scripts/feedback_to_features.py`       | merge logic (5A wraps this)   |
| `services/ml_service/model_registry.py` | register / promote / rollback |
| `scripts/validate_dataset.py`           | pre-train hard gate           |
| `scripts/evaluate_baseline.py`          | promotion gate                |
| `phase3b/train_models.py`               | candidate training            |


---

## Status — all subphases implemented

Phase 5 is **complete**. Run the loop end-to-end:

```bash
# one-shot, by hand:
python scripts/merge_feedback.py --apply --gate      # 5A
python scripts/retrain.py --profile fast             # 5B (registers candidate)
python scripts/promote_model.py --candidate latest --apply \
    --reload-url http://localhost:8000               # 5C (gated promote + hot-swap)

# or fully automated:
docker compose --profile retrain up automl-worker    # 5D/5E/5F worker loop
```

Every subphase ships with unit tests (`tests/test_feedback_merge.py`,
`test_retrain_trainer.py`, `test_promotion_gate.py`, `test_triggers_drift.py`,
`test_worker_cycle.py`, `test_watchdog.py` — 56 tests) and a design doc above.

After 5A–5F: optional **Phase 3I LambdaRank** experiment uses the same
promotion gate; **Phase 6** covers research polish and cross-workload evaluation.