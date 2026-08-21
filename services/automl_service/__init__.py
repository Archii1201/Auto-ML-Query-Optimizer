"""
automl_service
==============
Phase 5 — the AutoML retraining loop.

Turns the online feedback log into a *gated, automated* model-improvement
pipeline:

    merge (5A) → retrain (5B) → promote-gate (5C)
              ↑                          │
     triggers (5D)                       ▼
     worker  (5E)                  watchdog / rollback (5F)

Every subphase is standalone and fail-open: a retraining failure must
never take down the live `/plan-pick` service. See docs/PHASE5_*.md.
"""
