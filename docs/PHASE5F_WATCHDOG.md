# Phase 5F — Online Monitoring & Auto-Rollback

> The gate can be wrong online. Watch the freshly promoted model on live
> traffic and instantly roll back if it misbehaves.

## What this subphase delivers

| Artifact | Purpose |
|---|---|
| `services/automl_service/watchdog.py` | Pure health decision + injectable rollback orchestrator |
| `deploy/grafana/dashboards/retrain_promotions.json` | Post-promotion health dashboard |
| `tests/test_watchdog.py` | Decision + orchestrator tests |

## The safety net

```
before promote:  baseline = HealthSnapshot(error_rate, p95_latency, ratio)
        promote + reload  (5C)
        │  monitor window
        ▼
after:           current  = fetch()   (from ml-service /metrics)
        evaluate(baseline, current, policy):
            error_rate   ≤ cap AND ≤ baseline + tol
            p95_latency  ≤ baseline × (1 + tol)
            ratio        ∈ [0.5, 2.0]
        any FAIL ─► registry.promote(previous_version)  + reload replicas
                    + state.mark_rollback()
```

Rollback is the *same* atomic pointer flip as promotion (5C), pointed at the
previous version — instant, with no retrain and no file rewrite.

## Design decisions — what / why / why-not

**Monitor online even though the gate already passed offline.**
Why: OOF accuracy is measured on held-out *history*. Production can differ —
a serving-time feature bug, a traffic-mix shift, an infra interaction the
dataset never saw. Defense in depth: 5C stops bad models statistically
before they ship; 5F catches the ones that only reveal themselves live.

**Roll back on operational signals (errors, latency, calibration), not on
plan-pick accuracy.**
Why: true plan-pick accuracy needs the oracle (running every variant), which
we don't do on live traffic. The signals we *can* read cheaply and in real
time — error rate, p95 latency, and the pred/actual calibration ratio — are
exactly the ones that spike when a model is genuinely broken in production.
Calibration drifting outside [0.5, 2.0] is the strongest cheap "this model
is wrong now" signal, and it's the same series 5D uses for drift.

**Compare against a pre-promote baseline, not fixed absolutes only.**
Why: "good" latency/error levels are workload-specific. Gating on a
*regression vs. the model we just replaced* avoids false rollbacks on a
system that was always a bit slow, while still catching a real degradation.
Error rate additionally has a hard absolute ceiling as a backstop.

**Rollback = `registry.promote(previous_version)`.**
Why: because promotion never overwrites files (content-addressed, immutable
versions), the previous model is still sitting in the registry. Rolling back
is a one-line pointer flip plus a replica reload — no rebuild, no data loss,
seconds not minutes. This is the payoff of the Phase 4B registry design.

**Pure `evaluate()` + injectable `run_watchdog()`.**
Why: the rollback decision protects production, so every branch (each breach,
healthy, missing-previous-version) must be unit-tested without a live
service. The orchestrator's fetch/rollback/reload are injected, so tests
prove "unhealthy ⇒ exactly one rollback to the right version" with fakes.

**A dedicated Grafana dashboard.**
Why: humans need to *see* what the watchdog sees. The dashboard plots the
exact series the policy checks (calibration band, error rate, p95 latency,
plan-pick trend, fallback spikes) so an operator can confirm a rollback was
warranted — or spot trouble the thresholds didn't catch.

## Acceptance criteria

- Healthy metrics ⇒ no rollback. ✔ tested
- Error-rate spike / regression ⇒ rollback. ✔ tested
- Latency regression beyond tolerance ⇒ rollback. ✔ tested
- Calibration outside band ⇒ rollback. ✔ tested
- Zero-baseline latency ⇒ latency check skipped (no false positive). ✔ tested
- Orchestrator performs exactly one rollback to the previous version + reload. ✔ tested
- Unhealthy but no previous version ⇒ no rollback, reason recorded. ✔ tested
- Dashboard provisions under Grafana (Phase 4D provisioning picks it up).

## What is intentionally deferred

- Shadow/canary traffic split (serve candidate to X% and compare) before
  full promotion — heavier infra; the baseline-vs-current watchdog gives
  most of the safety at a fraction of the cost.
- Auto-quarantine of a repeatedly-rolled-back candidate version.
