# Phase 5E — Orchestration (the AutoML Worker)

> Wire 5A–5D into one long-running process that runs the whole loop on a
> schedule, safely, as a container you can bring up with one command.

## What this subphase delivers

| Artifact | Purpose |
|---|---|
| `services/automl_service/worker.py` | The loop: gather → decide → retrain → gate → promote → reload |
| `services/automl_service/filelock.py` | Dependency-free single-flight lock |
| `docker-compose.yml` `automl-worker` (profile `retrain`) | The worker container |
| shared volumes (`processeddata`, `registrydata`, `feedbackdata`) | So worker + serving see the same state |
| `.env.example` Phase 5 knobs | Configuration surface |
| `tests/test_worker_cycle.py` | Cycle + lock tests (fully faked) |

## The loop

```
every AUTOML_POLL_INTERVAL_S:
    signals = gather(watermark Δ, minutes-since, /metrics ratio)   [5D]
    should_retrain(signals)? ── no ──► sleep
                             └ yes ──► FileLock (single-flight)     [5E]
                                          retrain(profile)          [5A/5B]
                                          promote-gate(candidate)   [5C]
                                            pass ─► promote + reload every replica
                                          RetrainState.save()       [5D]
```

Run it:

```bash
docker compose --profile retrain up automl-worker      # with core services
# or the whole system incl. worker:
docker compose --profile all up
```

## Design decisions — what / why / why-not

**One worker process, separate from the serving process.**
Why: retraining is CPU-heavy and occasionally crashes (bad data, OOM in a
tuner). Isolating it in its own container means a retrain failure can never
touch `/plan-pick` latency or availability. The loop also wraps every cycle
in a broad `except` and keeps going — a bad cycle logs and is forgotten, it
never kills the worker.

**Reuse the ml-service image, just change the command.**
Why: the worker needs the exact same code and dependencies as the service
(it imports the registry, the trainer, the feature pipeline). Building a
second image would double maintenance and invite version skew. `command:
python -m services.automl_service.worker` on the shared image is the whole
"Dockerfile" — one image, one source of truth. Why-not a separate slim image:
the worker genuinely needs pandas/sklearn/lightgbm, so there's nothing to
slim away.

**Single-flight `FileLock` on shared storage.**
Why: retrains share `features.csv`, the trainer's output path, and the
registry. Two concurrent retrains (or a worker + a manual `scripts/retrain.py`)
would corrupt each other. The lock is built on atomic `O_CREAT|O_EXCL` (works
on Windows and POSIX) with stale-lock reclaim so a crashed holder can't wedge
the loop forever. Why-not the `filelock` PyPI package: one stdlib file keeps
the image slim and the failure mode obvious.

**Shared named volumes for `data/processed`, `models/registry`, `data/feedback`.**
Why: the worker promotes by flipping a pointer in the registry that the
*serving* replicas must be able to read — they have to share that directory.
Likewise the merge writes `features.csv` and the worker reads live feedback.
Named volumes give all containers one coherent view that survives restarts.

**Reload fans out to EVERY replica (comma-separated `AUTOML_RELOAD_URL`).**
Why: each ml-service caches its own `Predictor` in-process; flipping the
registry pointer alone doesn't refresh an already-running replica. So after a
promotion the worker POSTs `/admin/reload-models` to *each* replica directly
(not through nginx, which would hit only one). This is the multi-replica
correctness fix.

**Injectable `WorkerDeps` (signals / retrain / promote / reload).**
Why: the cycle's control flow — trigger-skip, lock-busy, retrain-fail,
gate-reject, promote+reload — is the bug-prone part and must be tested
without a trainer, Postgres, or network. `run_cycle(state, deps)` takes fakes
in tests and the real wiring (`WorkerDeps.default()`) in production.

## Configuration (`.env`)

| Var | Default | Meaning |
|---|---|---|
| `AUTOML_PROFILE` | `fast` | which retrain profile the worker uses |
| `AUTOML_POLL_INTERVAL_S` | `300` | trigger-evaluation cadence |
| `AUTOML_RELOAD_URL` | both replicas | where to POST reloads after promotion |
| `AUTOML_METRICS_URL` | replica-1 `/metrics` | drift-signal source |
| `ML_ADMIN_TOKEN` | `changeme` | guards the reload endpoint |

## Acceptance criteria

- No trigger ⇒ no retrain. ✔ tested
- Bootstrap ⇒ retrain + gate-pass ⇒ promote + reload + state counters. ✔ tested
- Gate reject ⇒ retrained but not promoted, no reload. ✔ tested
- Retrain failure ⇒ promote never runs. ✔ tested
- Lock held elsewhere ⇒ cycle skips (no overlap). ✔ tested
- `docker compose --profile retrain up automl-worker` starts the loop.

## What is intentionally deferred

- Kubernetes CronJob / Argo instead of an always-on poller (the poller is
  simplest and matches the single-node compose story).
- Distributed locking (Redis/etcd); the file lock is correct for the
  single-host deployment we ship.
