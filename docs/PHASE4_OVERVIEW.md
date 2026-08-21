# Phase 4 — Production Hardening (Overview)

Phase 3 made the model *correct*; Phase 4 makes the system *operable* —
resilient, scalable, observable, and shippable in one command. Each
subphase is **standalone shippable** (the system still works at the end of
each) and every new dependency is **optional with a working local default**,
so tests and laptops need zero infra.

| Sub | Theme | Key deliverables | Doc |
|---|---|---|---|
| 4A | Resilience | connection pool, circuit breaker, timeout budget, JSON logging, liveness/readiness, CI | [PHASE4A_RESILIENCE.md](PHASE4A_RESILIENCE.md) |
| 4B | Cache + Registry | `CacheBackend` (local/Redis), model registry | [PHASE4B_CACHE_REGISTRY.md](PHASE4B_CACHE_REGISTRY.md) |
| 4C | Streaming | Kafka feedback bus, schema versioning, dedup, consumer | [PHASE4C_STREAMING.md](PHASE4C_STREAMING.md) |
| 4D | Observability | Prometheus metrics, OTel tracing, Grafana dashboards | [PHASE4D_OBSERVABILITY.md](PHASE4D_OBSERVABILITY.md) |
| 4E | Orchestration | docker-compose, nginx LB, TLS, secrets, healthchecks | [PHASE4E_ORCHESTRATION.md](PHASE4E_ORCHESTRATION.md) |
| 4F | Load + Chaos | Locust load, chaos/soak, performance report | [PHASE4F_PERFORMANCE.md](PHASE4F_PERFORMANCE.md) |

## The two design rules that run through all of Phase 4
1. **Fail-open optionality.** Redis, Kafka, OTel and Prometheus are
   enhancements. If any is missing or down, the service logs and continues
   on a local default — a dependency outage never blocks startup or 500s a
   request.
2. **Strategy pattern at every seam** (cache, feedback bus) so the
   transport is one env var, call sites never change, and tests stay fast
   and infra-free.

## One-command bring-up
```bash
cp .env.example .env
docker compose --profile all up -d --build
curl -fsS http://localhost/healthz
```

## What's next: Phase 5

The **retraining loop** (feedback → merge → validate → train → OOF gate →
registry promote → reload) is specified in
[PHASE5_RETRAINING.md](PHASE5_RETRAINING.md). Phase 4C/4B built the seams
(Kafka consumer, model registry, metrics for drift).
