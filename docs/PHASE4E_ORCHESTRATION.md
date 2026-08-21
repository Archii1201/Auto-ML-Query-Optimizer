# Phase 4E — Container Orchestration & One-Command Bring-up

**Goal:** `docker compose up` stands up the whole system — datastores, two
ML replicas behind a load balancer, streaming, and observability — with
healthchecks, persistent volumes, and a single `.env` for all config.

---

## Topology

```
            ┌───────────── nginx (API gateway, :80/:443) ─────────────┐
            │              least_conn load balancing                   │
            ▼                                                          ▼
     ml-service-1 (:8000)                                     ml-service-2 (:8000)
            │   │                                                  │   │
   ┌────────┘   └───────────┐                          ┌──────────┘   └────────┐
   ▼                        ▼                          ▼                       ▼
 postgres (:5432)      redis (:6379)              kafka (:9092)         (feedback-consumer)
                                                       │
                                                       ▼
                                              data/feedback/ (volume)

 observability: prometheus(:9090) ─ grafana(:3000) ─ tempo ─ otel-collector(:4318)
```

## Profiles (start subsets)
```bash
docker compose up                          # core: pg, redis, 2x ml, nginx
docker compose --profile streaming up      # + kafka, kafka-ui, feedback-consumer
docker compose --profile observability up  # + prometheus, grafana, tempo, otel
docker compose --profile loadtest run locust   # + load generator
docker compose --profile all up            # everything
```

## Why these choices

| Decision | Why |
|---|---|
| **nginx + 2 named replicas** (not `deploy.replicas`) | stable hostnames for nginx upstream *and* Prometheus targets; `least_conn` suits variable query latency |
| **YAML anchors** (`x-ml-service`, `x-ml-env`) | the two replicas + consumer share one definition — no copy-paste drift |
| **Profiles** | keep the core light; opt into Kafka/observability/load only when needed |
| **Healthchecks everywhere** | dependents wait on `service_healthy`; nginx evicts unhealthy replicas via `max_fails` |
| **Named volumes** | pgdata/redisdata/kafkadata/feedback/promdata/grafanadata/tempodata survive restarts |
| **Multi-stage slim Dockerfile** | build toolchain stays in the builder stage; runtime image is small and runs as non-root (uid 10001) |
| **`.env` + `.env.example`** | one documented place for every knob; real `.env` is git-ignored |
| **Kafka KRaft** | no ZooKeeper container |

## TLS
HTTP-only by default so `docker compose up` never fails on a missing cert.
To enable HTTPS:
```bash
bash deploy/nginx/gen-certs.sh            # self-signed -> deploy/nginx/certs/
# then mount the TLS config instead of the plain one in docker-compose.yml:
#   - ./deploy/nginx/nginx-tls.conf:/etc/nginx/nginx.conf:ro
docker compose up -d nginx
```
In production, mount real certs into `/etc/nginx/certs` and use
`nginx-tls.conf`.

## Secrets
All credentials come from `.env` (never baked into images; `.env` and
`deploy/nginx/certs/` are git-ignored). `PGPASSWORD`/`GRAFANA_PASSWORD`
must be changed for any non-local deployment.

## Container image
`Dockerfile` (multi-stage): builder installs deps into a venv; runtime
copies the venv + source, adds `libgomp1` (lightgbm/xgboost OpenMP) +
`curl` (healthcheck), runs as non-root, and bakes a `/healthz`
HEALTHCHECK. `.dockerignore` excludes `data/`, notebooks, `.git`, certs and
`.env` (models/ are kept — needed for serving).

## Acceptance
- `docker compose up` brings up pg, redis, 2× ml-service, nginx; all
  healthy. `curl http://localhost/healthz` → 200 through nginx.
- A `/plan-pick` round-trips through nginx → a replica → PG.
- `docker compose restart ml-service-1` → nginx keeps serving on
  ml-service-2 (rolling restart, no client errors).
- With `--profile streaming`, `/run-and-learn` → Kafka → consumer writes a
  feedback file in the `feedbackdata` volume.
