# Phase 4F — Load Testing, Chaos & Performance Report

**Goal:** prove the system scales and degrades gracefully, with numbers.
This is the artifact you show in a review.

---

## Tooling

| Tool | File | Purpose |
|---|---|---|
| Locust | `loadtest/locustfile.py` | concurrent load on `/plan-pick` + `/run-and-learn` |
| Chaos driver | `loadtest/chaos.py` | kill/restore deps under load, record degradation |

Locust runs as a compose service (`--profile loadtest`) or locally.

---

## SLOs (targets)

| SLO | Target |
|---|---|
| `/plan-pick` p95 latency | < 500 ms |
| `/plan-pick` p99 latency | < 1000 ms |
| Error rate (5xx / total) | < 1 % |
| Throughput (plan-picks) | ≥ 100 req/s on 2 replicas (cache-warm) |
| Recovery after PG restart | ready again < 30 s, no manual action |

> Plan-pick *accuracy* (oracle hit-rate) is a model-quality metric tracked
> on the "Plan-pick Quality" Grafana dashboard, not a latency SLO.

---

## How to run

### 1. Bring up the system
```bash
cp .env.example .env
docker compose --profile observability up -d --build   # core + metrics
# (add --profile streaming for Kafka, --profile all for everything)
```

### 2. Smoke
```bash
curl -fsS http://localhost/healthz
curl -fsS http://localhost/readyz
curl -fsS "http://localhost/metrics?fmt=prom" | head
```

### 3. Load test
```bash
# headless: 100 users, ramp 20/s, 5 minutes, against the nginx gateway
docker compose --profile loadtest run --rm locust \
  -f /mnt/locust/locustfile.py --host http://nginx:80 \
  --headless -u 100 -r 20 -t 5m --csv /mnt/locust/result
# or open the Locust UI at http://localhost:8089
```

### 4. Soak (leak hunt)
Run the same load for **30 min** and watch for monotonically rising
memory / file handles / Kafka consumer lag:
```bash
docker compose --profile loadtest run --rm locust \
  -f /mnt/locust/locustfile.py --host http://nginx:80 \
  --headless -u 50 -r 10 -t 30m
docker stats --no-stream        # RSS should plateau, not climb
```

### 5. Chaos (graceful degradation)
With a load running in another terminal:
```bash
python loadtest/chaos.py --scenario postgres --down-seconds 20
python loadtest/chaos.py --scenario redis    --down-seconds 20
python loadtest/chaos.py --scenario kafka    --down-seconds 20
```

---

## Results (fill in after a run)

### Latency / throughput
| Endpoint | RPS | p50 (ms) | p95 (ms) | p99 (ms) | err % | Pass? |
|---|---|---|---|---|---|---|
| `/plan-pick` | | | | | | |
| `/run-and-learn` | | | | | | |

### Soak (30 min)
| Metric | Start | End | Verdict |
|---|---|---|---|
| ml-service RSS (MB) | | | |
| Open file handles | | | |
| Kafka consumer lag | | | |

### Chaos
| Scenario | Expected | Observed | Pass? |
|---|---|---|---|
| Kill PostgreSQL | 503s during outage, auto-recover < 30 s | | |
| Kill Redis | requests still 200 (cold cache) | | |
| Kill Kafka | `/run-and-learn` still 200, produce errors counted | | |

---

## Interpreting the numbers
- **p95 over SLO** → raise `ML_POOL_MAX`, add replicas, or warm the cache.
- **Errors during PG chaos are expected** (503 = correct backpressure).
  Errors during *Redis/Kafka* chaos are **not** — those deps are optional.
- **Climbing RSS in soak** → a leak; check the pool isn't discarding +
  recreating connections every request, and the Kafka producer is reused.
