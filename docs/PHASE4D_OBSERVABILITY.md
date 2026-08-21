# Phase 4D — Observability (Prometheus + Grafana + OpenTelemetry)

**Goal:** see the running system. Metrics answer "how much / how often";
traces answer "where did this request spend its time"; logs (Phase 4A)
answer "what happened to this specific request". Together they're the
three pillars.

---

## Metrics — Prometheus

### What we do
We keep the homegrown `MetricsRegistry` (it powers the dependency-free JSON
`/metrics` and the smoke demos) but **mirror every `inc()`/`observe()`
into `prometheus_client`** when it's installed. `GET /metrics?fmt=prom`
then returns real Prometheus exposition with **proper histogram buckets**
(`*_ms` metrics get ms-scale buckets) that Prometheus scrapes natively.

### Why mirror instead of rewrite
Zero call-site changes: a single hook in `inc`/`observe` updates both. The
JSON endpoint and existing tests keep working; Prometheus gets first-class
histograms. Controlled by `PROMETHEUS_ENABLED` (default on; auto-off if the
library is absent).

### Scraping
`deploy/prometheus/prometheus.yml` scrapes each replica at
`/metrics?fmt=prom` (via the `params: { fmt: [prom] }` block) on a 15 s
interval.

### Key metrics
`plan_picks_total`, `predictions_total`, `plan_pick_fallback_total`,
`pool_timeouts_total`, `execution_failures_total`, `feedback_rows_written_total`;
histograms `inference_latency_ms`, `execution_latency_ms`, `regret_ms`,
`pred_actual_ratio`.

---

## Tracing — OpenTelemetry

### What we do — `services/ml_service/observability.py`
`setup_tracing(app)` auto-instruments FastAPI so every request becomes a
trace, exported via **OTLP/HTTP** to an **OpenTelemetry Collector**, which
forwards to **Tempo** for storage and **Grafana** for viewing.

### Opt-in and fail-open
Tracing activates only when `OTEL_ENABLED=true` *and* the OTel packages
import. If the collector is down or libs are missing, the service runs
exactly as before — observability must never break serving.

### Why OpenTelemetry (not a vendor SDK)
Vendor-neutral: the same instrumentation exports to Tempo, Jaeger,
Datadog, etc. by changing only the collector's exporter — never the app.

---

## Dashboards — Grafana (auto-provisioned)

`deploy/grafana/provisioning/` wires datasources (Prometheus uid
`prometheus`, Tempo uid `tempo`) and a dashboard provider on startup. Two
dashboards ship in `deploy/grafana/dashboards/`:

- **ML Service — Health**: request rate, inference/execution latency
  percentiles, pool timeouts, fallbacks, cache hit ratio.
- **ML Service — Plan-pick Quality**: oracle-hit accuracy over time,
  regret (ms), predicted/actual ratio.

---

## Run it
```bash
docker compose --profile observability up -d
# Grafana   http://localhost:3000  (anon viewer on; admin pw = GRAFANA_PASSWORD)
# Prometheus http://localhost:9090
# enable traces:  set OTEL_ENABLED=true in .env, then `up` again
```

---

## Config
`PROMETHEUS_ENABLED`, `OTEL_ENABLED`, `OTEL_EXPORTER_OTLP_ENDPOINT`,
`OTEL_SERVICE_NAME`, `GRAFANA_PASSWORD`.

## Why not the homegrown metrics alone?
It can't do native histogram buckets, has no scrape/alerting ecosystem, and
no trace correlation. Prometheus + OTel are the industry standard and plug
straight into Grafana — far more than a custom `/metrics` JSON can offer.
