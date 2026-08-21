"""
observability.py
================
Phase 4D — distributed tracing setup (OpenTelemetry).

Metrics answer "how much / how often"; traces answer "where did this one
request spend its time". With OTel auto-instrumentation, every HTTP
request to the ML service becomes a trace whose span tree shows
plan-generation, prediction and execution, and a single trace-id flows
through to logs (the access log already carries request_id).

This is **opt-in and fail-open**: tracing only activates when
``OTEL_ENABLED=true`` *and* the OTel packages import successfully. If the
collector is down or the libs are missing, the service runs exactly as
before — observability must never break serving.

Why OpenTelemetry (not a vendor SDK)
------------------------------------
OTel is the vendor-neutral standard: the same instrumentation exports to
Tempo, Jaeger, Datadog, etc. by changing only the collector endpoint. We
export OTLP/HTTP to an OpenTelemetry Collector (see deploy/otel-collector),
which fans out to Tempo for storage and Grafana for viewing.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger("ml_service")


def setup_tracing(app) -> bool:
    """Instrument the FastAPI app with OTel. Returns True if activated."""
    if os.environ.get("OTEL_ENABLED", "false").strip().lower() != "true":
        return False
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        service_name = os.environ.get("OTEL_SERVICE_NAME", "ml_service")
        endpoint = os.environ.get(
            "OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318"
        ).rstrip("/")

        provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{endpoint}/v1/traces"))
        )
        trace.set_tracer_provider(provider)
        FastAPIInstrumentor.instrument_app(app)
        logger.info("otel tracing enabled",
                    extra={"fields": {"service": service_name, "endpoint": endpoint}})
        return True
    except Exception as exc:  # noqa: BLE001 — tracing is best-effort
        logger.warning("otel tracing setup failed; continuing without it",
                       extra={"fields": {"error": str(exc)}})
        return False
