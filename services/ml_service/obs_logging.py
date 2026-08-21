"""
obs_logging.py
=============
Phase 4A — structured JSON logging.

`print()` is fine for a demo but useless for aggregation: you can't
grep/group/alert on free-form text. We emit one machine-parseable JSON
object per log line so any collector (Loki, ELK, CloudWatch) can index
fields directly. Phase 4D will scrape metrics; this is the log half.

Two entry points:
    configure_logging()          -> install the JSON formatter on stdout
    log_request(logger, **fields)-> emit one access-log line per request

A per-request access line carries:
    request_id, path, status_code, latency_ms, regime, sql_hash,
    predicted_ms, actual_ms, fallback, error
"""

from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any

SERVICE_NAME = "ml_service"


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts":      time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
                       + f".{int(record.msecs):03d}Z",
            "level":   record.levelname,
            "service": SERVICE_NAME,
            "logger":  record.name,
            "msg":     record.getMessage(),
        }
        # Anything attached via `extra={"fields": {...}}` is merged in flat.
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            for k, v in fields.items():
                if v is not None:
                    payload[k] = v
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> logging.Logger:
    root = logging.getLogger()
    root.setLevel(level)
    # Replace any default handlers with a single JSON stdout handler.
    for h in list(root.handlers):
        root.removeHandler(h)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
    return logging.getLogger(SERVICE_NAME)


def log_request(logger: logging.Logger, *, level: int = logging.INFO, **fields: Any) -> None:
    """Emit a single structured access-log line."""
    logger.log(level, "request", extra={"fields": fields})
