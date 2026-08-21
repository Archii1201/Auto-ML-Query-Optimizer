"""
publisher.py
============
Phase 4C — pluggable feedback *publisher* (Strategy pattern).

The execution path produces a feedback record per query run. Where that
record goes should be swappable:

    FEEDBACK_BUS=file   ->  FilePublisher   (default; writes data/feedback/)
    FEEDBACK_BUS=kafka  ->  KafkaPublisher  (produces query.feedback.v1)

Why decouple the write path
---------------------------
Writing straight to local disk couples *producing* feedback to *storing*
it. With Kafka:
  - many consumers can read the same stream (training, dashboards, audit),
  - the write path is durable and replayable (offsets),
  - the request thread isn't blocked on disk I/O for storage.

The consumer (consumer.py) is what lands Kafka records on disk, so the
on-disk training corpus is identical regardless of transport.

Resilience: the request path must never fail because the bus is down.
KafkaPublisher.publish swallows produce errors (counts them) and the
factory falls back to FilePublisher if confluent-kafka isn't installed.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from services.exec_service.capture import (  # noqa: E402
    FeedbackWriter,
    build_feedback_record,
)

logger = logging.getLogger("ml_service")

DEFAULT_TOPIC = os.environ.get("FEEDBACK_TOPIC", "query.feedback.v1")


class FeedbackPublisher(ABC):
    name: str = "abstract"

    @abstractmethod
    def publish(self, **kwargs: Any) -> Path | None:
        """Accepts the same kwargs as FeedbackWriter.write."""

    def stats(self) -> dict[str, Any]:
        return {"publisher": self.name}

    def close(self) -> None:  # noqa: B027 - optional override
        pass


# ---------------------------------------------------------------------------
class FilePublisher(FeedbackPublisher):
    """Default: persist to local disk via FeedbackWriter (pre-4C behaviour)."""

    name = "file"

    def __init__(self, writer: FeedbackWriter | None = None) -> None:
        self.writer = writer or FeedbackWriter()

    def publish(self, **kwargs: Any) -> Path | None:
        return self.writer.write(**kwargs)

    def stats(self) -> dict[str, Any]:
        return {"publisher": self.name, **self.writer.stats()}


# ---------------------------------------------------------------------------
class KafkaPublisher(FeedbackPublisher):
    """
    Produce JSON feedback records to Kafka, keyed by sql_hash so all
    records for a query land on the same partition (ordered per query).
    """

    name = "kafka"

    def __init__(
        self,
        bootstrap_servers: str,
        topic: str = DEFAULT_TOPIC,
        *,
        producer: Any | None = None,
    ) -> None:
        self.topic = topic
        self._lock = threading.Lock()
        self.produced = 0
        self.errors = 0
        if producer is not None:
            self._p = producer
        else:
            from confluent_kafka import Producer  # lazy import
            self._p = Producer({
                "bootstrap.servers": bootstrap_servers,
                "enable.idempotence": True,
                "linger.ms": 20,
                "acks": "all",
            })

    def publish(self, **kwargs: Any) -> Path | None:
        record, _, _ = build_feedback_record(**kwargs)
        try:
            self._p.produce(
                self.topic,
                key=str(record.get("sql_hash", "")).encode("utf-8"),
                value=json.dumps(record).encode("utf-8"),
            )
            self._p.poll(0)  # serve delivery callbacks without blocking
            with self._lock:
                self.produced += 1
        except Exception as exc:  # noqa: BLE001 — never 500 a request over the bus
            with self._lock:
                self.errors += 1
            logger.warning("kafka produce failed",
                           extra={"fields": {"error": str(exc)}})
        return None  # the consumer writes to disk; no local path here

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {"publisher": self.name, "topic": self.topic,
                    "produced": self.produced, "errors": self.errors}

    def close(self) -> None:
        try:
            self._p.flush(5)
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
def make_publisher() -> FeedbackPublisher:
    """Build the publisher selected by env; fall back to file on any issue."""
    choice = os.environ.get("FEEDBACK_BUS", "file").strip().lower()
    if choice == "kafka":
        bootstrap = os.environ.get("KAFKA_BOOTSTRAP", "localhost:9092")
        try:
            pub = KafkaPublisher(bootstrap, DEFAULT_TOPIC)
            logger.info("feedback bus = kafka",
                        extra={"fields": {"topic": DEFAULT_TOPIC,
                                          "bootstrap": bootstrap}})
            return pub
        except Exception as exc:  # noqa: BLE001
            logger.warning("kafka publisher unavailable; using file",
                           extra={"fields": {"error": str(exc)}})
    return FilePublisher()
