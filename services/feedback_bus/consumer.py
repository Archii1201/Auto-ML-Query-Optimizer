"""
consumer.py
===========
Phase 4C — Kafka feedback consumer.

Reads `query.feedback.v1`, validates the schema version, **deduplicates**
(Kafka is at-least-once, so the same record can be redelivered after a
rebalance/crash), and writes each record to `data/feedback/` using the
*same* FeedbackWriter the online path uses. This decouples producing
feedback from storing it and is the seam Phase 5 retraining plugs into.

Run:
    python -m services.feedback_bus.consumer

Dedup strategy (DSA: hash set + bounded persistence)
----------------------------------------------------
- In-memory `set` of `dedup_key` (request_id:variant) for O(1) checks.
- The set is also persisted to a small `_seen.txt` so a restart doesn't
  re-write records it already processed. Offsets are committed only
  *after* a record is durably written, so a crash re-delivers at most the
  in-flight record — which the dedup set then drops.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
from collections import deque
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from services.exec_service.capture import (  # noqa: E402
    DEFAULT_FEEDBACK_DIR,
    SCHEMA_VERSION,
    FeedbackWriter,
)

logger = logging.getLogger("feedback_consumer")

DEFAULT_TOPIC = os.environ.get("FEEDBACK_TOPIC", "query.feedback.v1")
DEFAULT_GROUP = os.environ.get("FEEDBACK_GROUP", "feedback-writer")


def _major(version: str) -> str:
    return str(version).split(".", 1)[0]


class Deduper:
    """Bounded set of seen dedup_keys, backed by a small on-disk log."""

    def __init__(self, seen_path: Path, capacity: int = 200_000) -> None:
        self.seen_path = seen_path
        self.capacity = capacity
        self._set: set[str] = set()
        self._order: deque[str] = deque()
        if seen_path.exists():
            for line in seen_path.read_text(encoding="utf-8").splitlines():
                k = line.strip()
                if k:
                    self._remember(k)
        self._fh = seen_path.open("a", encoding="utf-8")

    def _remember(self, key: str) -> None:
        self._set.add(key)
        self._order.append(key)
        while len(self._order) > self.capacity:
            old = self._order.popleft()
            self._set.discard(old)

    def seen(self, key: str) -> bool:
        return key in self._set

    def add(self, key: str) -> None:
        if key in self._set:
            return
        self._remember(key)
        self._fh.write(key + "\n")
        self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:  # noqa: BLE001
            pass


class FeedbackConsumer:
    def __init__(
        self,
        *,
        bootstrap_servers: str,
        topic: str = DEFAULT_TOPIC,
        group_id: str = DEFAULT_GROUP,
        feedback_dir: Path = DEFAULT_FEEDBACK_DIR,
        consumer: Any | None = None,
    ) -> None:
        self.topic = topic
        self.writer = FeedbackWriter(base_dir=feedback_dir)
        self.deduper = Deduper(Path(feedback_dir) / "_seen.txt")
        self._running = False
        self.written = 0
        self.duplicates = 0
        self.rejected = 0

        if consumer is not None:
            self._c = consumer
        else:
            from confluent_kafka import Consumer  # lazy import
            self._c = Consumer({
                "bootstrap.servers": bootstrap_servers,
                "group.id": group_id,
                "auto.offset.reset": "earliest",
                "enable.auto.commit": False,  # commit only after a durable write
            })
            self._c.subscribe([topic])

    # ------------------------------------------------------------------
    def handle_record(self, record: dict[str, Any]) -> str:
        """
        Validate + dedup + persist a single record.
        Returns one of: 'written' | 'duplicate' | 'rejected'.
        """
        if _major(record.get("schema_version", "0")) != _major(SCHEMA_VERSION):
            self.rejected += 1
            logger.warning("rejecting record with incompatible schema",
                           extra={"fields": {"got": record.get("schema_version"),
                                             "want": SCHEMA_VERSION}})
            return "rejected"

        key = record.get("dedup_key")
        if key and self.deduper.seen(key):
            self.duplicates += 1
            return "duplicate"

        self.writer.write_record(record)
        if key:
            self.deduper.add(key)
        self.written += 1
        return "written"

    # ------------------------------------------------------------------
    def run(self, *, poll_timeout: float = 1.0) -> None:
        self._running = True
        signal.signal(signal.SIGINT, self._stop)
        signal.signal(signal.SIGTERM, self._stop)
        logger.info("consumer started",
                    extra={"fields": {"topic": self.topic}})
        try:
            while self._running:
                msg = self._c.poll(poll_timeout)
                if msg is None:
                    continue
                if msg.error():
                    logger.warning("kafka error",
                                   extra={"fields": {"error": str(msg.error())}})
                    continue
                try:
                    record = json.loads(msg.value().decode("utf-8"))
                except Exception as exc:  # noqa: BLE001 — poison message
                    self.rejected += 1
                    logger.warning("bad message json",
                                   extra={"fields": {"error": str(exc)}})
                    self._c.commit(msg, asynchronous=False)
                    continue
                self.handle_record(record)
                # Commit only after the record is durably written/deduped.
                self._c.commit(msg, asynchronous=False)
        finally:
            self.close()

    def _stop(self, *_a) -> None:
        logger.info("consumer stopping")
        self._running = False

    def stats(self) -> dict[str, Any]:
        return {"topic": self.topic, "written": self.written,
                "duplicates": self.duplicates, "rejected": self.rejected}

    def close(self) -> None:
        try:
            self._c.close()
        except Exception:  # noqa: BLE001
            pass
        self.deduper.close()
        logger.info("consumer closed", extra={"fields": self.stats()})


def main() -> int:
    from services.ml_service.obs_logging import configure_logging
    configure_logging(os.environ.get("ML_LOG_LEVEL", "INFO"))
    bootstrap = os.environ.get("KAFKA_BOOTSTRAP", "localhost:9092")
    consumer = FeedbackConsumer(bootstrap_servers=bootstrap)
    consumer.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
