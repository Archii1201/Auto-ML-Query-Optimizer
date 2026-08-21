# Phase 4C — Streaming Feedback Bus (Kafka)

**Goal:** decouple *producing* feedback (the request path) from *storing*
it, enabling real-time, multi-consumer, replayable learning — the
"Apache Kafka — collect execution logs, enable real-time learning" box
from the system-flow doc.

---

## What was wrong
`ExecutionRunner` wrote feedback straight to local disk inside the request
path. That couples the hot path to disk I/O and means only one process can
ever read the feedback (no fan-out to training + dashboards + audit), and
there's no durable, replayable log.

## What we do

### Producer side — pluggable publisher (Strategy pattern)
`services/feedback_bus/publisher.py`

```
FEEDBACK_BUS=file   ->  FilePublisher    (default; writes data/feedback/)
FEEDBACK_BUS=kafka  ->  KafkaPublisher   (produces query.feedback.v1)
```

- `FeedbackPublisher` ABC: `publish(**kwargs) / stats / close`.
- `KafkaPublisher` produces JSON **keyed by `sql_hash`** so all records for
  a query land on the same partition (ordered per query) and returns
  `None` for the local path — the consumer is what lands records on disk.
- One record schema, one definition: both publisher and consumer build
  records via `build_feedback_record(...)` (extracted from `capture.py`),
  so a record produced on one host and consumed on another is byte-identical.

### Consumer side — `services/feedback_bus/consumer.py`
A standalone service that reads the topic and writes to `data/feedback/`
using the *same* `FeedbackWriter.write_record(...)` the online path uses,
so the training corpus is identical regardless of transport.

### Schema versioning
Every record carries `schema_version` (currently `"1"`). The consumer
**rejects records with an incompatible major** instead of silently
mis-parsing — forward/backward compatibility as the schema evolves.

### Deduplication (Kafka is at-least-once)
After a rebalance or crash, Kafka can redeliver a record. Each record has a
stable `dedup_key` (`request_id:variant`). The consumer keeps a bounded
hash-set of seen keys (DSA: set + deque eviction), **persisted to
`_seen.txt`** so a restart doesn't re-write processed records. Offsets are
committed **only after** a record is durably written, so a crash
re-delivers at most the in-flight record — which the dedup set drops.

## Why these choices
- **Strategy pattern again** so tests and dev stay zero-infra (file
  default) and prod flips one env var.
- **Key by sql_hash** for per-query ordering and even partition spread.
- **Manual offset commits + dedup** give effectively-once *storage* on top
  of Kafka's at-least-once *delivery*.
- **Fail-open:** if `FEEDBACK_BUS=kafka` but Kafka is down, the publisher
  counts produce errors and never fails the request; the factory falls
  back to file if `confluent-kafka` isn't installed.
- **KRaft mode** (compose) — no ZooKeeper, one fewer moving part.

## Config
`FEEDBACK_BUS`, `KAFKA_BOOTSTRAP`, `FEEDBACK_TOPIC`, `FEEDBACK_GROUP`.
Publisher stats appear in `GET /metrics` under `feedback`.

## Run the consumer
```bash
FEEDBACK_BUS=kafka KAFKA_BOOTSTRAP=localhost:9092 \
  python -m services.feedback_bus.consumer
# or, in compose:
docker compose --profile streaming up feedback-consumer
```

## Tests
`tests/test_feedback_bus.py` — record builder, FilePublisher, KafkaPublisher
(fake producer), Deduper persistence, consumer write/dedup/schema-reject.
6 tests, no broker required.
