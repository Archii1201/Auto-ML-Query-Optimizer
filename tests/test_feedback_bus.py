"""
Unit tests for the Phase 4C feedback bus: publishers + consumer dedup.
No Kafka broker required — Kafka is exercised via fakes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from services.exec_service.capture import FeedbackWriter, build_feedback_record
from services.feedback_bus.consumer import Deduper, FeedbackConsumer
from services.feedback_bus.publisher import FilePublisher, KafkaPublisher

PLAN = [{
    "Plan": {"Node Type": "Seq Scan", "Total Cost": 10.0, "Plan Rows": 5,
             "Actual Rows": 5, "Actual Total Time": 1.0},
    "Planning Time": 0.2, "Execution Time": 1.5,
}]


def _kwargs(request_id="r1", variant="default"):
    return dict(sql="SELECT 1", variant=variant, knobs=[], plan_json=PLAN,
                wall_time_ms=1.5, request_id=request_id)


# ----- record builder -----------------------------------------------------
def test_record_has_schema_version_and_dedup_key():
    rec, name, idx = build_feedback_record(**_kwargs())
    assert rec["schema_version"] == "1"
    assert rec["dedup_key"] == "r1:default"
    assert name.endswith(".json")
    assert idx["variant"] == "default"


# ----- FilePublisher ------------------------------------------------------
def test_file_publisher_writes_to_disk(tmp_path):
    pub = FilePublisher(FeedbackWriter(base_dir=tmp_path))
    path = pub.publish(**_kwargs())
    assert path is not None and Path(path).exists()
    rec = json.loads(Path(path).read_text(encoding="utf-8"))
    assert rec["schema_version"] == "1"
    assert pub.stats()["publisher"] == "file"


# ----- KafkaPublisher with a fake producer --------------------------------
class FakeProducer:
    def __init__(self):
        self.msgs = []

    def produce(self, topic, key=None, value=None):
        self.msgs.append((topic, key, value))

    def poll(self, _):
        return 0

    def flush(self, _):
        return 0


def test_kafka_publisher_produces_json_keyed_by_sql_hash():
    fake = FakeProducer()
    pub = KafkaPublisher("x:9092", "query.feedback.v1", producer=fake)
    assert pub.publish(**_kwargs()) is None      # consumer writes disk, not us
    assert len(fake.msgs) == 1
    topic, key, value = fake.msgs[0]
    assert topic == "query.feedback.v1"
    rec = json.loads(value.decode("utf-8"))
    assert key.decode("utf-8") == rec["sql_hash"]
    assert rec["schema_version"] == "1"
    assert pub.stats()["produced"] == 1


# ----- Deduper ------------------------------------------------------------
def test_deduper_persists_and_dedups(tmp_path):
    seen = tmp_path / "_seen.txt"
    d = Deduper(seen)
    assert not d.seen("k1")
    d.add("k1")
    assert d.seen("k1")
    d.close()
    # restart: a new Deduper rehydrates from disk
    d2 = Deduper(seen)
    assert d2.seen("k1")
    d2.close()


# ----- Consumer handle_record --------------------------------------------
def _consumer(tmp_path):
    # pass a dummy consumer object so confluent_kafka is never imported
    return FeedbackConsumer(bootstrap_servers="x", feedback_dir=tmp_path,
                            consumer=object())


def test_consumer_writes_then_dedups(tmp_path):
    c = _consumer(tmp_path)
    rec, _, _ = build_feedback_record(**_kwargs(request_id="rA"))
    assert c.handle_record(rec) == "written"
    assert c.handle_record(rec) == "duplicate"   # same dedup_key
    assert c.written == 1 and c.duplicates == 1
    # the record landed on disk
    assert list(Path(tmp_path).glob("fb_*.json"))
    c.close()


def test_consumer_rejects_bad_schema(tmp_path):
    c = _consumer(tmp_path)
    rec, _, _ = build_feedback_record(**_kwargs(request_id="rB"))
    rec["schema_version"] = "99"
    assert c.handle_record(rec) == "rejected"
    assert c.rejected == 1
    c.close()
