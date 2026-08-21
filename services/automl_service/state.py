"""
state.py
========
Phase 5D — durable state for the retraining loop.

The worker (5E) is a long-running process, but it must survive restarts
without either (a) hammering retrains back-to-back or (b) forgetting that it
just promoted a model. This small JSON file is that memory: last retrain /
promote timestamps, counters, and the features.csv size at the last retrain
(so the volume trigger measures *new* rows since then).

Kept deliberately tiny and separate from the watermark (5A): the watermark
tracks *merge* progress; this tracks *retrain/promote* progress.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE = PROJECT_ROOT / "data" / "processed" / "retrain_state.json"


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime | None) -> str:
    return dt.isoformat() if dt else ""


@dataclass
class RetrainState:
    last_retrain_utc:      str = ""
    last_promote_utc:      str = ""
    last_trigger_reason:   str = ""
    retrains_total:        int = 0
    promotions_total:      int = 0
    rollbacks_total:       int = 0
    features_rows_at_retrain: int = 0
    last_candidate_version:   str = ""
    last_promoted_version:    str = ""

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: Path = DEFAULT_STATE) -> "RetrainState":
        p = Path(path)
        if not p.exists():
            return cls()
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            base = cls()
            return cls(**{k: data.get(k, getattr(base, k)) for k in base.__dict__})
        except Exception:  # noqa: BLE001 — corrupt → start fresh
            return cls()

    def save(self, path: Path = DEFAULT_STATE) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        tmp.replace(p)

    # ------------------------------------------------------------------
    def minutes_since_retrain(self, now: datetime | None = None) -> float | None:
        if not self.last_retrain_utc:
            return None
        now = now or now_utc()
        try:
            last = datetime.fromisoformat(self.last_retrain_utc)
        except ValueError:
            return None
        return (now - last).total_seconds() / 60.0

    def mark_retrain(self, *, reason: str, features_rows: int,
                     candidate_version: str, now: datetime | None = None) -> "RetrainState":
        self.last_retrain_utc = _iso(now or now_utc())
        self.last_trigger_reason = reason
        self.features_rows_at_retrain = features_rows
        self.last_candidate_version = candidate_version
        self.retrains_total += 1
        return self

    def mark_promote(self, version: str, now: datetime | None = None) -> "RetrainState":
        self.last_promote_utc = _iso(now or now_utc())
        self.last_promoted_version = version
        self.promotions_total += 1
        return self

    def mark_rollback(self) -> "RetrainState":
        self.rollbacks_total += 1
        return self
