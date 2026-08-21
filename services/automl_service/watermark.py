"""
watermark.py
============
Phase 5A — persistent bookmark for the feedback→features merge.

Why: the scheduler (5D/5E) needs a cheap way to answer "is there new
feedback worth merging / retraining on?" without re-scanning and
re-extracting the whole feedback corpus every poll. We persist a small
JSON bookmark after each successful merge.

The watermark is advisory only — losing it just means the next merge
re-scans everything and dedupes (idempotent), so it can never cause data
loss or duplication.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_WATERMARK = PROJECT_ROOT / "data" / "processed" / "merge_watermark.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Watermark:
    last_run_utc:   str = ""
    features_rows:  int = 0     # rows in features.csv after the last merge
    feedback_files: int = 0     # feedback files seen at last merge
    total_merged:   int = 0     # cumulative NEW rows ever merged in

    @classmethod
    def load(cls, path: Path = DEFAULT_WATERMARK) -> "Watermark":
        if not Path(path).exists():
            return cls()
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            return cls(**{k: data.get(k, getattr(cls(), k)) for k in cls().__dict__})
        except Exception:  # noqa: BLE001 — corrupt bookmark → start fresh
            return cls()

    def save(self, path: Path = DEFAULT_WATERMARK) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        tmp.replace(p)

    def bump(self, *, features_rows: int, feedback_files: int, new_rows: int) -> "Watermark":
        self.last_run_utc = _now_iso()
        self.features_rows = features_rows
        self.feedback_files = feedback_files
        self.total_merged += max(0, new_rows)
        return self
