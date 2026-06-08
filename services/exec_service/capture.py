"""
capture.py
==========
Persists execution traces to `data/feedback/` in the *same* JSON
record schema the offline collectors use (Phase 1 / 2A / 3B). That
means:

    * `feature_engineering/extract_features.py` can directly
      ingest these files — no schema changes needed.
    * The retraining loop in Phase 5 just runs feature extraction
      against `data/raw/` ∪ `data/tpch/plans*` ∪ `data/tpcds/plans`
      ∪ `data/feedback/` and gets a unified CSV.

In addition to the standard fields, feedback rows carry online-
specific metadata (`predicted_ms`, `model_name`, `selected_by`,
`request_id`) so we can audit how often the model's prediction
matched reality.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_FEEDBACK_DIR = PROJECT_ROOT / "data" / "feedback"
INDEX_FILE_NAME      = "_index.jsonl"


def _short_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


def _summary_from_plan(plan_json: list[dict[str, Any]],
                       wall_time_ms: float) -> dict[str, Any]:
    """Extract the same fields scripts/collect_data.extract_summary makes."""
    if not plan_json:
        return {}
    outer = plan_json[0]
    root  = outer.get("Plan", {}) if isinstance(outer, dict) else {}
    return {
        "estimated_total_cost":  root.get("Total Cost"),
        "estimated_rows":        root.get("Plan Rows"),
        "actual_rows":           root.get("Actual Rows"),
        "actual_total_time_ms":  root.get("Actual Total Time"),
        "execution_time_ms":     outer.get("Execution Time", wall_time_ms),
        "planning_time_ms":      outer.get("Planning Time"),
        "node_type":             root.get("Node Type"),
    }


# ---------------------------------------------------------------------------
class FeedbackWriter:
    """
    Atomically writes one record per execution.

    File names:
        feedback_{collected_at_compact}_{request_id_short}_{sql_hash}_{variant}.json

    Why uuid in the filename? We need uniqueness even when the same
    SQL is executed twice in the same millisecond. The request_id
    prefix also lets you correlate back to access logs / traces.
    """

    def __init__(self, base_dir: Path = DEFAULT_FEEDBACK_DIR) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.base_dir / INDEX_FILE_NAME
        self._index_lock = Lock()
        self._counter = 0

    # ------------------------------------------------------------------
    def write(
        self,
        *,
        sql:           str,
        variant:       str,
        knobs:         list[str],
        plan_json:     list[dict[str, Any]],
        wall_time_ms:  float,
        predicted_ms:  float | None = None,
        model_name:    str | None   = None,
        regime:        str | None   = None,
        selected_by:   str          = "ml",         # "ml" | "default" | "user"
        request_id:    str | None   = None,
        tag:           str          = "online",
        extra:         dict[str, Any] | None = None,
    ) -> Path:
        """Write one record to disk and append a one-line summary to _index.jsonl."""
        request_id = request_id or uuid.uuid4().hex
        sql_clean  = sql.rstrip().rstrip(";")
        sql_hash   = _short_hash(sql_clean)
        ts_iso     = datetime.now(timezone.utc).isoformat()
        ts_compact = ts_iso.replace(":", "").replace("-", "").split(".")[0]

        record: dict[str, Any] = {
            "query_id":        f"online_{sql_hash}",
            "variant":         variant,
            "tag":             tag,
            "sql":             sql_clean,
            "sql_hash":        sql_hash,
            "collected_at":    ts_iso,
            "wall_time_ms":    round(float(wall_time_ms), 3),
            "summary":         _summary_from_plan(plan_json, wall_time_ms),
            "plan":            plan_json,
            # ----- Phase 3D online-only metadata -----
            "online": {
                "request_id":   request_id,
                "predicted_ms": (round(float(predicted_ms), 3)
                                 if predicted_ms is not None else None),
                "model_name":   model_name,
                "regime":       regime,
                "selected_by":  selected_by,
                "knobs":        list(knobs),
                **(extra or {}),
            },
        }

        out_name = (
            f"fb_{ts_compact}_{request_id[:8]}_{sql_hash}_{variant}.json"
        )
        out_path = self.base_dir / out_name

        # Atomic write: tmp -> rename so a partially-written file is
        # never picked up by feature_extraction.
        tmp_path = out_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        tmp_path.replace(out_path)

        # Append to index (one line per record).
        index_line = {
            "query_id":     record["query_id"],
            "variant":      variant,
            "sql_hash":     sql_hash,
            "collected_at": ts_iso,
            "wall_time_ms": record["wall_time_ms"],
            "predicted_ms": record["online"]["predicted_ms"],
            "model_name":   model_name,
            "selected_by":  selected_by,
            "file":         out_name,
        }
        with self._index_lock, self.index_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(index_line) + "\n")
            self._counter += 1
        return out_path

    # ------------------------------------------------------------------
    def stats(self) -> dict[str, Any]:
        files = list(self.base_dir.glob("fb_*.json"))
        index_lines = 0
        if self.index_path.exists():
            with self.index_path.open("r", encoding="utf-8") as f:
                index_lines = sum(1 for _ in f)
        return {
            "feedback_dir":  str(self.base_dir),
            "files_on_disk": len(files),
            "index_lines":   index_lines,
            "session_writes": self._counter,
        }
