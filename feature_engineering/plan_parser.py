"""
plan_parser.py
==============
Loads and *normalizes* the JSON plan records produced by Phase 1
(scripts/collect_data.py) and Phase 2A (scripts/collect_tpch_plans.py).

Both phases write a record that looks like:

    {
        "query_id":      "...",
        "variant":       "..."          # phase 2A only
        "tag":           "...",
        "sql":           "...",
        "sql_hash":      "...",
        "collected_at":  "...",
        "wall_time_ms":  ...,
        "summary":       { ... },
        "plan":          [ { "Plan": {... root ...},
                             "Planning Time": ...,
                             "Execution Time": ...,
                             "Triggers": [] } ]
    }

This module hides the small differences between the two phases so the
feature extractor sees a single, predictable interface:

    record           = load_plan_record(path)
    root_node        = get_root_plan_node(record)
    top_metrics      = get_top_level_metrics(record)
    metadata         = get_record_metadata(record, path)

A clean separation between "I/O + normalization" (here) and
"feature math" (extract_features.py) is what keeps the pipeline
testable and easy to extend (CSV today, Parquet tomorrow,
streaming later).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------
class PlanParseError(ValueError):
    """Raised when a plan JSON file is structurally unusable."""


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def load_plan_record(path: Path) -> dict[str, Any]:
    """Read a single plan-record JSON file from disk."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PlanParseError(f"cannot read {path}: {exc}") from exc

    try:
        record = json.loads(text)
    except json.JSONDecodeError as exc:
        raise PlanParseError(f"invalid JSON in {path}: {exc}") from exc

    if not isinstance(record, dict):
        raise PlanParseError(f"top-level value in {path} is not an object")
    if "plan" not in record:
        raise PlanParseError(f"record in {path} has no 'plan' field")

    return record


# ---------------------------------------------------------------------------
# Plan tree extraction
# ---------------------------------------------------------------------------
def get_root_plan_node(record: dict[str, Any]) -> dict[str, Any]:
    """
    Return the root *operator* dict — i.e. record['plan'][0]['Plan'].

    EXPLAIN (FORMAT JSON) always wraps its output as:
        [ { "Plan": { ... }, "Planning Time": ..., "Execution Time": ... } ]
    even for a single statement. We dig through that wrapper here so the
    extractor never has to think about it.
    """
    plan = record.get("plan")
    if not isinstance(plan, list) or not plan:
        raise PlanParseError("record['plan'] is empty or not a list")

    outer = plan[0]
    if not isinstance(outer, dict):
        raise PlanParseError("record['plan'][0] is not an object")

    root = outer.get("Plan")
    if not isinstance(root, dict):
        raise PlanParseError("record['plan'][0]['Plan'] is missing")

    return root


def get_top_level_metrics(record: dict[str, Any]) -> dict[str, float]:
    """
    Return planning + execution time from the *outer* EXPLAIN wrapper
    (these don't live on the root node itself).
    """
    plan = record.get("plan") or []
    outer = plan[0] if plan else {}
    return {
        "planning_time_ms":  float(outer.get("Planning Time")  or 0.0),
        "execution_time_ms": float(outer.get("Execution Time") or 0.0),
    }


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
def get_record_metadata(record: dict[str, Any], path: Path) -> dict[str, Any]:
    """
    Pull the identification / provenance fields off the record so every
    CSV row remembers where its features came from.
    """
    return {
        "source_file":        path.name,
        "query_id":           record.get("query_id"),
        "variant":            record.get("variant", "default"),
        "tag":                record.get("tag"),
        "sql_hash":           record.get("sql_hash"),
        "collected_at":       record.get("collected_at"),
        "wall_time_ms":       float(record.get("wall_time_ms") or 0.0),
        "target_variance_ms": record.get("target_variance_ms", ""),
        "label_runs":         record.get("label_runs", ""),
    }
