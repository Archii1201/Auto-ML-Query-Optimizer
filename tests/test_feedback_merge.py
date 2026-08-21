"""
Unit tests for Phase 5A — feedback merge (pure dedup + watermark).

The disk-scan path needs plan JSON + feature extraction; we test the pure
dedup core and the watermark bookkeeping with tiny DataFrames instead, so
these tests stay fast and infra-free.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from services.automl_service.merge import dedupe_new_rows
from services.automl_service.watermark import Watermark
from scripts.feedback_to_features import KEY_COLS


def _row(sql_hash, variant, collected_at, value=1.0):
    return {"sql_hash": sql_hash, "variant": variant,
            "collected_at": collected_at, "feat": value}


def test_dedupe_all_new_when_base_empty():
    base = pd.DataFrame(columns=[*KEY_COLS, "feat"])
    fb = pd.DataFrame([_row("a", "default", "t1"), _row("b", "no_hashjoin", "t2")])
    out = dedupe_new_rows(base, fb)
    assert len(out) == 2


def test_dedupe_filters_existing_identity():
    base = pd.DataFrame([_row("a", "default", "t1")])
    fb = pd.DataFrame([
        _row("a", "default", "t1"),        # duplicate -> dropped
        _row("a", "no_hashjoin", "t1"),    # different variant -> kept
        _row("c", "default", "t9"),        # new -> kept
    ])
    out = dedupe_new_rows(base, fb)
    keys = set(map(tuple, out[list(KEY_COLS)].astype(str).itertuples(index=False, name=None)))
    assert ("a", "default", "t1") not in keys
    assert ("a", "no_hashjoin", "t1") in keys
    assert ("c", "default", "t9") in keys
    assert len(out) == 2


def test_dedupe_empty_feedback_returns_empty():
    base = pd.DataFrame([_row("a", "default", "t1")])
    fb = pd.DataFrame(columns=[*KEY_COLS, "feat"])
    assert dedupe_new_rows(base, fb).empty


def test_dedupe_is_idempotent():
    base = pd.DataFrame([_row("a", "default", "t1")])
    fb = pd.DataFrame([_row("x", "default", "t2")])
    first = dedupe_new_rows(base, fb)
    base2 = pd.concat([base, first], ignore_index=True)
    second = dedupe_new_rows(base2, fb)     # same feedback again
    assert first.shape[0] == 1
    assert second.empty


# ----- watermark ----------------------------------------------------------
def test_watermark_roundtrip(tmp_path):
    wm_path = tmp_path / "wm.json"
    wm = Watermark.load(wm_path)                 # missing -> defaults
    assert wm.total_merged == 0
    wm.bump(features_rows=100, feedback_files=10, new_rows=7).save(wm_path)
    again = Watermark.load(wm_path)
    assert again.features_rows == 100
    assert again.feedback_files == 10
    assert again.total_merged == 7
    assert again.last_run_utc != ""


def test_watermark_accumulates(tmp_path):
    wm_path = tmp_path / "wm.json"
    Watermark.load(wm_path).bump(features_rows=10, feedback_files=1, new_rows=3).save(wm_path)
    Watermark.load(wm_path).bump(features_rows=15, feedback_files=2, new_rows=5).save(wm_path)
    assert Watermark.load(wm_path).total_merged == 8


def test_watermark_corrupt_file_resets(tmp_path):
    wm_path = tmp_path / "wm.json"
    wm_path.write_text("{not json", encoding="utf-8")
    assert Watermark.load(wm_path).total_merged == 0
