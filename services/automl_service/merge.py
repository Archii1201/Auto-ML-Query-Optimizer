"""
merge.py
========
Phase 5A — merge online feedback into the offline training set, then run
the dataset-validation gate.

    data/feedback/fb_*.json ──► data/processed/features.csv  (deduped)
                                              │
                                              ▼
                              validate_dataset.py --features  (HARD GATE)

Design
------
- The dedup logic is a **pure function** (`dedupe_new_rows`) so it can be
  unit-tested with tiny DataFrames — no plan JSON, no feature extraction.
- The disk-driven path reuses `scripts/feedback_to_features.extract_feedback`
  so feature math is identical to the offline pipeline (no drift — the
  Phase 3D parity lesson).
- The validation gate is shelled out to `scripts/validate_dataset.py` so a
  merge that would introduce corrupt/incoherent rows aborts the retrain,
  exactly like the Phase 3E.4 gate.

Idempotency: rows are keyed by (sql_hash, variant, collected_at). Running
the merge twice adds nothing the second time.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.feedback_to_features import (  # noqa: E402
    FEATURES_CSV,
    FEEDBACK_DIR,
    KEY_COLS,
    extract_feedback,
)
from services.automl_service.watermark import DEFAULT_WATERMARK, Watermark

VALIDATE_SCRIPT = PROJECT_ROOT / "scripts" / "validate_dataset.py"


@dataclass
class MergeReport:
    scanned_feedback_files: int
    feedback_rows:          int
    new_rows:               int
    duplicates:             int
    features_before:        int
    features_after:         int
    applied:                bool

    def as_dict(self) -> dict:
        return self.__dict__.copy()


# ---------------------------------------------------------------------------
# Pure core — unit-testable without any plan JSON.
# ---------------------------------------------------------------------------
def dedupe_new_rows(base: pd.DataFrame, feedback: pd.DataFrame) -> pd.DataFrame:
    """
    Return only the feedback rows whose (sql_hash, variant, collected_at)
    identity is NOT already present in `base`. Order preserved.
    """
    if feedback.empty:
        return feedback.iloc[0:0]
    if base.empty:
        return feedback.reset_index(drop=True)

    key = list(KEY_COLS)
    base_keys = set(map(tuple, base[key].astype(str).itertuples(index=False, name=None)))
    fb_keys = list(map(tuple, feedback[key].astype(str).itertuples(index=False, name=None)))
    mask = [k not in base_keys for k in fb_keys]
    return feedback.loc[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Disk-driven merge.
# ---------------------------------------------------------------------------
def merge_feedback(
    *,
    feedback_dir: Path = FEEDBACK_DIR,
    features_csv: Path = FEATURES_CSV,
    watermark_path: Path = DEFAULT_WATERMARK,
    apply: bool = False,
) -> MergeReport:
    feedback_dir = Path(feedback_dir)
    features_csv = Path(features_csv)

    n_files = sum(1 for _ in feedback_dir.glob("fb_*.json")) if feedback_dir.exists() else 0
    fb = extract_feedback(feedback_dir) if feedback_dir.exists() else pd.DataFrame()

    base = pd.read_csv(features_csv) if features_csv.exists() else pd.DataFrame()
    before = len(base)

    new_rows = dedupe_new_rows(base, fb) if not fb.empty else fb
    n_new = len(new_rows)
    n_dupe = (len(fb) - n_new) if not fb.empty else 0

    after = before
    if apply and n_new:
        merged = pd.concat([base, new_rows], ignore_index=True) if not base.empty else new_rows
        features_csv.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(features_csv, index=False)
        after = len(merged)
        Watermark.load(watermark_path).bump(
            features_rows=after, feedback_files=n_files, new_rows=n_new,
        ).save(watermark_path)

    return MergeReport(
        scanned_feedback_files=n_files,
        feedback_rows=int(len(fb)),
        new_rows=int(n_new),
        duplicates=int(n_dupe),
        features_before=int(before),
        features_after=int(after),
        applied=bool(apply and n_new),
    )


# ---------------------------------------------------------------------------
# Validation gate (shell out to the Phase 3E.4 validator).
# ---------------------------------------------------------------------------
def run_validation_gate(*, features: bool = True, timeout_s: int = 600) -> tuple[bool, str]:
    """
    Run scripts/validate_dataset.py. Returns (passed, combined_output).
    A non-zero exit means a HARD check failed → caller must abort retrain.
    """
    cmd = [sys.executable, str(VALIDATE_SCRIPT)]
    if features:
        cmd.append("--features")
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s,
            cwd=str(PROJECT_ROOT),
        )
    except subprocess.TimeoutExpired:
        return False, "validation gate timed out"
    out = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode == 0, out
