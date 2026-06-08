"""
feedback_to_features.py
=======================
One-shot CLI that promotes online feedback rows into the offline
training set:

    data/feedback/fb_*.json   ─►   data/processed/features.csv
                                (appended, deduplicated)

This is what closes the offline / online divide. Phase 5 will call
this script (or its core function) on a schedule.

Behaviour:
    * Runs the existing feature-extraction pipeline against
      `data/feedback/` only.
    * Joins those rows against the existing `features.csv` by
      `sql_hash + variant + collected_at`. If a feedback row's
      identity already exists, we skip it.
    * Writes the merged CSV back, in place.

By default this is a *dry run*; pass `--apply` to actually mutate
features.csv.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from feature_engineering.extract_features import (  # noqa: E402
    ALL_COLUMNS,
    extract_features_from_record,
    iter_plan_files,
)
from feature_engineering.plan_parser import (  # noqa: E402
    PlanParseError,
    load_plan_record,
)

FEATURES_CSV = PROJECT_ROOT / "data" / "processed" / "features.csv"
FEEDBACK_DIR = PROJECT_ROOT / "data" / "feedback"

KEY_COLS = ("sql_hash", "variant", "collected_at")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--apply", action="store_true",
                   help="actually rewrite features.csv (default: dry run)")
    p.add_argument("--feedback-dir", type=Path, default=FEEDBACK_DIR)
    p.add_argument("--features-csv", type=Path, default=FEATURES_CSV)
    return p.parse_args()


def extract_feedback(feedback_dir: Path) -> pd.DataFrame:
    rows: list[dict] = []
    failed = 0
    for path in iter_plan_files([feedback_dir]):
        try:
            rec = load_plan_record(path)
            rows.append(extract_features_from_record(rec, path))
        except PlanParseError as exc:
            print(f"[!] skipping {path.name}: {exc}", file=sys.stderr)
            failed += 1
    if not rows:
        return pd.DataFrame(columns=list(ALL_COLUMNS))
    df = pd.DataFrame(rows, columns=list(ALL_COLUMNS))
    print(f"[i] feedback rows extracted: {len(df)} (failed: {failed})")
    return df


def main() -> int:
    args = parse_args()

    if not args.feedback_dir.exists():
        print(f"[!] feedback dir not found: {args.feedback_dir}", file=sys.stderr)
        return 1

    print(f"[i] reading feedback from {args.feedback_dir}")
    fb = extract_feedback(args.feedback_dir)
    if fb.empty:
        print("[i] no feedback rows; nothing to do.")
        return 0

    if not args.features_csv.exists():
        print(f"[!] {args.features_csv} not found — write fresh CSV from feedback only")
        if args.apply:
            args.features_csv.parent.mkdir(parents=True, exist_ok=True)
            fb.to_csv(args.features_csv, index=False)
            print(f"[OK] wrote {len(fb)} rows -> {args.features_csv}")
        else:
            print("    (dry run; pass --apply to write)")
        return 0

    base = pd.read_csv(args.features_csv)
    print(f"[i] base features.csv: {len(base)} rows")

    base_keys = set(map(tuple, base[list(KEY_COLS)].astype(str).itertuples(index=False, name=None)))
    fb_keys = list(map(tuple, fb[list(KEY_COLS)].astype(str).itertuples(index=False, name=None)))
    new_mask = [k not in base_keys for k in fb_keys]
    new_rows = fb.loc[new_mask].reset_index(drop=True)
    print(f"[i] feedback rows that are NEW: {len(new_rows)} / {len(fb)}")

    merged = pd.concat([base, new_rows], ignore_index=True)
    print(f"[i] merged size: {len(merged)} rows  (delta={len(new_rows)})")

    if args.apply and len(new_rows):
        merged.to_csv(args.features_csv, index=False)
        print(f"[OK] rewrote {args.features_csv}")
    elif not args.apply:
        print("    (dry run; pass --apply to write)")
    else:
        print("[i] nothing new; not rewriting.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
