"""
merge_feedback.py
=================
Phase 5A CLI — merge online feedback into features.csv, then (optionally)
run the dataset-validation gate.

    python scripts/merge_feedback.py                 # dry run (no writes)
    python scripts/merge_feedback.py --apply         # merge + write CSV
    python scripts/merge_feedback.py --apply --gate  # merge, then validate

Exit codes:
    0  success (or dry run)
    2  validation gate failed  (retrain must not proceed)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from services.automl_service.merge import (  # noqa: E402
    merge_feedback,
    run_validation_gate,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 5A feedback merge")
    p.add_argument("--apply", action="store_true",
                   help="actually rewrite features.csv (default: dry run)")
    p.add_argument("--gate", action="store_true",
                   help="run validate_dataset.py --features after applying")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    report = merge_feedback(apply=args.apply)
    print(json.dumps(report.as_dict(), indent=2))

    if args.apply and args.gate:
        print("\n[gate] running dataset validation ...")
        ok, out = run_validation_gate(features=True)
        print(out)
        if not ok:
            print("[gate] FAILED — dataset not trustworthy; aborting.", file=sys.stderr)
            return 2
        print("[gate] PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
