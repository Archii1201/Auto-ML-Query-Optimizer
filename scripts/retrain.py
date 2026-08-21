"""
retrain.py
==========
Phase 5B CLI — run one retrain and register the candidate.

    python scripts/retrain.py --profile fast          # merge + fast train
    python scripts/retrain.py --profile full           # merge + tuned train
    python scripts/retrain.py --profile fast --no-merge # train on current CSV
    python scripts/retrain.py --profile fast --no-gate  # skip validation gate

Registers the result as a CANDIDATE only. Promotion is a separate,
gated step (scripts/promote_model.py, Phase 5C).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from services.automl_service.config import get_profile  # noqa: E402
from services.automl_service.trainer import run_retrain  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 5B retrain")
    p.add_argument("--profile", default="fast", choices=["fast", "full"])
    p.add_argument("--no-merge", action="store_true",
                   help="train on current features.csv without merging feedback")
    p.add_argument("--no-gate", action="store_true",
                   help="skip the dataset validation gate")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    profile = get_profile(args.profile)
    result = run_retrain(
        profile,
        do_merge=not args.no_merge,
        gate=not args.no_gate,
    )
    print(json.dumps(result.as_dict(), indent=2))
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
