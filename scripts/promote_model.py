"""
promote_model.py
================
Phase 5C CLI — evaluate a candidate against the incumbent and, only if it
clears the promotion gate, flip `current` and (optionally) hot-swap the
live service.

    # dry-run the gate for the most recently registered candidate:
    python scripts/promote_model.py --regime plan_time --candidate latest

    # promote if it passes, and tell the running service to reload:
    python scripts/promote_model.py --regime plan_time --candidate latest \
        --apply --reload-url http://localhost:8000

Exit codes:
    0  gate passed (and promoted if --apply)
    3  gate FAILED (candidate not promoted)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from services.automl_service.promotion import (  # noqa: E402
    PromotionPolicy,
    decide,
    evaluate_candidate,
)
from services.ml_service.model_registry import REGISTRY  # noqa: E402

FEATURES_CSV = PROJECT_ROOT / "data" / "processed" / "features.csv"
REPORTS_DIR = PROJECT_ROOT / "models" / "registry" / "reports"


def _resolve_candidate(regime: str, candidate: str) -> str:
    if candidate != "latest":
        return candidate
    versions = REGISTRY.list_versions(regime)
    non_current = [v for v in versions if not v["is_current"]]
    pool = non_current or versions
    if not pool:
        raise SystemExit(f"no registered versions for regime {regime!r}")
    return pool[0]["version"]  # list_versions is newest-first


def _trigger_reload(base_url: str, token: str) -> dict:
    import urllib.request

    req = urllib.request.Request(
        base_url.rstrip("/") + "/admin/reload-models",
        method="POST",
        headers={"X-Admin-Token": token},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
        return json.loads(resp.read().decode("utf-8"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 5C promotion gate")
    p.add_argument("--regime", default="plan_time")
    p.add_argument("--candidate", default="latest",
                   help="candidate version, or 'latest'")
    p.add_argument("--apply", action="store_true",
                   help="actually promote if the gate passes")
    p.add_argument("--reload-url", default=None,
                   help="ml-service base URL to POST /admin/reload-models")
    p.add_argument("--admin-token", default=None,
                   help="X-Admin-Token for the reload call (or ML_ADMIN_TOKEN)")
    return p.parse_args()


def main() -> int:
    import os
    args = parse_args()

    if not FEATURES_CSV.exists():
        print(f"[!] {FEATURES_CSV} not found; run extract/merge first.", file=sys.stderr)
        return 1
    df = pd.read_csv(FEATURES_CSV)

    cand_version = _resolve_candidate(args.regime, args.candidate)
    inc_version = REGISTRY.current_version(args.regime)
    if inc_version == cand_version:
        inc_version = None  # candidate already current or first-ever model

    print(f"[i] regime={args.regime}  candidate={cand_version}  incumbent={inc_version}")
    cand, inc = evaluate_candidate(args.regime, cand_version, inc_version, df, REGISTRY)
    decision = decide(cand, inc, PromotionPolicy())

    print("\n".join(decision.reasons))
    print(f"\nGATE: {'PROMOTE' if decision.promote else 'REJECT'}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / f"{cand_version}.json").write_text(
        json.dumps({"candidate": cand, "incumbent": inc,
                    "decision": decision.as_dict()}, indent=2, default=str),
        encoding="utf-8",
    )

    if not decision.promote:
        return 3

    if args.apply:
        REGISTRY.promote(args.regime, cand_version)
        print(f"[OK] promoted {args.regime} -> {cand_version}")
        if args.reload_url:
            token = args.admin_token or os.environ.get("ML_ADMIN_TOKEN", "")
            try:
                out = _trigger_reload(args.reload_url, token)
                print(f"[OK] service reloaded: {json.dumps(out.get('changed', {}))}")
            except Exception as exc:  # noqa: BLE001
                print(f"[!] reload call failed (model still promoted): {exc}",
                      file=sys.stderr)
    else:
        print("    (dry run; pass --apply to promote)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
