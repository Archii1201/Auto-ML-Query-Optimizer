"""
audit_phase3d.py
================
Diagnostic harness — proves (or disproves) that the system works
the way we *say* it does. Runs four checks:

    1. Feature parity: re-extract features for one offline plan
       record and confirm the inference path produces the same
       feature vector. (Catches feature-alignment bugs.)
    2. Self-prediction: run the deployed model on a plan it was
       *trained on* and compare predicted_ms vs. the actual_ms
       in the CSV. A healthy model should be near-perfect here.
       Large gap => calibration / leakage issue.
    3. Tie analysis: for every (query_id) group of 4 variants in
       the training set, check whether their feature vectors are
       identical. That's the root cause of tied predictions.
    4. Per-regime plan-pick accuracy on training data: re-runs the
       deployed model over every (query_id, variant) row and
       computes plan-pick accuracy *exactly* as the live demo
       would, so we can confirm the offline 51% number is real.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from feature_engineering.extract_features import extract_features_from_record
from feature_engineering.plan_parser import load_plan_record
from phase3a.feature_selection import build_feature_matrix
from services.ml_service.inference import Predictor


# ---------------------------------------------------------------------------
def _section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# ---------------------------------------------------------------------------
def check_feature_parity(predictor: Predictor) -> None:
    _section("1) FEATURE PARITY  —  online inference vs. offline extraction")

    plans_dir = PROJECT_ROOT / "data" / "tpch" / "plans"
    paths = sorted(plans_dir.glob("*.json"))
    if not paths:
        print("[!] no offline plans to compare against; skipping")
        return

    sample = paths[0]
    record = load_plan_record(sample)
    online_features = extract_features_from_record(record, sample)
    print(f"sampled file: {sample.name}")
    print(f"online features computed: {len(online_features)} keys")

    df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "features.csv")
    csv_row = df[df["sql_hash"] == record["sql_hash"]].iloc[0]
    print(f"matched CSV row: query_id={csv_row['query_id']} variant={csv_row['variant']}")

    diffs = []
    for k, v in online_features.items():
        if k in csv_row.index:
            csv_v = csv_row[k]
            try:
                if pd.notna(csv_v) and pd.notna(v):
                    if isinstance(v, (int, float)) and not np.isclose(float(v), float(csv_v)):
                        diffs.append((k, v, csv_v))
                    elif isinstance(v, str) and str(v) != str(csv_v):
                        diffs.append((k, v, csv_v))
            except Exception:
                pass

    if diffs:
        print(f"  [!] {len(diffs)} feature mismatches:")
        for k, online, csv_v in diffs[:5]:
            print(f"       {k:<35} online={online}  csv={csv_v}")
    else:
        print("  [OK] feature vectors match exactly")


# ---------------------------------------------------------------------------
def check_self_prediction(predictor: Predictor, n_samples: int = 30) -> None:
    _section("2) SELF-PREDICTION  —  run model on plans it trained on")

    plan_dirs = [
        PROJECT_ROOT / "data" / "tpch"  / "plans",
        PROJECT_ROOT / "data" / "tpch"  / "plans_param",
        PROJECT_ROOT / "data" / "tpcds" / "plans",
    ]
    paths: list[Path] = []
    for d in plan_dirs:
        paths.extend(sorted(d.glob("*.json")))
    if not paths:
        print("[!] no plan files found"); return

    rng = np.random.default_rng(42)
    sample = rng.choice(paths, size=min(n_samples, len(paths)), replace=False)

    rows = []
    for p in sample:
        rec = load_plan_record(p)
        actual_ms = float(rec.get("plan", [{}])[0].get("Execution Time", 0.0))
        if actual_ms <= 0:
            continue
        result = predictor.predict_one(rec["plan"])
        rows.append({
            "file":        p.name,
            "actual_ms":   actual_ms,
            "predicted":   result.predicted_ms,
            "ratio":       result.predicted_ms / max(actual_ms, 1e-3),
        })

    if not rows:
        print("[!] no usable rows"); return
    df = pd.DataFrame(rows)
    df["abs_ratio"] = np.maximum(df["ratio"], 1.0 / df["ratio"])
    print(df.head(8).to_string(index=False, formatters={
        "actual_ms": "{:>9.1f}".format,
        "predicted": "{:>9.1f}".format,
        "ratio":     "{:>6.2f}".format,
        "abs_ratio": "{:>6.2f}".format,
    }))
    print("\nself-prediction summary:")
    print(f"  median actual / predicted ratio : {df['ratio'].median():.2f}")
    print(f"  median q-error (sym)            : {df['abs_ratio'].median():.2f}")
    print(f"  p95    q-error                  : {df['abs_ratio'].quantile(0.95):.2f}")
    print(f"  mean   actual / predicted ratio : {df['ratio'].mean():.2f}")
    if df["ratio"].mean() > 1.5 or df["ratio"].mean() < 0.66:
        print(f"  [!] systematic bias detected: mean ratio = {df['ratio'].mean():.2f}")
    else:
        print("  [OK] no major systematic bias")


# ---------------------------------------------------------------------------
def check_tied_features() -> None:
    _section("3) TIED FEATURES  —  do the 4 variants of a query produce identical vectors?")

    df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "features.csv")
    fm = build_feature_matrix(df, regime="plan_time")
    feat_df = fm.X.copy()
    feat_df["query_id"] = fm.groups.values
    feat_df["variant"]  = df.loc[feat_df.index, "variant"].values

    tied = 0
    distinct = 0
    examples = []
    for qid, g in feat_df.groupby("query_id"):
        if len(g) < 2:
            continue
        feature_cols = [c for c in g.columns if c not in {"query_id", "variant"}]
        # Hash the rows -> count distinct
        row_hashes = pd.util.hash_pandas_object(g[feature_cols], index=False)
        n_distinct = row_hashes.nunique()
        if n_distinct == 1:
            tied += 1
            if len(examples) < 3:
                examples.append((qid, list(g["variant"])))
        else:
            distinct += 1

    total = tied + distinct
    pct = (tied / total * 100) if total else 0
    print(f"groups analysed                : {total}")
    print(f"groups with ALL-IDENTICAL rows : {tied}  ({pct:.1f}%)")
    print(f"groups with distinct vectors   : {distinct}  ({100-pct:.1f}%)")
    if tied:
        print("\nExamples of ALL-tied groups (model can't differentiate variants here):")
        for qid, vs in examples:
            print(f"  {qid:<30} variants={vs}")
    if pct > 30:
        print(f"\n  [!!] {pct:.0f}% of groups have ZERO inter-variant feature signal.")
        print(f"       Plan-pick accuracy can NEVER exceed (1 - tied_pct/4 * 0.75) on this data.")


# ---------------------------------------------------------------------------
def check_plan_pick_accuracy(predictor: Predictor) -> None:
    _section("4) PLAN-PICK ACCURACY  —  re-derive offline 51% number with the deployed model")

    plan_dirs = [
        PROJECT_ROOT / "data" / "tpch"  / "plans",
        PROJECT_ROOT / "data" / "tpch"  / "plans_param",
        PROJECT_ROOT / "data" / "tpcds" / "plans",
    ]
    paths: list[Path] = []
    for d in plan_dirs:
        paths.extend(sorted(d.glob("*.json")))

    by_qid: dict[str, list[dict]] = defaultdict(list)
    for p in paths:
        rec = load_plan_record(p)
        qid = rec.get("query_id")
        actual_ms = float(rec.get("plan", [{}])[0].get("Execution Time", 0.0))
        if not qid or actual_ms <= 0:
            continue
        pred = predictor.predict_one(rec["plan"])
        by_qid[qid].append({
            "variant":  rec.get("variant", "default"),
            "actual":   actual_ms,
            "pred":     pred.predicted_ms,
        })

    hits = 0
    groups = 0
    regrets = []
    for qid, runs in by_qid.items():
        if len(runs) < 2:
            continue
        groups += 1
        oracle = min(runs, key=lambda r: r["actual"])
        picked = min(runs, key=lambda r: (r["pred"], r["actual"]))   # tiebreak by actual to be fair
        # actually we should tiebreak the way live system does: by estimated cost.
        # But we don't have est_cost here without re-extracting. Use stable-sort: pred then variant.
        picked = min(runs, key=lambda r: (r["pred"], r["variant"]))
        if picked["variant"] == oracle["variant"]:
            hits += 1
        regret = picked["actual"] - oracle["actual"]
        regrets.append(max(regret, 0))

    if groups == 0:
        print("[!] no groups"); return
    acc = hits / groups
    print(f"groups evaluated         : {groups}")
    print(f"plan-pick accuracy       : {hits}/{groups}  =  {acc:.1%}")
    print(f"mean regret (ms)         : {np.mean(regrets):.1f}")
    print(f"median regret (ms)       : {np.median(regrets):.1f}")


# ---------------------------------------------------------------------------
def main() -> int:
    print(f"[i] Loading AutoML winner ...")
    predictor = Predictor(regime="plan_time")
    print(f"    model={predictor.model_name}  features={len(predictor.feature_names)}")

    check_feature_parity(predictor)
    check_self_prediction(predictor)
    check_tied_features()
    check_plan_pick_accuracy(predictor)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
