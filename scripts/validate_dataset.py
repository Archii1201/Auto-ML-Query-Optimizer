"""
validate_dataset.py
===================
Phase 3E.4 — Dataset Validation gate.

A dataset is only worth training on if it is *complete, clean, and
trustworthy*. This script audits the collected plan corpus (and,
optionally, the extracted features.csv) and prints a report card. It
exits non-zero if any HARD check fails, so it can be wired into the
pipeline as a gate:

    python scripts/validate_dataset.py            # validate plans
    python scripts/validate_dataset.py --features # also validate features.csv

Checks (plans)
--------------
    * every plan file is valid, non-corrupt JSON
    * required keys present (query_id, variant, sql_hash, summary, plan)
    * all expected (query_id, variant) pairs collected (from the
      parameterized-query generator) — reports missing / extra
    * all 22 base TPC-H queries represented
    * all 4 join-knob variants present per query
    * no duplicate (query_id, variant) collected twice
    * execution-time distribution (sanity: no zero / negative times)
    * label-quality: label_runs / target_variance_ms coverage
    * SCHEMA INTEGRITY (guards the bug we just fixed): customer-join
      queries must scan a 150k-row TPC-H `customer`, not the 10k-row
      TPC-DS one. We assert the max Actual Rows on any `customer` scan
      across the corpus exceeds 10,000.

Checks (features, with --features)
----------------------------------
    * row / column counts
    * NaN / inf cells
    * duplicate feature vectors
    * knob features present (enable_hashjoin/mergejoin/nestloop)
    * plan_rows features present
    * metadata present (query_id, variant, sql_hash, label_runs)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from db.tpch_param_queries import generate as generate_param_queries  # noqa: E402

PLANS_DIR = PROJECT_ROOT / "data" / "tpch" / "plans_param"
FEATURES_CSV = PROJECT_ROOT / "data" / "processed" / "features.csv"

VARIANTS = ("default", "no_hashjoin", "no_mergejoin", "no_nestloop")
REQUIRED_KEYS = ("query_id", "variant", "sql_hash", "summary", "plan")

# TPC-H queries that join the `customer` table — the ones broken by the
# TPC-DS collision. Their presence + a >10k customer scan proves the fix.
CUSTOMER_JOIN_QUERIES = {"q03", "q05", "q07", "q08", "q10", "q13", "q18", "q22"}

GREEN = "[PASS]"
RED = "[FAIL]"
WARN = "[WARN]"


class Report:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.hard_failed = False

    def ok(self, msg: str) -> None:
        self.lines.append(f"  {GREEN} {msg}")

    def warn(self, msg: str) -> None:
        self.lines.append(f"  {WARN} {msg}")

    def fail(self, msg: str) -> None:
        self.lines.append(f"  {RED} {msg}")
        self.hard_failed = True

    def dump(self, title: str) -> None:
        print(f"\n{title}")
        print("-" * len(title))
        for ln in self.lines:
            print(ln)


def _walk_nodes(node: dict):
    """Yield every plan node depth-first."""
    yield node
    for child in node.get("Plans", []) or []:
        yield from _walk_nodes(child)


def _max_customer_rows(plan_field) -> float:
    """Max Actual Rows over any node scanning the `customer` relation."""
    if isinstance(plan_field, list):
        roots = [p.get("Plan", {}) for p in plan_field if isinstance(p, dict)]
    elif isinstance(plan_field, dict):
        roots = [plan_field.get("Plan", plan_field)]
    else:
        return 0.0
    best = 0.0
    for root in roots:
        for n in _walk_nodes(root):
            rel = n.get("Relation Name")
            if rel == "customer":
                best = max(best, float(n.get("Actual Rows", 0) or 0))
    return best


# ---------------------------------------------------------------------------
def validate_plans(plans_dir: Path) -> Report:
    rep = Report()
    files = sorted(plans_dir.glob("*.json"))
    if not files:
        rep.fail(f"no plan files found in {plans_dir}")
        return rep

    corrupt: list[str] = []
    missing_keys: list[str] = []
    seen_pairs: Counter = Counter()
    per_base_query: defaultdict[str, set] = defaultdict(set)
    exec_times: list[float] = []
    label_runs_seen: list[int] = []
    variance_seen = 0
    max_cust_rows = 0.0

    for f in files:
        try:
            rec = json.loads(f.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            corrupt.append(f.name)
            continue

        if any(k not in rec for k in REQUIRED_KEYS):
            missing_keys.append(f.name)
            continue

        qid = str(rec["query_id"])          # e.g. q03_p0
        variant = str(rec["variant"])
        base = qid.split("_p")[0]            # e.g. q03
        seen_pairs[(qid, variant)] += 1
        per_base_query[base].add(variant)

        et = rec.get("summary", {}).get("execution_time_ms")
        if et is not None:
            exec_times.append(float(et))
        if rec.get("label_runs") not in (None, ""):
            label_runs_seen.append(int(rec["label_runs"]))
        if rec.get("target_variance_ms") not in (None, ""):
            variance_seen += 1

        if base in CUSTOMER_JOIN_QUERIES:
            max_cust_rows = max(max_cust_rows, _max_customer_rows(rec.get("plan")))

    # ---- corrupt / structural ----
    if corrupt:
        rep.fail(f"{len(corrupt)} corrupt JSON file(s): {corrupt[:5]}")
    else:
        rep.ok(f"0 corrupt JSON files ({len(files)} parsed)")
    if missing_keys:
        rep.fail(f"{len(missing_keys)} file(s) missing required keys: {missing_keys[:5]}")
    else:
        rep.ok("all files have required keys")

    # ---- coverage vs. generator ----
    expected_qids = {q["id"] for q in generate_param_queries()}
    expected_pairs = {(qid, v) for qid in expected_qids for v in VARIANTS}
    collected_pairs = set(seen_pairs.keys())
    missing = expected_pairs - collected_pairs
    extra = collected_pairs - expected_pairs
    rep.ok(f"{len(collected_pairs)}/{len(expected_pairs)} expected (query,variant) pairs collected")
    if missing:
        rep.fail(f"{len(missing)} missing pairs, e.g. {sorted(missing)[:6]}")
    else:
        rep.ok("no missing (query,variant) pairs")
    if extra:
        rep.warn(f"{len(extra)} unexpected pairs (stale?), e.g. {sorted(extra)[:6]}")

    # ---- 22 base queries ----
    base_present = set(per_base_query.keys())
    expected_base = {q["id"].split("_p")[0] for q in generate_param_queries()}
    if expected_base.issubset(base_present):
        rep.ok(f"all {len(expected_base)} base TPC-H queries present")
    else:
        rep.fail(f"missing base queries: {sorted(expected_base - base_present)}")

    # ---- 4 variants each ----
    incomplete = {b: sorted(vs) for b, vs in per_base_query.items()
                  if not set(VARIANTS).issubset(vs)}
    if incomplete:
        rep.warn(f"{len(incomplete)} base queries missing some variant: "
                 f"{dict(list(incomplete.items())[:4])}")
    else:
        rep.ok("all base queries have all 4 join-knob variants")

    # ---- duplicates ----
    dups = {k: c for k, c in seen_pairs.items() if c > 1}
    if dups:
        rep.warn(f"{len(dups)} (query,variant) pairs have >1 file: {list(dups)[:4]}")
    else:
        rep.ok("no duplicate (query,variant) plan files")

    # ---- execution-time distribution ----
    if exec_times:
        st = sorted(exec_times)
        p95 = st[int(0.95 * (len(st) - 1))]
        bad = [t for t in exec_times if t <= 0]
        if bad:
            rep.fail(f"{len(bad)} plans with non-positive execution_time_ms")
        else:
            rep.ok(f"exec time ms  min={st[0]:.1f}  median={median(st):.1f}  "
                   f"p95={p95:.1f}  max={st[-1]:.1f}")
    else:
        rep.fail("no execution_time_ms values found")

    # ---- label quality ----
    if label_runs_seen:
        lr = Counter(label_runs_seen)
        rep.ok(f"label_runs distribution: {dict(lr)}  (variance recorded on "
               f"{variance_seen} plans)")
    else:
        rep.warn("no label_runs metadata (single-run labels?)")

    # ---- SCHEMA INTEGRITY (the bug we fixed) ----
    if max_cust_rows > 10_000:
        rep.ok(f"schema integrity: customer scans reach {max_cust_rows:.0f} rows "
               f"(>10k -> real TPC-H customer, not TPC-DS)")
    else:
        rep.fail(f"schema integrity: max customer scan = {max_cust_rows:.0f} rows "
                 f"(<=10k -> still hitting TPC-DS customer! re-run migration)")

    return rep


# ---------------------------------------------------------------------------
def validate_features(csv_path: Path) -> Report:
    rep = Report()
    if not csv_path.exists():
        rep.warn(f"{csv_path} not found (run extract_features first)")
        return rep
    import pandas as pd

    df = pd.read_csv(csv_path)
    rep.ok(f"shape: {df.shape[0]} rows x {df.shape[1]} columns")

    n_nan = int(df.isna().sum().sum())
    n_inf = int((df.select_dtypes("number").abs() == float("inf")).sum().sum())
    (rep.ok if n_nan == 0 else rep.fail)(f"{n_nan} NaN cells")
    (rep.ok if n_inf == 0 else rep.fail)(f"{n_inf} inf cells")

    feat_cols = [c for c in df.columns
                 if c not in ("query_id", "variant", "tag", "sql_hash",
                              "source_file", "collected_at", "target_variance_ms",
                              "label_runs", "execution_time_ms")]
    dup = int(df.duplicated(subset=feat_cols).sum())
    (rep.ok if dup == 0 else rep.warn)(f"{dup} duplicate feature vectors")

    knob = [c for c in ("enable_hashjoin", "enable_mergejoin", "enable_nestloop")
            if c in df.columns]
    (rep.ok if len(knob) == 3 else rep.fail)(f"knob features present: {knob}")

    planrows = [c for c in df.columns if c.startswith("plan_rows_")]
    (rep.ok if planrows else rep.fail)(f"plan_rows features present: {len(planrows)}")

    meta = [c for c in ("query_id", "variant", "sql_hash") if c in df.columns]
    (rep.ok if len(meta) == 3 else rep.fail)(f"metadata columns present: {meta}")

    return rep


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plans-dir", default=str(PLANS_DIR))
    ap.add_argument("--features", action="store_true",
                    help="also validate data/processed/features.csv")
    args = ap.parse_args()

    print("=" * 64)
    print("DATASET VALIDATION REPORT  (Phase 3E.4)")
    print("=" * 64)

    plan_rep = validate_plans(Path(args.plans_dir))
    plan_rep.dump("PLAN CORPUS")

    feat_failed = False
    if args.features:
        feat_rep = validate_features(FEATURES_CSV)
        feat_rep.dump("FEATURES.CSV")
        feat_failed = feat_rep.hard_failed

    hard = plan_rep.hard_failed or feat_failed
    print("\n" + "=" * 64)
    print(f"RESULT: {'DATASET NOT TRUSTWORTHY — fix above' if hard else 'DATASET TRUSTWORTHY'}")
    print("=" * 64)
    return 1 if hard else 0


if __name__ == "__main__":
    raise SystemExit(main())
