"""
extract_features.py
===================
Phase 2B entry point.

Reads execution-plan JSON files produced by Phase 1
(data/raw/*.json) and Phase 2A (data/tpch/plans/*.json), walks each
plan tree with a recursive DFS, builds an ML-ready feature vector
per plan, and writes the whole thing as a CSV to
data/processed/features.csv .

Usage:
    # Default: read both data/raw/ and data/tpch/plans/
    python feature_engineering/extract_features.py

    # Only TPC-H plans
    python feature_engineering/extract_features.py --input data/tpch/plans

    # Custom output
    python feature_engineering/extract_features.py --output data/processed/my_features.csv

Each output row =
    metadata columns
    + structural counts   (tree_depth, total_nodes, num_joins, num_scans)
    + per-operator counts (seq_scan_count, hash_join_count, ...)
    + cost / runtime columns
    + advanced aggregates (max_subtree_cost, total_rows_removed_by_filter, ...)
    + target_execution_time_ms      ← supervised label for Phase 3
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from feature_engineering.feature_utils import (  # noqa: E402
    AGG_NODES,
    JOIN_NODES,
    SCAN_NODES,
    SORT_NODES,
    TRACKED_NODE_TYPES,
    bump,
    dfs_iter,
    init_counter_dict,
    node_type_to_column,
    reduce_subtree,
    safe_get,
    safe_num,
    tree_size_and_depth,
)
from feature_engineering.plan_parser import (  # noqa: E402
    PlanParseError,
    get_record_metadata,
    get_root_plan_node,
    get_top_level_metrics,
    load_plan_record,
)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_INPUT_DIRS: tuple[Path, ...] = (
    PROJECT_ROOT / "data" / "raw",
    PROJECT_ROOT / "data" / "tpch" / "plans",
)
DEFAULT_OUTPUT_CSV: Path = PROJECT_ROOT / "data" / "processed" / "features.csv"


# ---------------------------------------------------------------------------
# The CSV schema. Listed explicitly so column order is deterministic
# regardless of dict-insertion order in any specific record.
# ---------------------------------------------------------------------------
METADATA_COLUMNS: tuple[str, ...] = (
    "source_file",
    "query_id",
    "variant",
    "tag",
    "sql_hash",
    "collected_at",
)

STRUCTURAL_COLUMNS: tuple[str, ...] = (
    "tree_depth",
    "total_nodes",
    "num_scans",
    "num_joins",
    "num_aggregates",
    "num_sorts",
)

OPERATOR_COUNT_COLUMNS: tuple[str, ...] = tuple(
    node_type_to_column(nt) for nt in TRACKED_NODE_TYPES
)

COST_COLUMNS: tuple[str, ...] = (
    "estimated_total_cost",
    "estimated_startup_cost",
    "estimated_rows",
    "actual_rows",
    "actual_total_time_ms",
    "planning_time_ms",
    "execution_time_ms",
    "wall_time_ms",
    "root_node_type",
)

ADVANCED_COLUMNS: tuple[str, ...] = (
    "max_subtree_cost",
    "max_actual_loops",
    "total_rows_removed_by_filter",
    "parallel_worker_count",
    "sum_shared_hit_blocks",
    "sum_shared_read_blocks",
    "sum_temp_read_blocks",
    "sum_temp_written_blocks",
)

TARGET_COLUMNS: tuple[str, ...] = (
    "target_execution_time_ms",
)

ALL_COLUMNS: tuple[str, ...] = (
    METADATA_COLUMNS
    + STRUCTURAL_COLUMNS
    + OPERATOR_COUNT_COLUMNS
    + COST_COLUMNS
    + ADVANCED_COLUMNS
    + TARGET_COLUMNS
)


# ---------------------------------------------------------------------------
# Per-plan feature extraction
# ---------------------------------------------------------------------------
def extract_features_from_record(record: dict[str, Any], path: Path) -> dict[str, Any]:
    """
    Build a single feature row from one plan record.

    The algorithm:
        1. Pull metadata + top-level metrics.
        2. Find the root operator node.
        3. Compute tree shape via a single recursive pass
           (tree_size_and_depth).
        4. Initialise an operator-count hash-map seeded at 0
           for every tracked node type.
        5. Walk the tree with dfs_iter (recursive DFS generator),
           bumping per-type counters and family totals.
        6. Compute reducer aggregates (max cost, sum filters, etc.).
        7. Assemble the final row.
    """
    metadata     = get_record_metadata(record, path)
    top_metrics  = get_top_level_metrics(record)
    root         = get_root_plan_node(record)

    total_nodes, tree_depth = tree_size_and_depth(root)

    counters = init_counter_dict()
    family_totals = {"scan": 0, "join": 0, "agg": 0, "sort": 0, "other": 0}
    parallel_workers = 0

    for node, _depth, _parent in dfs_iter(root):
        node_type = node.get("Node Type")
        bump(counters, node_type)

        if node_type in SCAN_NODES:
            family_totals["scan"] += 1
        elif node_type in JOIN_NODES:
            family_totals["join"] += 1
        elif node_type in AGG_NODES:
            family_totals["agg"] += 1
        elif node_type in SORT_NODES:
            family_totals["sort"] += 1
        else:
            family_totals["other"] += 1

        if "Workers Planned" in node or "Workers Launched" in node:
            parallel_workers += int(safe_get(node, "Workers Launched", 0))

    max_cost = reduce_subtree(
        root, lambda n: safe_num(n, "Total Cost"), initial=0.0, op="max",
    )
    max_loops = reduce_subtree(
        root, lambda n: safe_num(n, "Actual Loops"), initial=0.0, op="max",
    )
    sum_filter = reduce_subtree(
        root, lambda n: safe_num(n, "Rows Removed by Filter"),
    )
    sum_hit = reduce_subtree(
        root, lambda n: safe_num(n, "Shared Hit Blocks"),
    )
    sum_read = reduce_subtree(
        root, lambda n: safe_num(n, "Shared Read Blocks"),
    )
    sum_temp_read = reduce_subtree(
        root, lambda n: safe_num(n, "Temp Read Blocks"),
    )
    sum_temp_write = reduce_subtree(
        root, lambda n: safe_num(n, "Temp Written Blocks"),
    )

    row: dict[str, Any] = {
        **{k: metadata[k] for k in METADATA_COLUMNS},

        "tree_depth":     tree_depth,
        "total_nodes":    total_nodes,
        "num_scans":      family_totals["scan"],
        "num_joins":      family_totals["join"],
        "num_aggregates": family_totals["agg"],
        "num_sorts":      family_totals["sort"],

        **counters,

        "estimated_total_cost":   safe_num(root, "Total Cost"),
        "estimated_startup_cost": safe_num(root, "Startup Cost"),
        "estimated_rows":         safe_num(root, "Plan Rows"),
        "actual_rows":            safe_num(root, "Actual Rows"),
        "actual_total_time_ms":   safe_num(root, "Actual Total Time"),
        "planning_time_ms":       top_metrics["planning_time_ms"],
        "execution_time_ms":      top_metrics["execution_time_ms"],
        "wall_time_ms":           metadata["wall_time_ms"],
        "root_node_type":         root.get("Node Type"),

        "max_subtree_cost":             max_cost,
        "max_actual_loops":             max_loops,
        "total_rows_removed_by_filter": sum_filter,
        "parallel_worker_count":        parallel_workers,
        "sum_shared_hit_blocks":        sum_hit,
        "sum_shared_read_blocks":       sum_read,
        "sum_temp_read_blocks":         sum_temp_read,
        "sum_temp_written_blocks":      sum_temp_write,

        "target_execution_time_ms": top_metrics["execution_time_ms"],
    }
    return row


# ---------------------------------------------------------------------------
# Batch driver
# ---------------------------------------------------------------------------
def iter_plan_files(input_dirs: Iterable[Path]) -> Iterable[Path]:
    """
    Yield every plan JSON file under each input directory.

    We *skip* underscore-prefixed files (`_index.jsonl`, `.gitkeep`)
    because those are metadata, not plan records — same convention
    Phase 1 + Phase 2A established when writing them out.
    """
    for d in input_dirs:
        if not d.exists():
            print(f"[i] skipping missing directory: {d}")
            continue
        for path in sorted(d.glob("*.json")):
            if path.name.startswith("_") or path.name.startswith("."):
                continue
            yield path


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(ALL_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in ALL_COLUMNS})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 2B: extract ML-ready features from EXPLAIN plan JSONs.",
    )
    p.add_argument(
        "--input", action="append", type=Path, default=None,
        help="directory containing plan JSON files (repeatable). "
             "Default: data/raw and data/tpch/plans.",
    )
    p.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT_CSV,
        help=f"output CSV path (default: {DEFAULT_OUTPUT_CSV.relative_to(PROJECT_ROOT)})",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    input_dirs: tuple[Path, ...] = tuple(args.input) if args.input else DEFAULT_INPUT_DIRS

    print(f"[i] Input directories ({len(input_dirs)}):")
    for d in input_dirs:
        print(f"      - {d}")
    print(f"[i] Output CSV: {args.output}")

    rows: list[dict[str, Any]] = []
    ok, failed = 0, 0

    for path in iter_plan_files(input_dirs):
        try:
            record = load_plan_record(path)
            row = extract_features_from_record(record, path)
            rows.append(row)
            ok += 1
        except PlanParseError as exc:
            print(f"[!] {path.name}: {exc}", file=sys.stderr)
            failed += 1
        except Exception as exc:  # noqa: BLE001
            print(f"[!] {path.name}: unexpected error: "
                  f"{exc.__class__.__name__}: {exc}", file=sys.stderr)
            failed += 1

    if not rows:
        print("[!] No feature rows produced. Did you collect any plans yet?",
              file=sys.stderr)
        return 2

    write_csv(rows, args.output)

    print()
    print(f"[OK] Extracted features from {ok} plan(s); {failed} failed.")
    print(f"[OK] Wrote {len(rows)} rows x {len(ALL_COLUMNS)} columns "
          f"to {args.output}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
