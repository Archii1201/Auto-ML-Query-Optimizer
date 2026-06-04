"""
feature_utils.py
================
Low-level building blocks for Phase 2B feature extraction:

    * Node-type taxonomies   (which Node Type strings belong to which family)
    * Safe field accessors   (PostgreSQL's EXPLAIN JSON is *sparse* —
                              not every node has every key)
    * A recursive DFS walker (yields every node in a plan tree)
    * A counter aggregator   (operator histogram via dict / hash-map)

These helpers are intentionally side-effect-free and stateless so the
higher-level extractor in extract_features.py can compose them freely
(and unit tests can hit them in isolation).
"""

from __future__ import annotations

from typing import Any, Callable, Iterator


# ---------------------------------------------------------------------------
# Node-type taxonomy
# ---------------------------------------------------------------------------
# PostgreSQL's EXPLAIN reports "Node Type" as a free-form string. We bucket
# the well-known ones into families so feature counts stay stable across
# PG versions / minor wording changes.
#
# The "family" buckets are *non-overlapping*: each node contributes to
# exactly one of {scan, join, agg, sort, other} for the structural totals,
# while still also bumping its own fine-grained counter (e.g. "Seq Scan"
# bumps both `seq_scan_count` and `num_scans`).
# ---------------------------------------------------------------------------

SCAN_NODES: frozenset[str] = frozenset({
    "Seq Scan",
    "Index Scan",
    "Index Only Scan",
    "Bitmap Index Scan",
    "Bitmap Heap Scan",
    "Tid Scan",
    "CTE Scan",
    "Subquery Scan",
    "Function Scan",
    "Values Scan",
    "Table Function Scan",
    "Foreign Scan",
    "Sample Scan",
    "Named Tuplestore Scan",
    "WorkTable Scan",
})

JOIN_NODES: frozenset[str] = frozenset({
    "Hash Join",
    "Merge Join",
    "Nested Loop",
})

AGG_NODES: frozenset[str] = frozenset({
    "Aggregate",        # PG >= 11 ("Strategy" sub-field discriminates plain/hashed/sorted)
    "HashAggregate",    # legacy / explicit
    "GroupAggregate",   # legacy / explicit
    "WindowAgg",
})

SORT_NODES: frozenset[str] = frozenset({
    "Sort",
    "Incremental Sort",
})

# Fine-grained counters we *always* emit in the feature vector, even if zero.
# Keeping this list fixed guarantees every CSV row has the same columns
# regardless of which node types happened to appear in this particular plan.
TRACKED_NODE_TYPES: tuple[str, ...] = (
    # scans
    "Seq Scan",
    "Index Scan",
    "Index Only Scan",
    "Bitmap Index Scan",
    "Bitmap Heap Scan",
    # joins
    "Hash Join",
    "Merge Join",
    "Nested Loop",
    # agg / sort / misc
    "Aggregate",
    "HashAggregate",
    "GroupAggregate",
    "Sort",
    "Incremental Sort",
    "Hash",
    "Materialize",
    "Limit",
    "Gather",
    "Gather Merge",
    "Unique",
    "WindowAgg",
)


def node_type_to_column(node_type: str) -> str:
    """
    Convert a PostgreSQL Node Type string into a snake_case CSV column
    name suffix.

    >>> node_type_to_column("Bitmap Heap Scan")
    'bitmap_heap_scan_count'
    """
    return node_type.lower().replace(" ", "_") + "_count"


def family_of(node_type: str) -> str:
    """Return the high-level family bucket for a node type."""
    if node_type in SCAN_NODES:
        return "scan"
    if node_type in JOIN_NODES:
        return "join"
    if node_type in AGG_NODES:
        return "agg"
    if node_type in SORT_NODES:
        return "sort"
    return "other"


# ---------------------------------------------------------------------------
# Safe accessors
# ---------------------------------------------------------------------------
def safe_get(node: dict, key: str, default: Any = 0) -> Any:
    """
    Return node[key] if present *and* not None, else `default`.

    EXPLAIN JSON nodes routinely omit fields (e.g. "Rows Removed by Filter"
    only appears when there *was* a filter). Returning a numeric default
    keeps downstream sums type-stable.
    """
    val = node.get(key)
    return default if val is None else val


def safe_num(node: dict, key: str) -> float:
    """Numeric variant of safe_get — always returns a float, never None."""
    val = node.get(key)
    if val is None:
        return 0.0
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


# ---------------------------------------------------------------------------
# DFS traversal — the DSA centerpiece
# ---------------------------------------------------------------------------
def dfs_iter(
    root: dict,
    depth: int = 0,
    parent_type: str | None = None,
) -> Iterator[tuple[dict, int, str | None]]:
    """
    Recursive depth-first traversal of a PostgreSQL EXPLAIN plan tree.

    Yields a stream of `(node, depth, parent_node_type)` tuples — one per
    operator in the tree, in pre-order (parent before children).

    The tree shape is:
        node = {
            "Node Type": "...",
            ... per-node fields ...
            "Plans": [ child_node, child_node, ... ]   # optional
        }

    Why a generator instead of building a list?
        - O(depth) extra memory on the call stack instead of O(n)
          extra memory for an intermediate list.
        - Lets the caller short-circuit if it ever needs to.

    Why recursion instead of an explicit stack?
        - Plan trees are shallow (real-world depth < 30 even for ugly
          TPC-H queries with `enable_nestloop = off`), well below
          Python's default recursion limit (1000).
        - Recursive code reads 1:1 with the tree structure and is what
          the spec asked for ("recursive DFS tree traversal").
    """
    yield root, depth, parent_type

    children = root.get("Plans")
    if not children:
        return

    node_type = root.get("Node Type")
    for child in children:
        yield from dfs_iter(child, depth + 1, node_type)


def tree_size_and_depth(root: dict) -> tuple[int, int]:
    """
    Single-pass recursive computation of (total_nodes, tree_depth).

    Depth is 1-indexed (a single-node plan has depth 1) to match how
    most plan-analysis papers report it.
    """
    if not root:
        return 0, 0

    total = 1
    max_child_depth = 0

    for child in root.get("Plans", []) or []:
        child_total, child_depth = tree_size_and_depth(child)
        total += child_total
        if child_depth > max_child_depth:
            max_child_depth = child_depth

    return total, 1 + max_child_depth


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------
def init_counter_dict() -> dict[str, int]:
    """
    Initialise the operator-count hash-map with every tracked node type
    pinned at 0. Pre-seeding guarantees stable CSV columns even when a
    particular node type never appears in a plan.
    """
    return {node_type_to_column(nt): 0 for nt in TRACKED_NODE_TYPES}


def bump(counter: dict[str, int], node_type: str | None) -> None:
    """Increment a counter for a node type, ignoring unknown / None."""
    if not node_type:
        return
    col = node_type_to_column(node_type)
    counter[col] = counter.get(col, 0) + 1


# ---------------------------------------------------------------------------
# Small reducers used by extract_features
# ---------------------------------------------------------------------------
def reduce_subtree(
    root: dict,
    fn: Callable[[dict], float],
    initial: float = 0.0,
    op: str = "sum",
) -> float:
    """
    Walk the subtree under `root` and reduce a per-node numeric `fn(node)`
    using either summation or max.

    Generic enough to compute:
        sum  of Rows Removed by Filter across all nodes
        sum  of Shared Hit Blocks
        max  of Total Cost across all nodes  (== max_subtree_cost)
    """
    if op not in ("sum", "max"):
        raise ValueError(f"unsupported reducer op: {op}")

    acc = initial
    for node, _depth, _parent in dfs_iter(root):
        val = fn(node)
        if op == "sum":
            acc += val
        else:
            if val > acc:
                acc = val
    return acc
