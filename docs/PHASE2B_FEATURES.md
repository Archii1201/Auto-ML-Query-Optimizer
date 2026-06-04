# Phase 2B — Feature Engineering Deep Dive

> Advanced reference for the feature-extraction layer of the
> **AutoML-Powered Learned Query Optimizer**. Read alongside
> `docs/PHASE1_EXPLAINED.md` (raw plan collection) and
> `docs/PHASE2A_TPCH.md` (TPC-H workload). Phase 2B is the bridge
> between *raw EXPLAIN JSON trees* and *ML-ready numeric tensors*.

---

## 0. Mental model

Before touching code, lock this picture in your head:

```
   data/raw/*.json                data/tpch/plans/*.json
   (Phase 1 plans)                (Phase 2A plans)
          │                              │
          └──────────────┬───────────────┘
                         ▼
            ┌───────────────────────────┐
            │   plan_parser.py          │
            │   - load JSON             │
            │   - normalize schema      │
            │   - expose root operator  │
            └────────────┬──────────────┘
                         ▼
            ┌───────────────────────────┐
            │   feature_utils.py        │
            │   - DFS generator         │
            │   - node taxonomy         │
            │   - hash-map counters     │
            │   - reducers (sum / max)  │
            └────────────┬──────────────┘
                         ▼
            ┌───────────────────────────┐
            │   extract_features.py     │
            │   - per-plan row builder  │
            │   - fixed-schema CSV out  │
            └────────────┬──────────────┘
                         ▼
              data/processed/features.csv
              (50 columns × N plans)
                         │
                         ▼
                  Phase 3: ML training
```

Phase 2B converts each EXPLAIN tree into one **fixed-length feature
vector**. That single design decision drives everything downstream:

- The CSV has the **same columns** for every plan, even when a node
  type never appears (we emit `merge_join_count = 0`, not omit the
  column). Models hate missing columns; trees produce sparse plans.
- Counts come from a **recursive DFS** so we don't have to flatten
  the tree first — we stream over it once.
- The supervised label (`target_execution_time_ms`) lives in the same
  row as its features so a future training script is literally
  `pd.read_csv(...).drop(columns=["target_..."])` + a `y` column.

---

## 1. What Phase 2B adds to the repo

```
auto-ml-query-optimizer/
├── feature_engineering/         ← NEW (Python package)
│   ├── __init__.py
│   ├── feature_utils.py         ← DFS, node taxonomy, counters
│   ├── plan_parser.py           ← JSON load + tree normalization
│   └── extract_features.py      ← entry point + CSV writer
└── data/
    └── processed/               ← NEW
        ├── .gitkeep
        └── features.csv         ← generated output
```

Hard constraint we honour: **no Phase 1 / Phase 2A file is edited.**
We only *read* their JSON outputs. This is exactly the contract we
established at the end of `PHASE1_EXPLAINED.md` (§9, "Phase 2 should
never have to talk to PostgreSQL again").

---

## 2. End-to-end usage

```powershell
# default: pulls from BOTH data/raw/ and data/tpch/plans/
python feature_engineering/extract_features.py

# narrow to one source
python feature_engineering/extract_features.py --input data/tpch/plans

# custom output path
python feature_engineering/extract_features.py --output data/processed/tpch_only.csv

# multiple inputs (repeatable flag)
python feature_engineering/extract_features.py \
    --input data/raw --input data/tpch/plans
```

Output is a single CSV at `data/processed/features.csv` containing
**50 columns**: 6 metadata + 6 structural + 21 per-operator counts +
9 cost/runtime + 8 advanced aggregates + 1 target.

---

## 3. The DSA centerpiece — recursive DFS

### 3.1 The tree shape we walk

PostgreSQL's EXPLAIN JSON returns the plan as a nested tree:

```
{
  "Node Type": "Hash Join",
  "Startup Cost": ...,
  "Total Cost": ...,
  "Actual Total Time": ...,
  "Plans": [                       ← children live under "Plans"
    { "Node Type": "Seq Scan", ... },
    { "Node Type": "Hash", "Plans": [
        { "Node Type": "Seq Scan", ... }
    ]}
  ]
}
```

So the recursion contract is simple: "a node is a dict; if it has a
non-empty `Plans` key, recurse into each child".

### 3.2 The generator implementation

```python
def dfs_iter(
    root: dict,
    depth: int = 0,
    parent_type: str | None = None,
) -> Iterator[tuple[dict, int, str | None]]:
    yield root, depth, parent_type

    children = root.get("Plans")
    if not children:
        return

    node_type = root.get("Node Type")
    for child in children:
        yield from dfs_iter(child, depth + 1, node_type)
```

Three properties worth memorising:

1. **Pre-order.** Parents are yielded before their children. That
   matters if you ever want to compute *parent-aware* features (e.g.
   "Seq Scan that lives under a Hash Join inner side") — the
   `parent_type` argument is already threaded through.
2. **Lazy.** It's a generator, not a list-builder. Memory is `O(depth)`
   on the call stack, not `O(n)` for an intermediate list. Plan trees
   are shallow (real-world depth < 30), so the call stack is a
   non-issue.
3. **Composable.** Anything that wants a "for each node" pass can just
   `for node, depth, parent in dfs_iter(root): ...` — no traversal
   logic duplicated across the codebase.

### 3.3 Why recursion, not an explicit stack?

Both are O(n) time, O(depth) space. We chose recursion because:

- The spec asked for it ("recursive DFS").
- Plan trees are guaranteed shallow — Python's default
  `sys.getrecursionlimit() == 1000` won't be touched even at
  TPC-H SF=100.
- The code reads 1:1 with the tree definition: "yield self, then
  for each child, yield from recurse(child)". An explicit stack
  inverts that and obscures the pre-order invariant.

### 3.4 Single-pass shape computation

`tree_size_and_depth()` is a separate recursion that returns
`(total_nodes, tree_depth)` in one walk:

```python
def tree_size_and_depth(root: dict) -> tuple[int, int]:
    if not root:
        return 0, 0
    total, max_child_depth = 1, 0
    for child in root.get("Plans", []) or []:
        c_total, c_depth = tree_size_and_depth(child)
        total += c_total
        if c_depth > max_child_depth:
            max_child_depth = c_depth
    return total, 1 + max_child_depth
```

This is the classic post-order tree-shape recurrence:
`size(t) = 1 + Σ size(child)` and `depth(t) = 1 + max(depth(child))`.
Depth is 1-indexed (a single-node plan has depth 1) to match how
most plan-analysis papers report it.

Could we have folded this into `dfs_iter`? Yes, but keeping it
separate (and pure) makes the intent obvious and the unit-testing
trivial.

### 3.5 The generic reducer

`reduce_subtree()` lets us compute "sum/max of any per-node quantity"
without writing a new traversal each time:

```python
def reduce_subtree(root, fn, initial=0.0, op="sum"):
    acc = initial
    for node, _depth, _parent in dfs_iter(root):
        val = fn(node)
        if op == "sum":  acc += val
        else:            acc = max(acc, val)
    return acc
```

Used to compute:

| Feature                          | `fn`                              | `op`  |
|----------------------------------|-----------------------------------|-------|
| `max_subtree_cost`              | `lambda n: safe_num(n, "Total Cost")` | `max` |
| `max_actual_loops`              | `lambda n: safe_num(n, "Actual Loops")` | `max` |
| `total_rows_removed_by_filter` | `lambda n: safe_num(n, "Rows Removed by Filter")` | `sum` |
| `sum_shared_hit_blocks`         | `lambda n: safe_num(n, "Shared Hit Blocks")` | `sum` |
| ... (and 4 more)                 |                                   |       |

If Phase 3 wants a new aggregate, you write one lambda — no traversal
boilerplate.

---

## 4. The hash-map (operator histogram)

### 4.1 Pre-seeded counter

```python
def init_counter_dict() -> dict[str, int]:
    return {node_type_to_column(nt): 0 for nt in TRACKED_NODE_TYPES}
```

Every plan starts with the **same 21 keys**, all set to 0. This is
not cosmetic — it's the reason the CSV has a stable schema regardless
of which node types actually appeared. Without pre-seeding, a plan
without any Sort operators would have its `sort_count` *column
missing*, and `pd.read_csv` would silently fill NaN, and a tree-based
model would get confused about whether NaN means "zero" or "unknown".

### 4.2 `bump()` — the increment primitive

```python
def bump(counter, node_type):
    if not node_type:
        return
    col = node_type_to_column(node_type)
    counter[col] = counter.get(col, 0) + 1
```

`counter.get(col, 0) + 1` (not `counter[col] + 1`) so we tolerate
unknown node types from a future PG version — they just create a
new key on the fly without breaking the run. The fixed-schema CSV
writer later ignores any extra keys, so this is a defensive no-op
for now, but it costs us nothing and protects against PG upgrades.

### 4.3 Family totals — non-overlapping buckets

In parallel with per-type counts, we also compute coarse-grained
family totals:

```python
family_totals = {"scan": 0, "join": 0, "agg": 0, "sort": 0, "other": 0}

for node, _depth, _parent in dfs_iter(root):
    nt = node.get("Node Type")
    if   nt in SCAN_NODES: family_totals["scan"] += 1
    elif nt in JOIN_NODES: family_totals["join"] += 1
    elif nt in AGG_NODES:  family_totals["agg"]  += 1
    elif nt in SORT_NODES: family_totals["sort"] += 1
    else:                  family_totals["other"]+= 1
```

These map to the CSV columns `num_scans, num_joins, num_aggregates,
num_sorts`. The classification sets are **non-overlapping** —
critical so that `num_scans + num_joins + num_aggregates + num_sorts
+ other == total_nodes`. If sets overlapped, a model could
double-count, and feature importance would lie.

### 4.4 The taxonomy

```python
SCAN_NODES = frozenset({
    "Seq Scan", "Index Scan", "Index Only Scan",
    "Bitmap Index Scan", "Bitmap Heap Scan",
    "Tid Scan", "CTE Scan", "Subquery Scan",
    "Function Scan", "Values Scan",
    "Table Function Scan", "Foreign Scan",
    "Sample Scan", "Named Tuplestore Scan", "WorkTable Scan",
})
JOIN_NODES = frozenset({"Hash Join", "Merge Join", "Nested Loop"})
AGG_NODES  = frozenset({"Aggregate", "HashAggregate",
                        "GroupAggregate", "WindowAgg"})
SORT_NODES = frozenset({"Sort", "Incremental Sort"})
```

Why `frozenset`? `in` checks are O(1), and immutability prevents
"oops, I mutated the taxonomy at runtime" bugs in long-lived
notebooks.

---

## 5. Module-by-module deep dive

### 5.1 `feature_engineering/feature_utils.py`

The stateless building-block layer. Five sections:

1. **Node-type taxonomy** — `SCAN_NODES`, `JOIN_NODES`, `AGG_NODES`,
   `SORT_NODES`, and the master list `TRACKED_NODE_TYPES` (21 strings)
   that drives the CSV column order.

2. **Safe accessors** — `safe_get(node, key, default)` and
   `safe_num(node, key)`. EXPLAIN JSON is sparse: `"Rows Removed by
   Filter"` only appears when there *was* a filter. Returning a numeric
   default keeps `sum_*` totals type-stable. `safe_num` additionally
   coerces strings and `None` to `0.0` — defensive against PG variants
   that return scientific notation as a string.

3. **DFS traversal** — `dfs_iter()` (the generator) and
   `tree_size_and_depth()` (the post-order recurrence). Discussed
   above.

4. **Aggregator** — `init_counter_dict()` and `bump()`. The hash-map
   primitives.

5. **Reducers** — `reduce_subtree(root, fn, initial, op)`. The
   one-recursion-many-features pattern.

This file has **zero imports** outside the standard library — by
design, so it's import-cheap and easy to test in isolation.

### 5.2 `feature_engineering/plan_parser.py`

The I/O + normalization layer.

```python
record       = load_plan_record(path)        # JSON → dict
root_node    = get_root_plan_node(record)    # peel EXPLAIN wrapper
top_metrics  = get_top_level_metrics(record) # planning + execution time
metadata     = get_record_metadata(record, path)
```

The "EXPLAIN wrapper" is the irritating part of the PG plan format:

```
[                                              ← outer list, always length 1
  {
    "Plan":          { ... root operator ... },
    "Planning Time": 0.123,
    "Execution Time": 4.567,
    "Triggers":       []
  }
]
```

`get_root_plan_node()` digs through `record["plan"][0]["Plan"]` so the
rest of the pipeline can treat the root as a regular operator dict.
`get_top_level_metrics()` pulls `Planning Time` and `Execution Time`
from the wrapper — they don't live on the root, and confusing the
two is a classic Phase 2 bug.

A custom exception `PlanParseError` (subclass of `ValueError`) lets
the driver tell "this file is malformed, skip it" apart from "the
disk is on fire". Skipped files are logged but don't abort the batch.

### 5.3 `feature_engineering/extract_features.py`

The orchestration + schema layer.

Three things it does that the lower layers don't:

1. **Schema enforcement.** All 50 column names are declared up front
   as immutable tuples (`METADATA_COLUMNS`, `STRUCTURAL_COLUMNS`,
   `OPERATOR_COUNT_COLUMNS`, `COST_COLUMNS`, `ADVANCED_COLUMNS`,
   `TARGET_COLUMNS`). `ALL_COLUMNS` is their concatenation. The
   `csv.DictWriter` writes exactly these columns in this order, no
   matter what extra keys a row dict happens to carry. **Column
   order is deterministic across runs** — important for diffing
   features.csv between two collections.

2. **File discovery.** `iter_plan_files(input_dirs)` globs `*.json`
   and skips `_*` / `.*` files (the `_index.jsonl` and `.gitkeep`
   files Phase 1 and Phase 2A put alongside the real plans). This is
   the same "underscore = metadata, not data" convention used
   throughout the project.

3. **Error containment.** Each file is parsed in its own try/except.
   A single corrupt JSON or unexpected schema doesn't kill the batch:

   ```python
   for path in iter_plan_files(input_dirs):
       try:
           record = load_plan_record(path)
           row    = extract_features_from_record(record, path)
           rows.append(row)
           ok += 1
       except PlanParseError as exc:
           print(f"[!] {path.name}: {exc}", file=sys.stderr)
           failed += 1
       except Exception as exc:
           ...   # log + keep going
   ```

   Exit code semantics match Phase 1: `0` if all good, `2` if any
   file failed.

---

## 6. The output schema — every column explained

### 6.1 Metadata (6 columns)

| Column          | Source                          | Purpose |
|-----------------|----------------------------------|---------|
| `source_file`   | `path.name`                      | Provenance — which JSON produced this row. |
| `query_id`      | `record["query_id"]`             | Stable handle (`q03`, `q01_customers_by_country`). |
| `variant`       | `record.get("variant","default")`| Phase 2A optimizer variant; `"default"` for Phase 1. |
| `tag`           | `record["tag"]`                  | Plan-shape category (`join+agg+sort+limit`, ...). |
| `sql_hash`      | `record["sql_hash"]`             | SHA-1[:8] of the SQL text. Changes ⇒ query changed. |
| `collected_at`  | `record["collected_at"]`         | UTC ISO-8601 collection timestamp. |

Use these for group-by, stratified train/test split, and time-based
splits. None of them are passed to the model.

### 6.2 Structural (6 columns)

| Column           | How it's computed                                            |
|------------------|--------------------------------------------------------------|
| `tree_depth`     | `tree_size_and_depth(root)[1]` — recursive max-depth.        |
| `total_nodes`    | `tree_size_and_depth(root)[0]` — recursive size.             |
| `num_scans`      | `sum(1 for n in dfs if family_of(n.Node Type) == "scan")`    |
| `num_joins`      | same, family = `"join"`                                      |
| `num_aggregates` | same, family = `"agg"`                                       |
| `num_sorts`      | same, family = `"sort"`                                      |

These are the "tree shape signature". A 2-node plan with one join and
one scan looks completely different to a 30-node plan with five
joins and ten scans, and these columns capture that difference
linearly.

### 6.3 Per-operator counts (21 columns)

`seq_scan_count`, `index_scan_count`, `index_only_scan_count`,
`bitmap_index_scan_count`, `bitmap_heap_scan_count`,
`hash_join_count`, `merge_join_count`, `nested_loop_count`,
`aggregate_count`, `hashaggregate_count`, `groupaggregate_count`,
`sort_count`, `incremental_sort_count`, `hash_count`,
`materialize_count`, `limit_count`, `gather_count`,
`gather_merge_count`, `unique_count`, `windowagg_count`.

Every column is always emitted, even if 0. These give the model
fine-grained signal — e.g. "hash join vs. merge join" matters even
though both increment `num_joins`.

### 6.4 Cost / runtime (9 columns)

Pulled from the **root node** (planner's view of the whole plan) plus
the **EXPLAIN wrapper** (real wall times):

| Column                    | Source                                        |
|---------------------------|-----------------------------------------------|
| `estimated_total_cost`    | `root["Total Cost"]` — planner's abstract cost. |
| `estimated_startup_cost`  | `root["Startup Cost"]` — cost before first row. |
| `estimated_rows`          | `root["Plan Rows"]` — planner's row estimate. |
| `actual_rows`             | `root["Actual Rows"]` — real rows produced.   |
| `actual_total_time_ms`    | `root["Actual Total Time"]` — root-node wall ms (per loop). |
| `planning_time_ms`        | wrapper's `"Planning Time"` — how long the planner deliberated. |
| `execution_time_ms`       | wrapper's `"Execution Time"` — server-side execution. |
| `wall_time_ms`            | client-side wall clock around `cur.execute`.   |
| `root_node_type`          | `root["Node Type"]` — categorical "plan shape" label. |

The planner's estimation **error** — `estimated_rows` vs.
`actual_rows` — is the single most important signal in the dataset.
Many learned cost models literally use `log(actual_rows / max(1,
estimated_rows))` as a feature.

### 6.5 Advanced aggregates (8 columns)

Each is one call to `reduce_subtree()`:

| Column                          | Reducer                                      |
|---------------------------------|----------------------------------------------|
| `max_subtree_cost`             | `max` of `Total Cost` across all nodes.       |
| `max_actual_loops`             | `max` of `Actual Loops`. Spots Nested-Loop blowups. |
| `total_rows_removed_by_filter`| `sum` of `Rows Removed by Filter`.            |
| `parallel_worker_count`        | `sum` of `Workers Launched`.                  |
| `sum_shared_hit_blocks`        | `sum` of cache hits across all nodes.         |
| `sum_shared_read_blocks`       | `sum` of disk reads across all nodes.         |
| `sum_temp_read_blocks`         | `sum` of temp file reads (spill detection).   |
| `sum_temp_written_blocks`      | `sum` of temp file writes.                    |

The buffer-block features are unusually predictive of *real* runtime
— a plan that reads 10× more shared blocks will run ~10× slower
even if the planner's `Total Cost` is the same.

### 6.6 Target (1 column)

```python
"target_execution_time_ms": top_metrics["execution_time_ms"],
```

This is the supervised label. It's deliberately a duplicate of
`execution_time_ms` so future-you can `df.drop(columns=["target_*"])`
without losing the feature. The naming convention (`target_*` prefix)
matches scikit-learn / Kaggle pipelines.

---

## 7. Worked example

For your Phase 1 plan `q03_join_customer_orders__c266d5b5.json`:

```
tree:
   Hash Join (depth=1)
   ├── Seq Scan orders (depth=2)
   └── Hash (depth=2)
       └── Seq Scan customers (depth=3)
```

DFS visits 4 nodes. The feature row works out to:

```
tree_depth                = 3
total_nodes               = 4
num_scans                 = 2     (two Seq Scans)
num_joins                 = 1     (one Hash Join)
seq_scan_count            = 2
hash_join_count           = 1
hash_count                = 1     (Hash node — build side, not a join)
estimated_total_cost      = 2364.22  (root Hash Join Total Cost)
estimated_rows            = 6311
actual_rows               = 6170
max_subtree_cost          = 2364.22  (root is the max)
total_rows_removed_by_filter = 58713 (49963 from orders + 8750 from customers)
sum_shared_hit_blocks     = 1787  (841 + 736 + 105 + 105)
target_execution_time_ms  = 13.224
```

Every one of those numbers fell out of a single recursive walk +
seven `reduce_subtree` calls. Total cost: one pass over the tree.

---

## 8. Loading the CSV in Phase 3

A future training script will look approximately like:

```python
import pandas as pd

df = pd.read_csv("data/processed/features.csv")

# Hold out metadata
meta_cols = ["source_file", "query_id", "variant", "tag",
             "sql_hash", "collected_at", "root_node_type"]
meta = df[meta_cols]

# Target
y = df["target_execution_time_ms"]

# Features = everything numeric that isn't the target or metadata
X = df.drop(columns=meta_cols + ["target_execution_time_ms"])

# Stratified split by plan-shape tag, so each fold has Hash Joins,
# Nested Loops, etc.
from sklearn.model_selection import GroupShuffleSplit
splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(splitter.split(X, y, groups=meta["query_id"]))
```

Notice nothing about that snippet had to talk to PostgreSQL. That's
the whole point.

---

## 9. Reproducibility checklist

```powershell
# 0. start fresh
del data\processed\features.csv

# 1. re-run feature extraction
python feature_engineering/extract_features.py
```

If `features.csv` differs from a previous run, the cause is almost
always **upstream**, not in this phase:

1. Did `data/raw/` or `data/tpch/plans/` change? Phase 2B is
   deterministic given its inputs — a different CSV means a different
   input.
2. Did you re-run `collect_tpch_plans.py` and let plans get
   *re-collected* with a new `collected_at` timestamp? That changes
   one cell per row by design.
3. Did you add a new query and forget that adds new rows? Check
   `wc -l data/processed/features.csv`.

Phase 2B itself uses no randomness, no parallelism, no time-of-day
dependence.

---

## 10. Common pitfalls

- **NaN in numeric columns after `pd.read_csv`.** Means a feature
  came out as an empty string. Check that `safe_num()` is being used
  for that field — bare `node.get(...)` can return `None`, which the
  CSV writer turns into `""`.
- **All-zero per-operator columns.** Means PG used a node type we
  don't track. Add the string to `TRACKED_NODE_TYPES` *and*
  `SCAN_NODES`/`JOIN_NODES`/... — both lists, otherwise the family
  total won't update.
- **`target_execution_time_ms == 0`.** Means the EXPLAIN wrapper had
  no `Execution Time` field. That happens for plans where
  `EXPLAIN ANALYZE` was *not* used (only `EXPLAIN`). Phase 1 and 2A
  both use `ANALYZE`, so this should never trigger — if it does,
  inspect the source JSON.
- **Extraction is slow on huge SF=10 datasets.** It shouldn't be —
  one DFS per plan is O(n) — but if you ever batch hundreds of
  thousands of plans, drop the `csv.DictWriter` for
  `pandas.DataFrame.to_csv()` or `pyarrow.parquet`. The DFS itself
  is already fast.
- **Column order shifted unexpectedly.** Don't add columns ad-hoc
  inside `extract_features_from_record`. Add them to the appropriate
  `*_COLUMNS` tuple at the top of `extract_features.py` and the
  writer will pick them up in the right order.

---

## 11. What Phase 3 will plug into

The contract this phase exposes:

- **Single CSV:** `data/processed/features.csv` with the fixed 50-
  column schema documented in §6.
- **Row identity:** `(source_file)` is unique within the CSV.
  `(query_id, variant, sql_hash)` together identify a measurement
  (multiple rows possible if you re-collected over time).
- **Target column:** `target_execution_time_ms` (float, milliseconds).
- **Group keys for split:**
  - `query_id` for "leave-query-out" cross-validation (the gold
    standard for plan-cost models).
  - `tag` for stratification across plan shapes.
  - `variant` for "did our model rank the 4 optimizer variants in
    the right order?" — this is the actual deployment metric.

Phase 3 will almost certainly add:
- log-transforms on cost / row / time columns,
- ratios like `estimated_rows / max(1, actual_rows)`,
- one-hot encoding of `root_node_type`,
- pairwise ranking labels (for "plan A faster than plan B given
  same query_id").

All of those are feature-engineering on top of Phase 2B, not
modifications to it.

---

## 12. TL;DR

- **`feature_utils.py`** = stateless primitives: node taxonomy,
  recursive DFS generator, pre-seeded hash-map counters, generic
  reducers.
- **`plan_parser.py`** = JSON I/O + EXPLAIN-wrapper normalization.
  Hides the `record["plan"][0]["Plan"]` ugliness from everyone else.
- **`extract_features.py`** = orchestrator. Fixed 50-column schema,
  per-file error containment, deterministic output order.
- **The DSA story:** one recursive DFS per plan + a dict-keyed
  histogram + a generic reducer. O(n) time, O(depth) space, no
  external dependencies.
- **The contract:** every plan in `data/raw/` or `data/tpch/plans/`
  becomes one row in `data/processed/features.csv`. Phase 3 reads
  that CSV and nothing else.
