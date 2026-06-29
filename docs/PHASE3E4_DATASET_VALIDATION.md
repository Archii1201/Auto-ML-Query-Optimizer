# Phase 3E.4 — Dataset Validation Gate

> Why we refuse to train until the dataset *proves* it is trustworthy,
> what each check verifies, and why a hard gate beats "eyeballing the
> logs". Documents `scripts/validate_dataset.py`.

---

## 0. Mental model

```
   plans_param/*.json ──► validate_dataset.py ──► PASS ─► extract_features
                               │                          ▼
                               └── FAIL (exit 1) ─────────► STOP. fix data.
```

The validator is a **gate**, not a report. It exits non-zero on any hard
failure, so it can sit in a pipeline (`&&`) and physically prevent a bad
dataset from reaching the model. Garbage in → garbage model, and a
*silently* garbage model is the most expensive kind (see Phase 3E.1).

---

## 1. Why a validation phase exists at all

The Phase 3E.1 bug is the whole argument: the model trained for weeks on
a database where `customer` was the wrong table, and **nothing
complained**. Logs said "collection complete". Accuracy numbers looked
plausible. The only way to catch that class of problem is to assert
dataset *properties* explicitly and fail loudly when they break.

> A dataset you cannot validate is a dataset you cannot trust. A model
> trained on untrusted data produces untrusted conclusions — which is
> worse than no model, because it *looks* like progress.

---

## 2. Plan-corpus checks (always run)

| Check | What it asserts | Why it matters |
|---|---|---|
| **Corrupt JSON** | every file parses | a half-written plan (killed run, disk full) silently drops a training row |
| **Required keys** | `query_id, variant, sql_hash, summary, plan` present | downstream extraction assumes these; missing → cryptic crash later |
| **Coverage vs. generator** | every expected `(query_id, variant)` from `tpch_param_queries.generate()` was collected | catches the *exact* failure mode of Phase 3E.1 (q03… missing) |
| **All 22 base queries** | `q01…q22` represented | a whole query family missing skews the workload |
| **All 4 variants/query** | default + 3 knob-offs | a missing variant means an incomplete plan-pick group |
| **Duplicates** | no `(query_id, variant)` collected twice | append-only `_index.jsonl` + re-runs create dupes that bias metrics |
| **Exec-time distribution** | no zero/negative times; report min/median/p95/max | zero-time "labels" are impossible and poison regression |
| **Label quality** | `label_runs` / `target_variance_ms` coverage | confirms multi-run median labeling actually happened |
| **Schema integrity** ⭐ | max `customer` scan **> 10,000 rows** | proves we're on the real 150k TPC-H `customer`, not the 10k TPC-DS one — a permanent guard against Phase 3E.1 recurring |

### Why the schema-integrity check is shaped that way
A `customer` scan can return at most as many rows as the table holds.
The TPC-DS `customer` has 10,000 rows; the TPC-H one has 150,000. So
"does any `customer` scan in the corpus exceed 10,000 rows?" is a clean,
data-driven discriminator that needs **no external config** — it reads
the truth straight out of the `Actual Rows` in the plan trees. `q13`
(outer-joins all customers) alone guarantees a 150k scan when the schema
is correct.

---

## 3. Features.csv checks (`--features`)

Run after extraction, before training:

| Check | Why |
|---|---|
| **row/col counts** | sanity vs. expected plan count and feature schema |
| **NaN / inf cells** | trees tolerate some NaN; linear models don't, and inf breaks everything |
| **duplicate feature vectors** | identical vectors with different labels = unlearnable contradictions |
| **knob features present** | `enable_hashjoin/mergejoin/nestloop` — Phase 3E features must survive extraction |
| **plan_rows features present** | cardinality features the ablation found useful |
| **metadata present** | `query_id, variant, sql_hash` needed for GroupKFold + plan-pick grouping |

---

## 4. Why a hard gate, not "just read the logs"

- **Logs are per-run; properties are global.** "880/880 captured" in a
  log doesn't tell you the corpus has no dupes from *previous* runs, or
  that customer scans are the right size.
- **Humans skim.** A gate that returns exit code 1 cannot be skimmed past
  by `python a.py && python b.py`.
- **Reproducibility.** The same script run by anyone, anytime, yields the
  same verdict — that's what makes the downstream baseline citable.

### Alternatives considered
- ❌ **Great Expectations / Pandera (heavy frameworks):** overkill for a
  research repo; adds dependencies and config sprawl. Our checks are
  domain-specific (plan trees, customer-scan sizes) and read better as
  ~250 lines of explicit Python.
- ❌ **Schema-only validation (types/nulls):** would miss the *semantic*
  bug (wrong-but-well-typed `customer`). We need workload-aware asserts.
- ✅ **A small, explicit, domain-specific gate** that encodes exactly the
  failure modes this project has actually hit.

---

## 5. How to run

```bash
python scripts/validate_dataset.py             # gate the plan corpus
python scripts/validate_dataset.py --features  # also gate features.csv
echo $LASTEXITCODE   # 0 = trustworthy, 1 = stop and fix
```

Only a `RESULT: DATASET TRUSTWORTHY` (exit 0) unlocks Phase 3F training.
