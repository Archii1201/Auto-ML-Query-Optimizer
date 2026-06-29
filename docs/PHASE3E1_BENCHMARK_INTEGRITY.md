# Phase 3E.1 — Benchmark Integrity (Schema Isolation)

> Why the learned cost model was secretly training on the *wrong*
> database, how we found it, how we fixed it permanently, and why we
> chose PostgreSQL **schemas** over every other option.
>
> Read alongside `docs/PHASE2A_TPCH.md` (TPC-H load) and
> `docs/PHASE3B_AUTOML.md` (TPC-DS load). This document is the
> post-mortem + design rationale for the fix in
> `scripts/migrate_to_schemas.py`.

---

## 0. Mental model

```
          BEFORE (broken)                         AFTER (fixed)
   ┌───────────────────────────┐        ┌───────────────────────────┐
   │        public schema      │        │  tpch.*        tpcds.*     │
   │  ┌─────────┐ ┌─────────┐  │        │  customer(150k) customer(10k)
   │  │ TPC-H   │ │ TPC-DS  │  │        │  orders         store_sales │
   │  │ tables  │ │ tables  │  │        │  lineitem       catalog_*   │
   │  └────┬────┘ └────┬────┘  │        │  ...            ...         │
   │       │  COLLIDE  │       │        │   no shared names anywhere  │
   │       ▼  customer ▼       │        └───────────────────────────┘
   │   last loader WINS        │          search_path picks the right
   │   (TPC-DS clobbered H)    │          schema per benchmark
   └───────────────────────────┘
```

Both benchmarks define a table literally named `customer`. They were
loaded into the **same** `public` schema. PostgreSQL has no notion of
"this table belongs to benchmark X" — a name is a name. So the second
loader's `DROP TABLE IF EXISTS customer CASCADE; CREATE TABLE customer …`
**silently destroyed** the first one.

---

## 1. What actually went wrong

### Symptom
The expanded TPC-H collection appeared to **time out** on every `q03`
variant (`TIMEOUT (>300s)`), after `q01` and `q02` collected fine.

### Real cause
`q03` is the first TPC-H query that joins `customer`. The error was
**not** a timeout — it was:

```
column "c_mktsegment" does not exist
```

The collector's exception handler printed `TIMEOUT` for *any* failure
(it returned `None` on both `QueryCanceled` **and** generic exceptions),
which masked the true error. (We left that handler intact for the
param collector but learned to read the actual exception.)

### Why `c_mktsegment` vanished
`public.customer` was the **TPC-DS** customer table:

| | TPC-H `customer` | TPC-DS `customer` (what we had) |
|---|---|---|
| Key column | `c_custkey` | `c_customer_sk` |
| Has `c_mktsegment`? | **yes** | no |
| Rows at our SF | **150,000** | 10,000 |

TPC-DS setup ran after TPC-H setup, so it won the name. The **only**
shared table name between the two benchmarks is `customer`, so only the
8 TPC-H queries that join `customer` broke:

```
q03  q05  q07  q08  q10  q13  q18  q22
```

The other 14 TPC-H queries never touch `customer`, which is exactly why
`q01`/`q02` looked fine and the problem hid for so long.

### Why this is worse than a crash
A crash is loud. This was **silent**: any `q03/q05/.../q22` rows already
in `features.csv` were collected against an inconsistent database and
became **unreproducible ground truth**. A learned cost model trained on
them is partially learning from a database that no longer exists.

---

## 2. How we diagnosed it (evidence, not guesswork)

1. Read the live `customer` columns → `c_customer_sk …` (TPC-DS shape).
2. Counted rows → `10,000` (TPC-DS SF), not `150,000` (TPC-H SF1).
3. Cross-checked every other TPC-H table → all correct and intact
   (`orders` 1.5M, `lineitem` 6.0M, …). Only `customer` was wrong.
4. Confirmed both `db/tpch_schema.sql` and `db/tpcds_schema.sql` issue
   `DROP/CREATE customer` in `public`.

That isolated the blast radius to a single table and a single root
cause: **shared namespace.**

---

## 3. The fix: one schema per benchmark

We give each benchmark its own PostgreSQL **schema** so identical table
names can coexist:

```
tpch.region  tpch.nation  … tpch.customer(150k)  tpch.orders  tpch.lineitem
tpcds.customer(10k)  tpcds.store_sales  tpcds.catalog_sales  …
```

Queries stay **unchanged**. Each collector sets the search path so an
unqualified `customer` resolves to the right table:

```sql
-- TPC-H collectors
SET search_path = tpch, public;
-- TPC-DS collector
SET search_path = tpcds, public;
```

### Moving parts (all committed)

| File | Change | Why |
|---|---|---|
| `scripts/migrate_to_schemas.py` | **new**, idempotent migration | move existing tables + regenerate lost `tpch.customer` without a full reload |
| `db/tpch_schema.sql` | `CREATE SCHEMA tpch; SET search_path tpch;` + qualified DROPs | future re-runs land in `tpch`, never `public` |
| `scripts/setup_tpch.py` | `COPY tpch.<table>`, `ANALYZE tpch.<table>` | load can never spill into `public` |
| `scripts/setup_tpcds.py` | auto-emits `CREATE SCHEMA tpcds` + `tpcds.<table>` DDL/COPY | symmetric isolation |
| `scripts/collect_tpch_plans.py`, `collect_tpch_param_plans.py` | `SET search_path = tpch, public` after every `RESET ALL` | `RESET ALL` wipes `search_path`, so it must be re-set per query |
| `scripts/collect_tpcds_plans.py` | `SET search_path = tpcds, public` | same |

> **Subtlety that bites people:** `RESET ALL` (used between plan
> variants to clear knob settings) **also resets `search_path`**. So we
> set `search_path` *immediately after* each `RESET ALL`, not once at
> connection open.

### Restoring the lost data
TPC-H raw CSVs are not kept in the repo (they're generated, not stored).
DuckDB's `tpch` extension regenerates **identical, spec-compliant** data
deterministically, so the migration regenerates only `customer`
(`CALL dbgen(sf=1)` → export `customer.csv` → `COPY tpch.customer`),
giving back the correct 150,000 rows in ~30s.

### Why the migration is idempotent
Every step checks current state first (`table_schema()`,
`column_exists()`), so re-running is a no-op once converged. This matters
because infra scripts get run twice by tired humans.

---

## 4. Why schemas — and why **not** the alternatives

We considered four other strategies and rejected each:

### ❌ Separate databases (`tpch_db`, `tpcds_db`)
- **Pro:** total isolation.
- **Con:** every connection (`DB_CONFIG`) is single-database. We'd need
  per-benchmark configs, reconnect logic in the service, and we could
  **never** run a cross-benchmark query or compare side-by-side in one
  session. Heavyweight for a problem that's really just a name clash.

### ❌ Rename the tables (`customer_tpch`, `customer_tpcds`)
- **Pro:** stays in one schema.
- **Con:** **rewrites every query**. TPC-H/TPC-DS SQL is standardized and
  comparable to published baselines; mangling table names destroys that
  comparability and means our queries are no longer "TPC-H queries".
  Also re-introduces the bug class the moment someone adds a benchmark
  with a `customer_tpch`-shaped name.

### ❌ Reload `customer` before each collection run
- **Pro:** trivial.
- **Con:** treats the symptom, not the cause. The collision recurs every
  time TPC-DS is (re)loaded; two collectors running in any interleaving
  corrupt each other. Fragile and non-reproducible — the opposite of
  what Phase 3E.1 is for.

### ❌ Keep only one benchmark
- **Con:** the whole point of dataset growth (3E.2) is **workload
  diversity**. Dropping TPC-DS shrinks generalization and kills the JOB
  roadmap.

### ✅ Schemas + `search_path`
- Identical table names coexist with **zero query rewrites**.
- One connection sees both benchmarks; `search_path` selects context.
- Permanent: the setup scripts now create tables *inside* their schema,
  so the collision **cannot** be recreated.
- Cheap: `ALTER TABLE … SET SCHEMA` moves data in-place (no copy).

This is also standard practice for multi-tenant / multi-workload
PostgreSQL — we're using the database the way it was designed to be used.

---

## 5. Verification (the definition of done)

After migration we confirmed all 8 previously-broken queries run:

```
q03  OK  2.6s   q05  OK 16.1s   q07  OK 25.2s   q10  OK 1.1s
q13  OK  2.2s   q18  OK 31.9s   q22  OK 0.5s    q08  (customer join) OK
tpch.customer  = 150000     tpcds.customer = 10000
tpch.orders    = 1500000    tpch.lineitem  = 6001215
```

A permanent regression guard lives in `validate_dataset.py`
(Phase 3E.4): it asserts that some `customer` scan in the corpus exceeds
**10,000 rows** — impossible for the TPC-DS table, guaranteed for the
real TPC-H one. If the collision ever returns, the dataset gate fails
loudly instead of training silently on bad data.

---

## 6. How to run

```bash
# One-time fix on an already-loaded (colliding) database:
python scripts/migrate_to_schemas.py --sf 1

# From scratch (now collision-proof by construction):
python scripts/setup_tpch.py  --sf 1
python scripts/setup_tpcds.py --sf 1
```

---

## 7. Lesson for the project

> **A shared namespace is a silent data-integrity bug waiting to
> happen.** The model never errored; it just learned from the wrong
> table. We now treat dataset *provenance* as a first-class, testable
> property — which is exactly what Phases 3E.2 (generation) and 3E.4
> (validation) formalize.
