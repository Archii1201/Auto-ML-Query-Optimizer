"""
migrate_to_schemas.py
======================
One-shot, idempotent migration that fixes the TPC-H / TPC-DS table-name
collision.

Background
----------
Both benchmarks were loaded into the *public* schema. They share one
table name — ``customer`` — so whichever setup script ran last won.
TPC-DS ran last, so ``public.customer`` is the TPC-DS customer
(c_customer_sk, 10k rows) and the TPC-H customer (c_custkey, 150k rows)
was dropped. Every TPC-H query that joins ``customer`` (q03, q05, q07,
q08, q10, q13, q18, q22) then fails with "column c_mktsegment does not
exist".

Fix
---
Give each benchmark its own PostgreSQL schema so names never collide:

    tpch.region, tpch.nation, ..., tpch.customer (150k rows)
    tpcds.customer (10k rows), tpcds.store_sales, ...

Collectors then ``SET search_path = tpch, public`` (or tpcds) so the
existing unqualified SQL resolves to the right tables with no query
rewrites.

This script:
    1. Creates schemas ``tpch`` and ``tpcds``.
    2. Moves every existing TPC-DS table from public -> tpcds
       (this includes the colliding public.customer, which is TPC-DS's).
    3. Moves every existing TPC-H table from public -> tpch.
    4. Regenerates TPC-H ``customer`` via DuckDB and loads it into
       tpch.customer (the table that was lost to the collision).
    5. Re-adds the orders->customer FK and ANALYZEs.

Safe to re-run: every step checks current state first.

Usage:
    python scripts/migrate_to_schemas.py
    python scripts/migrate_to_schemas.py --sf 1     # TPC-H scale for customer regen
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import psycopg2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.db_config import DB_CONFIG  # noqa: E402

# TPC-H tables that are NOT shared with TPC-DS (safe to move as-is).
TPCH_ONLY = ["region", "nation", "part", "supplier", "partsupp", "orders", "lineitem"]

# All TPC-DS tables (the only shared name with TPC-H is `customer`).
TPCDS_TABLES = [
    "income_band", "ship_mode", "reason", "warehouse", "promotion",
    "household_demographics", "customer_demographics", "customer_address",
    "date_dim", "time_dim", "item", "web_site", "web_page", "store",
    "call_center", "catalog_page", "customer", "inventory",
    "store_sales", "store_returns", "web_sales", "web_returns",
    "catalog_sales", "catalog_returns",
]

# TPC-H customer DDL (from db/tpch_schema.sql), created inside the tpch schema.
TPCH_CUSTOMER_DDL = """
CREATE TABLE IF NOT EXISTS tpch.customer (
    c_custkey    INTEGER        NOT NULL,
    c_name       VARCHAR(25)    NOT NULL,
    c_address    VARCHAR(40)    NOT NULL,
    c_nationkey  INTEGER        NOT NULL,
    c_phone      CHAR(15)       NOT NULL,
    c_acctbal    DECIMAL(15,2)  NOT NULL,
    c_mktsegment CHAR(10)       NOT NULL,
    c_comment    VARCHAR(117)   NOT NULL,
    PRIMARY KEY (c_custkey)
);
"""


# ---------------------------------------------------------------------------
def table_schema(cur, table: str) -> str | None:
    """Return the schema a table currently lives in, or None if absent."""
    cur.execute(
        "SELECT table_schema FROM information_schema.tables "
        "WHERE table_name = %s AND table_schema IN ('public','tpch','tpcds') "
        "ORDER BY CASE table_schema WHEN 'public' THEN 0 ELSE 1 END LIMIT 1",
        (table,),
    )
    row = cur.fetchone()
    return row[0] if row else None


def column_exists(cur, schema: str, table: str, col: str) -> bool:
    cur.execute(
        "SELECT 1 FROM information_schema.columns "
        "WHERE table_schema=%s AND table_name=%s AND column_name=%s",
        (schema, table, col),
    )
    return cur.fetchone() is not None


def move_table(cur, table: str, dest_schema: str) -> str:
    """Move a table to dest_schema if it isn't already there."""
    cur_schema = table_schema(cur, table)
    if cur_schema is None:
        return f"    [skip] {table:<24} not found"
    if cur_schema == dest_schema:
        return f"    [ok]   {table:<24} already in {dest_schema}"
    cur.execute(f'ALTER TABLE {cur_schema}.{table} SET SCHEMA {dest_schema};')
    return f"    [move] {table:<24} {cur_schema} -> {dest_schema}"


# ---------------------------------------------------------------------------
def regenerate_tpch_customer(sf: float, out_dir: Path) -> Path:
    """Use DuckDB to regenerate just the TPC-H customer CSV."""
    import duckdb
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "customer.csv"
    print(f"[i] Regenerating TPC-H customer via DuckDB (sf={sf}) ...")
    t0 = time.perf_counter()
    con = duckdb.connect(":memory:")
    con.execute("INSTALL tpch;")
    con.execute("LOAD tpch;")
    con.execute(f"CALL dbgen(sf={sf});")
    con.execute(
        f"COPY customer TO '{csv_path.as_posix()}' "
        f"(HEADER, DELIMITER '|', FORMAT csv)"
    )
    con.close()
    mb = csv_path.stat().st_size / (1024 * 1024)
    print(f"    wrote {csv_path.name} ({mb:.1f} MB) in {time.perf_counter()-t0:.1f}s")
    return csv_path


def load_tpch_customer(conn, csv_path: Path) -> int:
    with conn.cursor() as cur:
        cur.execute(TPCH_CUSTOMER_DDL)
        cur.execute("TRUNCATE tpch.customer;")
        with csv_path.open("rb") as f:
            cur.copy_expert(
                "COPY tpch.customer FROM STDIN WITH "
                "(FORMAT csv, HEADER true, DELIMITER '|')",
                f,
            )
        cur.execute("SELECT count(*) FROM tpch.customer;")
        n = cur.fetchone()[0]
    conn.commit()
    return n


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sf", type=float, default=1.0,
                    help="TPC-H scale factor for customer regen (default 1.0)")
    args = ap.parse_args()

    try:
        conn = psycopg2.connect(**DB_CONFIG)
    except psycopg2.OperationalError as exc:
        print(f"[!] Could not connect to PostgreSQL: {exc}", file=sys.stderr)
        return 1
    conn.autocommit = False

    try:
        cur = conn.cursor()

        print("[1] Creating schemas tpch + tpcds ...")
        cur.execute("CREATE SCHEMA IF NOT EXISTS tpch;")
        cur.execute("CREATE SCHEMA IF NOT EXISTS tpcds;")
        conn.commit()

        # 2) Move TPC-DS tables first (this relocates the colliding
        #    public.customer, which is the TPC-DS one, into tpcds).
        print("[2] Moving TPC-DS tables -> tpcds ...")
        for t in TPCDS_TABLES:
            # Only move a `customer` that is actually TPC-DS-shaped.
            if t == "customer":
                sch = table_schema(cur, "customer")
                if sch == "public" and not column_exists(cur, "public", "customer", "c_customer_sk"):
                    print("    [warn] public.customer is not TPC-DS-shaped; skipping move")
                    continue
            print(move_table(cur, t, "tpcds"))
        conn.commit()

        # 3) Move the TPC-H-only tables into tpch.
        print("[3] Moving TPC-H tables -> tpch ...")
        for t in TPCH_ONLY:
            print(move_table(cur, t, "tpch"))
        conn.commit()

        # 4) Regenerate + load tpch.customer (lost to the collision).
        print("[4] Restoring tpch.customer ...")
        need_customer = table_schema(cur, "customer") is None or \
            not column_exists(cur, "tpch", "customer", "c_custkey")
        if need_customer:
            csv = regenerate_tpch_customer(args.sf, PROJECT_ROOT / "data" / "tpch" / "raw_data")
            n = load_tpch_customer(conn, csv)
            print(f"    [ok] loaded tpch.customer: {n} rows")
        else:
            cur.execute("SELECT count(*) FROM tpch.customer;")
            print(f"    [ok] tpch.customer already present: {cur.fetchone()[0]} rows")

        # 5) Re-add FK + ANALYZE (best-effort; FK failure is non-fatal).
        print("[5] Constraints + ANALYZE ...")
        try:
            cur.execute("""
                DO $$ BEGIN
                  IF NOT EXISTS (
                    SELECT 1 FROM pg_constraint WHERE conname = 'orders_custkey_fk'
                  ) THEN
                    ALTER TABLE tpch.orders
                      ADD CONSTRAINT orders_custkey_fk
                      FOREIGN KEY (o_custkey) REFERENCES tpch.customer(c_custkey)
                      NOT VALID;
                  END IF;
                END $$;
            """)
            conn.commit()
            print("    [ok] orders->customer FK present (NOT VALID; planner-visible)")
        except Exception as exc:  # noqa: BLE001
            conn.rollback()
            print(f"    [warn] FK add skipped: {exc}")

        with conn.cursor() as c2:
            c2.execute("ANALYZE tpch.customer;")
        conn.commit()
        print("    [ok] analyzed tpch.customer")

        # Summary
        print("\n[verify] row counts:")
        for sch, tbl in [("tpch", "customer"), ("tpcds", "customer"),
                         ("tpch", "orders"), ("tpch", "lineitem")]:
            try:
                with conn.cursor() as c3:
                    c3.execute(f"SELECT count(*) FROM {sch}.{tbl};")
                    print(f"    {sch}.{tbl:<10} = {c3.fetchone()[0]}")
            except Exception as exc:  # noqa: BLE001
                print(f"    {sch}.{tbl:<10} = ERR {exc}")
    finally:
        conn.close()

    print("\n[OK] Migration complete. Collectors will SET search_path per benchmark.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
