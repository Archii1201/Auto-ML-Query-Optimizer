"""
tpcds_queries.py
================
Phase 3B: a curated, PG-adapted subset of TPC-DS queries.

The full TPC-DS query suite has 99 queries; many use SQL features
PostgreSQL doesn't support out of the box (ROLLUP with GROUPING_ID,
some MS-SQL dialect constructs, optional COLLATE clauses). For
Phase 3B we hand-pick 20 queries that:

    * cover diverse plan shapes (joins / aggregates / window funcs / TOP-K)
    * run cleanly on PG 13+ without modifications
    * exercise the TPC-DS schema's fact tables (store_sales, web_sales,
      catalog_sales) and the larger dimensions (customer, item, date_dim)

Same output contract as db.tpch_param_queries.generate():
    [{"id": "ds_q03", "tag": "...", "sql": "..."}, ...]
"""

from __future__ import annotations


QUERIES: list[dict[str, str]] = [
    {
        "id":  "ds_q01_filter_agg",
        "tag": "filter+agg",
        "sql": """
            SELECT i_item_id, AVG(ss_quantity) avg_qty
            FROM store_sales, item, date_dim
            WHERE ss_item_sk = i_item_sk
              AND ss_sold_date_sk = d_date_sk
              AND d_year = 2000
            GROUP BY i_item_id
            ORDER BY i_item_id
            LIMIT 100
        """,
    },
    {
        "id":  "ds_q02_topk_revenue",
        "tag": "join+agg+sort+limit",
        "sql": """
            SELECT i_item_id, SUM(ss_ext_sales_price) tot
            FROM store_sales, item, date_dim
            WHERE ss_item_sk = i_item_sk
              AND ss_sold_date_sk = d_date_sk
              AND d_moy = 11
            GROUP BY i_item_id
            ORDER BY tot DESC
            LIMIT 50
        """,
    },
    {
        "id":  "ds_q03_promo_effect",
        "tag": "join+agg",
        "sql": """
            SELECT
                100.0 * SUM(CASE WHEN p_channel_email = 'N' OR p_channel_event = 'N'
                                 THEN ss_ext_sales_price ELSE 0 END)
                      / NULLIF(SUM(ss_ext_sales_price), 0) AS promo_share
            FROM store_sales, promotion
            WHERE ss_promo_sk = p_promo_sk
        """,
    },
    {
        "id":  "ds_q04_customer_topk",
        "tag": "join+agg+sort+limit",
        "sql": """
            SELECT c_customer_id, SUM(ss_net_paid) total_paid
            FROM customer, store_sales
            WHERE ss_customer_sk = c_customer_sk
            GROUP BY c_customer_id
            ORDER BY total_paid DESC
            LIMIT 100
        """,
    },
    {
        "id":  "ds_q05_returns_filter",
        "tag": "join+agg",
        "sql": """
            SELECT s_store_id, SUM(sr_return_amt) tot_returns
            FROM store_returns, store, date_dim
            WHERE sr_store_sk = s_store_sk
              AND sr_returned_date_sk = d_date_sk
              AND d_year = 2001
            GROUP BY s_store_id
            ORDER BY tot_returns DESC
        """,
    },
    {
        "id":  "ds_q06_web_vs_store",
        "tag": "join+agg+subquery",
        "sql": """
            SELECT i_item_id,
                   SUM(CASE WHEN src='web'   THEN amt ELSE 0 END) web_amt,
                   SUM(CASE WHEN src='store' THEN amt ELSE 0 END) store_amt
            FROM (
                SELECT 'web'   AS src, ws_item_sk AS isk, ws_ext_sales_price AS amt
                FROM web_sales
                UNION ALL
                SELECT 'store' AS src, ss_item_sk AS isk, ss_ext_sales_price AS amt
                FROM store_sales
            ) u
            JOIN item ON u.isk = i_item_sk
            GROUP BY i_item_id
            ORDER BY i_item_id
            LIMIT 100
        """,
    },
    {
        "id":  "ds_q07_yearly_trend",
        "tag": "join+agg+sort",
        "sql": """
            SELECT d_year, SUM(ss_ext_sales_price) yearly
            FROM store_sales, date_dim
            WHERE ss_sold_date_sk = d_date_sk
            GROUP BY d_year
            ORDER BY d_year
        """,
    },
    {
        "id":  "ds_q08_addr_filter",
        "tag": "join+agg",
        "sql": """
            SELECT ca_state, COUNT(*) n_customers
            FROM customer, customer_address
            WHERE c_current_addr_sk = ca_address_sk
            GROUP BY ca_state
            ORDER BY n_customers DESC
            LIMIT 20
        """,
    },
    {
        "id":  "ds_q09_avg_basket",
        "tag": "agg+filter",
        "sql": """
            SELECT AVG(ss_quantity) avg_qty, AVG(ss_ext_sales_price) avg_price
            FROM store_sales
            WHERE ss_quantity > 0
        """,
    },
    {
        "id":  "ds_q10_window_topk",
        "tag": "join+window+sort+limit",
        "sql": """
            SELECT i_item_id, ss_quantity,
                   RANK() OVER (PARTITION BY i_item_id ORDER BY ss_quantity DESC) rk
            FROM store_sales, item
            WHERE ss_item_sk = i_item_sk
            ORDER BY i_item_id, rk
            LIMIT 200
        """,
    },
    {
        "id":  "ds_q11_inventory_check",
        "tag": "join+agg",
        "sql": """
            SELECT w_warehouse_name, AVG(inv_quantity_on_hand) avg_qty
            FROM inventory, warehouse, date_dim
            WHERE inv_warehouse_sk = w_warehouse_sk
              AND inv_date_sk = d_date_sk
              AND d_year = 2001
            GROUP BY w_warehouse_name
            ORDER BY avg_qty DESC
        """,
    },
    {
        "id":  "ds_q12_catalog_topk",
        "tag": "join+agg+sort+limit",
        "sql": """
            SELECT i_category, SUM(cs_ext_sales_price) cat_sales
            FROM catalog_sales, item
            WHERE cs_item_sk = i_item_sk
            GROUP BY i_category
            ORDER BY cat_sales DESC
            LIMIT 25
        """,
    },
    {
        "id":  "ds_q13_high_quantity",
        "tag": "filter+agg",
        "sql": """
            SELECT COUNT(*) high_qty_orders
            FROM store_sales
            WHERE ss_quantity > 50
        """,
    },
    {
        "id":  "ds_q14_state_revenue",
        "tag": "join+agg+sort",
        "sql": """
            SELECT ca_state, SUM(ss_ext_sales_price) rev
            FROM store_sales, customer, customer_address
            WHERE ss_customer_sk = c_customer_sk
              AND c_current_addr_sk = ca_address_sk
            GROUP BY ca_state
            ORDER BY rev DESC
            LIMIT 10
        """,
    },
    {
        "id":  "ds_q15_returns_ratio",
        "tag": "join+agg+subquery",
        "sql": """
            SELECT i_item_id,
                   SUM(sr_return_amt) / NULLIF(SUM(ss_ext_sales_price), 0) ret_rate
            FROM store_sales
            LEFT JOIN store_returns
              ON ss_ticket_number = sr_ticket_number
             AND ss_item_sk = sr_item_sk
            JOIN item ON ss_item_sk = i_item_sk
            GROUP BY i_item_id
            HAVING SUM(ss_ext_sales_price) > 0
            ORDER BY ret_rate DESC NULLS LAST
            LIMIT 50
        """,
    },
    {
        "id":  "ds_q16_web_returns",
        "tag": "join+agg",
        "sql": """
            SELECT wr_returning_customer_sk, COUNT(*) n_returns
            FROM web_returns
            WHERE wr_return_quantity > 0
            GROUP BY wr_returning_customer_sk
            ORDER BY n_returns DESC
            LIMIT 100
        """,
    },
    {
        "id":  "ds_q17_dow_pattern",
        "tag": "join+agg+sort",
        "sql": """
            SELECT d_day_name, SUM(ss_ext_sales_price) day_rev
            FROM store_sales, date_dim
            WHERE ss_sold_date_sk = d_date_sk
            GROUP BY d_day_name
            ORDER BY day_rev DESC
        """,
    },
    {
        "id":  "ds_q18_brand_topk",
        "tag": "join+agg+sort+limit",
        "sql": """
            SELECT i_brand, SUM(ss_ext_sales_price) brand_rev
            FROM store_sales, item
            WHERE ss_item_sk = i_item_sk
            GROUP BY i_brand
            ORDER BY brand_rev DESC
            LIMIT 30
        """,
    },
    {
        "id":  "ds_q19_self_join",
        "tag": "join+agg",
        "sql": """
            SELECT s1.s_state, COUNT(*) n_pairs
            FROM store s1, store s2
            WHERE s1.s_state = s2.s_state
              AND s1.s_store_sk < s2.s_store_sk
            GROUP BY s1.s_state
            ORDER BY n_pairs DESC
            LIMIT 20
        """,
    },
    {
        "id":  "ds_q20_demographic",
        "tag": "join+agg+filter",
        "sql": """
            SELECT cd_gender, cd_marital_status,
                   AVG(ss_ext_sales_price) avg_purchase
            FROM store_sales, customer, customer_demographics
            WHERE ss_customer_sk = c_customer_sk
              AND c_current_cdemo_sk = cd_demo_sk
            GROUP BY cd_gender, cd_marital_status
            ORDER BY avg_purchase DESC
        """,
    },
]


def generate() -> list[dict[str, str]]:
    """Return all queries, each cleaned of leading/trailing whitespace."""
    return [
        {"id": q["id"], "tag": q["tag"], "sql": q["sql"].strip()}
        for q in QUERIES
    ]


if __name__ == "__main__":
    qs = generate()
    print(f"loaded {len(qs)} TPC-DS queries")
    for q in qs[:3]:
        print(f"  - {q['id']}  ({q['tag']})")
