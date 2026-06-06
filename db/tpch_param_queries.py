"""
tpch_param_queries.py
=====================
Parameterized TPC-H workload generator.

For each of the 22 standard TPC-H queries we define a SQL template
plus a list of parameter dictionaries that follow the official
qparam ranges from the TPC-H specification (Appendix B). At
generation time we fill the templates with each parameter set and
emit a flat list of "concrete" queries that the plan collector can
treat exactly like any other workload.

Why parameterize?
-----------------
The bottleneck for the Phase 3A learned models was data: only 22
unique queries. By varying parameters within their TPC-H-spec
ranges we get 110 distinct queries (22 templates x 5 params each),
which translates to ~440 plans once you cross-multiply with the
4 optimizer variants. Critically, parameterizing changes
*selectivity*, which forces the planner into different plan shapes
- the kind of contrast a learned cost model needs.

Output format matches `parse_queries()` in collect_tpch_plans.py:
    [{"id": "q01_p0", "tag": "...", "sql": "..."}, ...]
"""

from __future__ import annotations

from string import Template
from typing import Iterable


# ---------------------------------------------------------------------------
# 22 queries as `string.Template` strings.
#   Placeholders use $NAME (NOT %s, NOT f-strings) because the SQL
#   itself contains many `$` would-be-interpreted characters from
#   PostgreSQL's dollar-quoting style — Template's $-escape rules are
#   the cleanest.
# ---------------------------------------------------------------------------
TEMPLATES: dict[str, tuple[str, str]] = {
    # ----- Q1 -----
    "q01": ("agg+filter+sort", """
SELECT
    l_returnflag, l_linestatus,
    SUM(l_quantity) AS sum_qty,
    SUM(l_extendedprice) AS sum_base_price,
    SUM(l_extendedprice * (1 - l_discount)) AS sum_disc_price,
    SUM(l_extendedprice * (1 - l_discount) * (1 + l_tax)) AS sum_charge,
    AVG(l_quantity) AS avg_qty, AVG(l_extendedprice) AS avg_price,
    AVG(l_discount) AS avg_disc, COUNT(*) AS count_order
FROM lineitem
WHERE l_shipdate <= DATE '1998-12-01' - INTERVAL '$DELTA' DAY
GROUP BY l_returnflag, l_linestatus
ORDER BY l_returnflag, l_linestatus
"""),

    # ----- Q2 -----
    "q02": ("join+subquery+agg+sort", """
SELECT s_acctbal, s_name, n_name, p_partkey, p_mfgr,
       s_address, s_phone, s_comment
FROM part, supplier, partsupp, nation, region
WHERE p_partkey = ps_partkey AND s_suppkey = ps_suppkey
  AND p_size = $SIZE AND p_type LIKE '%$TYPE'
  AND s_nationkey = n_nationkey AND n_regionkey = r_regionkey
  AND r_name = '$REGION'
  AND ps_supplycost = (
      SELECT MIN(ps_supplycost)
      FROM partsupp, supplier, nation, region
      WHERE p_partkey = ps_partkey AND s_suppkey = ps_suppkey
        AND s_nationkey = n_nationkey AND n_regionkey = r_regionkey
        AND r_name = '$REGION'
  )
ORDER BY s_acctbal DESC, n_name, s_name, p_partkey
LIMIT 100
"""),

    # ----- Q3 -----
    "q03": ("join+agg+sort+limit", """
SELECT l_orderkey,
       SUM(l_extendedprice * (1 - l_discount)) AS revenue,
       o_orderdate, o_shippriority
FROM customer, orders, lineitem
WHERE c_mktsegment = '$SEGMENT'
  AND c_custkey = o_custkey AND l_orderkey = o_orderkey
  AND o_orderdate < DATE '$DATE'
  AND l_shipdate  > DATE '$DATE'
GROUP BY l_orderkey, o_orderdate, o_shippriority
ORDER BY revenue DESC, o_orderdate
LIMIT 10
"""),

    # ----- Q4 -----
    "q04": ("semijoin+agg+sort", """
SELECT o_orderpriority, COUNT(*) AS order_count
FROM orders
WHERE o_orderdate >= DATE '$DATE'
  AND o_orderdate <  DATE '$DATE' + INTERVAL '3' MONTH
  AND EXISTS (
      SELECT 1 FROM lineitem
      WHERE l_orderkey = o_orderkey AND l_commitdate < l_receiptdate
  )
GROUP BY o_orderpriority
ORDER BY o_orderpriority
"""),

    # ----- Q5 -----
    "q05": ("join+agg+sort", """
SELECT n_name, SUM(l_extendedprice * (1 - l_discount)) AS revenue
FROM customer, orders, lineitem, supplier, nation, region
WHERE c_custkey = o_custkey AND l_orderkey = o_orderkey
  AND l_suppkey = s_suppkey AND c_nationkey = s_nationkey
  AND s_nationkey = n_nationkey AND n_regionkey = r_regionkey
  AND r_name = '$REGION'
  AND o_orderdate >= DATE '$DATE'
  AND o_orderdate <  DATE '$DATE' + INTERVAL '1' YEAR
GROUP BY n_name
ORDER BY revenue DESC
"""),

    # ----- Q6 -----
    "q06": ("filter+agg", """
SELECT SUM(l_extendedprice * l_discount) AS revenue
FROM lineitem
WHERE l_shipdate >= DATE '$DATE'
  AND l_shipdate <  DATE '$DATE' + INTERVAL '1' YEAR
  AND l_discount BETWEEN $DISCOUNT - 0.01 AND $DISCOUNT + 0.01
  AND l_quantity < $QUANTITY
"""),

    # ----- Q7 -----
    "q07": ("join+agg+sort+subquery", """
SELECT supp_nation, cust_nation, l_year, SUM(volume) AS revenue
FROM (
    SELECT n1.n_name AS supp_nation, n2.n_name AS cust_nation,
           EXTRACT(YEAR FROM l_shipdate) AS l_year,
           l_extendedprice * (1 - l_discount) AS volume
    FROM supplier, lineitem, orders, customer, nation n1, nation n2
    WHERE s_suppkey = l_suppkey AND o_orderkey = l_orderkey
      AND c_custkey = o_custkey AND s_nationkey = n1.n_nationkey
      AND c_nationkey = n2.n_nationkey
      AND ((n1.n_name = '$NATION1' AND n2.n_name = '$NATION2')
        OR (n1.n_name = '$NATION2' AND n2.n_name = '$NATION1'))
      AND l_shipdate BETWEEN DATE '1995-01-01' AND DATE '1996-12-31'
) AS shipping
GROUP BY supp_nation, cust_nation, l_year
ORDER BY supp_nation, cust_nation, l_year
"""),

    # ----- Q8 -----
    "q08": ("join+agg+subquery+sort", """
SELECT o_year,
       SUM(CASE WHEN nation = '$NATION' THEN volume ELSE 0 END) / SUM(volume) AS mkt_share
FROM (
    SELECT EXTRACT(YEAR FROM o_orderdate) AS o_year,
           l_extendedprice * (1 - l_discount) AS volume,
           n2.n_name AS nation
    FROM part, supplier, lineitem, orders, customer, nation n1, nation n2, region
    WHERE p_partkey = l_partkey AND s_suppkey = l_suppkey
      AND l_orderkey = o_orderkey AND o_custkey = c_custkey
      AND c_nationkey = n1.n_nationkey AND n1.n_regionkey = r_regionkey
      AND r_name = '$REGION' AND s_nationkey = n2.n_nationkey
      AND o_orderdate BETWEEN DATE '1995-01-01' AND DATE '1996-12-31'
      AND p_type = '$TYPE'
) AS all_nations
GROUP BY o_year
ORDER BY o_year
"""),

    # ----- Q9 -----
    "q09": ("join+agg+subquery+sort", """
SELECT nation, o_year, SUM(amount) AS sum_profit
FROM (
    SELECT n_name AS nation,
           EXTRACT(YEAR FROM o_orderdate) AS o_year,
           l_extendedprice * (1 - l_discount) - ps_supplycost * l_quantity AS amount
    FROM part, supplier, lineitem, partsupp, orders, nation
    WHERE s_suppkey = l_suppkey AND ps_suppkey = l_suppkey
      AND ps_partkey = l_partkey AND p_partkey = l_partkey
      AND o_orderkey = l_orderkey AND s_nationkey = n_nationkey
      AND p_name LIKE '%$COLOR%'
) AS profit
GROUP BY nation, o_year
ORDER BY nation, o_year DESC
"""),

    # ----- Q10 -----
    "q10": ("join+agg+sort+limit", """
SELECT c_custkey, c_name,
       SUM(l_extendedprice * (1 - l_discount)) AS revenue,
       c_acctbal, n_name, c_address, c_phone, c_comment
FROM customer, orders, lineitem, nation
WHERE c_custkey = o_custkey AND l_orderkey = o_orderkey
  AND o_orderdate >= DATE '$DATE'
  AND o_orderdate <  DATE '$DATE' + INTERVAL '3' MONTH
  AND l_returnflag = 'R' AND c_nationkey = n_nationkey
GROUP BY c_custkey, c_name, c_acctbal, c_phone, n_name, c_address, c_comment
ORDER BY revenue DESC
LIMIT 20
"""),

    # ----- Q11 -----
    "q11": ("join+agg+subquery+sort", """
SELECT ps_partkey, SUM(ps_supplycost * ps_availqty) AS value
FROM partsupp, supplier, nation
WHERE ps_suppkey = s_suppkey AND s_nationkey = n_nationkey
  AND n_name = '$NATION'
GROUP BY ps_partkey
HAVING SUM(ps_supplycost * ps_availqty) > (
    SELECT SUM(ps_supplycost * ps_availqty) * $FRACTION
    FROM partsupp, supplier, nation
    WHERE ps_suppkey = s_suppkey AND s_nationkey = n_nationkey
      AND n_name = '$NATION'
)
ORDER BY value DESC
"""),

    # ----- Q12 -----
    "q12": ("join+agg+sort", """
SELECT l_shipmode,
       SUM(CASE WHEN o_orderpriority = '1-URGENT' OR o_orderpriority = '2-HIGH'
                THEN 1 ELSE 0 END) AS high_line_count,
       SUM(CASE WHEN o_orderpriority <> '1-URGENT' AND o_orderpriority <> '2-HIGH'
                THEN 1 ELSE 0 END) AS low_line_count
FROM orders, lineitem
WHERE o_orderkey = l_orderkey
  AND l_shipmode IN ('$SHIPMODE1', '$SHIPMODE2')
  AND l_commitdate < l_receiptdate AND l_shipdate < l_commitdate
  AND l_receiptdate >= DATE '$DATE'
  AND l_receiptdate <  DATE '$DATE' + INTERVAL '1' YEAR
GROUP BY l_shipmode
ORDER BY l_shipmode
"""),

    # ----- Q13 -----
    "q13": ("outerjoin+agg+sort", """
SELECT c_count, COUNT(*) AS custdist
FROM (
    SELECT c_custkey, COUNT(o_orderkey) AS c_count
    FROM customer LEFT OUTER JOIN orders
      ON c_custkey = o_custkey
     AND o_comment NOT LIKE '%$WORD1%$WORD2%'
    GROUP BY c_custkey
) AS c_orders
GROUP BY c_count
ORDER BY custdist DESC, c_count DESC
"""),

    # ----- Q14 -----
    "q14": ("join+agg", """
SELECT 100.00 * SUM(CASE WHEN p_type LIKE 'PROMO%'
                         THEN l_extendedprice * (1 - l_discount)
                         ELSE 0 END)
              / SUM(l_extendedprice * (1 - l_discount)) AS promo_revenue
FROM lineitem, part
WHERE l_partkey = p_partkey
  AND l_shipdate >= DATE '$DATE'
  AND l_shipdate <  DATE '$DATE' + INTERVAL '1' MONTH
"""),

    # ----- Q15 -----
    "q15": ("join+agg+subquery+sort", """
WITH revenue0 AS (
    SELECT l_suppkey AS supplier_no,
           SUM(l_extendedprice * (1 - l_discount)) AS total_revenue
    FROM lineitem
    WHERE l_shipdate >= DATE '$DATE'
      AND l_shipdate <  DATE '$DATE' + INTERVAL '3' MONTH
    GROUP BY l_suppkey
)
SELECT s_suppkey, s_name, s_address, s_phone, total_revenue
FROM supplier, revenue0
WHERE s_suppkey = supplier_no
  AND total_revenue = (SELECT MAX(total_revenue) FROM revenue0)
ORDER BY s_suppkey
"""),

    # ----- Q16 -----
    "q16": ("join+agg+notin+sort", """
SELECT p_brand, p_type, p_size, COUNT(DISTINCT ps_suppkey) AS supplier_cnt
FROM partsupp, part
WHERE p_partkey = ps_partkey
  AND p_brand <> '$BRAND'
  AND p_type NOT LIKE '$TYPE%'
  AND p_size IN ($SIZES)
  AND ps_suppkey NOT IN (
      SELECT s_suppkey FROM supplier
      WHERE s_comment LIKE '%Customer%Complaints%'
  )
GROUP BY p_brand, p_type, p_size
ORDER BY supplier_cnt DESC, p_brand, p_type, p_size
"""),

    # ----- Q17 -----
    "q17": ("join+subquery+agg", """
SELECT SUM(l_extendedprice) / 7.0 AS avg_yearly
FROM lineitem, part
WHERE p_partkey = l_partkey
  AND p_brand = '$BRAND'
  AND p_container = '$CONTAINER'
  AND l_quantity < (
      SELECT 0.2 * AVG(l_quantity)
      FROM lineitem WHERE l_partkey = p_partkey
  )
"""),

    # ----- Q18 -----
    "q18": ("join+agg+subquery+sort+limit", """
SELECT c_name, c_custkey, o_orderkey, o_orderdate, o_totalprice,
       SUM(l_quantity) AS total_qty
FROM customer, orders, lineitem
WHERE o_orderkey IN (
        SELECT l_orderkey FROM lineitem
        GROUP BY l_orderkey HAVING SUM(l_quantity) > $QUANTITY
      )
  AND c_custkey = o_custkey AND o_orderkey = l_orderkey
GROUP BY c_name, c_custkey, o_orderkey, o_orderdate, o_totalprice
ORDER BY o_totalprice DESC, o_orderdate
LIMIT 100
"""),

    # ----- Q19 -----
    "q19": ("join+filter+agg", """
SELECT SUM(l_extendedprice * (1 - l_discount)) AS revenue
FROM lineitem, part
WHERE
    (p_partkey = l_partkey AND p_brand = '$BRAND1'
     AND p_container IN ('SM CASE', 'SM BOX', 'SM PACK', 'SM PKG')
     AND l_quantity >= $Q1 AND l_quantity <= $Q1 + 10
     AND p_size BETWEEN 1 AND 5
     AND l_shipmode IN ('AIR', 'AIR REG')
     AND l_shipinstruct = 'DELIVER IN PERSON')
 OR (p_partkey = l_partkey AND p_brand = '$BRAND2'
     AND p_container IN ('MED BAG', 'MED BOX', 'MED PKG', 'MED PACK')
     AND l_quantity >= $Q2 AND l_quantity <= $Q2 + 10
     AND p_size BETWEEN 1 AND 10
     AND l_shipmode IN ('AIR', 'AIR REG')
     AND l_shipinstruct = 'DELIVER IN PERSON')
 OR (p_partkey = l_partkey AND p_brand = '$BRAND3'
     AND p_container IN ('LG CASE', 'LG BOX', 'LG PACK', 'LG PKG')
     AND l_quantity >= $Q3 AND l_quantity <= $Q3 + 10
     AND p_size BETWEEN 1 AND 15
     AND l_shipmode IN ('AIR', 'AIR REG')
     AND l_shipinstruct = 'DELIVER IN PERSON')
"""),

    # ----- Q20 -----
    "q20": ("join+subquery+sort", """
SELECT s_name, s_address
FROM supplier, nation
WHERE s_suppkey IN (
        SELECT ps_suppkey FROM partsupp
        WHERE ps_partkey IN (
                SELECT p_partkey FROM part
                WHERE p_name LIKE '$COLOR%'
              )
          AND ps_availqty > (
                SELECT 0.5 * SUM(l_quantity)
                FROM lineitem
                WHERE l_partkey = ps_partkey AND l_suppkey = ps_suppkey
                  AND l_shipdate >= DATE '$DATE'
                  AND l_shipdate <  DATE '$DATE' + INTERVAL '1' YEAR
              )
      )
  AND s_nationkey = n_nationkey AND n_name = '$NATION'
ORDER BY s_name
"""),

    # ----- Q21 -----
    "q21": ("join+agg+subquery+sort+limit", """
SELECT s_name, COUNT(*) AS numwait
FROM supplier, lineitem l1, orders, nation
WHERE s_suppkey = l1.l_suppkey AND o_orderkey = l1.l_orderkey
  AND o_orderstatus = 'F'
  AND l1.l_receiptdate > l1.l_commitdate
  AND EXISTS (SELECT 1 FROM lineitem l2
              WHERE l2.l_orderkey = l1.l_orderkey AND l2.l_suppkey <> l1.l_suppkey)
  AND NOT EXISTS (SELECT 1 FROM lineitem l3
                  WHERE l3.l_orderkey = l1.l_orderkey AND l3.l_suppkey <> l1.l_suppkey
                    AND l3.l_receiptdate > l3.l_commitdate)
  AND s_nationkey = n_nationkey AND n_name = '$NATION'
GROUP BY s_name
ORDER BY numwait DESC, s_name
LIMIT 100
"""),

    # ----- Q22 -----
    "q22": ("agg+subquery+sort", """
SELECT cntrycode, COUNT(*) AS numcust, SUM(c_acctbal) AS totacctbal
FROM (
    SELECT SUBSTRING(c_phone FROM 1 FOR 2) AS cntrycode, c_acctbal
    FROM customer
    WHERE SUBSTRING(c_phone FROM 1 FOR 2) IN ($CCS)
      AND c_acctbal > (
          SELECT AVG(c_acctbal) FROM customer
          WHERE c_acctbal > 0.00
            AND SUBSTRING(c_phone FROM 1 FOR 2) IN ($CCS)
      )
      AND NOT EXISTS (SELECT 1 FROM orders WHERE o_custkey = c_custkey)
) AS custsale
GROUP BY cntrycode
ORDER BY cntrycode
"""),
}


# ---------------------------------------------------------------------------
# Parameter sets — 5 per query, drawn from the TPC-H spec ranges.
# Index 2 (the middle one) is the canonical reference set used in
# tpch_queries.sql, so re-running won't disturb the original 87 plans.
# ---------------------------------------------------------------------------
PARAMS: dict[str, list[dict[str, str | int | float]]] = {
    "q01": [{"DELTA": d} for d in (60, 75, 90, 105, 120)],
    "q02": [
        {"SIZE": 15, "TYPE": "BRASS",  "REGION": "AFRICA"},
        {"SIZE": 25, "TYPE": "COPPER", "REGION": "AMERICA"},
        {"SIZE": 15, "TYPE": "BRASS",  "REGION": "EUROPE"},
        {"SIZE": 35, "TYPE": "STEEL",  "REGION": "ASIA"},
        {"SIZE": 45, "TYPE": "TIN",    "REGION": "MIDDLE EAST"},
    ],
    "q03": [
        {"SEGMENT": "AUTOMOBILE", "DATE": "1995-03-15"},
        {"SEGMENT": "BUILDING",   "DATE": "1995-03-15"},
        {"SEGMENT": "FURNITURE",  "DATE": "1995-03-15"},
        {"SEGMENT": "HOUSEHOLD",  "DATE": "1995-03-15"},
        {"SEGMENT": "MACHINERY",  "DATE": "1995-03-15"},
    ],
    "q04": [
        {"DATE": "1993-07-01"},
        {"DATE": "1994-01-01"},
        {"DATE": "1995-04-01"},
        {"DATE": "1996-07-01"},
        {"DATE": "1997-04-01"},
    ],
    "q05": [
        {"REGION": "AFRICA",      "DATE": "1994-01-01"},
        {"REGION": "AMERICA",     "DATE": "1994-01-01"},
        {"REGION": "ASIA",        "DATE": "1994-01-01"},
        {"REGION": "EUROPE",      "DATE": "1995-01-01"},
        {"REGION": "MIDDLE EAST", "DATE": "1996-01-01"},
    ],
    "q06": [
        {"DATE": "1993-01-01", "DISCOUNT": 0.05, "QUANTITY": 24},
        {"DATE": "1994-01-01", "DISCOUNT": 0.06, "QUANTITY": 24},
        {"DATE": "1995-01-01", "DISCOUNT": 0.07, "QUANTITY": 24},
        {"DATE": "1996-01-01", "DISCOUNT": 0.08, "QUANTITY": 25},
        {"DATE": "1997-01-01", "DISCOUNT": 0.09, "QUANTITY": 25},
    ],
    "q07": [
        {"NATION1": "FRANCE",  "NATION2": "GERMANY"},
        {"NATION1": "INDIA",   "NATION2": "JAPAN"},
        {"NATION1": "UNITED STATES", "NATION2": "CANADA"},
        {"NATION1": "BRAZIL",  "NATION2": "ARGENTINA"},
        {"NATION1": "CHINA",   "NATION2": "VIETNAM"},
    ],
    "q08": [
        {"NATION": "BRAZIL",      "REGION": "AMERICA", "TYPE": "ECONOMY ANODIZED STEEL"},
        {"NATION": "CANADA",      "REGION": "AMERICA", "TYPE": "ECONOMY ANODIZED STEEL"},
        {"NATION": "INDIA",       "REGION": "ASIA",    "TYPE": "ECONOMY POLISHED COPPER"},
        {"NATION": "FRANCE",      "REGION": "EUROPE",  "TYPE": "PROMO BURNISHED NICKEL"},
        {"NATION": "JORDAN",      "REGION": "MIDDLE EAST", "TYPE": "STANDARD BRUSHED TIN"},
    ],
    "q09": [{"COLOR": c} for c in ("green", "blue", "red", "yellow", "white")],
    "q10": [
        {"DATE": "1993-10-01"},
        {"DATE": "1994-01-01"},
        {"DATE": "1994-04-01"},
        {"DATE": "1994-07-01"},
        {"DATE": "1994-10-01"},
    ],
    "q11": [
        {"NATION": "GERMANY", "FRACTION": 0.0001},
        {"NATION": "FRANCE",  "FRACTION": 0.0001},
        {"NATION": "JAPAN",   "FRACTION": 0.0001},
        {"NATION": "INDIA",   "FRACTION": 0.0001},
        {"NATION": "UNITED STATES", "FRACTION": 0.0001},
    ],
    "q12": [
        {"SHIPMODE1": "MAIL",  "SHIPMODE2": "SHIP",  "DATE": "1994-01-01"},
        {"SHIPMODE1": "RAIL",  "SHIPMODE2": "TRUCK", "DATE": "1994-01-01"},
        {"SHIPMODE1": "AIR",   "SHIPMODE2": "REG AIR","DATE": "1995-01-01"},
        {"SHIPMODE1": "FOB",   "SHIPMODE2": "MAIL",  "DATE": "1996-01-01"},
        {"SHIPMODE1": "TRUCK", "SHIPMODE2": "SHIP",  "DATE": "1997-01-01"},
    ],
    "q13": [
        {"WORD1": "special", "WORD2": "requests"},
        {"WORD1": "pending", "WORD2": "deposits"},
        {"WORD1": "express", "WORD2": "packages"},
        {"WORD1": "regular", "WORD2": "accounts"},
        {"WORD1": "unusual", "WORD2": "requests"},
    ],
    "q14": [
        {"DATE": "1993-09-01"},
        {"DATE": "1994-09-01"},
        {"DATE": "1995-09-01"},
        {"DATE": "1996-09-01"},
        {"DATE": "1997-09-01"},
    ],
    "q15": [
        {"DATE": "1993-04-01"},
        {"DATE": "1994-04-01"},
        {"DATE": "1995-04-01"},
        {"DATE": "1996-01-01"},
        {"DATE": "1997-04-01"},
    ],
    "q16": [
        {"BRAND": "Brand#45", "TYPE": "MEDIUM POLISHED",
         "SIZES": "49, 14, 23, 45, 19, 3, 36, 9"},
        {"BRAND": "Brand#22", "TYPE": "SMALL ANODIZED",
         "SIZES": "1, 5, 12, 18, 25, 33, 41, 50"},
        {"BRAND": "Brand#33", "TYPE": "LARGE BURNISHED",
         "SIZES": "2, 8, 14, 20, 26, 32, 38, 44"},
        {"BRAND": "Brand#15", "TYPE": "STANDARD BRUSHED",
         "SIZES": "3, 10, 17, 24, 31, 38, 45, 50"},
        {"BRAND": "Brand#41", "TYPE": "PROMO PLATED",
         "SIZES": "4, 11, 18, 25, 32, 39, 46, 50"},
    ],
    "q17": [
        {"BRAND": "Brand#23", "CONTAINER": "MED BOX"},
        {"BRAND": "Brand#11", "CONTAINER": "SM PACK"},
        {"BRAND": "Brand#34", "CONTAINER": "LG CASE"},
        {"BRAND": "Brand#42", "CONTAINER": "JUMBO BAG"},
        {"BRAND": "Brand#55", "CONTAINER": "WRAP DRUM"},
    ],
    "q18": [{"QUANTITY": q} for q in (300, 305, 310, 312, 315)],
    "q19": [
        {"BRAND1": "Brand#12", "BRAND2": "Brand#23", "BRAND3": "Brand#34",
         "Q1": 1,  "Q2": 10, "Q3": 20},
        {"BRAND1": "Brand#15", "BRAND2": "Brand#25", "BRAND3": "Brand#35",
         "Q1": 3,  "Q2": 13, "Q3": 23},
        {"BRAND1": "Brand#18", "BRAND2": "Brand#28", "BRAND3": "Brand#38",
         "Q1": 5,  "Q2": 15, "Q3": 25},
        {"BRAND1": "Brand#21", "BRAND2": "Brand#31", "BRAND3": "Brand#41",
         "Q1": 7,  "Q2": 17, "Q3": 27},
        {"BRAND1": "Brand#24", "BRAND2": "Brand#33", "BRAND3": "Brand#44",
         "Q1": 9,  "Q2": 19, "Q3": 29},
    ],
    "q20": [
        {"COLOR": "forest", "DATE": "1994-01-01", "NATION": "CANADA"},
        {"COLOR": "olive",  "DATE": "1994-01-01", "NATION": "FRANCE"},
        {"COLOR": "sky",    "DATE": "1995-01-01", "NATION": "JAPAN"},
        {"COLOR": "rose",   "DATE": "1996-01-01", "NATION": "GERMANY"},
        {"COLOR": "lemon",  "DATE": "1997-01-01", "NATION": "PERU"},
    ],
    "q21": [{"NATION": n} for n in (
        "SAUDI ARABIA", "EGYPT", "JORDAN", "IRAN", "IRAQ",
    )],
    "q22": [
        {"CCS": "'13','31','23','29','30','18','17'"},
        {"CCS": "'10','11','12','13','14','15','16'"},
        {"CCS": "'20','21','22','23','24','25','26'"},
        {"CCS": "'30','31','32','33','34','35','36'"},
        {"CCS": "'17','19','22','25','28','31','33'"},
    ],
}


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------
def generate() -> list[dict[str, str]]:
    """
    Produce the full parameterized workload as a flat list of dicts:
        {"id": "q05_p2", "tag": "join+agg+sort", "sql": "...resolved..."}

    Each query template is filled with each of its 5 parameter sets,
    so the output has |TEMPLATES| * 5 = 110 entries when complete.
    """
    out: list[dict[str, str]] = []
    for qid, (tag, template_sql) in TEMPLATES.items():
        if qid not in PARAMS:
            continue
        tmpl = Template(template_sql.strip())
        for i, params in enumerate(PARAMS[qid]):
            try:
                sql = tmpl.substitute({k: str(v) for k, v in params.items()})
            except KeyError as exc:
                raise ValueError(f"missing placeholder {exc} for {qid} params {params}") from exc
            out.append({
                "id":     f"{qid}_p{i}",
                "tag":    tag,
                "params": params,
                "sql":    sql,
            })
    return out


def query_count() -> int:
    return sum(len(PARAMS.get(q, [])) for q in TEMPLATES)


if __name__ == "__main__":
    qs = generate()
    print(f"generated {len(qs)} parameterized TPC-H queries")
    print("first few:")
    for q in qs[:3]:
        print(f"  - {q['id']}  (tag: {q['tag']}, params: {q['params']})")
