"""
pg_variants.py
==============
Produces N candidate execution plans for an arbitrary SQL by toggling
PostgreSQL planner knobs and re-running EXPLAIN (FORMAT JSON).

The variants we use here mirror the offline collectors
(scripts/collect_tpch_plans.py, scripts/collect_tpcds_plans.py) so
that the model evaluates *familiar* plan shapes at inference time —
the same shapes it was trained on.

Variants exposed:
    default       : whatever PG decides
    no_hashjoin   : SET enable_hashjoin  = off
    no_mergejoin  : SET enable_mergejoin = off
    no_nestloop   : SET enable_nestloop  = off

You can extend this dict in place; the rest of the pipeline picks
new entries up automatically.
"""

from __future__ import annotations

from dataclasses import dataclass

VARIANTS: dict[str, list[str]] = {
    "default":      [],
    "no_hashjoin":  ["SET enable_hashjoin  = off"],
    "no_mergejoin": ["SET enable_mergejoin = off"],
    "no_nestloop":  ["SET enable_nestloop  = off"],
}


@dataclass
class GeneratedPlan:
    variant:        str
    knobs:          list[str]
    plan_json:      list  # raw EXPLAIN (FORMAT JSON) payload
    estimated_cost: float
