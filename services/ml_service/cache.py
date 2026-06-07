"""
cache.py
========
Two-tier LRU caches keyed by SHA-256 hashes:

    * sql_cache    : sql_hash -> (winner, candidates) for /plan-pick
    * plan_cache   : plan_hash -> predicted_ms      for /predict

DSA — *hashing*:
    SHA-256 over a canonicalised payload gives us a fixed-size key
    independent of the query length. Two queries differing only by
    whitespace / case produce the same hash, which is *exactly* the
    behaviour a query-cache wants.

We use `cachetools.LRUCache` for bounded, O(1) get/set with eviction.
A real deployment would back this with Redis (`redis-py` has the
identical interface). For the FastAPI service this in-process
LRU is fine and dependency-free.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from threading import Lock
from typing import Any

from cachetools import LRUCache


# ---------------------------------------------------------------------------
# Hashing helpers
# ---------------------------------------------------------------------------
def _normalize_sql(sql: str) -> str:
    """
    Cheap canonicalisation for the cache key:
        - collapse whitespace
        - strip trailing semicolons + spaces
        - lowercase  (we don't care about case for caching)

    NOTE: this is NOT SQL-aware. Two semantically equivalent queries
    written differently (e.g. column reorder in SELECT, alias
    differences) will hash to different keys. That's acceptable —
    we'd rather have a few cache misses than ever return a wrong plan.
    """
    return " ".join(sql.split()).rstrip(";").strip().lower()


def hash_sql(sql: str) -> str:
    """SHA-256 hex digest of the canonicalised SQL."""
    payload = _normalize_sql(sql).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def hash_plan(plan_json: list[dict[str, Any]]) -> str:
    """
    SHA-256 of the plan JSON, sorted-keys for stability.

    EXPLAIN output is order-stable across runs, so the plan hash is
    a strong identifier for "this exact plan tree". Two plans with
    the same hash will always have the same predicted runtime —
    which is precisely the caching invariant we want.
    """
    payload = json.dumps(plan_json, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# ---------------------------------------------------------------------------
# Cache wrapper
# ---------------------------------------------------------------------------
@dataclass
class CacheStats:
    hits:   int = 0
    misses: int = 0
    sets:   int = 0
    size:   int = 0
    capacity: int = 0


class HashedLRUCache:
    """Thread-safe LRU keyed by SHA-256 hex digests."""

    def __init__(self, capacity: int = 1024) -> None:
        self._cache: LRUCache = LRUCache(maxsize=capacity)
        self._lock:  Lock     = Lock()
        self._hits:  int      = 0
        self._misses: int     = 0
        self._sets:  int      = 0

    def get(self, key: str) -> Any | None:
        with self._lock:
            v = self._cache.get(key)
            if v is None:
                self._misses += 1
            else:
                self._hits += 1
            return v

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._cache[key] = value
            self._sets += 1

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "hits":     self._hits,
                "misses":   self._misses,
                "sets":     self._sets,
                "size":     len(self._cache),
                "capacity": self._cache.maxsize,
            }

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._hits = self._misses = self._sets = 0
