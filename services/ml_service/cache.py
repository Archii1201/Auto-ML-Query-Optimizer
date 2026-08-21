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

from services.ml_service.cache_backend import CacheBackend, make_cache_backend


# ---------------------------------------------------------------------------
# Hashing helpers
# ---------------------------------------------------------------------------
def _normalize_sql(sql: str) -> str:
    """
    Cheap canonicalisation for the cache key:
        - collapse whitespace
        - strip trailing semicolons + spaces

    NOTE: case is *preserved* — PostgreSQL is case-sensitive for quoted
    identifiers, so two queries differing only by case can be
    semantically different and must hash to different keys.

    NOTE: this is NOT SQL-aware. Two semantically equivalent queries
    written differently (e.g. column reorder in SELECT, alias
    differences) will hash to different keys. That's acceptable —
    we'd rather have a few cache misses than ever return a wrong plan.
    """
    return " ".join(sql.split()).rstrip(";").strip()


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
    """
    Thread-safe cache keyed by SHA-256 hex digests.

    Phase 4B: storage is delegated to a pluggable ``CacheBackend``
    (local LRU by default, Redis when ``CACHE_BACKEND=redis``). Hit/miss
    bookkeeping lives here so the reported stats are identical regardless
    of which backend is active. ``namespace`` keeps the predict cache and
    the plan-pick cache in separate keyspaces when they share Redis.
    """

    def __init__(
        self,
        capacity: int = 1024,
        *,
        namespace: str = "cache",
        backend: CacheBackend | None = None,
    ) -> None:
        self._backend: CacheBackend = backend or make_cache_backend(
            namespace=namespace, capacity=capacity
        )
        self._lock:   Lock = Lock()
        self._hits:   int  = 0
        self._misses: int  = 0
        self._sets:   int  = 0

    def get(self, key: str) -> Any | None:
        v = self._backend.get(key)
        with self._lock:
            if v is None:
                self._misses += 1
            else:
                self._hits += 1
        return v

    def set(self, key: str, value: Any) -> None:
        self._backend.set(key, value)
        with self._lock:
            self._sets += 1

    def stats(self) -> dict[str, int]:
        with self._lock:
            base: dict[str, Any] = {
                "hits":   self._hits,
                "misses": self._misses,
                "sets":   self._sets,
            }
        base.update(self._backend.raw_stats())
        return base

    def clear(self) -> None:
        self._backend.clear()
        with self._lock:
            self._hits = self._misses = self._sets = 0
