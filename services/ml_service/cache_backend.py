"""
cache_backend.py
================
Phase 4B — pluggable cache backends (Strategy pattern).

Why
---
The in-process LRU (Phase 3C) is perfect for a single worker but it is
*not shared*: run two uvicorn workers (Phase 4E) and each keeps its own
cache, halving the hit rate and serving stale entries after a model swap.
Redis gives a **shared, restart-surviving** cache across workers.

We don't want to *force* Redis on a developer, though — tests and local
runs should stay zero-infra and fast. So we hide the storage behind a
small interface and pick the implementation at startup:

    CACHE_BACKEND=local   ->  LocalLRUBackend   (default; cachetools)
    CACHE_BACKEND=redis   ->  RedisBackend      (redis-py)

Both expose the same 4 raw operations (get/set/clear/raw stats). The
hit/miss bookkeeping lives one level up in HashedLRUCache so the metrics
are identical regardless of backend.

Why not alternatives
--------------------
- **memcached** — no native data-structure ops we need later, and Redis
  is already the project's stated cache (system-flow doc).
- **DIY shared memory / mmap** — fragile across containers; Redis solves
  exactly this with one dependency.

Resilience: if `CACHE_BACKEND=redis` but Redis can't be imported or
reached at startup, we log and **fall back to local** rather than refuse
to boot. A cache is an optimization, never a hard dependency.
"""

from __future__ import annotations

import logging
import os
import pickle
import threading
from abc import ABC, abstractmethod
from typing import Any

from cachetools import LRUCache

logger = logging.getLogger("ml_service")


class CacheBackend(ABC):
    """Raw key/value storage. Keys are opaque strings (SHA-256 hex)."""

    name: str = "abstract"

    @abstractmethod
    def get(self, key: str) -> Any | None: ...

    @abstractmethod
    def set(self, key: str, value: Any) -> None: ...

    @abstractmethod
    def clear(self) -> None: ...

    @abstractmethod
    def raw_stats(self) -> dict[str, Any]: ...


# ---------------------------------------------------------------------------
class LocalLRUBackend(CacheBackend):
    """Bounded in-process LRU. O(1) get/set with eviction. Thread-safe."""

    name = "local"

    def __init__(self, capacity: int = 1024) -> None:
        self._cache: LRUCache = LRUCache(maxsize=capacity)
        self._lock = threading.Lock()

    def get(self, key: str) -> Any | None:
        with self._lock:
            return self._cache.get(key)

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._cache[key] = value

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    def raw_stats(self) -> dict[str, Any]:
        with self._lock:
            return {"backend": self.name,
                    "size": len(self._cache),
                    "capacity": self._cache.maxsize}


# ---------------------------------------------------------------------------
class RedisBackend(CacheBackend):
    """
    Shared cache in Redis. Values are pickled (our cached values include
    dataclasses with nested plan JSON, which JSON can't round-trip without
    a custom codec). Keys are namespaced: ``{namespace}:{sha256}``.

    A per-entry TTL (default 1h) bounds staleness so a model swap can't
    serve outdated predictions forever even if we forget to clear().
    """

    name = "redis"

    def __init__(
        self,
        url: str,
        *,
        namespace: str = "cache",
        ttl_s: int = 3600,
        client: Any | None = None,
    ) -> None:
        self.namespace = namespace
        self.ttl_s = ttl_s
        if client is not None:
            self._r = client
        else:
            import redis  # imported lazily so non-redis deploys don't need it
            self._r = redis.Redis.from_url(url, socket_connect_timeout=2,
                                           socket_timeout=2)
        # Fail fast at construction so the factory can fall back to local.
        self._r.ping()

    def _k(self, key: str) -> str:
        return f"{self.namespace}:{key}"

    def get(self, key: str) -> Any | None:
        try:
            raw = self._r.get(self._k(key))
        except Exception as exc:  # noqa: BLE001 — never let cache errors 500 a request
            logger.warning("redis get failed", extra={"fields": {"error": str(exc)}})
            return None
        if raw is None:
            return None
        try:
            return pickle.loads(raw)
        except Exception:  # noqa: BLE001 — corrupt entry -> treat as miss
            return None

    def set(self, key: str, value: Any) -> None:
        try:
            self._r.set(self._k(key), pickle.dumps(value), ex=self.ttl_s)
        except Exception as exc:  # noqa: BLE001
            logger.warning("redis set failed", extra={"fields": {"error": str(exc)}})

    def clear(self) -> None:
        try:
            keys = list(self._r.scan_iter(match=f"{self.namespace}:*", count=500))
            if keys:
                self._r.delete(*keys)
        except Exception as exc:  # noqa: BLE001
            logger.warning("redis clear failed", extra={"fields": {"error": str(exc)}})

    def raw_stats(self) -> dict[str, Any]:
        info: dict[str, Any] = {"backend": self.name, "namespace": self.namespace,
                                "ttl_s": self.ttl_s}
        try:
            info["keys"] = sum(1 for _ in self._r.scan_iter(
                match=f"{self.namespace}:*", count=500))
        except Exception:  # noqa: BLE001
            info["keys"] = -1
        return info


# ---------------------------------------------------------------------------
def make_cache_backend(*, namespace: str, capacity: int = 1024) -> CacheBackend:
    """
    Build the backend selected by env. Falls back to local if Redis is
    requested but unavailable (a cache must never block startup).
    """
    choice = os.environ.get("CACHE_BACKEND", "local").strip().lower()
    if choice == "redis":
        url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        ttl = int(os.environ.get("CACHE_TTL_S", "3600"))
        try:
            backend = RedisBackend(url, namespace=namespace, ttl_s=ttl)
            logger.info("cache backend = redis",
                        extra={"fields": {"namespace": namespace, "url": url}})
            return backend
        except Exception as exc:  # noqa: BLE001
            logger.warning("redis cache unavailable; falling back to local",
                           extra={"fields": {"error": str(exc)}})
    return LocalLRUBackend(capacity=capacity)
