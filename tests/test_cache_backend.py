"""
Unit tests for Phase 4B pluggable cache backends.

Redis is exercised through an in-memory fake client (no server needed),
which is enough to verify the backend's serialization + key namespacing.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from services.ml_service.cache import HashedLRUCache
from services.ml_service.cache_backend import (
    LocalLRUBackend,
    RedisBackend,
    make_cache_backend,
)


# ----- LocalLRUBackend ----------------------------------------------------
def test_local_get_set_clear():
    b = LocalLRUBackend(capacity=2)
    assert b.get("a") is None
    b.set("a", 1)
    assert b.get("a") == 1
    b.clear()
    assert b.get("a") is None


def test_local_evicts_lru():
    b = LocalLRUBackend(capacity=2)
    b.set("a", 1)
    b.set("b", 2)
    b.get("a")            # touch a so b is least-recently-used
    b.set("c", 3)         # evicts b
    assert b.get("b") is None
    assert b.get("a") == 1
    assert b.get("c") == 3


# ----- HashedLRUCache hit/miss bookkeeping --------------------------------
def test_hashed_cache_counts_hits_and_misses():
    c = HashedLRUCache(capacity=8, backend=LocalLRUBackend(8))
    assert c.get("k") is None            # miss
    c.set("k", "v")
    assert c.get("k") == "v"             # hit
    s = c.stats()
    assert s["hits"] == 1
    assert s["misses"] == 1
    assert s["sets"] == 1
    assert s["backend"] == "local"


# ----- Factory ------------------------------------------------------------
def test_factory_defaults_to_local(monkeypatch):
    monkeypatch.delenv("CACHE_BACKEND", raising=False)
    b = make_cache_backend(namespace="t")
    assert isinstance(b, LocalLRUBackend)


def test_factory_redis_unavailable_falls_back(monkeypatch):
    # Request redis but point at a dead address -> must fall back to local.
    monkeypatch.setenv("CACHE_BACKEND", "redis")
    monkeypatch.setenv("REDIS_URL", "redis://127.0.0.1:1/0")
    b = make_cache_backend(namespace="t")
    assert isinstance(b, LocalLRUBackend)


# ----- RedisBackend via fake client --------------------------------------
class FakeRedis:
    def __init__(self) -> None:
        self.store: dict[str, bytes] = {}

    def ping(self):
        return True

    def get(self, k):
        return self.store.get(k)

    def set(self, k, v, ex=None):
        self.store[k] = v

    def scan_iter(self, match=None, count=None):
        import fnmatch
        for k in list(self.store):
            if match is None or fnmatch.fnmatch(k, match):
                yield k

    def delete(self, *keys):
        for k in keys:
            self.store.pop(k, None)


def test_redis_backend_roundtrip_and_namespacing():
    fake = FakeRedis()
    b = RedisBackend("redis://x", namespace="predict:plan_time", client=fake)
    b.set("abc", {"complex": [1, 2, 3]})
    assert b.get("abc") == {"complex": [1, 2, 3]}
    # key is namespaced in the underlying store
    assert any(k.startswith("predict:plan_time:") for k in fake.store)
    b.clear()
    assert b.get("abc") is None


def test_redis_backend_missing_key_is_none():
    b = RedisBackend("redis://x", namespace="ns", client=FakeRedis())
    assert b.get("nope") is None
