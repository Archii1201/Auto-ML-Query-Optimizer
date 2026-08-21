"""
Unit tests for the Phase 4A bounded connection pool.

Uses a fake connection factory so the tests need no real PostgreSQL:
we only verify the pool's *behaviour* (capacity, reuse, timeout on
exhaustion, dead-connection replacement), not psycopg2 itself.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from services.ml_service.db_pool import PgConnectionPool, PoolTimeout


class FakeCursor:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, *a, **k):
        return None

    def fetchone(self):
        return (1,)


class FakeConn:
    def __init__(self) -> None:
        self.closed = 0
        self.autocommit = False
        self.alive = True

    def cursor(self):
        if not self.alive:
            raise RuntimeError("connection is dead")
        return FakeCursor()

    def rollback(self):
        pass

    def close(self):
        self.closed = 1
        self.alive = False


def make_pool(minconn=0, maxconn=2, acquire_timeout_s=0.2):
    created: list[FakeConn] = []

    def connect():
        c = FakeConn()
        created.append(c)
        return c

    pool = PgConnectionPool(
        {}, minconn=minconn, maxconn=maxconn,
        acquire_timeout_s=acquire_timeout_s, connect_fn=connect,
    )
    return pool, created


def test_acquire_and_release_reuses_connection():
    pool, created = make_pool(minconn=0, maxconn=2)
    c1 = pool.acquire()
    pool.release(c1)
    c2 = pool.acquire()
    assert c1 is c2                      # reused, not a fresh connect
    assert len(created) == 1


def test_respects_maxconn_and_times_out():
    pool, _ = make_pool(minconn=0, maxconn=2, acquire_timeout_s=0.15)
    a = pool.acquire()
    b = pool.acquire()
    assert a is not b
    with pytest.raises(PoolTimeout):
        pool.acquire()                   # third while 2 are checked out -> timeout
    assert pool.stats()["timeouts_total"] == 1


def test_release_lets_waiter_proceed():
    pool, _ = make_pool(minconn=0, maxconn=1, acquire_timeout_s=1.0)
    held = pool.acquire()
    got: list = []

    def worker():
        got.append(pool.acquire())

    t = threading.Thread(target=worker)
    t.start()
    pool.release(held)                   # frees the single slot
    t.join(timeout=2.0)
    assert len(got) == 1


def test_dead_connection_is_discarded_on_acquire():
    pool, created = make_pool(minconn=0, maxconn=2)
    c1 = pool.acquire()
    pool.release(c1)
    c1.alive = False                     # simulate PG dropping the connection
    c2 = pool.acquire()                  # pool must detect + replace it
    assert c2 is not c1
    assert c1.closed == 1
    assert pool.stats()["discarded_total"] >= 1


def test_context_manager_releases():
    pool, created = make_pool(minconn=0, maxconn=1)
    with pool.connection() as conn:
        assert conn is not None
    # released back -> can acquire again immediately
    with pool.connection() as conn2:
        assert conn2 is not None
    assert len(created) == 1


def test_min_connections_prefilled():
    pool, created = make_pool(minconn=2, maxconn=4)
    assert pool.stats()["idle"] == 2
    assert len(created) == 2
