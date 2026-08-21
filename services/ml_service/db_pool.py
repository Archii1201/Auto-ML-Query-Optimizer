"""
db_pool.py
==========
Phase 4A — bounded PostgreSQL connection pool.

Why a pool (vs. psycopg2.connect() per request)
------------------------------------------------
Opening a fresh connection per request means a burst of N concurrent
requests opens N connections; a big enough burst exhausts PG's
`max_connections` and the *whole database* starts refusing everyone
(not just us). A bounded pool caps the number of live connections and
makes excess load *wait* (with a timeout) instead of melting PG.

Design (DSA: bounded blocking queue)
------------------------------------
- A `queue.Queue(maxsize=maxconn)` holds idle connections.
- Up to `maxconn` connections are created lazily on demand.
- `acquire(timeout)` returns an idle connection, or creates one if we're
  below `maxconn`, or blocks up to `timeout` seconds for one to free up.
  If nothing is available in time it raises `PoolTimeout` -> the API
  maps that to HTTP 503 (graceful backpressure, never a PG meltdown).
- Connections are validated (`SELECT 1`) on acquire; dead ones (PG
  restarted, network blip) are discarded and replaced transparently.

We deliberately use psycopg2's own primitives (the project already
depends on psycopg2-binary) rather than migrating to psycopg3/psycopg-
pool, which would be a much larger change for the same guarantee.
"""

from __future__ import annotations

import queue
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable

import psycopg2


class PoolTimeout(Exception):
    """Raised when no connection becomes available within the budget."""


class PoolClosed(Exception):
    """Raised when acquire() is called on a shut-down pool."""


class PgConnectionPool:
    def __init__(
        self,
        dsn_kwargs: dict[str, Any],
        *,
        minconn: int = 2,
        maxconn: int = 10,
        acquire_timeout_s: float = 2.0,
        connect_fn: Callable[..., Any] | None = None,
    ) -> None:
        if minconn < 0 or maxconn < 1 or minconn > maxconn:
            raise ValueError("require 0 <= minconn <= maxconn and maxconn >= 1")
        self._dsn = dsn_kwargs
        self._minconn = minconn
        self._maxconn = maxconn
        self._acquire_timeout_s = acquire_timeout_s
        self._connect = connect_fn or (lambda: psycopg2.connect(**self._dsn))

        self._idle: queue.Queue = queue.Queue(maxsize=maxconn)
        self._lock = threading.Lock()
        self._created = 0
        self._closed = False

        # Counters for observability / /metrics.
        self.acquired_total = 0
        self.timeouts_total = 0
        self.created_total = 0
        self.discarded_total = 0

        for _ in range(minconn):
            try:
                self._idle.put_nowait(self._open_one())
            except Exception:
                break  # PG may be down at boot; we'll lazily retry later

    # ------------------------------------------------------------------
    def _open_one(self):
        conn = self._connect()
        with self._lock:
            self._created += 1
            self.created_total += 1
        return conn

    def _discard(self, conn) -> None:
        with self._lock:
            self._created = max(0, self._created - 1)
            self.discarded_total += 1
        try:
            conn.close()
        except Exception:
            pass

    @staticmethod
    def _is_alive(conn) -> bool:
        if getattr(conn, "closed", 1):
            return False
        try:
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
                cur.fetchone()
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    def acquire(self, timeout: float | None = None):
        if self._closed:
            raise PoolClosed("connection pool is closed")
        deadline = time.monotonic() + (
            self._acquire_timeout_s if timeout is None else timeout
        )

        while True:
            # 1) reuse an idle connection if one is ready
            try:
                conn = self._idle.get_nowait()
                if self._is_alive(conn):
                    with self._lock:
                        self.acquired_total += 1
                    return conn
                self._discard(conn)
                continue
            except queue.Empty:
                pass

            # 2) create a new one if we're under the cap
            with self._lock:
                can_create = self._created < self._maxconn
            if can_create:
                try:
                    conn = self._open_one()
                except psycopg2.OperationalError:
                    # PG unreachable — fail fast rather than spin
                    with self._lock:
                        self.timeouts_total += 1
                    raise PoolTimeout("PostgreSQL unreachable while opening connection")
                with self._lock:
                    self.acquired_total += 1
                return conn

            # 3) at capacity — wait for someone to release
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                with self._lock:
                    self.timeouts_total += 1
                raise PoolTimeout(
                    f"no pooled connection available within "
                    f"{self._acquire_timeout_s if timeout is None else timeout:.2f}s "
                    f"(pool exhausted: {self._maxconn} in use)"
                )
            try:
                conn = self._idle.get(timeout=min(remaining, 0.25))
            except queue.Empty:
                continue
            if self._is_alive(conn):
                with self._lock:
                    self.acquired_total += 1
                return conn
            self._discard(conn)

    def release(self, conn) -> None:
        if conn is None:
            return
        if self._closed or getattr(conn, "closed", 1):
            self._discard(conn)
            return
        try:
            if not conn.autocommit:
                conn.rollback()
        except Exception:
            self._discard(conn)
            return
        try:
            self._idle.put_nowait(conn)
        except queue.Full:
            self._discard(conn)

    @contextmanager
    def connection(self, timeout: float | None = None):
        conn = self.acquire(timeout=timeout)
        try:
            yield conn
        finally:
            self.release(conn)

    # ------------------------------------------------------------------
    def ping(self, timeout: float | None = None) -> bool:
        """Readiness probe: can we get a working connection right now?"""
        try:
            with self.connection(timeout=timeout or 1.0) as conn:
                return self._is_alive(conn)
        except Exception:
            return False

    def closeall(self) -> None:
        self._closed = True
        while True:
            try:
                conn = self._idle.get_nowait()
            except queue.Empty:
                break
            try:
                conn.close()
            except Exception:
                pass
        with self._lock:
            self._created = 0

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "maxconn":         self._maxconn,
                "minconn":         self._minconn,
                "created":         self._created,
                "idle":            self._idle.qsize(),
                "acquired_total":  self.acquired_total,
                "created_total":   self.created_total,
                "discarded_total": self.discarded_total,
                "timeouts_total":  self.timeouts_total,
            }
