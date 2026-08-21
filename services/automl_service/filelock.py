"""
filelock.py
===========
Phase 5E — a tiny, dependency-free single-flight lock.

Why: only ONE retrain may run at a time (they share features.csv, the
trainer's output path, and the registry). If two workers — or a worker plus
a manual `scripts/retrain.py` — overlap, they corrupt each other. This is a
cross-platform advisory lock built on atomic `O_CREAT | O_EXCL` file
creation (works on Windows and POSIX), with stale-lock reclaim so a crashed
holder can't wedge the loop forever.

We avoid the `filelock` PyPI package on purpose: one stdlib file keeps the
image slim and the failure mode obvious.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path


class LockBusy(RuntimeError):
    """Raised when the lock is held by another live holder."""


@dataclass
class FileLock:
    path: Path
    stale_after_s: float = 3600.0   # reclaim a lock older than this

    def _write_holder(self, fd: int) -> None:
        os.write(fd, f"{os.getpid()} {time.time()}".encode())

    def _is_stale(self) -> bool:
        try:
            age = time.time() - self.path.stat().st_mtime
        except FileNotFoundError:
            return False
        return age > self.stale_after_s

    def acquire(self) -> "FileLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            self._write_holder(fd)
            os.close(fd)
            return self
        except FileExistsError:
            if self._is_stale():
                # Reclaim: remove and retry once.
                try:
                    self.path.unlink()
                except FileNotFoundError:
                    pass
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                self._write_holder(fd)
                os.close(fd)
                return self
            raise LockBusy(f"lock held: {self.path}") from None

    def release(self) -> None:
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass

    def __enter__(self) -> "FileLock":
        return self.acquire()

    def __exit__(self, *exc) -> None:
        self.release()
