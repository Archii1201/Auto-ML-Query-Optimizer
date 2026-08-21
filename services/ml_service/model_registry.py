"""
model_registry.py
=================
Phase 4B — a tiny, file-based model registry.

Problem
-------
Today the service hard-loads ``models/phase3b/{regime}/automl_best.joblib``.
That's one mutable file with no history: when Phase 5 retrains and
overwrites it, we lose the ability to (a) say *which* model served a
prediction, (b) roll back a bad model, or (c) run two model versions side
by side. Feedback records already store a ``model_version`` hash — but
nothing maps that hash back to an artifact.

What this provides
------------------
A JSON-indexed store of versioned artifacts:

    models/registry/
        registry.json                 # the index (per-regime versions + current)
        plan_time/<version>.joblib     # immutable, content-addressed copies
        post_mortem/<version>.joblib

- ``register(regime, path, promote=)``  — content-address an artifact by
  the SHA-256 of its bytes (same 16-hex id the Predictor reports), copy it
  into the store, and record metadata.
- ``promote(regime, version)``          — point "current" at a version.
- ``resolve_artifact(regime, version)`` — path to load; ``"current"``
  resolves the promoted version.

Backward compatible: if the registry is empty / a regime is unknown,
``resolve_artifact(regime, "current")`` falls back to the legacy
``models/phase3b/{regime}/automl_best.joblib`` so nothing breaks before a
single ``register`` has ever been run.

Why file-based (not MLflow / a DB)
----------------------------------
The project is self-contained and offline-friendly; a JSON index + copied
joblibs needs zero services and is trivial to inspect, diff, and ship in
the same Docker image. MLflow is the right answer at org scale; it's
overkill (and another container) for this phase.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_REGISTRY_DIR = Path(
    os.environ.get("MODEL_REGISTRY_DIR", str(PROJECT_ROOT / "models" / "registry"))
)
LEGACY_MODELS_DIR = PROJECT_ROOT / "models" / "phase3b"


def _sha16(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ModelRegistry:
    def __init__(self, registry_dir: Path = DEFAULT_REGISTRY_DIR) -> None:
        self.dir = Path(registry_dir)
        self.index_path = self.dir / "registry.json"
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    def _load(self) -> dict[str, Any]:
        if not self.index_path.exists():
            return {"regimes": {}}
        try:
            return json.loads(self.index_path.read_text(encoding="utf-8"))
        except Exception:
            return {"regimes": {}}

    def _save(self, idx: dict[str, Any]) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.index_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(idx, indent=2), encoding="utf-8")
        tmp.replace(self.index_path)

    # ------------------------------------------------------------------
    def register(
        self,
        regime: str,
        artifact_path: Path | str,
        *,
        promote: bool = False,
    ) -> str:
        """Content-address + copy an artifact into the store. Returns its version."""
        src = Path(artifact_path)
        if not src.exists():
            raise FileNotFoundError(f"artifact not found: {src}")
        raw = src.read_bytes()
        version = _sha16(raw)

        # Pull human-readable metadata out of the joblib if we can.
        meta: dict[str, Any] = {}
        try:
            import joblib
            art = joblib.load(src)
            meta = {
                "model_name": str(art.get("model_name", "")),
                "trained_at": str(art.get("trained_at", "")),
                "automl_winner": art.get("automl_winner"),
            }
        except Exception:  # noqa: BLE001 — metadata is best-effort
            pass

        with self._lock:
            idx = self._load()
            regimes = idx.setdefault("regimes", {})
            entry = regimes.setdefault(regime, {"versions": {}, "current": None})

            dest_dir = self.dir / regime
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / f"{version}.joblib"
            if not dest.exists():
                shutil.copy2(src, dest)

            entry["versions"][version] = {
                "version":       version,
                "regime":        regime,
                "path":          str(dest.relative_to(self.dir)),
                "size_bytes":    len(raw),
                "registered_at": _now(),
                "source":        str(src),
                "status":        "active",
                **meta,
            }
            if promote or entry.get("current") is None:
                entry["current"] = version
            self._save(idx)
        return version

    # ------------------------------------------------------------------
    def promote(self, regime: str, version: str) -> None:
        with self._lock:
            idx = self._load()
            entry = idx.get("regimes", {}).get(regime)
            if not entry or version not in entry["versions"]:
                raise KeyError(f"version {version!r} not registered for regime {regime!r}")
            entry["current"] = version
            self._save(idx)

    def current_version(self, regime: str) -> str | None:
        return self._load().get("regimes", {}).get(regime, {}).get("current")

    def list_versions(self, regime: str) -> list[dict[str, Any]]:
        entry = self._load().get("regimes", {}).get(regime, {})
        cur = entry.get("current")
        out = []
        for v in entry.get("versions", {}).values():
            out.append({**v, "is_current": v["version"] == cur})
        return sorted(out, key=lambda d: d.get("registered_at", ""), reverse=True)

    # ------------------------------------------------------------------
    def resolve_artifact(self, regime: str, version: str = "current") -> Path:
        """
        Return the on-disk artifact path. ``current`` resolves the promoted
        version; falls back to the legacy automl_best.joblib when the
        registry has no entry yet (backward compatibility).
        """
        entry = self._load().get("regimes", {}).get(regime, {})
        versions = entry.get("versions", {})
        if version == "current":
            version = entry.get("current")
        if version and version in versions:
            return (self.dir / versions[version]["path"]).resolve()

        legacy = LEGACY_MODELS_DIR / regime / "automl_best.joblib"
        if legacy.exists():
            return legacy
        raise FileNotFoundError(
            f"no artifact for regime={regime!r} version={version!r} "
            f"(and legacy {legacy} is missing)"
        )

    def snapshot(self) -> dict[str, Any]:
        idx = self._load()
        return {
            "registry_dir": str(self.dir),
            "regimes": {
                r: {"current": e.get("current"),
                    "version_count": len(e.get("versions", {}))}
                for r, e in idx.get("regimes", {}).items()
            },
        }


# Process-wide default registry.
REGISTRY = ModelRegistry()


# ---------------------------------------------------------------------------
# CLI:  python -m services.ml_service.model_registry <cmd> ...
# ---------------------------------------------------------------------------
def _main(argv: list[str]) -> int:
    import argparse
    sys.path.insert(0, str(PROJECT_ROOT))

    p = argparse.ArgumentParser(description="Model registry admin")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("register", help="register (and optionally promote) an artifact")
    pr.add_argument("--regime", required=True)
    pr.add_argument("--path", required=True, help="path to a .joblib artifact")
    pr.add_argument("--promote", action="store_true")

    pp = sub.add_parser("promote", help="point current at a version")
    pp.add_argument("--regime", required=True)
    pp.add_argument("--version", required=True)

    pl = sub.add_parser("list", help="list versions for a regime")
    pl.add_argument("--regime", required=True)

    sub.add_parser("snapshot", help="summarize the registry")

    args = p.parse_args(argv)
    reg = ModelRegistry()

    if args.cmd == "register":
        v = reg.register(args.regime, args.path, promote=args.promote)
        print(f"registered {args.regime} -> {v}" + (" (promoted)" if args.promote else ""))
    elif args.cmd == "promote":
        reg.promote(args.regime, args.version)
        print(f"promoted {args.regime} -> {args.version}")
    elif args.cmd == "list":
        for v in reg.list_versions(args.regime):
            flag = "*" if v["is_current"] else " "
            print(f" {flag} {v['version']}  {v.get('model_name','?'):24s} "
                  f"{v.get('registered_at','')}")
    elif args.cmd == "snapshot":
        print(json.dumps(reg.snapshot(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
