"""
Unit tests for the Phase 4B file-based model registry.
Uses a tiny fake joblib file (the registry only reads bytes for the
content hash; metadata extraction is best-effort and tolerated to fail).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from services.ml_service.model_registry import ModelRegistry


def _fake_artifact(tmp_path: Path, content: bytes = b"model-bytes-v1") -> Path:
    p = tmp_path / "automl_best.joblib"
    p.write_bytes(content)
    return p


def test_register_and_resolve_current(tmp_path):
    reg = ModelRegistry(tmp_path / "registry")
    art = _fake_artifact(tmp_path, b"aaaa")
    v = reg.register("plan_time", art)
    assert reg.current_version("plan_time") == v
    resolved = reg.resolve_artifact("plan_time", "current")
    assert resolved.exists()
    assert resolved.read_bytes() == b"aaaa"


def test_content_addressing_is_stable(tmp_path):
    reg = ModelRegistry(tmp_path / "registry")
    a1 = tmp_path / "m.joblib"
    a1.write_bytes(b"same")
    v1 = reg.register("plan_time", a1)
    v2 = reg.register("plan_time", a1)
    assert v1 == v2                      # identical bytes -> identical version


def test_promote_switches_current(tmp_path):
    reg = ModelRegistry(tmp_path / "registry")
    p1 = tmp_path / "m1.joblib"; p1.write_bytes(b"v1")
    p2 = tmp_path / "m2.joblib"; p2.write_bytes(b"v2")
    v1 = reg.register("plan_time", p1, promote=True)
    v2 = reg.register("plan_time", p2)   # registered but not promoted
    assert reg.current_version("plan_time") == v1
    reg.promote("plan_time", v2)
    assert reg.current_version("plan_time") == v2


def test_list_versions_marks_current(tmp_path):
    reg = ModelRegistry(tmp_path / "registry")
    p1 = tmp_path / "m1.joblib"; p1.write_bytes(b"v1")
    p2 = tmp_path / "m2.joblib"; p2.write_bytes(b"v2")
    reg.register("plan_time", p1, promote=True)
    reg.register("plan_time", p2)
    versions = reg.list_versions("plan_time")
    assert len(versions) == 2
    assert sum(1 for v in versions if v["is_current"]) == 1


def test_promote_unknown_version_raises(tmp_path):
    reg = ModelRegistry(tmp_path / "registry")
    p1 = tmp_path / "m1.joblib"; p1.write_bytes(b"v1")
    reg.register("plan_time", p1)
    with pytest.raises(KeyError):
        reg.promote("plan_time", "deadbeef")


def test_resolve_unknown_without_legacy_raises(tmp_path):
    reg = ModelRegistry(tmp_path / "registry")
    with pytest.raises(FileNotFoundError):
        reg.resolve_artifact("no_such_regime", "current")
