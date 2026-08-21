"""
trainer.py
==========
Phase 5B — orchestrate a single retrain and register the result as a
*candidate* (never auto-promoted here; promotion is 5C's job behind a gate).

    (optional) merge feedback + validation gate   [5A]
              │
              ▼
    phase3b/train_models.py  --regimes plan_time [--skip-tuning | --n-trials]
              │  writes models/phase3b/plan_time/automl_best.joblib
              ▼
    REGISTRY.register(regime, artifact, promote=False)   → candidate version
              │
              ▼
    RetrainResult(version, regime, artifact_path, ...)

Testability: the merge, train, and register steps are injectable callables.
Default implementations shell out / hit the real registry; unit tests pass
fakes so they never spawn a trainer or touch Postgres.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from services.automl_service.config import RetrainProfile
from services.automl_service.merge import merge_feedback, run_validation_gate
from services.ml_service.model_registry import REGISTRY, ModelRegistry

TRAIN_SCRIPT = PROJECT_ROOT / "phase3b" / "train_models.py"
LEGACY_ARTIFACT = PROJECT_ROOT / "models" / "phase3b" / "{regime}" / "automl_best.joblib"

# Injectable step signatures.
TrainFn = Callable[[RetrainProfile], int]                 # -> returncode
RegisterFn = Callable[[str, Path], str]                    # (regime, path) -> version


@dataclass
class RetrainResult:
    ok:            bool
    regime:        str
    profile:       str
    version:       str | None = None
    artifact_path: str | None = None
    merged_rows:   int = 0
    steps:         list[str] = field(default_factory=list)
    error:         str | None = None

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def _default_train_fn(profile: RetrainProfile) -> int:
    cmd = [sys.executable, str(TRAIN_SCRIPT), *profile.train_argv()]
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    return proc.returncode


def _default_register_fn(registry: ModelRegistry) -> RegisterFn:
    def _register(regime: str, path: Path) -> str:
        return registry.register(regime, path, promote=False)
    return _register


def artifact_for(regime: str) -> Path:
    return Path(str(LEGACY_ARTIFACT).format(regime=regime))


def run_retrain(
    profile: RetrainProfile,
    *,
    do_merge: bool = True,
    gate: bool = True,
    registry: ModelRegistry = REGISTRY,
    train_fn: TrainFn | None = None,
    register_fn: RegisterFn | None = None,
    regime: str | None = None,
) -> RetrainResult:
    regime = regime or profile.regimes[0]
    train_fn = train_fn or _default_train_fn
    register_fn = register_fn or _default_register_fn(registry)

    res = RetrainResult(ok=False, regime=regime, profile=profile.name)

    # 1) merge feedback (5A) --------------------------------------------
    if do_merge:
        report = merge_feedback(apply=True)
        res.merged_rows = report.new_rows
        res.steps.append(f"merged {report.new_rows} new rows")
        if gate:
            ok, _out = run_validation_gate(features=True)
            res.steps.append(f"validation gate {'passed' if ok else 'FAILED'}")
            if not ok:
                res.error = "validation gate failed"
                return res

    # 2) train (5B) ------------------------------------------------------
    rc = train_fn(profile)
    res.steps.append(f"train rc={rc}")
    if rc != 0:
        res.error = f"trainer exited {rc}"
        return res

    artifact = artifact_for(regime)
    if not artifact.exists():
        res.error = f"expected artifact missing: {artifact}"
        return res

    # 3) register candidate (NOT promoted) ------------------------------
    version = register_fn(regime, artifact)
    res.version = version
    res.artifact_path = str(artifact)
    res.steps.append(f"registered candidate {version}")
    res.ok = True
    return res
