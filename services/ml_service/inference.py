"""
inference.py
============
The "ML Service" core: load the AutoML winner, transform a plan JSON
into a feature vector that *exactly* matches the training column
layout, and return a predicted runtime in milliseconds.

Why this file exists separately from server.py:

    1. The FastAPI app should be a thin HTTP wrapper. All business
       logic — feature alignment, log-target inversion, NaN guards,
       prediction clipping — lives here so we can unit-test it
       without spinning up a server.

    2. The CLI demo (`scripts/demo_phase3c.py`) calls the same
       `Predictor` class directly, so a single bug fix here
       improves both the HTTP service and the offline demo.

    3. Phase 5 (the retraining loop) will call into this module
       for online predictions before deciding whether to swap the
       deployed model.
"""

from __future__ import annotations

import logging
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

import joblib
import numpy as np
import pandas as pd

# LightGBM emits a spammy "min_data_in_leaf is set, min_child_samples will
# be ignored" warning per booster predict() call. It's emitted from the
# C++ side, so Python's `warnings` module can't catch it; we have to use
# LightGBM's own logger registry. (The deployed booster was trained with
# the legacy alias; phase3b/tuning.py now uses the canonical name so future
# retrains drop the warning entirely.)
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm.*")
logging.getLogger("lightgbm").setLevel(logging.ERROR)
try:
    import lightgbm as _lgb

    class _NoisyLgbFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            msg = record.getMessage()
            return "min_child_samples will be ignored" not in msg
    _lgb.register_logger(logging.getLogger("lightgbm"))
    logging.getLogger("lightgbm").addFilter(_NoisyLgbFilter())
except Exception:
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from feature_engineering.extract_features import extract_features_from_record  # noqa: E402
from phase3a.feature_selection import (  # noqa: E402
    CATEGORICAL_COLUMNS,
    ID_COLUMNS,
    LEAKY_COLUMNS,
    TARGET_COLUMN,
)
from services.ml_service.cache import HashedLRUCache, hash_plan  # noqa: E402

DEFAULT_MODELS_DIR: Path = PROJECT_ROOT / "models" / "phase3b"
PRED_FLOOR_MS: float = 0.1
PRED_CEILING_MS: float = 1e7   # 10000 seconds — sanity cap


class InvalidPlanError(ValueError):
    """Raised when the incoming plan_json doesn't match the expected EXPLAIN shape."""


# ---------------------------------------------------------------------------
@dataclass
class PredictionResult:
    predicted_ms: float
    model_name:   str
    regime:       str
    cache_hit:    bool
    elapsed_ms:   float


# ---------------------------------------------------------------------------
class Predictor:
    """
    Loads `automl_best.joblib` for one regime and exposes `.predict_one(plan)`.

    Thread-safe: a single mutex guards the cache; the underlying sklearn
    estimator is read-only after load.
    """

    def __init__(
        self,
        regime: str = "plan_time",
        models_dir: Path = DEFAULT_MODELS_DIR,
        cache_capacity: int = 4096,
    ) -> None:
        self.regime: str = regime
        self.models_dir: Path = models_dir
        self.cache: HashedLRUCache = HashedLRUCache(capacity=cache_capacity)
        self._lock: Lock = Lock()
        self._missing_warned: bool = False

        artifact_path = models_dir / regime / "automl_best.joblib"
        if not artifact_path.exists():
            raise FileNotFoundError(
                f"AutoML winner missing: {artifact_path}. Run phase3b/train_models.py first."
            )

        artifact = joblib.load(artifact_path)
        self.model = artifact["model"]
        self.feature_names: list[str] = artifact["feature_names"]
        self.log_target: bool = bool(artifact.get("log_target", True))
        self.model_name: str = str(artifact.get("model_name", "automl_best"))
        self.trained_at: str = str(artifact.get("trained_at", ""))
        self.metadata: dict[str, Any] = {
            k: artifact.get(k)
            for k in ("automl_winner", "best_params", "tuner", "n_trials")
            if k in artifact
        }

    # ------------------------------------------------------------------
    def predict_one(self, plan_json: list[dict[str, Any]]) -> PredictionResult:
        """
        Run the full inference path on a single plan.

        plan_json is the full EXPLAIN (FORMAT JSON) payload (a list with
        a single wrapping dict). Cache key = SHA-256 of the plan JSON.
        """
        t0 = time.perf_counter()
        plan_hash = hash_plan(plan_json)

        cached = self.cache.get(plan_hash)
        if cached is not None:
            return PredictionResult(
                predicted_ms=float(cached),
                model_name=self.model_name,
                regime=self.regime,
                cache_hit=True,
                elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            )

        ms = self._infer(plan_json)
        self.cache.set(plan_hash, ms)
        return PredictionResult(
            predicted_ms=ms,
            model_name=self.model_name,
            regime=self.regime,
            cache_hit=False,
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
        )

    # ------------------------------------------------------------------
    def _infer(self, plan_json: list[dict[str, Any]]) -> float:
        """
        Plan JSON -> feature row -> aligned X -> y_pred.

        We reuse `extract_features_from_record` from the training
        pipeline so feature math is *identical* to what the model
        was trained on. The only adapter we need is to wrap the
        plan in the same record envelope the offline collectors use.
        """
        if not isinstance(plan_json, list) or not plan_json:
            raise InvalidPlanError("plan_json must be a non-empty list")
        if not isinstance(plan_json[0], dict) or "Plan" not in plan_json[0]:
            raise InvalidPlanError(
                "plan_json[0] must be the EXPLAIN envelope: "
                "{'Plan': {...}, 'Planning Time': ..., 'Execution Time': ...}"
            )

        record: dict[str, Any] = {
            "query_id":      "online",
            "variant":       "online",
            "tag":           "online",
            "sql_hash":      "online",
            "collected_at":  "online",
            "wall_time_ms":  0.0,
            "plan":          plan_json,
        }
        try:
            feat_row = extract_features_from_record(record, Path("online.json"))
        except Exception as exc:
            raise InvalidPlanError(f"feature extraction failed: {exc}") from exc
        X = self._align_features(feat_row)

        with self._lock:
            y_pred = float(self.model.predict(X)[0])
        if self.log_target:
            y_pred = float(np.expm1(y_pred))
        if not np.isfinite(y_pred):
            y_pred = PRED_CEILING_MS
        return float(np.clip(y_pred, PRED_FLOOR_MS, PRED_CEILING_MS))

    # ------------------------------------------------------------------
    def _align_features(self, feat_row: dict[str, Any]) -> pd.DataFrame:
        """
        Build a 1-row DataFrame whose columns *exactly* match the
        training feature_names. Missing columns are filled with 0
        (the same imputation policy used in training) but are also
        recorded so /info can report "drift" — the gap between what
        the live extractor produces and what the trained model expects.

        For categoricals (currently just `root_node_type`) we
        hot-encode using the names the model already knows about.
        """
        row = dict(feat_row)
        for c in (*ID_COLUMNS, *LEAKY_COLUMNS, TARGET_COLUMN):
            row.pop(c, None)

        cat_value = row.pop("root_node_type", None)
        for fname in self.feature_names:
            if fname.startswith("root_node_type__"):
                want = fname.split("__", 1)[1]
                row[fname] = 1.0 if cat_value == want else 0.0

        # Drift detection: which trained features did the live
        # extractor *not* produce? Log once per name.
        missing = [c for c in self.feature_names if c not in row]
        if missing and not self._missing_warned:
            print(
                f"[inference] WARNING: {len(missing)} trained features missing "
                f"from live extraction; filling with 0. Examples: {missing[:5]}",
                file=sys.stderr,
            )
            self._missing_warned = True

        df = pd.DataFrame([row])
        out = pd.DataFrame(index=df.index, columns=self.feature_names, dtype=float)
        common = [c for c in self.feature_names if c in df.columns]
        out[common] = df[common]
        out = out.fillna(0.0).astype(float)
        return out

    # ------------------------------------------------------------------
    def info(self) -> dict[str, Any]:
        return {
            "regime":        self.regime,
            "model_name":    self.model_name,
            "feature_count": len(self.feature_names),
            "trained_at":    self.trained_at,
            "log_target":    self.log_target,
            "cache":         self.cache.stats(),
            **self.metadata,
        }


# ---------------------------------------------------------------------------
# Process-wide singletons (one per regime)
# ---------------------------------------------------------------------------
_predictors: dict[str, Predictor] = {}
_init_lock = Lock()


def get_predictor(regime: str = "plan_time") -> Predictor:
    """Lazily create + cache one Predictor per regime."""
    with _init_lock:
        if regime not in _predictors:
            _predictors[regime] = Predictor(regime=regime)
        return _predictors[regime]
