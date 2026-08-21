"""
Unit tests for Phase 5D — drift detection, triggers, and retrain state.
All pure logic; no network, no dataset.
"""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from services.automl_service.drift import (
    DriftThresholds,
    evaluate_ratio,
    parse_prom_gauge,
)
from services.automl_service.state import RetrainState, now_utc
from services.automl_service.triggers import (
    TriggerConfig,
    TriggerSignals,
    should_retrain,
)


# ----- drift --------------------------------------------------------------
def test_calibrated_ratio_no_drift():
    v = evaluate_ratio(1.05)
    assert v.drifted is False


def test_over_prediction_drift():
    v = evaluate_ratio(2.5, DriftThresholds(0.5, 2.0))
    assert v.drifted is True and "over" in v.reason


def test_under_prediction_drift():
    v = evaluate_ratio(0.3, DriftThresholds(0.5, 2.0))
    assert v.drifted is True and "under" in v.reason


def test_none_and_nan_ratio_safe():
    assert evaluate_ratio(None).drifted is False
    assert evaluate_ratio(float("nan")).drifted is False


def test_parse_prom_gauge():
    text = "\n".join([
        "# HELP pred_actual_ratio ratio",
        "# TYPE pred_actual_ratio gauge",
        "pred_actual_ratio 1.42",
        "other_metric 9",
    ])
    assert parse_prom_gauge(text, "pred_actual_ratio") == 1.42
    assert parse_prom_gauge(text, "missing_metric") is None


def test_parse_prom_gauge_with_labels_takes_last():
    text = 'm{a="1"} 1.0\nm{a="2"} 3.0\n'
    assert parse_prom_gauge(text, "m") == 3.0


# ----- triggers -----------------------------------------------------------
def test_bootstrap_when_never_retrained():
    d = should_retrain(TriggerSignals(minutes_since_retrain=None))
    assert d.should_retrain is True and "bootstrap" in d.reason


def test_cooldown_blocks_everything():
    sig = TriggerSignals(new_rows=10_000, minutes_since_retrain=10,
                         pred_actual_ratio=9.0)
    d = should_retrain(sig, TriggerConfig(cooldown_minutes=360))
    assert d.should_retrain is False and d.in_cooldown is True


def test_volume_rule_fires_after_cooldown():
    sig = TriggerSignals(new_rows=500, minutes_since_retrain=400)
    d = should_retrain(sig, TriggerConfig(min_new_rows=200, cooldown_minutes=360,
                                          max_interval_minutes=100000))
    assert d.should_retrain is True and any("volume" in f for f in d.fired)


def test_schedule_rule_fires():
    sig = TriggerSignals(new_rows=0, minutes_since_retrain=2000)
    d = should_retrain(sig, TriggerConfig(min_new_rows=999999,
                                          max_interval_minutes=1440,
                                          cooldown_minutes=360))
    assert d.should_retrain is True and any("schedule" in f for f in d.fired)


def test_drift_rule_fires():
    sig = TriggerSignals(new_rows=0, minutes_since_retrain=400,
                         pred_actual_ratio=3.0)
    d = should_retrain(sig, TriggerConfig(min_new_rows=999999,
                                          max_interval_minutes=100000,
                                          cooldown_minutes=360))
    assert d.should_retrain is True and any("drift" in f for f in d.fired)


def test_no_rule_fires():
    sig = TriggerSignals(new_rows=5, minutes_since_retrain=400,
                         pred_actual_ratio=1.0)
    d = should_retrain(sig, TriggerConfig(min_new_rows=200,
                                          max_interval_minutes=100000,
                                          cooldown_minutes=360))
    assert d.should_retrain is False


# ----- state --------------------------------------------------------------
def test_state_roundtrip_and_counters(tmp_path):
    p = tmp_path / "state.json"
    st = RetrainState.load(p)
    assert st.retrains_total == 0
    st.mark_retrain(reason="volume", features_rows=1000,
                    candidate_version="abc").save(p)
    st2 = RetrainState.load(p)
    assert st2.retrains_total == 1
    assert st2.features_rows_at_retrain == 1000
    assert st2.last_candidate_version == "abc"


def test_minutes_since_retrain(tmp_path):
    st = RetrainState()
    assert st.minutes_since_retrain() is None
    past = now_utc() - timedelta(minutes=120)
    st.last_retrain_utc = past.isoformat()
    mins = st.minutes_since_retrain()
    assert 119 <= mins <= 121


def test_state_corrupt_resets(tmp_path):
    p = tmp_path / "state.json"
    p.write_text("garbage{", encoding="utf-8")
    assert RetrainState.load(p).retrains_total == 0
