"""
triggers.py
===========
Phase 5D — decide *when* to retrain.

Three independent rules, ORed together, all behind a single cooldown so we
never retrain in a tight loop:

    volume   — enough NEW feedback rows since the last retrain
    schedule — it's simply been too long (a heartbeat retrain)
    drift    — pred/actual ratio has left its calibrated band  (see drift.py)

Pure and deterministic: `should_retrain(signals, config)` takes plain numbers
and returns a decision. The worker (5E) is responsible for *gathering* the
signals (watermark delta, minutes-since, Prometheus ratio); this module only
judges them, so it's fully unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from services.automl_service.drift import DriftThresholds, evaluate_ratio


@dataclass(frozen=True)
class TriggerConfig:
    min_new_rows:         int = 200     # volume rule
    cooldown_minutes:     float = 360   # never retrain more often than this
    max_interval_minutes: float = 1440  # schedule rule (heartbeat: daily)
    drift:                DriftThresholds = field(default_factory=DriftThresholds)


@dataclass
class TriggerSignals:
    new_rows:                  int = 0
    minutes_since_retrain:     float | None = None   # None = never retrained
    pred_actual_ratio:         float | None = None


@dataclass
class TriggerDecision:
    should_retrain: bool
    reason:         str
    fired:          list[str] = field(default_factory=list)
    in_cooldown:    bool = False

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def should_retrain(signals: TriggerSignals,
                   config: TriggerConfig = TriggerConfig()) -> TriggerDecision:
    mins = signals.minutes_since_retrain

    # Never retrained → bootstrap immediately (no cooldown to respect yet).
    if mins is None:
        return TriggerDecision(True, "bootstrap: no prior retrain", ["bootstrap"])

    # Cooldown gate — applies to every rule below.
    if mins < config.cooldown_minutes:
        return TriggerDecision(
            False,
            f"cooldown: {mins:.0f}m < {config.cooldown_minutes:.0f}m",
            [], in_cooldown=True,
        )

    fired: list[str] = []
    if signals.new_rows >= config.min_new_rows:
        fired.append(f"volume({signals.new_rows}≥{config.min_new_rows})")
    if mins >= config.max_interval_minutes:
        fired.append(f"schedule({mins:.0f}m≥{config.max_interval_minutes:.0f}m)")

    verdict = evaluate_ratio(signals.pred_actual_ratio, config.drift)
    if verdict.drifted:
        fired.append(f"drift({verdict.reason})")

    if fired:
        return TriggerDecision(True, "; ".join(fired), fired)
    return TriggerDecision(False, "no trigger rule fired", [])
