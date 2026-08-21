"""
config.py
=========
Phase 5B — retrain profiles.

Two named profiles trade wall-clock time against model quality:

    fast  — Optuna OFF, defaults only. ~seconds–minutes. Used by the
            automated worker on every trigger so the loop stays cheap and
            never blocks. Good enough to catch a real regression/improvement.
    full  — Optuna ON (n_trials). Minutes–hours. Run nightly / on demand
            when we want the best possible candidate before a big promotion.

Keeping this as data (not scattered flags) means the worker, the CLI, and
the tests all agree on exactly what "a fast retrain" means.
"""

from __future__ import annotations

from dataclasses import dataclass, field

DEFAULT_REGIMES = ("plan_time",)


@dataclass(frozen=True)
class RetrainProfile:
    name:         str
    skip_tuning:  bool
    n_trials:     int = 25
    tune_timeout: int = 120
    regimes:      tuple[str, ...] = field(default_factory=lambda: DEFAULT_REGIMES)

    def train_argv(self) -> list[str]:
        """Args passed to phase3b/train_models.py."""
        argv: list[str] = ["--regimes", *self.regimes]
        if self.skip_tuning:
            argv.append("--skip-tuning")
        else:
            argv += ["--n-trials", str(self.n_trials),
                     "--tune-timeout", str(self.tune_timeout)]
        return argv


PROFILES: dict[str, RetrainProfile] = {
    "fast": RetrainProfile(name="fast", skip_tuning=True),
    "full": RetrainProfile(name="full", skip_tuning=False, n_trials=25),
}


def get_profile(name: str) -> RetrainProfile:
    try:
        return PROFILES[name]
    except KeyError:
        raise ValueError(
            f"unknown retrain profile {name!r}; choose from {sorted(PROFILES)}"
        ) from None
