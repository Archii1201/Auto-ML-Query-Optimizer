"""
phase3b/reports.py
==================
Phase 3B reports & plots:

    * leaderboard plot — q-err median per (regime, model)
    * plan-pick accuracy bar chart
    * regret distribution per regime
    * Optuna convergence curves per tuned model
    * pred-vs-actual scatter for the AutoML winner
    * final markdown summary at reports/phase3b/REPORT.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

REPORT_DIR = PROJECT_ROOT / "reports" / "phase3b"
PLOTS_DIR  = REPORT_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

sns.set_style("whitegrid")


def _read(name: str, must_exist: bool = True) -> pd.DataFrame | None:
    p = REPORT_DIR / name
    if not p.exists():
        if must_exist:
            print(f"[!] missing required report file: {p}", file=sys.stderr)
            sys.exit(1)
        return None
    return pd.read_csv(p)


# ---------------------------------------------------------------------------
def plot_leaderboard(cmp_df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for ax, regime in zip(axes, ("plan_time", "post_mortem")):
        sub = cmp_df[cmp_df["regime"] == regime].copy()
        if sub.empty:
            ax.set_visible(False)
            continue
        sub = sub.sort_values("q_error_median_mean")
        colors = sub["kind"].map({
            "baseline":   "#888888",
            "default_ml": "#3b82f6",
            "tuned_ml":   "#10b981",
        }).fillna("#cccccc")
        ax.barh(sub["model"], sub["q_error_median_mean"], color=colors)
        ax.invert_yaxis()
        ax.set_xlabel("median q-error (lower is better)")
        ax.set_title(f"Regime: {regime}")
    fig.suptitle("Phase 3B leaderboard — median q-error", fontsize=14)
    fig.tight_layout()
    out = PLOTS_DIR / "leaderboard_qerror.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_plan_pick(pp: pd.DataFrame) -> Path:
    if pp.empty:
        return PLOTS_DIR / "plan_pick_accuracy.png"
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for ax, regime in zip(axes, ("plan_time", "post_mortem")):
        sub = pp[pp["regime"] == regime].copy()
        if sub.empty:
            ax.set_visible(False)
            continue
        sub = sub.sort_values("accuracy", ascending=True)
        ax.barh(sub["model"], sub["accuracy"], color="#10b981")
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel("plan-pick accuracy")
        ax.set_title(f"Regime: {regime}")
    fig.suptitle("Plan-pick accuracy — fraction of queries where the model's "
                 "fastest-predicted variant is the truly fastest variant",
                 fontsize=12)
    fig.tight_layout()
    out = PLOTS_DIR / "plan_pick_accuracy.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_regret(detail: pd.DataFrame, winner: dict) -> list[Path]:
    if detail.empty:
        return []
    paths: list[Path] = []
    for regime, w in winner.items():
        sub = detail[(detail["regime"] == regime) & (detail["model"] == w["model"])]
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(sub["regret_ratio"].clip(0, 5), bins=20, ax=ax,
                     color="#f97316", edgecolor="white")
        ax.set_xlabel("regret ratio  (picked_ms / oracle_ms − 1)")
        ax.set_ylabel("query groups")
        ax.set_title(f"Regret distribution — {regime} winner: {w['model']}")
        out = PLOTS_DIR / f"regret_distribution_{regime}.png"
        fig.tight_layout()
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(out)
    return paths


def plot_optuna_history(history: pd.DataFrame) -> list[Path]:
    if history is None or history.empty:
        return []
    paths: list[Path] = []
    for (regime, model), g in history.groupby(["regime", "model"]):
        g = g.sort_values("trial").copy()
        g["best_so_far"] = g["qerror_median"].cummin()
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(g["trial"], g["qerror_median"], "o-",
                color="#94a3b8", markersize=4, label="trial q-err median")
        ax.plot(g["trial"], g["best_so_far"], "-",
                color="#dc2626", linewidth=2, label="best-so-far")
        ax.set_xlabel("trial")
        ax.set_ylabel("q-err median")
        ax.set_title(f"Optuna convergence — {regime} / {model}")
        ax.legend()
        out = PLOTS_DIR / f"optuna_{regime}_{model}.png"
        fig.tight_layout()
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(out)
    return paths


def plot_pred_vs_actual_winner(preds: pd.DataFrame, winner: dict) -> list[Path]:
    paths: list[Path] = []
    for regime, w in winner.items():
        sub = preds[(preds["regime"] == regime) & (preds["model"] == w["model"])]
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(sub["y_true"], sub["y_pred"], s=12, alpha=0.5, color="#0ea5e9")
        lo = max(min(sub["y_true"].min(), sub["y_pred"].min()), 0.5)
        hi = max(sub["y_true"].max(), sub["y_pred"].max())
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="y = x")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("actual execution time (ms)")
        ax.set_ylabel("predicted execution time (ms)")
        ax.set_title(f"AutoML winner — {regime}: {w['model']}\n"
                     f"q-err med={w['q_error_median']:.2f}, "
                     f"plan-pick={w['plan_pick_acc']}")
        ax.legend()
        out = PLOTS_DIR / f"pred_vs_actual_winner_{regime}.png"
        fig.tight_layout()
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(out)
    return paths


# ---------------------------------------------------------------------------
def write_markdown(cmp_df: pd.DataFrame,
                   pp: pd.DataFrame,
                   winner: dict,
                   plot_paths: dict[str, Path | list[Path]]) -> Path:
    lines: list[str] = ["# Phase 3B — AutoML-Tuned Cost Model — Final Report", ""]
    lines.append("This phase upgrades the Phase 3A baseline pipeline with: "
                 "(a) ~5× more data via parameterized TPC-H + curated TPC-DS, "
                 "(b) plan-time log-transformed cost features and ratio features, "
                 "(c) **Optuna**-driven hyperparameter tuning of all five tree models, "
                 "(d) an **AutoML model selector** that picks the best model per regime "
                 "by a composite of median q-error and plan-pick accuracy.")
    lines.append("")

    lines.append("## AutoML winners")
    lines.append("")
    for regime, w in winner.items():
        lines.append(
            f"- **`{regime}`** → `{w['model']}` "
            f"(q-err median={w['q_error_median']:.2f}, "
            f"RMSE={w['rmse_ms']:.1f} ms, R²={w['r2']:+.3f}, "
            f"plan-pick acc={w['plan_pick_acc']!s})"
        )
    lines.append("")

    lines.append("## Leaderboard")
    lines.append("")
    leader = plot_paths.get("leaderboard")
    if leader:
        lines.append(f"![leaderboard](plots/{Path(leader).name})")
        lines.append("")

    lines.append("## Plan-pick accuracy")
    lines.append("")
    lines.append("This is the metric that matters in production: for each query "
                 "(group of 4 variants) does the model pick the truly fastest plan?")
    lines.append("")
    pp_path = plot_paths.get("plan_pick")
    if pp_path:
        lines.append(f"![plan-pick]({Path(pp_path).relative_to(REPORT_DIR).as_posix()})")
        lines.append("")

    lines.append("## Regret distribution (winners only)")
    lines.append("")
    for p in plot_paths.get("regret", []) or []:
        lines.append(f"![regret]({Path(p).relative_to(REPORT_DIR).as_posix()})")
        lines.append("")

    lines.append("## Predicted vs Actual — AutoML winners")
    lines.append("")
    for p in plot_paths.get("pred_vs_actual", []) or []:
        lines.append(f"![pred-actual]({Path(p).relative_to(REPORT_DIR).as_posix()})")
        lines.append("")

    lines.append("## Optuna convergence")
    lines.append("")
    for p in plot_paths.get("optuna", []) or []:
        lines.append(f"![optuna]({Path(p).relative_to(REPORT_DIR).as_posix()})")
        lines.append("")

    lines.append("## Numeric leaderboard")
    lines.append("")
    cols = ["regime", "kind", "model", "q_error_median_mean", "q_error_p95_mean",
            "rmse_mean", "r2_mean", "plan_pick_acc", "regret_ms_mean", "train_seconds"]
    show = cmp_df[cols].copy()
    for c in show.select_dtypes(include="float").columns:
        show[c] = show[c].round(3)
    for regime, grp in show.groupby("regime"):
        lines.append(f"### `{regime}`")
        lines.append("")
        lines.append(grp.to_markdown(index=False))
        lines.append("")

    out = REPORT_DIR / "REPORT.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
def main() -> int:
    cmp_df  = _read("model_comparison.csv")
    pp      = _read("plan_pick_summary.csv", must_exist=False)
    detail  = _read("plan_pick_detail.csv",  must_exist=False)
    history = _read("optuna_history.csv",    must_exist=False)
    preds   = _read("cv_predictions.csv")

    winner_path = REPORT_DIR / "automl_winner.json"
    winner = json.loads(winner_path.read_text(encoding="utf-8")) if winner_path.exists() else {}

    plot_paths: dict[str, Path | list[Path]] = {}
    plot_paths["leaderboard"]    = plot_leaderboard(cmp_df)
    plot_paths["plan_pick"]      = plot_plan_pick(pp)               if pp     is not None else None
    plot_paths["regret"]         = plot_regret(detail, winner)      if detail is not None else []
    plot_paths["optuna"]         = plot_optuna_history(history)
    plot_paths["pred_vs_actual"] = plot_pred_vs_actual_winner(preds, winner)

    md = write_markdown(cmp_df, pp if pp is not None else pd.DataFrame(),
                        winner, plot_paths)
    print(f"[OK] Wrote {md}")
    print(f"[OK] Plots in {PLOTS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
