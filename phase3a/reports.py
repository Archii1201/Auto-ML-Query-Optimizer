"""
reports.py
==========
Render the Phase 3A artifact set:

    * runtime_distribution.png       — histogram + log-axis of the target
    * correlation_heatmap.png        — feature × target correlations
    * pred_vs_actual_<model>.png     — one per (regime, model)
    * feature_importance_<model>.png — top-20 importances (tree models only)
    * error_analysis.md              — worst-q-error rows + per-regime/model commentary

Inputs come from artifacts produced by `train_models.py`:
    reports/phase3a/cv_predictions.parquet
    reports/phase3a/feature_importance.csv
    reports/phase3a/model_comparison.csv
    data/processed/features.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive, safe inside scripts
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from phase3a.feature_selection import (  # noqa: E402
    GROUP_COLUMN,
    ID_COLUMNS,
    LEAKY_COLUMNS,
    TARGET_COLUMN,
)


REPORT_DIR  = PROJECT_ROOT / "reports" / "phase3a"
PLOT_DIR    = REPORT_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES_CSV = PROJECT_ROOT / "data" / "processed" / "features.csv"
PRED_FILE    = REPORT_DIR / "cv_predictions.csv"
IMP_FILE     = REPORT_DIR / "feature_importance.csv"
CMP_FILE     = REPORT_DIR / "model_comparison.csv"


sns.set_theme(context="notebook", style="whitegrid")


# ---------------------------------------------------------------------------
# Plot 1 — runtime distribution
# ---------------------------------------------------------------------------
def plot_runtime_distribution(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    y = df[TARGET_COLUMN].astype(float)

    sns.histplot(y, bins=30, kde=True, ax=axes[0], color="steelblue")
    axes[0].set_title("Execution time — linear scale")
    axes[0].set_xlabel("execution_time_ms")

    sns.histplot(np.log10(y.clip(lower=0.1)), bins=30, kde=True,
                 ax=axes[1], color="indianred")
    axes[1].set_title("Execution time — log10 scale")
    axes[1].set_xlabel("log10(execution_time_ms)")

    fig.tight_layout()
    out = PLOT_DIR / "runtime_distribution.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Plot 2 — correlation heatmap
# ---------------------------------------------------------------------------
def plot_correlation_heatmap(df: pd.DataFrame) -> Path:
    drop = list(ID_COLUMNS)
    keep = [c for c in df.columns if c not in drop]
    work = df[keep].copy()
    for c in work.columns:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    work = work.dropna(axis=1, how="all")

    is_leaky = {c: (c in LEAKY_COLUMNS) for c in work.columns}
    corr = work.corr(numeric_only=True)

    target_corr = (
        corr[TARGET_COLUMN]
        .drop(labels=[TARGET_COLUMN], errors="ignore")
        .abs()
        .sort_values(ascending=False)
        .head(20)
    )

    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(
        corr.loc[target_corr.index.tolist() + [TARGET_COLUMN],
                 target_corr.index.tolist() + [TARGET_COLUMN]],
        annot=True, fmt=".2f", cmap="coolwarm", center=0,
        square=True, cbar_kws={"shrink": 0.7}, ax=ax,
    )
    ax.set_title("Top-20 |corr| with execution_time_ms\n(leaky cols inflated by design)")
    for label in ax.get_xticklabels():
        if is_leaky.get(label.get_text(), False):
            label.set_color("crimson")
    for label in ax.get_yticklabels():
        if is_leaky.get(label.get_text(), False):
            label.set_color("crimson")

    fig.tight_layout()
    out = PLOT_DIR / "correlation_heatmap.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Plot 3 — predicted vs actual (per regime/model)
# ---------------------------------------------------------------------------
def plot_pred_vs_actual(preds: pd.DataFrame) -> list[Path]:
    paths: list[Path] = []
    for (regime, model), g in preds.groupby(["regime", "model"]):
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        x = np.maximum(g["y_true"].astype(float).to_numpy(), 0.1)
        y = np.maximum(g["y_pred"].astype(float).to_numpy(), 0.1)

        ax.scatter(x, y, alpha=0.7, s=24, edgecolor="black", linewidth=0.3)
        lo = float(min(x.min(), y.min()))
        hi = float(max(x.max(), y.max()))
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x")
        ax.plot([lo, hi], [10 * lo, 10 * hi], "r:", lw=0.8, label="10× error")
        ax.plot([lo, hi], [lo / 10, hi / 10], "r:", lw=0.8)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Actual execution time (ms)")
        ax.set_ylabel("Predicted execution time (ms)")
        ax.set_title(f"{model}  ·  {regime}\nlog–log scale")
        ax.legend(loc="upper left")

        out = PLOT_DIR / f"pred_vs_actual__{regime}__{model}.png"
        fig.tight_layout()
        fig.savefig(out, dpi=130)
        plt.close(fig)
        paths.append(out)
    return paths


# ---------------------------------------------------------------------------
# Plot 4 — feature importance (top-20)
# ---------------------------------------------------------------------------
def plot_feature_importance(imp: pd.DataFrame) -> list[Path]:
    paths: list[Path] = []
    for (regime, model), g in imp.groupby(["regime", "model"]):
        top = g.sort_values("importance", ascending=False).head(20).iloc[::-1]
        if top.empty or top["importance"].sum() == 0:
            continue
        fig, ax = plt.subplots(figsize=(7.5, 6.5))
        ax.barh(top["feature"], top["importance"], color="seagreen")
        ax.set_title(f"Feature importance — {model}  ·  {regime}\n(top 20)")
        ax.set_xlabel("importance")
        fig.tight_layout()
        out = PLOT_DIR / f"feature_importance__{regime}__{model}.png"
        fig.savefig(out, dpi=130)
        plt.close(fig)
        paths.append(out)
    return paths


# ---------------------------------------------------------------------------
# Markdown error analysis
# ---------------------------------------------------------------------------
def write_error_analysis(preds: pd.DataFrame, cmp_df: pd.DataFrame) -> Path:
    out = REPORT_DIR / "error_analysis.md"
    lines: list[str] = ["# Phase 3A — Error Analysis", ""]

    lines.append(
        "All metrics below come from the held-out fold of 5-fold "
        "**GroupKFold** CV (queries never appear in both train and test).\n"
    )

    for regime, gcmp in cmp_df.groupby("regime"):
        lines.append(f"## Regime: `{regime}`")
        ranked = gcmp.sort_values("q_error_median_mean").head(3)
        lines.append("\n**Top 3 by median q-error:**\n")
        for _, r in ranked.iterrows():
            lines.append(
                f"- `{r['model']}` — q-err median **{r['q_error_median_mean']:.2f}**, "
                f"R² {r['r2_mean']:+.3f}, MAE {r['mae_mean']:.1f} ms"
            )
        lines.append("")

        gpred = preds[preds["regime"] == regime].copy()
        if gpred.empty:
            continue
        gpred["q_error"] = np.maximum(
            np.maximum(gpred["y_pred"], 1.0) / np.maximum(gpred["y_true"], 1.0),
            np.maximum(gpred["y_true"], 1.0) / np.maximum(gpred["y_pred"], 1.0),
        )
        worst = (
            gpred.sort_values("q_error", ascending=False)
                 .head(10)
                 [["model", "query_id", "y_true", "y_pred", "q_error"]]
                 .round({"y_true": 2, "y_pred": 2, "q_error": 2})
        )
        lines.append("**10 worst predictions across all models in this regime:**\n")
        lines.append(worst.to_markdown(index=False))
        lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    if not FEATURES_CSV.exists():
        print(f"[!] {FEATURES_CSV} missing — run extract_features first.", file=sys.stderr)
        return 1
    if not PRED_FILE.exists() or not CMP_FILE.exists():
        print(f"[!] training artifacts missing — run train_models.py first.", file=sys.stderr)
        return 1

    df    = pd.read_csv(FEATURES_CSV)
    preds = pd.read_csv(PRED_FILE)
    cmp   = pd.read_csv(CMP_FILE)
    imp   = pd.read_csv(IMP_FILE) if IMP_FILE.exists() else pd.DataFrame()

    print("[i] runtime distribution -> ", plot_runtime_distribution(df))
    print("[i] correlation heatmap  -> ", plot_correlation_heatmap(df))

    paths = plot_pred_vs_actual(preds)
    print(f"[i] pred-vs-actual plots -> {len(paths)} files")

    if not imp.empty:
        ipaths = plot_feature_importance(imp)
        print(f"[i] feature importance   -> {len(ipaths)} files")

    print("[i] error analysis       -> ", write_error_analysis(preds, cmp))
    print("\n[OK] Reports refreshed under", REPORT_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
