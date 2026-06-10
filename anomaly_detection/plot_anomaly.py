"""
anomaly_detection/plot_anomaly.py

Plots for UNSW-NB15 online anomaly detection results.
Mirrors the particle physics figures: FAR trajectory + TPR comparison.

Usage:
    python anomaly_detection/plot_anomaly.py \
        --csv outputs/anomaly_unsw/tables/chunk_stats.csv \
        --out outputs/anomaly_unsw/
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
try:
    import mplhep as hep
    hep.style.use("CMS")
except ImportError:
    pass

METHOD_STYLE = {
    "Constant": dict(color="#7f7f7f", ls="-",  lw=1.2),
    "PID":      dict(color="#bcbd22", ls="-",  lw=1.4),
    "SPOT":     dict(color="#17becf", ls="-",  lw=1.4),
    "DQN":      dict(color="#e377c2", ls="-",  lw=1.6),
    "GRPO":     dict(color="#1f77b4", ls="-",  lw=1.8),
    "L-GRPO":   dict(color="#ff7f0e", ls=":",  lw=1.8),
    "GFPO-F":   dict(color="#2ca02c", ls="-",  lw=2.0),
    "PPO":      dict(color="#9467bd", ls="-",  lw=1.6),
}


def smooth(x, w=5):
    return pd.Series(x).rolling(w, center=True, min_periods=1).mean().values


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="outputs/anomaly_unsw/tables/chunk_stats.csv")
    p.add_argument("--out", default="outputs/anomaly_unsw/")
    p.add_argument("--far-target", type=float, default=0.005)
    p.add_argument("--far-tol",    type=float, default=0.0005)
    return p.parse_args()


def plot_far_trajectory(df, far_target, far_tol, out_dir):
    """FAR over chunks (analogous to background rate trajectory)."""
    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)

    for meth, style in METHOD_STYLE.items():
        sub = df[df["method"] == meth].sort_values("chunk")
        if sub.empty:
            continue
        ax.plot(sub["chunk"], smooth(sub["far"].values * 100, w=3),
                label=meth, **style)

    ax.axhline((far_target + far_tol) * 100, color="black", lw=1.0, ls=":", alpha=0.5)
    ax.axhline((far_target - far_tol) * 100, color="black", lw=1.0, ls=":", alpha=0.5)
    ax.axhline(far_target * 100,              color="black", lw=0.7, ls="-", alpha=0.3)
    n = df["chunk"].max()
    ax.fill_between([0, n],
                    [(far_target - far_tol)*100]*2,
                    [(far_target + far_tol)*100]*2,
                    alpha=0.07, color="black", label=r"$\pm\tau$ FAR band")

    ax.set_xlabel("Chunk index (streaming)", fontsize=11)
    ax.set_ylabel("False alert rate (%)", fontsize=11)
    ax.set_title("False alert rate trajectory — UNSW-NB15 online anomaly detection",
                 fontsize=11)
    ax.legend(fontsize=9, ncol=2)

    out = Path(out_dir) / "anomaly_far_trajectory.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    fig.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    print(f"Saved → {out}")
    plt.close(fig)


def plot_summary_bars(df, out_dir):
    """Bar chart: InBand rate and mean TPR per method."""
    summary = df.groupby("method").agg(
        InBand=("inband", "mean"),
        TPR=("tpr",    "mean"),
    ).reset_index()

    order = [m for m in METHOD_STYLE if m in summary["method"].values]
    summary = summary.set_index("method").reindex(order).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)

    for ax, col, ylabel, title in [
        (axes[0], "InBand", "In-band fraction (FAR within budget)",
         "(a) FAR in-band rate"),
        (axes[1], "TPR",    "Attack recall (TPR)",
         "(b) Mean attack detection rate"),
    ]:
        colors = [METHOD_STYLE.get(m, {}).get("color", "grey") for m in summary["method"]]
        ax.bar(range(len(summary)), summary[col].values, color=colors, alpha=0.85)
        ax.set_xticks(range(len(summary)))
        ax.set_xticklabels(summary["method"].values, rotation=30, ha="right", fontsize=9)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.set_ylim(0, 1.1)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11)
        for i, v in enumerate(summary[col].values):
            ax.text(i, v + 0.02, f"{v:.0%}", ha="center", fontsize=8)

    fig.suptitle("UNSW-NB15 online anomaly detection — method comparison",
                 fontsize=12, fontweight="bold")
    out = Path(out_dir) / "anomaly_summary_bars.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    fig.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    print(f"Saved → {out}")
    plt.close(fig)


def main():
    args = parse_args()
    df   = pd.read_csv(args.csv)
    Path(args.out).mkdir(parents=True, exist_ok=True)

    plot_far_trajectory(df, args.far_target, args.far_tol, args.out)
    plot_summary_bars(df, args.out)


if __name__ == "__main__":
    main()
