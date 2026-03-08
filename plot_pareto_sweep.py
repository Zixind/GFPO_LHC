#!/usr/bin/env python3
"""
4 separate Pareto-frontier plots (2 triggers x 2 signals).

Each plot:
  X-axis = InBand fraction (bg rate within [90,110] kHz tolerance)
  Y-axis = signal efficiency (overall)
  9 method series, each with 25 points (5 lambda_1 x 5 lambda_2)
  Per-method Pareto frontier connecting non-dominated points

Usage:
  python plot_pareto_sweep.py --sweep-id <SWEEP_ID>
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend, no plt.show() blocking
import matplotlib.pyplot as plt
import wandb

PLOT_METHODS = ["DQN", "PPO", "ADT", "GRPO", "GFPO-F", "GFPO-FR"]

METHOD_MARKERS = {
    "Constant": "o", "PID": "s", "DQN": "^", "DQN-F": "v",
    "PPO": "D", "ADT": "P", "GRPO": "*", "GFPO-F": "X", "GFPO-FR": "h",
}

METHOD_COLORS = {
    "Constant": "tab:gray", "PID": "tab:blue", "DQN": "tab:orange", "DQN-F": "tab:brown",
    "PPO": "tab:green", "ADT": "tab:red", "GRPO": "tab:purple",
    "GFPO-F": "tab:cyan", "GFPO-FR": "tab:pink",
}


def fetch_sweep_data(entity, project, sweep_id):
    api = wandb.Api()
    sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
    rows = []
    for run in sweep.runs:
        if run.state != "finished":
            continue
        rows.append({
            "config": run.config,
            "summary": run.summary._json_dict,
            "name": run.name,
        })
    return rows


def pareto_front(x, y):
    """Pareto-optimal indices. Objective: maximize x (InBand), maximize y (sig eff)."""
    pts = np.column_stack([x, y])
    n = len(pts)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if pts[j, 0] >= pts[i, 0] and pts[j, 1] >= pts[i, 1]:
                if pts[j, 0] > pts[i, 0] or pts[j, 1] > pts[i, 1]:
                    is_pareto[i] = False
                    break
    return np.where(is_pareto)[0]


def extract_method_data(rows, trigger, method):
    """Return (inband, tt_overall, aa_overall, l1, l3) arrays for one (trigger, method)."""
    ib_key = f"{trigger}_{method}_InBand"
    tt_key = f"{trigger}_{method}_tt_overall"
    aa_key = f"{trigger}_{method}_aa_overall"

    ib, tt, aa, l1, l3 = [], [], [], [], []
    for r in rows:
        s = r["summary"]
        c = r["config"]
        if ib_key not in s or tt_key not in s:
            continue
        ib.append(float(s[ib_key]))
        tt.append(float(s[tt_key]))
        aa.append(float(s.get(aa_key, 0)))
        l1.append(float(c.get("lambda_1", 0)))
        l3.append(float(c.get("lambda_2", 0)))

    return (np.asarray(ib), np.asarray(tt), np.asarray(aa),
            np.asarray(l1), np.asarray(l3))


def plot_one(rows, trigger, sig_key, sig_label, outpath):
    """One standalone plot for a given (trigger, signal)."""
    fig, ax = plt.subplots(figsize=(12, 9))

    for method in PLOT_METHODS:
        inband, tt, aa, l1, l3 = extract_method_data(rows, trigger, method)
        sig = tt if sig_key == "tt" else aa
        if inband.size == 0:
            continue

        color = METHOD_COLORS.get(method, "black")
        marker = METHOD_MARKERS.get(method, "o")

        ax.scatter(inband, sig, c=color, marker=marker, s=80,
                   edgecolors="k", linewidths=0.3, zorder=3, label=method, alpha=0.85)

        # Annotate with (lambda_1, lambda_2)
        for i in range(len(inband)):
            ax.annotate(f"({l1[i]:.2f},{l3[i]:.2f})",
                        (inband[i], sig[i]),
                        textcoords="offset points", xytext=(4, 4),
                        fontsize=6, alpha=0.55, color=color)

        # Pareto frontier for this method
        pidx = pareto_front(inband, sig)
        if len(pidx) > 1:
            order = np.argsort(inband[pidx])
            pidx = pidx[order]
            ax.plot(inband[pidx], sig[pidx], "--", color=color, linewidth=1.5, alpha=0.7, zorder=2)

    ax.set_xlabel("InBand Rate", fontsize=26)
    ax.set_ylabel("Overall Signal Efficiency", fontsize=26)
    # No title — keep plots clean for paper
    ax.legend(fontsize=22, ncol=3, loc="best", framealpha=0.8)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=18)

    fig.tight_layout()
    fig.savefig(f"{outpath}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{outpath}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {outpath}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", default="zixin911")
    ap.add_argument("--project", default="Adaptive-ParticlePhysics-Triggers-RL")
    ap.add_argument("--sweep-id", required=True)
    ap.add_argument("--outdir", default="outputs/pareto_sweep")
    args = ap.parse_args()

    from pathlib import Path
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    print(f"Fetching sweep {args.sweep_id} ...")
    rows = fetch_sweep_data(args.entity, args.project, args.sweep_id)
    print(f"  Found {len(rows)} finished runs")

    if not rows:
        print("No finished runs found. Wait for the sweep to complete.")
        return

    triggers = ["AD", "HT"]
    signals = [
        ("tt", r"$t\bar{t}$", "ttbar"),
        ("aa", r"$h\rightarrow 4b$", "h4b"),
    ]

    # --- 4 separate plots ---
    for trigger in triggers:
        for sig_key, sig_label, sig_fname in signals:
            outpath = f"{args.outdir}/pareto_{trigger}_{sig_fname}"
            plot_one(rows, trigger, sig_key, sig_label, outpath)

    # --- Print Pareto-optimal configs per method ---
    for trigger in triggers:
        for sig_key, _, sig_fname in signals:
            print(f"\n{'='*60}")
            print(f"  {trigger} — {sig_fname}")
            print(f"{'='*60}")
            for method in PLOT_METHODS:
                inband, tt, aa, l1, l3 = extract_method_data(rows, trigger, method)
                sig = tt if sig_key == "tt" else aa
                if inband.size == 0:
                    continue
                pidx = pareto_front(inband, sig)
                print(f"\n  {method} Pareto-optimal:")
                for i in sorted(pidx, key=lambda j: -inband[j]):
                    print(f"    l1={l1[i]:.2f}  l3={l3[i]:.2f}  "
                          f"InBand={inband[i]:.3f}  sig_eff={sig[i]:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
