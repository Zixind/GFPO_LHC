#!/usr/bin/env python3
"""
Explains why CMS inband h->4b efficiency exceeds Oracle(r+).

2-panel figure:
  (a) Per-chunk h->4b AD efficiency vs chunk index.
      Oracle per-chunk efficiency shown as a line.
      Inband chunks (for GFPO-F and GFPO-FR) shaded.
      Oracle overall average and inband averages marked.
  (b) Per-chunk background acceptance rate vs chunk index.
      Tolerance band [r*-tau, r*+tau] shaded.
      Inband chunks highlighted.

Root cause: h->4b efficiency drifts ~5x across the CMS run (early
high-pileup chunks: ~11%, late low-pileup chunks: ~59%).  Inband
chunks are not uniformly distributed — they cluster in the
high-efficiency tail, pushing the inband average above the oracle
average over all chunks.

Usage:
    conda run -n adaptive python plot_cms_inband_drift.py
"""

import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
from pathlib import Path

hep.style.use("CMS")

LABEL_FS  = 16
TICK_FS   = 14
LEGEND_FS = 13

TARGET_PCT = 0.25
TOL_PCT    = 0.025
R_STAR     = TARGET_PCT
R_PLUS     = TARGET_PCT + TOL_PCT   # 0.275 %
R_MINUS    = TARGET_PCT - TOL_PCT   # 0.225 %

OUT = Path("outputs/cms_inband_drift")

# ── load oracle per-chunk CSV ─────────────────────────────────────────────────
oracle_chunks = []
with open("chunk_cuts_as2_99725_realdata.csv") as f:
    for row in csv.DictReader(f):
        if row["chunk"] == "AVG_FINITE":
            continue
        oracle_chunks.append({
            "chunk":   int(row["chunk"]),
            "aa_eff":  float(row["aa_eff"]),
            "bg_accept": float(row["bg_accept"]) * 100.0,  # → %
        })

oracle_idx    = np.array([r["chunk"]   for r in oracle_chunks])
oracle_h4b    = np.array([r["aa_eff"]  for r in oracle_chunks])
oracle_bg_pct = np.array([r["bg_accept"] for r in oracle_chunks])
oracle_avg    = float(np.mean(oracle_h4b))

# ── load RL chunk_stats ───────────────────────────────────────────────────────
chunk_stats_path = "outputs/rollout_real_all_RealData/tables/chunk_stats.csv"
rl_data = {}   # method -> list of rows sorted by chunk

with open(chunk_stats_path) as f:
    for row in csv.DictReader(f):
        if row["trigger"] != "AD":
            continue
        method = row["method"]
        if method not in rl_data:
            rl_data[method] = []
        rl_data[method].append({
            "chunk":      int(row["chunk"]),
            "bg_pct":     float(row["bg_pct"]),
            "aa_overall": float(row["aa_overall"]) if row["aa_overall"] else np.nan,
            "inband":     int(row["inband"]),
        })

for method in rl_data:
    rl_data[method].sort(key=lambda r: r["chunk"])

methods_show   = ["GFPO-F", "GFPO-FR"]
colors_method  = {"GFPO-F": "#ff7f0e", "GFPO-FR": "#9467bd"}
markers_method = {"GFPO-F": "^",       "GFPO-FR": "s"}    # triangle, square
offsets_method = {"GFPO-F": -0.25,     "GFPO-FR": +0.25}  # x-offset to avoid overlap

# ── compute inband averages ───────────────────────────────────────────────────
for method in methods_show:
    rows = rl_data[method]
    inband_h4b = [oracle_h4b[r["chunk"]] for r in rows if r["inband"] == 1]
    avg_inband = np.mean(inband_h4b) if inband_h4b else np.nan
    n_inband   = len(inband_h4b)
    print(f"{method}: n_inband={n_inband}  inband_h4b_avg={avg_inband:.3f}%  "
          f"oracle_overall={oracle_avg:.3f}%")

# ── figure ────────────────────────────────────────────────────────────────────
fig, ax_eff = plt.subplots(1, 1, figsize=(13, 5))

# ── panel (a): per-chunk h->4b efficiency ────────────────────────────────────
# Draw oracle line with open circles for all chunks
ax_eff.plot(oracle_idx, oracle_h4b, color="steelblue", lw=1.5, zorder=2,
            alpha=0.6)
ax_eff.scatter(oracle_idx, oracle_h4b, color="steelblue", s=35, zorder=3,
               facecolors="none", linewidths=1.2, marker="o",
               label="Oracle $h\\to4b$ eff. per chunk")

# Overlay filled colored markers for inband chunks per method
for method in methods_show:
    rows   = rl_data[method]
    color  = colors_method[method]
    marker = markers_method[method]
    xoff   = offsets_method[method]
    inband_chunks = [r["chunk"] for r in rows if r["inband"] == 1]
    inband_h4b_vals = oracle_h4b[inband_chunks]

    ax_eff.scatter(np.array(inband_chunks) + xoff, inband_h4b_vals,
                   color=color, s=35, zorder=5, marker=marker,
                   facecolors="none", linewidths=1.2,
                   label=f"{method}: inband chunks")

    # inband average line
    avg = float(np.mean(inband_h4b_vals))
    ax_eff.axhline(avg, color=color, linestyle="--", lw=1.8,
                   label=f"{method} inband avg = {avg:.1f}%")

# oracle overall average
ax_eff.axhline(oracle_avg, color="steelblue", linestyle=":", lw=2.0,
               label=f"Oracle overall avg = {oracle_avg:.1f}%")

ax_eff.set_ylabel("$h\\to4b$ AD efficiency [%]", fontsize=LABEL_FS)
ax_eff.tick_params(axis="both", labelsize=TICK_FS)
ax_eff.legend(fontsize=LEGEND_FS - 2, framealpha=0.8, ncol=2)
ax_eff.set_title("Per-chunk $h\\to4b$ AD efficiency — CMS Run 283408",
                 fontsize=LABEL_FS)
ax_eff.set_ylim(0, 75)
ax_eff.set_xlim(oracle_idx[0] - 0.5, oracle_idx[-1] + 0.5)
ax_eff.set_xlabel("Chunk index", fontsize=LABEL_FS)

OUT.parent.mkdir(parents=True, exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(f"{OUT}.{ext}", dpi=150, bbox_inches="tight")
    print(f"Saved {OUT}.{ext}")

plt.close(fig)
