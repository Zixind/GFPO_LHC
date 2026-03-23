#!/usr/bin/env python3
"""
Explains why CMS inband h->4b efficiency exceeds Oracle(r+).

4-panel figure:
  (a) AD trigger background rate [kHz] vs chunk index (GFPO-F, GFPO-FR only)
  (b) HT trigger background rate [kHz] vs chunk index (GFPO-F, GFPO-FR only)
  (c) Per-chunk h->4b AD efficiency vs chunk index.
      Oracle per-chunk efficiency shown as a line.
      Inband chunks (for GFPO-F and GFPO-FR) highlighted.
      Oracle overall average and inband averages marked.
  (d) Inband count per 10-chunk bin histogram.

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

RATE_SCALE_KHZ = 400.0  # convert percent to kHz
TARGET_KHZ = TARGET_PCT * RATE_SCALE_KHZ   # 100 kHz
TOL_KHZ    = TOL_PCT    * RATE_SCALE_KHZ   # 10 kHz

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

# ── load RL chunk_stats (both AD and HT) ─────────────────────────────────────
chunk_stats_path = "outputs/rollout_real_all_RealData/tables/chunk_stats.csv"
rl_data_ad = {}   # method -> list of rows sorted by chunk (AD trigger)
rl_data_ht = {}   # method -> list of rows sorted by chunk (HT trigger)

with open(chunk_stats_path) as f:
    for row in csv.DictReader(f):
        method = row["method"]
        entry = {
            "chunk":      int(row["chunk"]),
            "bg_pct":     float(row["bg_pct"]),
            "bg_khz":     float(row["bg_khz"]),
            "aa_overall": float(row["aa_overall"]) if row["aa_overall"] else np.nan,
            "aa":         float(row["aa"]) if row["aa"] else np.nan,
            "inband":     int(row["inband"]),
        }
        if row["trigger"] == "AD":
            if method not in rl_data_ad:
                rl_data_ad[method] = []
            rl_data_ad[method].append(entry)
        elif row["trigger"] == "HT":
            if method not in rl_data_ht:
                rl_data_ht[method] = []
            rl_data_ht[method].append(entry)

for d in (rl_data_ad, rl_data_ht):
    for method in d:
        d[method].sort(key=lambda r: r["chunk"])

methods_show   = ["GFPO-F", "GFPO-FR"]
colors_method  = {"GFPO-F": "#ff7f0e", "GFPO-FR": "#9467bd"}
markers_method = {"GFPO-F": "^",       "GFPO-FR": "s"}    # triangle, square
offsets_method = {"GFPO-F": -0.25,     "GFPO-FR": +0.25}  # x-offset to avoid overlap

# ── compute inband averages using RL aa_overall (matches paper table) ─────────
for method in methods_show:
    rows = rl_data_ad[method]
    inband_aa = [r["aa_overall"] for r in rows if r["inband"] == 1 and np.isfinite(r["aa_overall"])]
    avg_inband = np.mean(inband_aa) if inband_aa else np.nan
    n_inband   = len(inband_aa)
    print(f"{method}: n_inband={n_inband}  inband_h4b_avg={avg_inband:.3f}%  "
          f"oracle_overall={oracle_avg:.3f}%")

# ── determine inband chunks per method ────────────────────────────────────────
inband_set = {
    "GFPO-F":  set(r["chunk"] for r in rl_data_ad["GFPO-F"]  if r["inband"] == 1),
    "GFPO-FR": set(r["chunk"] for r in rl_data_ad["GFPO-FR"] if r["inband"] == 1),
}

# ── figure ────────────────────────────────────────────────────────────────────
fig, (ax_f, ax_fr, ax_eff) = plt.subplots(
    3, 1, figsize=(13, 13),
    gridspec_kw={"height_ratios": [2, 2, 3]},
    sharex=True,
)

# ── helper: plot one method's bg rate with inband shading + oracle comparison ─
def plot_method_panel(ax, method, panel_label):
    color = colors_method[method]
    ib_set = inband_set[method]

    # shade inband (green) vs out-of-band (red) per chunk
    for c in oracle_idx:
        c = int(c)
        if c in ib_set:
            ax.axvspan(c - 0.5, c + 0.5, alpha=0.10, color="green", zorder=0)
        else:
            ax.axvspan(c - 0.5, c + 0.5, alpha=0.06, color="red", zorder=0)

    # tolerance band
    ax.fill_between(oracle_idx, TARGET_KHZ - TOL_KHZ, TARGET_KHZ + TOL_KHZ,
                    alpha=0.15, color="steelblue", label="Tolerance band [90, 110] kHz")
    ax.axhline(TARGET_KHZ, color="steelblue", linestyle=":", lw=1.0, alpha=0.5)

    # bg rate line + markers
    rows = rl_data_ad[method]
    chunks = np.array([r["chunk"] for r in rows])
    bg_khz = np.array([r["bg_khz"] for r in rows])
    ib_mask = np.array([int(c) in ib_set for c in chunks])

    ax.plot(chunks, bg_khz, color=color, lw=1.5, alpha=0.5, zorder=2)
    ax.scatter(chunks[ib_mask], bg_khz[ib_mask], color=color, s=55,
               marker=markers_method[method], facecolors=color,
               linewidths=0.8, zorder=4, alpha=0.85, label=f"{method} inband")
    ax.scatter(chunks[~ib_mask], bg_khz[~ib_mask], color=color, s=35,
               marker=markers_method[method], facecolors="none",
               linewidths=0.8, zorder=3, alpha=0.5, label=f"{method} out-of-band")

    # inband fraction annotation
    n_ib = int(ib_mask.sum())
    n_tot = len(chunks)
    ax.text(0.98, 0.05, f"Inband: {n_ib}/{n_tot} chunks",
            transform=ax.transAxes, fontsize=LEGEND_FS, ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    # legend entries for shading
    ax.fill_between([], [], [], alpha=0.20, color="green", label="Inband chunk")
    ax.fill_between([], [], [], alpha=0.15, color="red",   label="Out-of-band chunk")

    ax.set_ylabel("AD bg rate [kHz]", fontsize=LABEL_FS)
    ax.set_title(f"({panel_label}) {method} — AD trigger background rate",
                 fontsize=LABEL_FS)
    ax.tick_params(axis="both", labelsize=TICK_FS)
    ax.legend(fontsize=LEGEND_FS - 2, framealpha=0.8, ncol=3, loc="upper left")
    ax.set_ylim(50, 160)
    ax.grid(True, alpha=0.2)

plot_method_panel(ax_f,  "GFPO-F",  "a")
plot_method_panel(ax_fr, "GFPO-FR", "b")

# ── panel (c): per-chunk h->4b efficiency ─────────────────────────────────────
ax_eff.plot(oracle_idx, oracle_h4b, color="steelblue", lw=1.5, zorder=2,
            alpha=0.6)
ax_eff.scatter(oracle_idx, oracle_h4b, color="steelblue", s=35, zorder=3,
               facecolors="none", linewidths=1.2, marker="o",
               label="Oracle $h\\to4b$ eff. per chunk")

for method in methods_show:
    rows   = rl_data_ad[method]
    color  = colors_method[method]
    marker = markers_method[method]
    xoff   = offsets_method[method]
    inband_rows = [r for r in rows if r["inband"] == 1 and np.isfinite(r["aa_overall"])]
    inband_chunks   = np.array([r["chunk"]      for r in inband_rows])
    inband_aa_vals  = np.array([r["aa_overall"] for r in inband_rows])

    ax_eff.scatter(inband_chunks + xoff, inband_aa_vals,
                   color=color, s=35, zorder=5, marker=marker,
                   facecolors="none", linewidths=1.2,
                   label=f"{method}: inband chunks")

    avg = float(np.mean(inband_aa_vals))
    ax_eff.axhline(avg, color=color, linestyle="--", lw=1.8,
                   label=f"{method} inband avg = {avg:.1f}%")

ax_eff.axhline(oracle_avg, color="steelblue", linestyle=":", lw=2.0,
               label=f"Oracle overall avg = {oracle_avg:.1f}%")

ax_eff.set_ylabel("$h\\to4b$ AD efficiency [%]", fontsize=LABEL_FS)
ax_eff.set_xlabel("Chunk index", fontsize=LABEL_FS)
ax_eff.tick_params(axis="both", labelsize=TICK_FS)
ax_eff.legend(fontsize=LEGEND_FS - 2, framealpha=0.8, ncol=2)
ax_eff.set_title("(c) Per-chunk $h\\to4b$ AD efficiency", fontsize=LABEL_FS)
ax_eff.set_ylim(0, 75)
ax_eff.set_xlim(oracle_idx[0] - 0.5, oracle_idx[-1] + 0.5)

fig.subplots_adjust(hspace=0.18)

OUT.parent.mkdir(parents=True, exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(f"{OUT}.{ext}", dpi=150, bbox_inches="tight")
    print(f"Saved {OUT}.{ext}")

plt.close(fig)
