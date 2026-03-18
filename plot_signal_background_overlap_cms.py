#!/usr/bin/env python3
"""
Signal-background overlap figure for CMS Run 283408.

3-panel figure:
  (a) HT score distributions  — background, H→4b, ttbar
  (b) AD score distributions  — background, H→4b, ttbar
  (c) ROC at operating point  — per-chunk-avg signal efficiency vs r[%]
      with oracle operating-point markers loaded from oracle CSV files
      (matching Table 1 values exactly).

Usage:
    conda run -n adaptive python plot_signal_background_overlap_cms.py
"""

import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
from pathlib import Path

from RL.utils import read_any_h5

hep.style.use("CMS")
LABEL_FS  = 16
TICK_FS   = 14
LEGEND_FS = 13
ANNOT_FS  = 15

OUT        = Path("outputs/signal_background_overlap_cms")
H5         = Path("Data/Matched_data_2016_dim2.h5")
CHUNK_SIZE = 20_000
START      = CHUNK_SIZE * 10   # = 200_000

TARGET_PCT = 0.25
TOL_PCT    = 0.025
R_PLUS     = TARGET_PCT + TOL_PCT   # 0.275 %

# ── oracle operating-point values from CSV (match Table 1 exactly) ────────────
def read_oracle_avg(csv_path):
    """Return (tt_eff_avg, aa_eff_avg) from the AVG_FINITE summary row."""
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row["chunk"] == "AVG_FINITE":
                return float(row["tt_eff_avg"]), float(row["aa_eff_avg"])
    raise ValueError(f"No AVG_FINITE row in {csv_path}")

e_ht_tt_oracle, e_ht_aa_oracle = read_oracle_avg("chunk_cuts_ht2_99725_realdata.csv")
e_as_tt_oracle, e_as_aa_oracle = read_oracle_avg("chunk_cuts_as2_99725_realdata.csv")
print(f"Oracle (r+={R_PLUS}%) from CSV:  "
      f"HT tt={e_ht_tt_oracle:.3f}%  HT h4b={e_ht_aa_oracle:.3f}%  "
      f"AD tt={e_as_tt_oracle:.3f}%  AD h4b={e_as_aa_oracle:.3f}%")

# ── load ─────────────────────────────────────────────────────────────────────
print("Loading CMS data …")
d = read_any_h5(str(H5), score_dim_hint=2)

Bht  = np.asarray(d["Bht"],  dtype=np.float64)
Bas  = np.asarray(d["Bas2"], dtype=np.float64)
Tht  = np.asarray(d["Tht"],  dtype=np.float64)
Tas  = np.asarray(d["Tas2"], dtype=np.float64)
Aht  = np.asarray(d["Aht"],  dtype=np.float64)
Aas  = np.asarray(d["Aas2"], dtype=np.float64)

sl  = slice(START, None)
bht = Bht[sl];  bas = Bas[sl]
tht = Tht[sl];  tas = Tas[sl]
aht = Aht[sl];  aas = Aas[sl]

print(f"Using {len(bht):,} events from event {START}")

# ── per-chunk ROC for smooth curve ───────────────────────────────────────────
def roc_curve_perchunk(bg_all, sig_all, chunk_size=CHUNK_SIZE, n_pts=400):
    """Per-chunk-average (r_bg[%], eff_sig[%]) for a sweep of single cuts.

    CMS signals are index-aligned to background.  Gives a smooth ROC curve
    showing discriminability of a fixed-cut strategy.  Oracle CSV markers
    (per-chunk adaptive cuts) are annotated separately.
    """
    bg_fin = bg_all[np.isfinite(bg_all)]
    lo = np.percentile(bg_fin, 99.0)
    hi = np.percentile(bg_fin, 99.999)
    cuts = np.linspace(hi, lo, n_pts)

    r_chunks, eff_chunks = [], []
    n = len(bg_all)
    start = 0
    while start < n:
        end = min(start + chunk_size, n)
        bg_c      = bg_all[start:end]
        sig_c     = sig_all[start:end]
        bg_c_fin  = bg_c[np.isfinite(bg_c)]
        sig_c_fin = sig_c[np.isfinite(sig_c)]
        if len(bg_c_fin) == 0:
            start += chunk_size
            continue

        bg_sorted = np.sort(bg_c_fin)
        r_c = (len(bg_c_fin) - np.searchsorted(bg_sorted, cuts, side="left")) \
              / len(bg_c_fin) * 100.0
        if len(sig_c_fin) > 0:
            sig_sorted = np.sort(sig_c_fin)
            eff_c = (len(sig_c_fin) - np.searchsorted(sig_sorted, cuts, side="left")) \
                    / len(sig_c_fin) * 100.0
        else:
            eff_c = np.full(n_pts, np.nan)

        r_chunks.append(r_c)
        eff_chunks.append(eff_c)
        start += chunk_size

    return np.nanmean(r_chunks, axis=0), np.nanmean(eff_chunks, axis=0), cuts

print("Computing per-chunk ROC curves …")
r_ht_tt, eff_ht_tt, _ = roc_curve_perchunk(bht, tht)
r_ht_aa, eff_ht_aa, _ = roc_curve_perchunk(bht, aht)
r_as_tt, eff_as_tt, _ = roc_curve_perchunk(bas, tas)
r_as_aa, eff_as_aa, _ = roc_curve_perchunk(bas, aas)

# ── oracle cut for histogram vertical line ────────────────────────────────────
def oracle_cut_global(scores, rate_pct):
    s = scores[np.isfinite(scores)]
    k = max(1, int(np.round(len(s) * rate_pct / 100.0)))
    return float(np.sort(s)[-k])

cut_ht = oracle_cut_global(bht, R_PLUS)
cut_as = oracle_cut_global(bas, R_PLUS)
print(f"Oracle cut HT  (r+={R_PLUS}%): {cut_ht:.3f}")
print(f"Oracle cut AD  (r+={R_PLUS}%): {cut_as:.3f}")

# ── colors / styles ──────────────────────────────────────────────────────────
C_BG  = "#6baed6"
C_H4B = "#d62728"
C_TT  = "#2ca02c"

# ── figure ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13, 10))
gs  = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.32)
ax_ht  = fig.add_subplot(gs[0, 0])
ax_as  = fig.add_subplot(gs[0, 1])
ax_roc = fig.add_subplot(gs[1, :])

def plot_score_hist(ax, bg, sig_aa, sig_tt, cut, title, xlabel):
    all_fin = np.concatenate([bg[np.isfinite(bg)],
                               sig_aa[np.isfinite(sig_aa)],
                               sig_tt[np.isfinite(sig_tt)]])
    lo = np.percentile(all_fin, 0.5)
    hi = np.percentile(all_fin, 99.95)
    bins = np.logspace(np.log10(max(lo, 1e-1)), np.log10(hi), 80)

    kw = dict(bins=bins, density=True, histtype="stepfilled", alpha=0.45)
    ax.hist(bg[np.isfinite(bg)], color=C_BG, label="Background (CMS Run)", **kw)
    kw["histtype"] = "step"; kw["alpha"] = 1.0; kw["linewidth"] = 1.6
    ax.hist(sig_aa[np.isfinite(sig_aa)], color=C_H4B, linestyle="--",
            label=r"$H \to 4b$", **kw)
    ax.hist(sig_tt[np.isfinite(sig_tt)], color=C_TT,  linestyle="-.",
            label=r"$t\bar{t}$", **kw)
    ax.axvline(cut, color="black", linestyle="--", linewidth=1.4,
               label=f"Oracle cut (bg={R_PLUS}%)")
    ax.set_xscale("log")
    ax.set_xlabel(xlabel, fontsize=LABEL_FS)
    ax.set_ylabel("Normalized density", fontsize=LABEL_FS)
    ax.set_title(title, fontsize=LABEL_FS)
    ax.tick_params(axis="both", labelsize=TICK_FS)
    ax.legend(fontsize=LEGEND_FS - 1, framealpha=0.7)

plot_score_hist(ax_ht, bht, aht, tht, cut_ht,
                r"(a) $H_T$ score", r"$H_T$ [GeV]")
plot_score_hist(ax_as, bas, aas, tas, cut_as,
                r"(b) AD score",    r"AD score")

# ── ROC panel ────────────────────────────────────────────────────────────────
x_max = 0.8
ax_roc.axvspan(TARGET_PCT - TOL_PCT, TARGET_PCT + TOL_PCT,
               alpha=0.18, color="steelblue", label="Tolerance band")
ax_roc.axvline(TARGET_PCT, color="steelblue", linestyle=":", linewidth=1.2)

ax_roc.plot(r_ht_tt, eff_ht_tt, color=C_TT,  linestyle="-",  lw=2.0,
            label=r"$H_T$: $t\bar{t}$")
ax_roc.plot(r_ht_aa, eff_ht_aa, color=C_H4B, linestyle="--", lw=2.0,
            label=r"$H_T$: $H \to 4b$")
ax_roc.plot(r_as_tt, eff_as_tt, color=C_TT,  linestyle="-.", lw=2.0,
            label=r"AD: $t\bar{t}$")
ax_roc.plot(r_as_aa, eff_as_aa, color=C_H4B, linestyle=":",  lw=2.0,
            label=r"AD: $H \to 4b$")

# Oracle operating points — star on the ROC line at r=R_PLUS, text from CSV
def interp_at_rate(r_arr, eff_arr, rate):
    idx = np.argmin(np.abs(r_arr - rate))
    return float(eff_arr[idx])

def annotate_oracle(ax, r_arr, eff_arr, rate, oracle_val, color, label, dy):
    star_y = interp_at_rate(r_arr, eff_arr, rate)
    ax.plot(rate, star_y, "*", color=color, markersize=11, zorder=6,
            markeredgecolor="black", markeredgewidth=0.5)
    ax.annotate(f"{label}: {oracle_val:.1f}%",
                xy=(rate, star_y), xytext=(rate + 0.03, star_y + dy),
                fontsize=ANNOT_FS, color=color,
                arrowprops=dict(arrowstyle="-", color=color, lw=0.8))

annotate_oracle(ax_roc, r_ht_tt, eff_ht_tt, R_PLUS, e_ht_tt_oracle, C_TT,
                r"$H_T$: $t\bar{t}$", dy=-5)
annotate_oracle(ax_roc, r_ht_aa, eff_ht_aa, R_PLUS, e_ht_aa_oracle, C_H4B,
                r"$H_T$: $H\to4b$",   dy=5)
annotate_oracle(ax_roc, r_as_tt, eff_as_tt, R_PLUS, e_as_tt_oracle, C_TT,
                r"AD: $t\bar{t}$",    dy=-20)
annotate_oracle(ax_roc, r_as_aa, eff_as_aa, R_PLUS, e_as_aa_oracle, C_H4B,
                r"AD: $H\to4b$",      dy=5)

ax_roc.text(TARGET_PCT + 0.005, 5,
            f"$r^*={TARGET_PCT}\\%$\ntol=$\\pm{TOL_PCT}\\%$",
            fontsize=LEGEND_FS - 1, color="steelblue", va="bottom")

ax_roc.set_xlim(0, x_max)
ax_roc.set_ylim(0, 105)
ax_roc.set_xlabel(r"Background rate $r$ [%]", fontsize=LABEL_FS)
ax_roc.set_ylabel("Overall Signal Efficiency [%]", fontsize=LABEL_FS)
ax_roc.set_title("(c) ROC at operating point", fontsize=LABEL_FS)
ax_roc.tick_params(axis="both", labelsize=TICK_FS)
ax_roc.legend(fontsize=LEGEND_FS, loc="lower right", framealpha=0.8, ncol=2)


OUT.parent.mkdir(parents=True, exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(f"{OUT}.{ext}", dpi=150, bbox_inches="tight")
    print(f"Saved {OUT}.{ext}")

plt.close(fig)
