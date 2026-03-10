#!/usr/bin/env python
"""
EDA: Near-cut binary indicators — why the agent needs them.

For each sliding window, we compute the fraction of events within fixed
distances of the operating point (99.75th percentile of the current window).
This shows that the density of events near the decision boundary is
non-stationary and varies substantially over time.

Usage
-----
    conda run -n adaptive python eda_nearcut.py
"""

import sys, pathlib
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("CMS")

ROOT = pathlib.Path(__file__).resolve().parent
H5 = ROOT / "Data" / "Trigger_food_MC.h5"

CHUNK = 50_000
MICRO = 5_000

HT_WIDTHS = [5.0, 10.0, 20.0]       # GeV
AS_WIDTHS = [0.25, 0.5, 1.0]

print("Loading data …")
with h5py.File(H5, "r") as f:
    Bas  = f["mc_bkg_score02"][:].astype(np.float32)
    Bht  = f["mc_bkg_ht"][:].astype(np.float32)

N = len(Bas)
starts = np.arange(0, N - CHUNK, MICRO)
n_steps = len(starts)

near_ht = np.zeros((n_steps, len(HT_WIDTHS)))
near_as = np.zeros((n_steps, len(AS_WIDTHS)))
cut_ht  = np.zeros(n_steps)
cut_as  = np.zeros(n_steps)

print(f"Computing near-cut fractions across {n_steps} windows …")
for i, s in enumerate(starts):
    sl = slice(s, s + CHUNK)
    bht_j = Bht[sl]
    bas_j = Bas[sl]

    # Operating point: each window's own 99.75th percentile
    c_ht = float(np.percentile(bht_j, 99.75))
    c_as = float(np.percentile(bas_j, 99.75))
    cut_ht[i] = c_ht
    cut_as[i] = c_as

    for wi, w in enumerate(HT_WIDTHS):
        near_ht[i, wi] = np.mean(np.abs(bht_j - c_ht) <= w)
    for wi, w in enumerate(AS_WIDTHS):
        near_as[i, wi] = np.mean(np.abs(bas_j - c_as) <= w)

    if (i + 1) % 200 == 0:
        print(f"  {i+1}/{n_steps}")

print("Done.")

t_frac = (starts + CHUNK / 2) / float(N)

# ═══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# --- Left: HT near-cut occupancy ---
ax = axes[0]
for wi, w in enumerate(HT_WIDTHS):
    ax.plot(t_frac, near_ht[:, wi], lw=0.6, label=f"|score $-$ c| $\\leq$ {w:.0f} GeV")
ax.set_xlabel("Time (Fraction of Run)")
ax.set_ylabel("Fraction of events near cut")
ax.set_title(r"$H_T$ near-cut occupancy over time")
ax.legend(loc="upper left", framealpha=0.9)

# --- Right: AD near-cut occupancy ---
ax = axes[1]
for wi, w in enumerate(AS_WIDTHS):
    ax.plot(t_frac, near_as[:, wi], lw=0.6, label=f"|score $-$ c| $\\leq$ {w}")
ax.set_xlabel("Time (Fraction of Run)")
ax.set_ylabel("Fraction of events near cut")
ax.set_title("AD near-cut occupancy over time")
ax.legend(loc="upper left", framealpha=0.9)

fig.subplots_adjust(left=0.08, right=0.97, wspace=0.3)
out = ROOT / "outputs" / "eda_nearcut.pdf"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
plt.close(fig)
