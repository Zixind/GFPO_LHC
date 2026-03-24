#!/usr/bin/env python
"""
EDA: Sensitivity probe — why the agent needs local gradient information.

The sensitivity probe d r/d c / r* measures how steeply the background rate
changes per unit threshold shift, normalized by the target rate.  We compute
it at each window's own 99.75th percentile (the natural operating point),
so this reflects an intrinsic property of the score distribution's tail shape
— not a fixed-threshold artifact.

Usage
-----
    conda run -n adaptive python eda_sensitivity_probe.py
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
sys.path.insert(0, str(ROOT))
from triggers import Sing_Trigger

H5 = ROOT / "Data" / "Trigger_food_MC.h5"

CHUNK   = 50_000
MICRO   = 5_000
TARGET  = 0.25       # target rate (%)
AS_STEP = 0.5
HT_STEP = 1.0

print("Loading data …")
with h5py.File(H5, "r") as f:
    Bas  = f["mc_bkg_score02"][:].astype(np.float32)
    Bht  = f["mc_bkg_ht"][:].astype(np.float32)

N = len(Bas)
starts = np.arange(0, N - CHUNK, MICRO)
n_steps = len(starts)

# Sensitivity at each window's own 99.75th percentile
sens_as = np.zeros(n_steps)
sens_ht = np.zeros(n_steps)
cut_as  = np.zeros(n_steps)
cut_ht  = np.zeros(n_steps)
rate_as = np.zeros(n_steps)
rate_ht = np.zeros(n_steps)

print(f"Computing sensitivity across {n_steps} windows …")
for i, s in enumerate(starts):
    sl = slice(s, s + CHUNK)
    bas_j = Bas[sl]
    bht_j = Bht[sl]

    # Operating point: each window's own 99.75th percentile
    c_as = float(np.percentile(bas_j, 99.75))
    c_ht = float(np.percentile(bht_j, 99.75))
    cut_as[i] = c_as
    cut_ht[i] = c_ht

    # Rate at operating point
    rate_as[i] = float(Sing_Trigger(bas_j, c_as))
    rate_ht[i] = float(Sing_Trigger(bht_j, c_ht))

    # Central-difference sensitivity: (r(c+Δ) - r(c-Δ)) / (2Δ) / r*
    rp_as = float(Sing_Trigger(bas_j, c_as + AS_STEP))
    rm_as = float(Sing_Trigger(bas_j, c_as - AS_STEP))
    sens_as[i] = ((rp_as - rm_as) / (2 * AS_STEP)) / max(TARGET, 1e-6)

    rp_ht = float(Sing_Trigger(bht_j, c_ht + HT_STEP))
    rm_ht = float(Sing_Trigger(bht_j, c_ht - HT_STEP))
    sens_ht[i] = ((rp_ht - rm_ht) / (2 * HT_STEP)) / max(TARGET, 1e-6)

    if (i + 1) % 200 == 0:
        print(f"  {i+1}/{n_steps}")

print("Done.")

t_frac = (starts + CHUNK / 2) / float(N)

# ═══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# --- Left: HT sensitivity over time ---
ax = axes[0]
ax.plot(t_frac, sens_ht, lw=0.5, color="C3")
ax.axhline(0, ls=":", color="grey")
ax.set_ylabel(r"$\boldsymbol{\frac{\partial\,r / \partial\,c}{r_{B}^{*}}}$")
ax.set_xlabel("Time (Fraction of Run)")
ax.set_title(r"$H_T$ sensitivity probe over time")

# --- Right: AD sensitivity over time ---
ax = axes[1]
ax.plot(t_frac, sens_as, lw=0.5, color="C0")
ax.axhline(0, ls=":", color="grey")
ax.set_ylabel(r"$\boldsymbol{\frac{\partial\,r / \partial\,c}{r_{B}^{*}}}$")
ax.set_xlabel("Time (Fraction of Run)")
ax.set_title("AD sensitivity probe over time")

fig.tight_layout()
out = ROOT / "outputs" / "eda_sensitivity_probe.pdf"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
plt.close(fig)
