#!/usr/bin/env python
"""
EDA: Normalized threshold — why the agent needs to know where its cut sits.

The normalized threshold c_norm = (c - mid) / span tells the agent where
the current cut lies relative to the score distribution's calibration range.
As pileup changes, the "ideal" cut (99.75th percentile) drifts, so the
same raw cut value means very different things at different times.

We show:
  Row 1: The same raw threshold maps to completely different positions
         within the score distribution at different times.
  Row 2: The normalized ideal threshold drifts far from the fixed value,
         meaning the agent must continuously re-assess where its cut sits.

Usage
-----
    conda run -n adaptive python eda_normalized_cut.py
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

print("Loading data …")
with h5py.File(H5, "r") as f:
    Bas = f["mc_bkg_score02"][:].astype(np.float32)
    Bht = f["mc_bkg_ht"][:].astype(np.float32)

N = len(Bas)

# Calibration from first chunk (matches training)
cal = slice(0, CHUNK)

as_lo, as_hi = float(np.percentile(Bas[cal], 95.0)), float(np.percentile(Bas[cal], 99.99))
as_mid = 0.5 * (as_lo + as_hi)
as_span = max(1e-6, as_hi - as_lo)

ht_lo = float(np.percentile(Bht[cal], 95.0))
ht_hi = float(np.percentile(Bht[cal], 99.99))
ht_mid = 0.5 * (ht_lo + ht_hi)
ht_span = max(1.0, ht_hi - ht_lo)

# Fixed cut from calibration
fixed_AS = float(np.percentile(Bas[cal], 99.75))
fixed_HT = float(np.percentile(Bht[cal], 99.75))

starts = np.arange(0, N - CHUNK, MICRO)
n_steps = len(starts)

# Per-window statistics
ideal_as      = np.zeros(n_steps)
ideal_ht      = np.zeros(n_steps)
ideal_as_norm = np.zeros(n_steps)
ideal_ht_norm = np.zeros(n_steps)
# What percentile does the fixed cut correspond to in each window?
fixed_as_pctl = np.zeros(n_steps)
fixed_ht_pctl = np.zeros(n_steps)
# Per-window local mid and span (to show they drift)
local_ht_mid  = np.zeros(n_steps)
local_as_mid  = np.zeros(n_steps)
local_ht_span = np.zeros(n_steps)
local_as_span = np.zeros(n_steps)

print(f"Computing thresholds across {n_steps} windows …")
for i, s in enumerate(starts):
    sl = slice(s, s + CHUNK)
    bas_j = Bas[sl]
    bht_j = Bht[sl]

    ideal_as[i] = float(np.percentile(bas_j, 99.75))
    ideal_ht[i] = float(np.percentile(bht_j, 99.75))
    ideal_as_norm[i] = (ideal_as[i] - as_mid) / as_span
    ideal_ht_norm[i] = (ideal_ht[i] - ht_mid) / ht_span

    # What percentile is the fixed cut in this window's distribution?
    fixed_ht_pctl[i] = 100.0 * np.mean(bht_j <= fixed_HT)
    fixed_as_pctl[i] = 100.0 * np.mean(bas_j <= fixed_AS)

    # Local mid/span
    lo_ht = float(np.percentile(bht_j, 95.0))
    hi_ht = float(np.percentile(bht_j, 99.99))
    local_ht_mid[i]  = 0.5 * (lo_ht + hi_ht)
    local_ht_span[i] = max(1.0, hi_ht - lo_ht)

    lo_as = float(np.percentile(bas_j, 95.0))
    hi_as = float(np.percentile(bas_j, 99.99))
    local_as_mid[i]  = 0.5 * (lo_as + hi_as)
    local_as_span[i] = max(1e-6, hi_as - lo_as)

    if (i + 1) % 200 == 0:
        print(f"  {i+1}/{n_steps}")

print("Done.")

fixed_as_norm = (fixed_AS - as_mid) / as_span
fixed_ht_norm = (fixed_HT - ht_mid) / ht_span

t_frac = (starts + CHUNK / 2) / float(N)

# ═══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(18, 11))

# --- Row 0: Fixed cut's percentile rank drifts ---
ax = axes[0, 0]
ax.plot(t_frac, fixed_ht_pctl, lw=0.6, color="C0")
ax.axhline(99.75, ls="--", color="C3", lw=1.5, label="Initial cut (99.75th pctl at calibration)")
ax.set_ylabel("Percentile rank of fixed cut")
ax.set_title(r"$H_T$: background rate of same raw threshold")
ax.legend(fontsize=15)

ax = axes[0, 1]
ax.plot(t_frac, fixed_as_pctl, lw=0.6, color="C0")
ax.axhline(99.75, ls="--", color="C3", lw=1.5, label="Initial cut (99.75th pctl at calibration)")
ax.set_ylabel("Percentile rank of fixed cut")
ax.set_title("AD: background rate of same raw threshold")
ax.legend(fontsize=15)

# --- Row 1: Normalized ideal threshold drifts ---
ax = axes[1, 0]
ax.plot(t_frac, ideal_ht_norm, lw=0.7, color="C0", label="Ideal $c_{\\mathrm{norm}}$")
ax.axhline(fixed_ht_norm, ls="--", color="C3", lw=1.5,
           label=f"Fixed $c_{{\\mathrm{{norm}}}}$ = {fixed_ht_norm:.2f}")
ax.set_ylabel(r"$c_{\mathrm{normalized}} = (c - \mathrm{mid}) / \mathrm{span}$")
ax.set_xlabel("Time (Fraction of Run)")
ax.set_title(r"$H_T$: normalized threshold as $c_{\mathrm{normalized}}$")
ax.legend(fontsize=15, loc="upper right")

ax = axes[1, 1]
ax.plot(t_frac, ideal_as_norm, lw=0.7, color="C0", label="Ideal $c_{\\mathrm{norm}}$")
ax.axhline(fixed_as_norm, ls="--", color="C3", lw=1.5,
           label=f"Fixed $c_{{\\mathrm{{norm}}}}$ = {fixed_as_norm:.2f}")
ax.set_ylabel(r"$c_{\mathrm{normalized}} = (c - \mathrm{mid}) / \mathrm{span}$")
ax.set_xlabel("Time (Fraction of Run)")
ax.set_title(r"AD: normalized threshold as $c_{\mathrm{normalized}}$")
ax.legend(fontsize=15, loc="upper right")

fig.tight_layout()
out = ROOT / "outputs" / "eda_normalized_cut.pdf"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
plt.close(fig)
