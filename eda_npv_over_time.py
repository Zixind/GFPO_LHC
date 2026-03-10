#!/usr/bin/env python
"""
EDA: Pileup (NPV) mean and standard deviation over time for MC background.

Usage
-----
    conda run -n adaptive python eda_npv_over_time.py
"""

import pathlib
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
    Bnpv = f["mc_bkg_Npv"][:].astype(np.float32)

N = len(Bnpv)
starts = np.arange(0, N - CHUNK, MICRO)
n_steps = len(starts)

npv_mean = np.zeros(n_steps)
npv_std  = np.zeros(n_steps)

print(f"Computing NPV statistics across {n_steps} windows …")
for i, s in enumerate(starts):
    npv_j = Bnpv[s : s + CHUNK]
    npv_mean[i] = np.mean(npv_j)
    npv_std[i]  = np.std(npv_j)

t_frac = (starts + CHUNK / 2) / float(N)

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

ax = axes[0]
ax.plot(t_frac, npv_mean, lw=0.7, color="C0")
ax.set_xlabel("Time (Fraction of Run)")
ax.set_ylabel(r"$NPV_{\mu}$")

ax = axes[1]
ax.plot(t_frac, npv_std, lw=0.7, color="C3")
ax.set_xlabel("Time (Fraction of Run)")
ax.set_ylabel(r"$NPV_{\sigma}$")

fig.tight_layout()
out = ROOT / "outputs" / "eda_npv_over_time.pdf"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
plt.close(fig)
