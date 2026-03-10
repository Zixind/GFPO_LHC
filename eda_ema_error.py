#!/usr/bin/env python
"""
EDA: Exponential Moving Average (EMA) of rate error — why the agent needs it.

The instantaneous rate error is noisy window-to-window.  The EMA smooths
this noise and reveals sustained drifts and regime changes that the raw
signal obscures.  We compute the rate at each window's own 99.75th
percentile (no fixed threshold), then show raw error vs EMA at the
actual λ=0.95 used in training, plus two comparison values.

Usage
-----
    conda run -n adaptive python eda_ema_error.py
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

CHUNK  = 50_000
MICRO  = 5_000
TARGET = 0.25   # target rate (%)

print("Loading data …")
with h5py.File(H5, "r") as f:
    Bht = f["mc_bkg_ht"][:].astype(np.float32)
    Bas = f["mc_bkg_score02"][:].astype(np.float32)

N = len(Bht)
starts = np.arange(0, N - CHUNK, MICRO)
n_steps = len(starts)

# Calibration: fix the threshold from the first window
cal = slice(0, CHUNK)
fixed_HT = float(np.percentile(Bht[cal], 99.75))
fixed_AS = float(np.percentile(Bas[cal], 99.75))

# Raw instantaneous error
err_ht = np.zeros(n_steps)
err_as = np.zeros(n_steps)

print(f"Computing rate errors across {n_steps} windows …")
for i, s in enumerate(starts):
    sl = slice(s, s + CHUNK)
    rate_ht = float(Sing_Trigger(Bht[sl], fixed_HT))
    rate_as = float(Sing_Trigger(Bas[sl], fixed_AS))
    err_ht[i] = (rate_ht - TARGET) / max(TARGET, 1e-6)
    err_as[i] = (rate_as - TARGET) / max(TARGET, 1e-6)
    if (i + 1) % 200 == 0:
        print(f"  {i+1}/{n_steps}")


def ema(x, lam):
    """EMA: y_t = λ * y_{t-1} + (1-λ) * x_t"""
    out = np.zeros_like(x)
    out[0] = x[0]
    for t in range(1, len(x)):
        out[t] = lam * out[t - 1] + (1.0 - lam) * x[t]
    return out


LAMBDAS = [0.8, 0.95, 0.99]
COLORS  = ["C2", "C1", "C4"]

t_frac = (starts + CHUNK / 2) / float(N)

# ═══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(18, 10))

TOL_FRAC = 0.025 / TARGET   # ±0.1 in fractional units (±10 kHz / 100 kHz)

for row, (err, label) in enumerate([(err_ht, r"$H_T$"), (err_as, "AD")]):
    # Left: raw vs EMA overlay
    ax = axes[row, 0]
    ax.plot(t_frac, err, lw=0.3, color="C0", alpha=0.4, label="Instantaneous error")
    for lam, c in zip(LAMBDAS, COLORS):
        y = ema(err, lam)
        ax.plot(t_frac, y, lw=1.2, color=c, label=f"EMA ($\\lambda$={lam})")
    ax.axhline(0, ls=":", color="grey")
    ax.fill_between(t_frac, -TOL_FRAC, TOL_FRAC, alpha=0.12,
                    label=f"$\\pm${TOL_FRAC:.1f} tolerance (10 kHz)")
    ax.set_ylabel("Rate error (fractional)")
    ax.set_title(f"{label}: raw error vs EMA at different $\\lambda$")
    ax.legend(fontsize=15, loc="upper right")
    if row == 1:
        ax.set_xlabel("Time (Fraction of Run)")

    # Right: zoom into a transition region (first 20% of run where drift is fastest)
    ax = axes[row, 1]
    mask = t_frac <= 0.20
    ax.plot(t_frac[mask], err[mask], lw=0.4, color="C0", alpha=0.4,
            label="Instantaneous error")
    for lam, c in zip(LAMBDAS, COLORS):
        y = ema(err, lam)
        ax.plot(t_frac[mask], y[mask], lw=1.5, color=c,
                label=f"EMA ($\\lambda$={lam})")
    ax.axhline(0, ls=":", color="grey")
    ax.fill_between(t_frac[mask], -TOL_FRAC, TOL_FRAC, alpha=0.12,
                    label=f"$\\pm${TOL_FRAC:.1f} tolerance (10 kHz)")
    ax.set_ylabel("Rate error (fractional)")
    ax.set_title(f"{label}: zoom into first 20% — EMA tracks regime shift")
    ax.legend(fontsize=15, loc="upper right")
    if row == 1:
        ax.set_xlabel("Time (Fraction of Run)")

fig.tight_layout()
out = ROOT / "outputs" / "eda_ema_error.pdf"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
plt.close(fig)
