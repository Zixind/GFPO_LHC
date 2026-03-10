#!/usr/bin/env python
"""
Exploratory Data Analysis: How the underlying data distributions evolve over time.

Produces multi-panel figures showing that the raw quantities entering the
RL state representation are **non-stationary** — no threshold is imposed.
This justifies the need for an adaptive agent and motivates each feature group.

Usage
-----
    conda run -n adaptive python eda_state_features.py
"""

import sys, pathlib
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("CMS")

# ── paths ────────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parent
H5 = ROOT / "Data" / "Trigger_food_MC.h5"

# ── hyperparams ──────────────────────────────────────────────────────────
CHUNK = 50_000          # sliding window size
MICRO = 5_000           # stride between windows

# ── load data ────────────────────────────────────────────────────────────
print("Loading data …")
with h5py.File(H5, "r") as f:
    Bas  = f["mc_bkg_score02"][:].astype(np.float32)
    Bht  = f["mc_bkg_ht"][:].astype(np.float32)
    Bnpv = f["mc_bkg_Npv"][:].astype(np.float32)

N = len(Bas)
print(f"  {N:,} background events loaded.")

# ── sweep sliding windows ────────────────────────────────────────────────
starts = np.arange(0, N - CHUNK, MICRO)
n_steps = len(starts)

# Storage — pure distributional statistics, no threshold
npv_mean   = np.zeros(n_steps)
npv_std    = np.zeros(n_steps)
npv_median = np.zeros(n_steps)

as_mean    = np.zeros(n_steps)
as_std     = np.zeros(n_steps)
as_p90     = np.zeros(n_steps)
as_p95     = np.zeros(n_steps)
as_p99     = np.zeros(n_steps)
as_p9975   = np.zeros(n_steps)
as_skew    = np.zeros(n_steps)
as_iqr     = np.zeros(n_steps)

ht_mean    = np.zeros(n_steps)
ht_std     = np.zeros(n_steps)
ht_p90     = np.zeros(n_steps)
ht_p95     = np.zeros(n_steps)
ht_p99     = np.zeros(n_steps)
ht_p9975   = np.zeros(n_steps)
ht_skew    = np.zeros(n_steps)
ht_iqr     = np.zeros(n_steps)

# correlation between NPV and scores (per window)
corr_npv_as = np.zeros(n_steps)
corr_npv_ht = np.zeros(n_steps)

print(f"Computing distributional statistics across {n_steps} windows …")
for i, s in enumerate(starts):
    sl = slice(s, s + CHUNK)
    bas_j = Bas[sl]
    bht_j = Bht[sl]
    npv_j = Bnpv[sl]

    # ── NPV ──
    npv_mean[i]   = np.mean(npv_j)
    npv_std[i]    = np.std(npv_j)
    npv_median[i] = np.median(npv_j)

    # ── AD score distribution ──
    as_mean[i] = np.mean(bas_j)
    as_std[i]  = np.std(bas_j)
    p25, p75 = np.percentile(bas_j, [25, 75])
    as_iqr[i] = p75 - p25
    as_p90[i]   = np.percentile(bas_j, 90)
    as_p95[i]   = np.percentile(bas_j, 95)
    as_p99[i]   = np.percentile(bas_j, 99)
    as_p9975[i] = np.percentile(bas_j, 99.75)
    mu = as_mean[i]
    sd = max(as_std[i], 1e-8)
    as_skew[i] = float(np.mean(((bas_j - mu) / sd) ** 3))

    # ── HT score distribution ──
    ht_mean[i] = np.mean(bht_j)
    ht_std[i]  = np.std(bht_j)
    p25h, p75h = np.percentile(bht_j, [25, 75])
    ht_iqr[i] = p75h - p25h
    ht_p90[i]   = np.percentile(bht_j, 90)
    ht_p95[i]   = np.percentile(bht_j, 95)
    ht_p99[i]   = np.percentile(bht_j, 99)
    ht_p9975[i] = np.percentile(bht_j, 99.75)
    mu_h = ht_mean[i]
    sd_h = max(ht_std[i], 1e-8)
    ht_skew[i] = float(np.mean(((bht_j - mu_h) / sd_h) ** 3))

    # ── NPV–score correlation ──
    if npv_std[i] > 1e-8:
        corr_npv_as[i] = np.corrcoef(npv_j, bas_j)[0, 1]
        corr_npv_ht[i] = np.corrcoef(npv_j, bht_j)[0, 1]

    if (i + 1) % 200 == 0:
        print(f"  {i+1}/{n_steps}")

print("Done computing.")

# ── time axis ────────────────────────────────────────────────────────────
t_frac = (starts + CHUNK / 2) / float(N)

# ═══════════════════════════════════════════════════════════════════════════
#  FIGURE 1 — Raw distributional features evolve over time (5×2 panels)
# ═══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(5, 2, figsize=(18, 19), sharex=True)
fig.suptitle(
    "Underlying data distributions are non-stationary\n"
    "(50 k-event sliding window, no threshold imposed)",
    fontsize=18, y=0.995,
)

# --- Row 0: NPV ---
ax = axes[0, 0]
ax.plot(t_frac, npv_mean, lw=0.6, color="C1", label=r"$\langle NPV \rangle$")
ax.fill_between(t_frac, npv_mean - npv_std, npv_mean + npv_std,
                alpha=0.15, color="C1", label=r"$\pm 1\sigma$")
ax.set_ylabel("NPV")
ax.legend(fontsize=9)
ax.set_title("Pileup (NPV) drifts over time", fontsize=12)

ax = axes[0, 1]
ax.plot(t_frac, npv_std, lw=0.6, color="C2", label=r"$\sigma_{\mathrm{NPV}}$")
ax.set_ylabel("Std NPV")
ax.legend(fontsize=9)
ax.set_title("Pileup spread changes over time", fontsize=12)

# --- Row 1: AD score percentiles ---
ax = axes[1, 0]
ax.plot(t_frac, as_mean, lw=0.6, color="C0", label="Mean")
ax.plot(t_frac, as_p90,  lw=0.6, color="C3", label="90th pctl")
ax.plot(t_frac, as_p95,  lw=0.6, color="C4", label="95th pctl")
ax.set_ylabel("AD score")
ax.legend(fontsize=9)
ax.set_title("AD score distribution shifts over time", fontsize=12)

ax = axes[1, 1]
ax.plot(t_frac, as_p99,   lw=0.6, color="C5", label="99th pctl")
ax.plot(t_frac, as_p9975, lw=0.6, color="C6", label="99.75th pctl")
ax.set_ylabel("AD score (upper tail)")
ax.legend(fontsize=9)
ax.set_title("AD upper tail — where triggers operate — drifts", fontsize=12)

# --- Row 2: HT score percentiles ---
ax = axes[2, 0]
ax.plot(t_frac, ht_mean, lw=0.6, color="C0", label="Mean")
ax.plot(t_frac, ht_p90,  lw=0.6, color="C3", label="90th pctl")
ax.plot(t_frac, ht_p95,  lw=0.6, color="C4", label="95th pctl")
ax.set_ylabel(r"$H_T$ (GeV)")
ax.legend(fontsize=9)
ax.set_title(r"$H_T$ score distribution shifts over time", fontsize=12)

ax = axes[2, 1]
ax.plot(t_frac, ht_p99,   lw=0.6, color="C5", label="99th pctl")
ax.plot(t_frac, ht_p9975, lw=0.6, color="C6", label="99.75th pctl")
ax.set_ylabel(r"$H_T$ (GeV, upper tail)")
ax.legend(fontsize=9)
ax.set_title(r"$H_T$ upper tail — where triggers operate — drifts", fontsize=12)

# --- Row 3: distribution shape (std, IQR, skewness) ---
ax = axes[3, 0]
ax.plot(t_frac, as_std, lw=0.6, color="C7", label="AD std")
ax2 = ax.twinx()
ax2.plot(t_frac, as_iqr, lw=0.6, color="C8", ls="--", label="AD IQR")
ax.set_ylabel("AD score std", color="C7")
ax2.set_ylabel("AD score IQR", color="C8")
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9)
ax.set_title("AD distribution width changes", fontsize=12)

ax = axes[3, 1]
ax.plot(t_frac, ht_std, lw=0.6, color="C7", label=r"$H_T$ std")
ax2 = ax.twinx()
ax2.plot(t_frac, ht_iqr, lw=0.6, color="C8", ls="--", label=r"$H_T$ IQR")
ax.set_ylabel(r"$H_T$ std (GeV)", color="C7")
ax2.set_ylabel(r"$H_T$ IQR (GeV)", color="C8")
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9)
ax.set_title(r"$H_T$ distribution width changes", fontsize=12)

# --- Row 4: skewness ---
ax = axes[4, 0]
ax.plot(t_frac, as_skew, lw=0.6, color="C9")
ax.axhline(0, ls=":", color="grey")
ax.set_ylabel("Skewness")
ax.set_xlabel("Time (Fraction of Run)")
ax.set_title("AD skewness evolves — tail shape is non-stationary", fontsize=12)

ax = axes[4, 1]
ax.plot(t_frac, ht_skew, lw=0.6, color="C9")
ax.axhline(0, ls=":", color="grey")
ax.set_ylabel("Skewness")
ax.set_xlabel("Time (Fraction of Run)")
ax.set_title(r"$H_T$ skewness evolves — tail shape is non-stationary", fontsize=12)

fig.tight_layout(rect=[0, 0, 1, 0.97])
out1 = ROOT / "outputs" / "eda_state_features_time.pdf"
out1.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out1, dpi=150, bbox_inches="tight")
print(f"Saved → {out1}")
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════
#  FIGURE 2 — Score percentile drift rate (justifies drift/velocity features)
# ═══════════════════════════════════════════════════════════════════════════
# Rate of change of the 99.75th percentile (where triggers live)
d_as_p9975 = np.diff(as_p9975, prepend=as_p9975[0])
d_ht_p9975 = np.diff(ht_p9975, prepend=ht_p9975[0])

fig2, axes2 = plt.subplots(2, 2, figsize=(18, 9), sharex=True)
fig2.suptitle(
    "Score-distribution drift velocity — temporal structure the agent must track",
    fontsize=16,
)

ax = axes2[0, 0]
ax.plot(t_frac, d_as_p9975, lw=0.5, color="C3")
ax.axhline(0, ls=":", color="grey")
ax.set_ylabel(r"$\Delta$ (99.75th pctl)")
ax.set_title("AD tail drift velocity", fontsize=12)

ax = axes2[0, 1]
ax.plot(t_frac, d_ht_p9975, lw=0.5, color="C4")
ax.axhline(0, ls=":", color="grey")
ax.set_ylabel(r"$\Delta$ (99.75th pctl)")
ax.set_title(r"$H_T$ tail drift velocity", fontsize=12)

ax = axes2[1, 0]
# Cumulative drift of 99.75th percentile from its initial value
ax.plot(t_frac, as_p9975 - as_p9975[0], lw=0.6, color="C0")
ax.axhline(0, ls=":", color="grey")
ax.set_ylabel("Cumulative shift")
ax.set_xlabel("Time (Fraction of Run)")
ax.set_title("AD: cumulative 99.75th percentile shift", fontsize=12)

ax = axes2[1, 1]
ax.plot(t_frac, ht_p9975 - ht_p9975[0], lw=0.6, color="C0")
ax.axhline(0, ls=":", color="grey")
ax.set_ylabel("Cumulative shift (GeV)")
ax.set_xlabel("Time (Fraction of Run)")
ax.set_title(r"$H_T$: cumulative 99.75th percentile shift", fontsize=12)

fig2.tight_layout(rect=[0, 0, 1, 0.96])
out2 = ROOT / "outputs" / "eda_state_features_drift.pdf"
fig2.savefig(out2, dpi=150, bbox_inches="tight")
print(f"Saved → {out2}")
plt.close(fig2)

# ═══════════════════════════════════════════════════════════════════════════
#  FIGURE 3 — NPV correlates with scores (justifies NPV as leading indicator)
# ═══════════════════════════════════════════════════════════════════════════
fig3, axes3 = plt.subplots(2, 2, figsize=(18, 10))
fig3.suptitle("Pileup (NPV) drives score-distribution variation", fontsize=16)

# Top row: scatter of NPV vs upper-tail percentile
ax = axes3[0, 0]
ax.scatter(npv_mean, as_p9975, s=2, alpha=0.3, c=t_frac, cmap="viridis")
ax.set_xlabel(r"$\langle NPV \rangle$")
ax.set_ylabel("AD 99.75th percentile")
ax.set_title("AD tail rises with pileup", fontsize=12)

ax = axes3[0, 1]
sc = ax.scatter(npv_mean, ht_p9975, s=2, alpha=0.3, c=t_frac, cmap="viridis")
ax.set_xlabel(r"$\langle NPV \rangle$")
ax.set_ylabel(r"$H_T$ 99.75th percentile (GeV)")
ax.set_title(r"$H_T$ tail rises with pileup", fontsize=12)
fig3.colorbar(sc, ax=axes3[0, 1], label="Fraction of Run")

# Bottom row: NPV–score correlation coefficient over time
ax = axes3[1, 0]
ax.plot(t_frac, corr_npv_as, lw=0.6, color="C1")
ax.axhline(0, ls=":", color="grey")
ax.set_ylabel(r"Pearson $\rho$(NPV, AD score)")
ax.set_xlabel("Time (Fraction of Run)")
ax.set_title("NPV–AD correlation over time", fontsize=12)

ax = axes3[1, 1]
ax.plot(t_frac, corr_npv_ht, lw=0.6, color="C2")
ax.axhline(0, ls=":", color="grey")
ax.set_ylabel(r"Pearson $\rho$(NPV, $H_T$)")
ax.set_xlabel("Time (Fraction of Run)")
ax.set_title(r"NPV–$H_T$ correlation over time", fontsize=12)

fig3.tight_layout(rect=[0, 0, 1, 0.96])
out3 = ROOT / "outputs" / "eda_state_features_npv_correlation.pdf"
fig3.savefig(out3, dpi=150, bbox_inches="tight")
print(f"Saved → {out3}")
plt.close(fig3)

print("\nAll EDA plots saved to outputs/")
