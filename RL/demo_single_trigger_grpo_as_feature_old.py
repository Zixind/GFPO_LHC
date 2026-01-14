#!/usr/bin/env python3
"""
[DEPRECATED] demo_single_trigger_grpo_as_feature_old.py

Single-trigger threshold control with event-sequence features. 
This is non-sliding window. Chunk size is non overlapping for plotting. Micor windows inside each chunk still non-overlapping.

Main focus: AD/AS trigger control (AS_cut) comparing:
  - Constant menu threshold (fixed from calibration window)
  - PD baseline (uses PD_controller2 for AS)
  - DQN baseline (sequence DQN; epsilon-greedy)
  - GRPO (bandit-style group sampling + policy update)

Optional: also run the HT trigger control (Ht_cut) when --run-ht is set:
  - PD baseline for HT uses PD_controller1 for HT specifically
  - DQN baseline (sequence DQN)
  - GRPO (bandit-style)

Create summary table for paper:
1) Compact summary table (CSV + LaTeX) with:
   - InBand (↑), MAE (↓), P95|e| (↓), ViolMag (↓), StepRMS (↓),
     TT_inband (↑), AA_inband (↑), Mix80_20 (↑)
2) CDF of |rate error| (kHz) for PD vs GRPO
3) Running in-band fraction vs time (PD vs GRPO)
4) Cut-step magnitude histogram |Δcut| (PD vs GRPO)
5) In-band efficiency bars (PD vs GRPO)

Notes:
- Rates are in *percent units* from Sing_Trigger: target r* = 0.25 (%).
- Convert to kHz via r_kHz = 400 * r_%.
- If your tolerance band is [90,110] kHz around 100 kHz, use tol=0.025 (%).
- Controller mapping (IMPORTANT):
    * AS_cut (AD trigger) PD baseline  -> PD_controller2
    * Ht_cut (HT trigger) PD baseline  -> PD_controller1
"""

import argparse
import random
import csv
import numpy as np
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import matplotlib.pyplot as plt
from controllers import PD_controller1, PD_controller2
from triggers import Sing_Trigger
from RL.utils import add_cms_header, save_png, print_h5_tree, read_any_h5, cummean, rel_to_t0, near_occupancy, style_diag_axes, style_diag_legend, finalize_diag_fig, apply_paper_style, plot_inband_eff_single_signal_ad_vs_ht
from RL.grpo_agent import GRPOAgent, GRPOConfig, GRPORewardCfg #GRPO agent
from RL.dqn_agent import SeqDQNAgent, DQNConfig  # DQN agent
from RL.dqn_agent import make_event_seq_as_v0, make_event_seq_ht_v0, shield_delta

SEED = 20251221
random.seed(SEED)
np.random.seed(SEED)

RATE_SCALE_KHZ = 400.0

import mplhep as hep
hep.style.use("CMS")

from RL.utils import apply_paper_style
apply_paper_style()

@dataclass
class RollingWindow:
    def __init__(self, max_events: int):
        self.max_events = int(max_events)
        self._bas = deque(maxlen=self.max_events)
        self._bnpv = deque(maxlen=self.max_events)

    def append(self, bas, bnpv):
        self._bas.extend(np.asarray(bas, dtype=np.float32).tolist())
        self._bnpv.extend(np.asarray(bnpv, dtype=np.float32).tolist())

    def get(self):
        return (
            np.fromiter(self._bas, dtype=np.float32),
            np.fromiter(self._bnpv, dtype=np.float32),
        )

# This is for HT
@dataclass
class RollingWindowHT:
    def __init__(self, max_events: int):
        self.max_events = int(max_events)
        self._bht = deque(maxlen=self.max_events)
        self._bnpv = deque(maxlen=self.max_events)

    def append(self, bht, bnpv):
        self._bht.extend(np.asarray(bht, dtype=np.float32).tolist())
        self._bnpv.extend(np.asarray(bnpv, dtype=np.float32).tolist())

    def get(self):
        return (
            np.fromiter(self._bht, dtype=np.float32),
            np.fromiter(self._bnpv, dtype=np.float32),
        )


# ----------------------------- metrics helpers -----------------------------
def _group_advantages_from_grpo_samples(grpo_samples, *, trigger="AS", baseline="mean", eps=1e-8):
    """
    Reconstruct GRPO advantages from logged candidate rewards.

    Returns:
      adv_raw_all:   list of (r - baseline)
      adv_norm_all:  list of (r - baseline) / std
      adv_raw_exec:  list of executed (r_exec - baseline_of_candidates)
      adv_norm_exec: list of executed (r_exec - baseline) / std
      frac_vanish:   fraction of groups with std ~ 0 (vanishing-adv groups)
    """
    # Filter rows for this trigger
    rows = [r for r in grpo_samples if r.get("trigger") == trigger]

    # Group candidate rewards by micro-step
    cand_by_micro = {}
    exec_by_micro = {}  # store executed row (reward_exec)
    for r in rows:
        micro = int(r["micro"])
        if r["phase"] == "candidate":
            rr = r.get("reward_raw", None)
            if rr is None:
                continue
            cand_by_micro.setdefault(micro, []).append(float(rr))
        elif r["phase"] == "executed":
            # executed reward is stored in reward_exec
            re = r.get("reward_exec", None)
            if re is None:
                continue
            exec_by_micro[micro] = float(re)

    adv_raw_all, adv_norm_all = [], []
    adv_raw_exec, adv_norm_exec = [], []
    vanish_groups = 0
    total_groups = 0

    for micro, rs in cand_by_micro.items():
        rs = np.asarray(rs, dtype=np.float64)
        if rs.size == 0:
            continue

        if baseline == "median":
            b = float(np.median(rs))
        else:
            b = float(np.mean(rs))

        s = float(np.std(rs))
        total_groups += 1
        if s < 1e-12:
            vanish_groups += 1
            s = 0.0

        # Candidate advantages
        adv = rs - b
        adv_raw_all.extend(adv.tolist())
        if s > 0:
            adv_norm_all.extend((adv / (s + eps)).tolist())
        else:
            adv_norm_all.extend(np.zeros_like(adv).tolist())

        # Executed advantage (compare executed reward to candidate baseline)
        if micro in exec_by_micro:
            re = float(exec_by_micro[micro])
            ae = re - b
            adv_raw_exec.append(ae)
            adv_norm_exec.append(ae / (s + eps) if s > 0 else 0.0)

    frac_vanish = (vanish_groups / max(1, total_groups))
    return adv_raw_all, adv_norm_all, adv_raw_exec, adv_norm_exec, frac_vanish


def _plot_adv_hist_and_ecdf(values, *, title, xlabel, outpath_prefix, run_label):
    """
    Saves:
      - {outpath_prefix}_hist.png
      - {outpath_prefix}_ecdf.png
    """
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return

    # Histogram
    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.hist(x, bins=60, density=True, alpha=0.75)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_title(title)
    add_cms_header(fig, run_label=run_label)
    finalize_diag_fig(fig)
    save_png(fig, str(outpath_prefix) + "_hist")
    plt.close(fig)

    # ECDF
    xs, ys = ecdf(x)
    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.plot(xs, ys, linewidth=2.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("CDF")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_title(title)
    add_cms_header(fig, run_label=run_label)
    finalize_diag_fig(fig)
    save_png(fig, str(outpath_prefix) + "_ecdf")
    plt.close(fig)


def log_grpo_row(rows, *, trigger, chunk, micro, micro_global, phase,
                k, a, delta, step,
                cut_before, cut_next, cut_lo, cut_hi,
                bg_before, bg_after,
                tt_after, aa_after,
                occ_mid,
                reward_raw=None, reward_best_sample=None, reward_exec=None,
                executed=0, shielded=0):
        
    rows.append({
        "trigger": str(trigger),          # "AS" or "HT"
        "chunk": int(chunk),
        "micro": int(micro),              # per-trigger micro counter (AS uses micro_counter, HT uses ht_micro_counter)
        "micro_global": int(micro_global),# optional global counter (nice for single x-axis plots)
        "phase": str(phase),              # "candidate" or "executed"

        "k": int(k),                      # candidate index, or k_best for executed
        "a": int(a),
        "delta": float(delta),
        "step": float(step),              # AS_STEP or HT_STEP

        "cut_before": float(cut_before),
        "cut_next": float(cut_next),
        "cut_lo": float(cut_lo),
        "cut_hi": float(cut_hi),

        "bg_before": float(bg_before),
        "bg_after": float(bg_after),
        "tt_after": float(tt_after),
        "aa_after": float(aa_after),

        "occ_mid": float(occ_mid),

        "reward_raw": (None if reward_raw is None else float(reward_raw)),
        "reward_best_sample": (None if reward_best_sample is None else float(reward_best_sample)),
        "reward_exec": (None if reward_exec is None else float(reward_exec)),

        "executed": int(executed),
        "shielded": int(shielded),
    })
def ecdf(x):
    """Creating error cdf"""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.array([]), np.array([])
    x = np.sort(x)
    y = (np.arange(1, x.size + 1) / x.size)
    return x, y




def summarize_paper_table(r_pct, s_tt, s_aa, cut_hist, target_pct, tol_pct):
    """
    Paper-table metrics (matching screenshot):

      MAE↓      = mean(|r - r*|)
      P95|e|↓   = 95th percentile of |r - r*|
      InBand↑   = mean( |r-r*| <= tol )
      UpViol↓   = mean( max(0, r - (r* + tol)) )   [only upward violations]
      DownViol↓ = mean( max(0, (r* - tol) - r) )        [downward violations only]
      TV↓       = sum(|Δcut|)  (total variation / actuation)
      tt↑       = mean(tt efficiency | in-band)
      h→4b↑     = mean(AA efficiency | in-band)
    """
    r = np.asarray(r_pct, dtype=np.float64)
    s_tt = np.asarray(s_tt, dtype=np.float64)
    s_aa = np.asarray(s_aa, dtype=np.float64)
    c = np.asarray(cut_hist, dtype=np.float64)

    err = r - float(target_pct)
    abs_err = np.abs(err)
    inband = abs_err <= float(tol_pct)

    def safe_mean(x, m):
        return float(np.mean(x[m])) if np.any(m) else np.nan

    dc = np.diff(c) if c.size >= 2 else np.array([], dtype=np.float64)

    out = {}
    out["MAE"] = float(np.mean(abs_err)) if r.size else np.nan
    out["P95_abs_err"] = float(np.percentile(abs_err, 95)) if r.size else np.nan
    out["InBand"] = float(np.mean(inband)) if r.size else np.nan

    # Upward violation magnitude (rate too high beyond upper tolerance)
    out["UpViol"] = float(np.mean(np.maximum(0.0, err - float(tol_pct)))) if r.size else np.nan
    out["DownViol"] = float(np.mean(np.maximum(0.0, -err - float(tol_pct)))) if r.size else np.nan

    # Total variation of cut trajectory (actuation cost)
    out["TV"] = float(np.sum(np.abs(dc))) if dc.size else 0.0

    # Signal efficiencies conditioned on being in-band
    out["tt"] = safe_mean(s_tt, inband)
    out["h_to_4b"] = safe_mean(s_aa, inband)
    return out


def write_paper_table(rows, out_csv: Path, out_tex: Path, target_pct, tol_pct):
    """
    Writes:
      - CSV with columns: Trigger, Method, MAE, P95_abs_err, InBand, UpViol, TV, tt, h_to_4b
      - LaTeX table matching screenshot header
    """
    if not rows:
        return

    # ---- CSV ----
    fieldnames = ["Trigger", "Method", "MAE", "P95_abs_err", "InBand", "UpViol", "DownViol", "TV", "tt", "h_to_4b"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, None) for k in fieldnames})

    # ---- Bold best per trigger ----
    higher_better = {"InBand", "tt", "h_to_4b"}
    lower_better  = {"MAE", "P95_abs_err", "UpViol", "DownViol", "TV"}

    triggers = sorted(set(r["Trigger"] for r in rows))
    best = {tr: {} for tr in triggers}

    for tr in triggers:
        sub = [r for r in rows if r["Trigger"] == tr]

        for k in higher_better:
            vals = np.array([float(x[k]) for x in sub], dtype=np.float64)
            i = int(np.nanargmax(vals)) if np.any(np.isfinite(vals)) else 0
            best[tr][k] = sub[i]["Method"]

        for k in lower_better:
            vals = np.array([float(x[k]) for x in sub], dtype=np.float64)
            i = int(np.nanargmin(vals)) if np.any(np.isfinite(vals)) else 0
            best[tr][k] = sub[i]["Method"]

    def fmt(v, nd=3):
        if v is None:
            return "nan"
        if isinstance(v, (float, np.floating)):
            if not np.isfinite(v):
                return "nan"
            if abs(v) < 1e-3 and v != 0:
                return f"{v:.2e}"
            return f"{v:.{nd}f}"
        return str(v)

    def cell(tr, method, key, val):
        s = fmt(val, 3)
        if best.get(tr, {}).get(key, None) == method:
            return r"\textbf{" + s + "}"
        return s

    # ---- LaTeX ----
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{6pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.10}")
    lines.append(r"\begin{tabular}{llrrrrrrrr}")
    lines.append(r"\hline")
    lines.append(
        r"Trigger & Method & MAE$\downarrow$ & P95$|e|$$\downarrow$ & InBand$\uparrow$ & "
        r"UpViol$\downarrow$ & DownViol$\downarrow$ & TV$\downarrow$ & $t\bar{t}\uparrow$ & $h\rightarrow 4b\uparrow$ \\"
    )
    lines.append(r"\hline")

    for tr in triggers:
        sub = [r for r in rows if r["Trigger"] == tr]
        lines.append(rf"\multicolumn{{10}}{{l}}{{\textbf{{{tr} trigger}}}} \\")
        for r in sub:
            m = r["Method"]
            lines.append(
                f"{tr} & {m} & "
                f"{cell(tr,m,'MAE',r['MAE'])} & "
                f"{cell(tr,m,'P95_abs_err',r['P95_abs_err'])} & "
                f"{cell(tr,m,'InBand',r['InBand'])} & "
                f"{cell(tr,m,'UpViol',r['UpViol'])} & "
                f"{cell(tr,m,'DownViol',r['DownViol'])} & "
                f"{cell(tr,m,'TV',r['TV'])} & "
                f"{cell(tr,m,'tt',r['tt'])} & "
                f"{cell(tr,m,'h_to_4b',r['h_to_4b'])} \\\\"
            )
        lines.append(r"\hline")

    lines.append(r"\end{tabular}")
    lines.append(
        rf"\caption{{Summary of single-trigger control. Rates are in percent units with target "
        rf"$r^*={target_pct:.3f}\%$ and tolerance $\pm {tol_pct:.3f}\%$. "
        rf"InBand is the fraction of chunks within $|r-r^*|\le\tau$. "
        rf"UpViol measures upward band violations. TV is total cut variation. "
        rf"$t\bar t$ and $h\rightarrow 4b$ are mean signal efficiencies conditioned on in-band chunks.}}"
    )
    lines.append(r"\label{tab:single_trigger_summary_paper}")
    lines.append(r"\end{table}")

    with open(out_tex, "w") as f:
        f.write("\n".join(lines) + "\n")

def running_mean_bool(mask, w=7):
    m = np.asarray(mask, dtype=np.float64)
    k = np.ones(int(w), dtype=np.float64)
    return np.convolve(m, k, mode="same") / np.convolve(np.ones_like(m), k, mode="same")

def plot_cdf_abs_err_multi(rate_khz_by_method, target_khz, tol_khz, title, outpath, run_label):
    """
    rate_khz_by_method: dict(name -> 1D array of rates in kHz)
    """
    fig, ax = plt.subplots(figsize=(8, 5.2))
    for name, r_khz in rate_khz_by_method.items():
        e = np.abs(np.asarray(r_khz, dtype=np.float64) - float(target_khz))
        x, y = ecdf(e)
        if x.size:
            ax.plot(x, y, linewidth=2.2, label=name)

    ax.axvline(float(tol_khz), linestyle="--", linewidth=1.6, label=f"Tolerance = {tol_khz:.1f} kHz")
    ax.set_xlabel(r"$|r-r^*|$ [kHz]")
    ax.set_ylabel("CDF")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best", frameon=True, title=title)
    add_cms_header(fig, run_label=run_label)
    finalize_diag_fig(fig)
    save_png(fig, str(outpath))
    plt.close(fig)


def plot_running_inband_multi(time, inband_by_method, w, title, outpath, run_label):
    """
    inband_by_method: dict(name -> boolean mask per chunk)
    """
    fig, ax = plt.subplots(figsize=(8, 5.2))
    for name, m in inband_by_method.items():
        ax.plot(time, running_mean_bool(m, w=int(w)), linewidth=2.2, label=f"{name} (w={int(w)})")

    ax.set_xlabel("Time (Fraction of Run)")
    ax.set_ylabel("Running in-band fraction")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best", frameon=True, title=title)
    add_cms_header(fig, run_label=run_label)
    finalize_diag_fig(fig)
    save_png(fig, str(outpath))
    plt.close(fig)


def plot_cut_step_hist_multi(cut_by_method, xlabel, title, outpath, run_label, bins=30,
                             allow_constant_zeros=True
                             ):
    """
    cut_by_method: dict(name -> 1D cut history)
    If allow_constant_zeros: constant menu can produce a delta array of zeros.
    """
    fig, ax = plt.subplots(figsize=(8, 5.2))
    any_plotted = False
    for name, c in cut_by_method.items():
        c = np.asarray(c, dtype=np.float64)

        if c.size >= 2:
            dc = np.diff(c)
        else:
            dc = np.array([], dtype=np.float64)

        if dc.size == 0 and allow_constant_zeros:
            # if we only have one point, or no history, treat as "no motion"
            dc = np.zeros(max(1, c.size - 1), dtype=np.float64)

        if dc.size:
            ax.hist(np.abs(dc), bins=int(bins), alpha=0.50, label=name)
            any_plotted = True

    if not any_plotted:
        ax.text(0.5, 0.5, "No cut history to plot", ha="center", va="center", transform=ax.transAxes)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best", frameon=True, title=title)
    add_cms_header(fig, run_label=run_label)
    finalize_diag_fig(fig)
    save_png(fig, str(outpath))
    plt.close(fig)


def plot_inband_eff_bars_multi(summary_by_method, title, outpath, run_label):
    """
    summary_by_method: dict(name -> summarize_compact(...) dict)
    """
    labels = [r"$t\bar{t}$", r"$h\rightarrow 4b$"]
    keys   = ["tt", "h_to_4b"]
    methods = list(summary_by_method.keys())

    vals = np.array([[summary_by_method[m][k] for k in keys] for m in methods], dtype=np.float64)  # (M,3)

    x = np.arange(len(labels))
    bw = 0.80 / max(1, len(methods))  # fill 80% of tick width

    fig, ax = plt.subplots(figsize=(8, 5.2))
    for i, m in enumerate(methods):
        ax.bar(x - 0.40 + (i + 0.5) * bw, vals[i], width=bw, label=m)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean signal efficiency (in-band)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="best", frameon=True, title=title)
    add_cms_header(fig, run_label=run_label)
    finalize_diag_fig(fig)
    save_png(fig, str(outpath))
    plt.close(fig)

# ----------------------------- main -----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--input", default="Data/Trigger_food_MC.h5",
                    choices=["Data/Trigger_food_MC.h5", "Data/Matched_data_2016_dim2.h5"])
    ap.add_argument("--outdir", default="outputs/demo_sing_grpo_as_feature")
    ap.add_argument("--control", default="MC", choices=["MC", "RealData"])
    ap.add_argument("--score-dim-hint", type=int, default=2)
    ap.add_argument("--as-dim", type=int, default=2, choices=[1, 2, 4])

    ap.add_argument("--as-deltas", type=str, default="-3,-1.5,0,1.5,3")
    ap.add_argument("--as-step", type=float, default=0.5)

    ap.add_argument("--print-keys", action="store_true")
    ap.add_argument("--print-keys-max", type=int, default=None)

    ap.add_argument("--window-events-chunk-size", type=int, default=3)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--inner-stride", type=int, default=10000)

    # GRPO kwargs
    ap.add_argument("--group-size", type=int, default=16, choices=[16, 32])
    ap.add_argument("--train-every", type=int, default=50)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--beta-kl", type=float, default=0.02)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--lr", type=float, default=3e-4)

    # objective/reward
    ap.add_argument("--target", type=float, default=0.25)   # percent
    ap.add_argument("--tol", type=float, default=0.025,     # percent  (0.025% -> ±10kHz band)
                    help="tolerance in percent units; 0.025 corresponds to [90,110] kHz when target=0.25%")
    ap.add_argument("--alpha", type=float, default=0.4)
    ap.add_argument("--beta", type=float, default=0.2)

    # optional stabilization (AD-specific)
    ap.add_argument("--occ-pen", type=float, default=0.0,
                    help="extra penalty weight for near-cut occupancy * |delta| (suggest 0.5~3.0)")
    ap.add_argument("--run-avg-window", type=int, default=7,
                    help="window size (chunks) for running in-band fraction plot")
    # DQN knobs (AS-only)
    ap.add_argument("--dqn-lr", type=float, default=1e-4)
    ap.add_argument("--dqn-gamma", type=float, default=0.95)
    ap.add_argument("--dqn-batch-size", type=int, default=32)
    ap.add_argument("--dqn-target-update", type=int, default=200)
    ap.add_argument("--dqn-train-steps-per-micro", type=int, default=1)
    ap.add_argument("--dqn-eps-min", type=float, default=0.05)
    ap.add_argument("--dqn-eps-decay", type=float, default=0.98)
    ap.add_argument("--run-ht", action="store_true", help="also run HT trigger GRPO baselines/plots")
    ap.add_argument("--ht-deltas", type=str, default="-2,-1,0,1,2")
    ap.add_argument("--ht-step", type=float, default=1.0)

    args = ap.parse_args()

    if args.print_keys:
        print_h5_tree(args.input, max_items=args.print_keys_max)
        raise SystemExit(0)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    plots_dir = outdir / "extra_plots"
    tables_dir = outdir / "tables"
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    d = read_any_h5(args.input, score_dim_hint=args.score_dim_hint)
    matched_by_index = bool(d["meta"].get("matched_by_index", False))

    Bnpv = d["Bnpv"]
    Tnpv = d["Tnpv"]
    Anpv = d["Anpv"]

    Bht, Tht, Aht = d["Bht"], d["Tht"], d["Aht"]
    if args.run_ht and (Bht is None or Tht is None or Aht is None):
        raise SystemExit("HT arrays missing: need Bht/Tht/Aht in the input file.")

    # choose AS
    if args.as_dim == 2:
        Bas, Tas, Aas = d["Bas2"], d["Tas2"], d["Aas2"]
    elif args.as_dim == 1:
        Bas, Tas, Aas = d["Bas1"], d["Tas1"], d["Aas1"]
    else:
        Bas, Tas, Aas = d["Bas4"], d["Tas4"], d["Aas4"]

    if Bas is None or Tas is None or Aas is None:
        raise SystemExit("AS arrays missing for requested --as-dim.")

    N = len(Bas)
    chunk_size = 50000 if args.control == "MC" else 20000
    start_event = chunk_size * 10
    start_event = max(0, (start_event // chunk_size) * chunk_size)
    if start_event + chunk_size > N:
        start_event = max(0, ((N - chunk_size) // chunk_size) * chunk_size)

    # fixed cut from calibration window
    win_lo = min(start_event, N - 1)
    win_hi = min(start_event + (100000 if args.control == "MC" else 10000), N)
    fixed_AS_cut = float(np.percentile(Bas[win_lo:win_hi], 99.75))
    if args.run_ht:
        fixed_Ht_cut = float(np.percentile(Bht[win_lo:win_hi], 99.75))

        # clip range (use run tail percentiles, for DQN)
        ht_lo = float(np.percentile(Bht[start_event:], 95.0))
        ht_hi = float(np.percentile(Bht[start_event:], 99.99))
        ht_mid = 0.5 * (ht_lo + ht_hi)
        ht_span = max(1.0, ht_hi - ht_lo)

        print(f"[HT] fixed={fixed_Ht_cut:.3f} clip=({ht_lo:.3f},{ht_hi:.3f}) ht_step={args.ht_step}")


    # clip range (use calibration window range)
    ref_as = Bas[win_lo:win_hi]
    as_lo = float(np.min(ref_as))
    as_hi = float(np.max(ref_as))
    as_mid = 0.5 * (as_lo + as_hi)
    as_span = max(1e-6, as_hi - as_lo)

    print(f"[INFO] matched_by_index={matched_by_index} N={N} chunk={chunk_size} start={start_event}")
    print(f"[AS dim={args.as_dim}] fixed={fixed_AS_cut:.6f} clip=({as_lo:.6f},{as_hi:.6f}) as_step={args.as_step}")

    # PD init
    AS_cut_pd = fixed_AS_cut
    pre_as_err = 0.0

    # HT PD init (optional baseline; even if you don't plot yet, keep for completeness)
    if args.run_ht:
        Ht_cut_pd = fixed_Ht_cut
        pre_ht_err = 0.0

    # separate counter so HT GRPO doesn't mess with DQN epsilon schedule
    ht_micro_counter = 0

    # GRPO init
    AS_cut_grpo = fixed_AS_cut
    last_das = 0.0
    prev_bg_as = None

    # action space
    AS_DELTAS = np.array([float(x) for x in args.as_deltas.split(",")], dtype=np.float32)
    AS_STEP = float(args.as_step)
    MAX_DELTA_AS = float(np.max(np.abs(AS_DELTAS))) * AS_STEP

    # features
    K = int(args.seq_len)
    near_widths_as = (0.25, 0.5, 1.0)
    feat_dim_as = 10 + len(near_widths_as)

    # logs (background in percent units first)
    target = float(args.target)
    tol = float(args.tol)
    run_label = "MC" if args.control == "MC" else "283408"

    # GRPO agent
    cfg = GRPOConfig(
        lr=args.lr,
        beta_kl=args.beta_kl,
        ent_coef=args.ent_coef,
        device="cpu",
        batch_size=128,
        train_epochs=2,
        ref_update_interval=200,
    )
    agent = GRPOAgent(seq_len=K, feat_dim=feat_dim_as, n_actions=len(AS_DELTAS), cfg=cfg, seed=SEED,
        reward_cfg=GRPORewardCfg(
        target=target,
        tol=tol,
        mode="lex",        # "lex" recommended; "lag" if you want adaptive lambda
        mix=0.75, #increase for tt
        alpha_sig=1.0,
        beta_move=0.02,
        gamma_stab=0.25,
        k_violate=5.0,
        w_occ=float(args.occ_pen)
    ))
    # ---------------- DQN agent (AS only for now temporarilly, parallel baseline) ----------------
    dqn_cfg = DQNConfig(
        lr=float(args.dqn_lr),
        gamma=float(args.dqn_gamma),
        batch_size=int(args.dqn_batch_size),
        target_update=int(args.dqn_target_update),
    )
    dqn_as = SeqDQNAgent(seq_len=K, feat_dim=feat_dim_as, n_actions=len(AS_DELTAS),
                        cfg=dqn_cfg, seed=SEED)

    AS_cut_dqn = fixed_AS_cut
    prev_bg_dqn = None
    last_das_dqn = 0.0
    dqn_losses = []
    dqn_rewards = []

    if args.run_ht:
        HT_DELTAS = np.array([float(x) for x in args.ht_deltas.split(",")], dtype=np.float32)
        HT_STEP = float(args.ht_step)
        MAX_DELTA_HT = float(np.max(np.abs(HT_DELTAS))) * HT_STEP

        near_widths_ht = (5.0, 10.0, 20.0)
        feat_dim_ht = 10 + len(near_widths_ht)

        cfg_ht = GRPOConfig(
            lr=args.lr, beta_kl=args.beta_kl, ent_coef=args.ent_coef,
            device="cpu", batch_size=256, train_epochs=2, ref_update_interval=200,
        )
        agent_ht = GRPOAgent(
                seq_len=K, feat_dim=feat_dim_ht, n_actions=len(HT_DELTAS),
                cfg=cfg_ht, seed=SEED,
                reward_cfg=GRPORewardCfg(
                target=target,
                tol=tol,
                mode="lex",        # "lex" recommended
                mix=0.75,
                alpha_sig=1.0,
                beta_move=0.02,
                gamma_stab=0.25,
                k_violate=5.0,
                w_occ=float(args.occ_pen)
            )
        )

        Ht_cut_grpo = fixed_Ht_cut
        last_dht = 0.0
        prev_bg_ht = None

        # logs
        R_ht_const_pct, R_ht_pd_pct, R_ht_grpo_pct = [], [], []
        Cut_ht_pd, Cut_ht_grpo = [], []
        TT_ht_const, TT_ht_pd, TT_ht_grpo = [], [], []
        AA_ht_const, AA_ht_pd, AA_ht_grpo = [], [], []
        ht_losses, ht_rewards = [], []

        # ---------------- DQN agent (HT) ----------------
        dqn_ht_cfg = DQNConfig(
            lr=float(args.dqn_lr),
            gamma=float(args.dqn_gamma),
            batch_size=int(args.dqn_batch_size),
            target_update=int(args.dqn_target_update),
        )
        dqn_ht = SeqDQNAgent(
            seq_len=K, feat_dim=feat_dim_ht, n_actions=len(HT_DELTAS),
            cfg=dqn_ht_cfg, seed=SEED
        )

        Ht_cut_dqn = fixed_Ht_cut
        prev_bg_ht_dqn = None
        last_dht_dqn = 0.0
        dqn_ht_losses = []
        dqn_ht_rewards = []

        # add DQN logs for HT
        R_ht_dqn_pct = []
        Cut_ht_dqn = []
        TT_ht_dqn = []
        AA_ht_dqn = []

        ht_dqn_step = 0




    # rolling window for event features (AS)
    roll = RollingWindow(max_events=int(args.window_events_chunk_size * chunk_size))
    if args.run_ht:
        roll_ht = RollingWindowHT(max_events=int(args.window_events_chunk_size * chunk_size))

    # logs (background in percent units first)
    target = float(args.target)
    tol = float(args.tol)
    run_label = "MC" if args.control == "MC" else "283408"

    R_const_pct, R_pd_pct, R_grpo_pct = [], [], []
    Cut_pd, Cut_grpo = [], []
    TT_const, TT_pd, TT_grpo = [], [], []
    AA_const, AA_pd, AA_grpo = [], [], []

    R_dqn_pct = []
    Cut_dqn = []
    TT_dqn = []
    AA_dqn = []

    losses = []
    rewards = []
    # --- GRPO near-cut occupancy logs (per chunk) ---
    near_occ_as_grpo = []   # list of shape (len(near_widths_as),)
    near_occ_ht_grpo = []   # (only if --run-ht)



    batch_starts = list(range(start_event, N, chunk_size))
    micro_counter = 0
    grpo_samples = []   # one table, add column "trigger" = {"AS","HT"}

    micro_global = 0    # optional: single timeline across AS+HT micro-steps

    for t, I in enumerate(batch_starts):
        end = min(I + chunk_size, N, len(Bnpv))
        if end <= I:
            break

        idx = np.arange(I, end)
        bas = Bas[idx]
        bnpv = Bnpv[idx]
        if args.run_ht:
            bht = Bht[idx]
        # signals for the chunk
        if matched_by_index:
            end_sig = min(end, len(Tas), len(Aas), len(Tnpv), len(Anpv))
            idx_sig = np.arange(I, end_sig)
            sas_tt = Tas[idx_sig]
            sas_aa = Aas[idx_sig]
        else:
            npv_min = float(np.min(bnpv))
            npv_max = float(np.max(bnpv))
            mask_tt = (Tnpv >= npv_min) & (Tnpv <= npv_max)
            mask_aa = (Anpv >= npv_min) & (Anpv <= npv_max)
            sas_tt = Tas[mask_tt]
            sas_aa = Aas[mask_aa]
        if args.run_ht:
            if matched_by_index:
                end_sig = min(end, len(Tht), len(Aht), len(Tnpv), len(Anpv))
                idx_sig = np.arange(I, end_sig)
                sht_tt = Tht[idx_sig]
                sht_aa = Aht[idx_sig]
            else:
                npv_min = float(np.min(bnpv))
                npv_max = float(np.max(bnpv))
                mask_tt = (Tnpv >= npv_min) & (Tnpv <= npv_max)
                mask_aa = (Anpv >= npv_min) & (Anpv <= npv_max)
                sht_tt = Tht[mask_tt]
                sht_aa = Aht[mask_aa]


        stride = max(500, int(args.inner_stride))
        n_micro = max(1, int(np.ceil((end - I) / stride)))

        micro_rewards = []
        micro_rewards_ht = []   # HT-GRPO executed rewards per micro-step (this chunk)


        for j in range(n_micro):
            j_lo = I + j * stride
            j_hi = min(I + (j + 1) * stride, end)
            if j_hi <= j_lo:
                continue

            idxj = np.arange(j_lo, j_hi)

            # --- micro-slice arrays ---
            bas_j = Bas[idxj]
            bnpv_j = Bnpv[idxj]

            # --- update rolling windows ONCE per micro-step ---
            roll.append(bas_j, bnpv_j)
            bas_w, bnpv_w = roll.get()

            if args.run_ht:
                bht_j = Bht[idxj]
                roll_ht.append(bht_j, bnpv_j)
                bht_w, bnpv_w_ht = roll_ht.get()

                # ============================================================
                # HT micro-step: DQN + GRPO
                # ============================================================
                # ----- HT DQN -----
                bg_before_ht_dqn = Sing_Trigger(bht_j, Ht_cut_dqn)
                if prev_bg_ht_dqn is None:
                    prev_bg_ht_dqn = bg_before_ht_dqn

                obs_ht_dqn = make_event_seq_ht_v0(
                    bht=bht_w, bnpv=bnpv_w_ht,
                    bg_rate=bg_before_ht_dqn,
                    prev_bg_rate=prev_bg_ht_dqn,
                    cut=Ht_cut_dqn,
                    ht_mid=ht_mid, ht_span=ht_span,
                    target=target, K=K,
                    last_delta=last_dht_dqn,
                    max_delta=MAX_DELTA_HT,
                    near_widths=near_widths_ht,
                )

                eps_ht = max(float(args.dqn_eps_min), 1.0 * (float(args.dqn_eps_decay) ** ht_dqn_step))
                a_ht_dqn = dqn_ht.act(obs_ht_dqn, eps=eps_ht)
                dht_dqn = float(HT_DELTAS[a_ht_dqn] * HT_STEP)

                sd = shield_delta(bg_before_ht_dqn, target, tol, MAX_DELTA_HT)
                if sd is not None:
                    dht_dqn = float(sd)

                cut_next_ht_dqn = float(np.clip(Ht_cut_dqn + dht_dqn, ht_lo, ht_hi))
                bg_after_ht_dqn = Sing_Trigger(bht_j, cut_next_ht_dqn)

                tt_after_ht_dqn = Sing_Trigger(sht_tt, cut_next_ht_dqn)
                aa_after_ht_dqn = Sing_Trigger(sht_aa, cut_next_ht_dqn)

                obs_next_ht_dqn = make_event_seq_ht_v0(
                    bht=bht_w, bnpv=bnpv_w_ht,
                    bg_rate=bg_after_ht_dqn,
                    prev_bg_rate=bg_before_ht_dqn,
                    cut=cut_next_ht_dqn,
                    ht_mid=ht_mid, ht_span=ht_span,
                    target=target, K=K,
                    last_delta=dht_dqn,
                    max_delta=MAX_DELTA_HT,
                    near_widths=near_widths_ht,
                )
                occ_mid_ht_dqn = float(near_occupancy(bht_j, Ht_cut_dqn, near_widths_ht)[1])  # width=10


                r_ht_dqn = SeqDQNAgent.compute_reward(
                    bg_rate=bg_after_ht_dqn,
                    target=target, tol=tol,
                    sig_rate_1=tt_after_ht_dqn,
                    sig_rate_2=aa_after_ht_dqn,
                    delta_applied=dht_dqn,
                    max_delta=MAX_DELTA_HT,
                    alpha=float(args.alpha),
                    beta=float(args.beta),
                    prev_bg_rate=bg_before_ht_dqn,
                    gamma_stab=0.3,
                )

                dqn_ht.buf.push(obs_ht_dqn, int(a_ht_dqn), float(r_ht_dqn), obs_next_ht_dqn, done=False)

                for _ in range(int(args.dqn_train_steps_per_micro)):
                    loss_ht = dqn_ht.train_step()
                    if loss_ht is not None:
                        dqn_ht_losses.append(float(loss_ht))

                Ht_cut_dqn = cut_next_ht_dqn
                prev_bg_ht_dqn = bg_after_ht_dqn
                last_dht_dqn = dht_dqn
                dqn_ht_rewards.append(float(r_ht_dqn))
                ht_dqn_step += 1

                # ----- HT GRPO -----
                bg_before_ht = Sing_Trigger(bht_j, Ht_cut_grpo)
                if prev_bg_ht is None:
                    prev_bg_ht = bg_before_ht

                obs_ht = make_event_seq_ht_v0(
                    bht=bht_w, bnpv=bnpv_w_ht,
                    bg_rate=bg_before_ht,
                    prev_bg_rate=prev_bg_ht,
                    cut=Ht_cut_grpo,
                    ht_mid=ht_mid, ht_span=ht_span,
                    target=target, K=K,
                    last_delta=last_dht,
                    max_delta=MAX_DELTA_HT,
                    near_widths=near_widths_ht,
                )

                G = int(args.group_size)
                acts_ht, old_logps_ht = agent_ht.sample_group_actions(
                    obs_ht, group_size=G, temperature=float(args.temperature)
                )

                cand_rewards_ht = np.zeros(G, dtype=np.float32)
                occ_mid_ht = float(near_occupancy(bht_j, Ht_cut_grpo, near_widths_ht)[1])  # width=10

                for k in range(G):
                    a = int(acts_ht[k])
                    dht = float(HT_DELTAS[a] * HT_STEP)

                    cut_next = float(np.clip(Ht_cut_grpo + dht, ht_lo, ht_hi))
                    bg_after = Sing_Trigger(bht_j, cut_next)

                    tt_after = Sing_Trigger(sht_tt, cut_next)
                    aa_after = Sing_Trigger(sht_aa, cut_next)

                    r = agent_ht.compute_reward(
                        bg_after=bg_after,
                        tt_after=tt_after,
                        aa_after=aa_after,
                        delta_applied=dht,
                        max_delta=MAX_DELTA_HT,
                        prev_bg=bg_before_ht,
                        occ_mid=occ_mid_ht,
                        update_dual=False,
                    )

                    cand_rewards_ht[k] = float(r)

                    log_grpo_row(
                        grpo_samples,
                        trigger="HT",
                        chunk=t,
                        micro=ht_micro_counter,
                        micro_global=micro_global,
                        phase="candidate",
                        k=k,
                        a=a,
                        delta=dht,
                        step=HT_STEP,
                        cut_before=Ht_cut_grpo,
                        cut_next=cut_next,
                        cut_lo=ht_lo,
                        cut_hi=ht_hi,
                        bg_before=bg_before_ht,
                        bg_after=bg_after,
                        tt_after=tt_after,
                        aa_after=aa_after,
                        occ_mid=occ_mid_ht,
                        reward_raw=r,
                        executed=0,
                        shielded=0,
                    )
                    micro_global += 1


                agent_ht.store_group(
                    obs=obs_ht,
                    actions=acts_ht,
                    logp=old_logps_ht,
                    rewards=cand_rewards_ht,
                    baseline="mean",
                )

                k_best = int(np.argmax(cand_rewards_ht))
                a_exec = int(acts_ht[k_best])
                dht_exec = float(HT_DELTAS[a_exec] * HT_STEP)

                sd = shield_delta(bg_before_ht, target, tol, MAX_DELTA_HT)
                if sd is not None:
                    dht_exec = float(sd)

                cut_next_exec = float(np.clip(Ht_cut_grpo + dht_exec, ht_lo, ht_hi))
                bg_after_exec = Sing_Trigger(bht_j, cut_next_exec)
                tt_after_exec = Sing_Trigger(sht_tt, cut_next_exec)
                aa_after_exec = Sing_Trigger(sht_aa, cut_next_exec)

                r_exec = agent_ht.compute_reward(
                    bg_after=bg_after_exec,
                    tt_after=tt_after_exec,
                    aa_after=aa_after_exec,
                    delta_applied=dht_exec,
                    max_delta=MAX_DELTA_HT,
                    prev_bg=bg_before_ht,
                    occ_mid=occ_mid_ht,
                    update_dual=True,
                )  
                micro_rewards_ht.append(float(r_exec))


                
                log_grpo_row(
                    grpo_samples,
                    trigger="HT",
                    chunk=t,
                    micro=ht_micro_counter,
                    micro_global=micro_global,
                    phase="executed",
                    k=k_best,
                    a=a_exec,
                    delta=dht_exec,
                    step=HT_STEP,
                    cut_before=Ht_cut_grpo,
                    cut_next=cut_next_exec,
                    cut_lo=ht_lo,
                    cut_hi=ht_hi,
                    bg_before=bg_before_ht,
                    bg_after=bg_after_exec,
                    tt_after=tt_after_exec,
                    aa_after=aa_after_exec,
                    occ_mid=occ_mid_ht,
                    reward_best_sample=float(cand_rewards_ht[k_best]),  # pre-shield best sampled
                    reward_exec=r_exec,                                 # reward of executed (post-shield)
                    executed=1,
                    shielded=int(sd is not None),
                )
                
                ht_rewards.append(float(np.mean(micro_rewards_ht)) if micro_rewards_ht else np.nan)

                micro_global += 1

                Ht_cut_grpo = cut_next_exec
                prev_bg_ht = bg_after_exec
                last_dht = dht_exec

                ht_micro_counter += 1
                if ht_micro_counter % int(args.train_every) == 0:
                    loss = agent_ht.update()
                    if loss is not None:
                        ht_losses.append(float(loss))

            # ============================================================
            # AS micro-step: DQN + GRPO  
            # ============================================================
            bg_before_dqn = Sing_Trigger(bas_j, AS_cut_dqn)
            if prev_bg_dqn is None:
                prev_bg_dqn = bg_before_dqn

            obs_dqn = make_event_seq_as_v0(
                bas=bas_w, bnpv=bnpv_w,
                bg_rate=bg_before_dqn,
                prev_bg_rate=prev_bg_dqn,
                cut=AS_cut_dqn,
                as_mid=as_mid, as_span=as_span,
                target=target, K=K,
                last_delta=last_das_dqn,
                max_delta=MAX_DELTA_AS,
                near_widths=near_widths_as,
            )

            step = micro_counter
            eps = max(float(args.dqn_eps_min), 1.0 * (float(args.dqn_eps_decay) ** step))

            a_dqn = dqn_as.act(obs_dqn, eps=eps)
            das_dqn = float(AS_DELTAS[a_dqn] * AS_STEP)

            sd = shield_delta(bg_before_dqn, target, tol, MAX_DELTA_AS)
            if sd is not None:
                das_dqn = float(sd)

            cut_next_dqn = float(np.clip(AS_cut_dqn + das_dqn, as_lo, as_hi))
            bg_after_dqn = Sing_Trigger(bas_j, cut_next_dqn)

            tt_after_dqn = Sing_Trigger(sas_tt, cut_next_dqn)
            aa_after_dqn = Sing_Trigger(sas_aa, cut_next_dqn)

            obs_next_dqn = make_event_seq_as_v0(
                bas=bas_w, bnpv=bnpv_w,
                bg_rate=bg_after_dqn,
                prev_bg_rate=bg_before_dqn,
                cut=cut_next_dqn,
                as_mid=as_mid, as_span=as_span,
                target=target, K=K,
                last_delta=das_dqn,
                max_delta=MAX_DELTA_AS,
                near_widths=near_widths_as,
            )

            r_dqn = SeqDQNAgent.compute_reward(
                bg_rate=bg_after_dqn,
                target=target, tol=tol,
                sig_rate_1=tt_after_dqn,
                sig_rate_2=aa_after_dqn,
                delta_applied=das_dqn,
                max_delta=MAX_DELTA_AS,
                alpha=float(args.alpha),
                beta=float(args.beta),
                prev_bg_rate=bg_before_dqn,
                gamma_stab=0.3,
            )

            dqn_as.buf.push(obs_dqn, int(a_dqn), float(r_dqn), obs_next_dqn, done=False)

            for _ in range(int(args.dqn_train_steps_per_micro)):
                loss = dqn_as.train_step()
                if loss is not None:
                    dqn_losses.append(float(loss))

            AS_cut_dqn = cut_next_dqn
            prev_bg_dqn = bg_after_dqn
            last_das_dqn = das_dqn
            dqn_rewards.append(float(r_dqn))

            # ----- AS GRPO -----
            bg_before = Sing_Trigger(bas_j, AS_cut_grpo)
            if prev_bg_as is None:
                prev_bg_as = bg_before

            obs = make_event_seq_as_v0(
            bas=bas_w, bnpv=bnpv_w,
            bg_rate=bg_before,
            prev_bg_rate=prev_bg_as,
            cut=AS_cut_grpo,
            as_mid=as_mid, as_span=as_span,
            target=target, K=K,
            last_delta=last_das,
            max_delta=MAX_DELTA_AS,
            near_widths=near_widths_as,
            )

            G = int(args.group_size)
            acts, old_logps = agent.sample_group_actions(obs, group_size=G, temperature=float(args.temperature))

            cand_rewards = np.zeros(G, dtype=np.float32)
            occ_mid = float(near_occupancy(bas_j, AS_cut_grpo, near_widths_as)[1])  # w=0.5

            for k in range(G):
                a = int(acts[k])
                das = float(AS_DELTAS[a] * AS_STEP)

                cut_next = float(np.clip(AS_cut_grpo + das, as_lo, as_hi))
                bg_after = Sing_Trigger(bas_j, cut_next)

                tt_after = Sing_Trigger(sas_tt, cut_next)
                aa_after = Sing_Trigger(sas_aa, cut_next)

                r = agent.compute_reward(
                    bg_after=bg_after,
                    tt_after=tt_after,
                    aa_after=aa_after,
                    delta_applied=das,
                    max_delta=MAX_DELTA_AS,
                    prev_bg=bg_before,
                    occ_mid=occ_mid,
                    update_dual=False,   # only matters if mode="lag"
                )

                cand_rewards[k] = float(r)

                log_grpo_row(
                    grpo_samples,
                    trigger="AS",
                    chunk=t,
                    micro=micro_counter,
                    micro_global=micro_global,
                    phase="candidate",
                    k=k,
                    a=a,
                    delta=das,
                    step=AS_STEP,
                    cut_before=AS_cut_grpo,
                    cut_next=cut_next,
                    cut_lo=as_lo,
                    cut_hi=as_hi,
                    bg_before=bg_before,
                    bg_after=bg_after,
                    tt_after=tt_after,
                    aa_after=aa_after,
                    occ_mid=occ_mid,
                    reward_raw=r,
                    executed=0,
                    shielded=0,
                )
                micro_global += 1


            agent.store_group(
                obs=obs,
                actions=acts,
                logp=old_logps,
                rewards=cand_rewards,
                baseline="mean",   # or "median"
            )

            

            k_best = int(np.argmax(cand_rewards))
            a_exec = int(acts[k_best])

            das_exec = float(AS_DELTAS[a_exec] * AS_STEP)

            sd = shield_delta(bg_before, target, tol, MAX_DELTA_AS)
            if sd is not None:
                das_exec = float(sd)
            
            cut_next_exec = float(np.clip(AS_cut_grpo + das_exec, as_lo, as_hi))
            bg_after_exec = Sing_Trigger(bas_j, cut_next_exec)
            tt_after_exec = Sing_Trigger(sas_tt, cut_next_exec)
            aa_after_exec = Sing_Trigger(sas_aa, cut_next_exec)

            r_exec = agent.compute_reward(
                bg_after=bg_after_exec,
                tt_after=tt_after_exec,
                aa_after=aa_after_exec,
                delta_applied=das_exec,
                max_delta=MAX_DELTA_AS,
                prev_bg=bg_before,
                occ_mid=occ_mid,
                update_dual=True,   # ✅ executed action is where you'd update lambda if mode="lag"
            )

            log_grpo_row(
                grpo_samples,
                trigger="AS",
                chunk=t,
                micro=micro_counter,
                micro_global=micro_global,
                phase="executed",
                k=k_best,
                a=a_exec,
                delta=das_exec,
                step=AS_STEP,
                cut_before=AS_cut_grpo,
                cut_next=cut_next_exec,
                cut_lo=as_lo,
                cut_hi=as_hi,
                bg_before=bg_before,
                bg_after=bg_after_exec,
                tt_after=tt_after_exec,
                aa_after=aa_after_exec,
                occ_mid=occ_mid,
                reward_best_sample=float(cand_rewards[k_best]),
                reward_exec=r_exec,
                executed=1,
                shielded=int(sd is not None),
            )
            micro_global += 1


            # IMPORTANT: update GRPO state
            AS_cut_grpo = cut_next_exec
            prev_bg_as = bg_after_exec
            last_das = das_exec

            micro_rewards.append(float(r_exec))
            micro_counter += 1

            if micro_counter % int(args.train_every) == 0:
                loss = agent.update()
                if loss is not None:
                    losses.append(float(loss))
        
        # ============================
        # CHUNK-LEVEL logging (ONCE per chunk)
        # ============================

        # --- AD (AS trigger) rates for this chunk ---
        bg_const = Sing_Trigger(bas, fixed_AS_cut)
        bg_grpo  = Sing_Trigger(bas, AS_cut_grpo)
        bg_dqn   = Sing_Trigger(bas, AS_cut_dqn)

        near_occ_as_grpo.append(near_occupancy(bas, AS_cut_grpo, near_widths_as)) #GRPO near cut occupancy chunk level


        # --- PD update once per chunk ---
        bg_pd_before = Sing_Trigger(bas, AS_cut_pd)
        AS_cut_pd, pre_as_err = PD_controller2(bg_pd_before, pre_as_err, AS_cut_pd)
        AS_cut_pd = float(np.clip(AS_cut_pd, as_lo, as_hi))
        bg_pd = Sing_Trigger(bas, AS_cut_pd)

        # --- signal efficiencies for this chunk (same cuts as rates) ---
        tt_const = Sing_Trigger(sas_tt, fixed_AS_cut)
        aa_const = Sing_Trigger(sas_aa, fixed_AS_cut)

        tt_pd  = Sing_Trigger(sas_tt, AS_cut_pd)
        aa_pd  = Sing_Trigger(sas_aa, AS_cut_pd)

        tt_grpo = Sing_Trigger(sas_tt, AS_cut_grpo)
        aa_grpo = Sing_Trigger(sas_aa, AS_cut_grpo)

        tt_dqn = Sing_Trigger(sas_tt, AS_cut_dqn)
        aa_dqn = Sing_Trigger(sas_aa, AS_cut_dqn)

        # --- append AD logs ---
        R_const_pct.append(bg_const)
        R_pd_pct.append(bg_pd)
        R_grpo_pct.append(bg_grpo)
        R_dqn_pct.append(bg_dqn)

        Cut_pd.append(AS_cut_pd)
        Cut_grpo.append(AS_cut_grpo)
        Cut_dqn.append(AS_cut_dqn)

        TT_const.append(tt_const); TT_pd.append(tt_pd); TT_grpo.append(tt_grpo); TT_dqn.append(tt_dqn)
        AA_const.append(aa_const); AA_pd.append(aa_pd); AA_grpo.append(aa_grpo); AA_dqn.append(aa_dqn)

        rewards.append(float(np.mean(micro_rewards)) if micro_rewards else np.nan)

        # --- HT chunk-level logs (ONLY if enabled) ---
        if args.run_ht:

            near_occ_ht_grpo.append(near_occupancy(bht, Ht_cut_grpo, near_widths_ht))

            bg_ht_const = Sing_Trigger(bht, fixed_Ht_cut)
            bg_ht_grpo  = Sing_Trigger(bht, Ht_cut_grpo)
            bg_ht_dqn   = Sing_Trigger(bht, Ht_cut_dqn)

            bg_ht_pd_before = Sing_Trigger(bht, Ht_cut_pd)
            Ht_cut_pd, pre_ht_err = PD_controller1(bg_ht_pd_before, pre_ht_err, Ht_cut_pd)
            Ht_cut_pd = float(np.clip(Ht_cut_pd, ht_lo, ht_hi))
            bg_ht_pd = Sing_Trigger(bht, Ht_cut_pd)

            tt_ht_const = Sing_Trigger(sht_tt, fixed_Ht_cut)
            aa_ht_const = Sing_Trigger(sht_aa, fixed_Ht_cut)

            tt_ht_pd  = Sing_Trigger(sht_tt, Ht_cut_pd)
            aa_ht_pd  = Sing_Trigger(sht_aa, Ht_cut_pd)

            tt_ht_grpo = Sing_Trigger(sht_tt, Ht_cut_grpo)
            aa_ht_grpo = Sing_Trigger(sht_aa, Ht_cut_grpo)

            tt_ht_dqn = Sing_Trigger(sht_tt, Ht_cut_dqn)
            aa_ht_dqn = Sing_Trigger(sht_aa, Ht_cut_dqn)

            R_ht_const_pct.append(bg_ht_const)
            R_ht_pd_pct.append(bg_ht_pd)
            R_ht_grpo_pct.append(bg_ht_grpo)
            R_ht_dqn_pct.append(bg_ht_dqn)

            Cut_ht_pd.append(Ht_cut_pd)
            Cut_ht_grpo.append(Ht_cut_grpo)
            Cut_ht_dqn.append(Ht_cut_dqn)

            TT_ht_const.append(tt_ht_const); TT_ht_pd.append(tt_ht_pd); TT_ht_grpo.append(tt_ht_grpo); TT_ht_dqn.append(tt_ht_dqn)
            AA_ht_const.append(aa_ht_const); AA_ht_pd.append(aa_ht_pd); AA_ht_grpo.append(aa_ht_grpo); AA_ht_dqn.append(aa_ht_dqn)

        if t % 5 == 0:
            print(f"[chunk {t:4d}] "
                  f"AS bg% const={bg_const:.3f} pd={bg_pd:.3f} dqn={bg_dqn:.3f} grpo={bg_grpo:.3f} "
                  f"| cut pd={AS_cut_pd:.5f} dqn={AS_cut_dqn:.5f} grpo={AS_cut_grpo:.5f} "
                  f"| grpo_reward={rewards[-1]} grpo_loss={losses[-1] if losses else None} "
                  f"| dqn_loss={dqn_losses[-1] if dqn_losses else None}")
            if args.run_ht:
                print(
                    f"           "
                    f"HT bg% const={bg_ht_const:.3f} pd={bg_ht_pd:.3f} dqn={bg_ht_dqn:.3f} grpo={bg_ht_grpo:.3f} "
                    f"| Ht_cut pd={Ht_cut_pd:.3f} dqn={Ht_cut_dqn:.3f} grpo={Ht_cut_grpo:.3f} "
                    f"| ht_grpo_loss={ht_losses[-1] if ht_losses else None} "
                    f"| ht_dqn_loss={dqn_ht_losses[-1] if dqn_ht_losses else None}"
                )

        
    # outside the batch starts loop
    # ----------------------------- arrays -----------------------------
    R_const_pct = np.asarray(R_const_pct, dtype=np.float64)
    R_pd_pct = np.asarray(R_pd_pct, dtype=np.float64)
    R_grpo_pct = np.asarray(R_grpo_pct, dtype=np.float64)
    Cut_pd = np.asarray(Cut_pd, dtype=np.float64)
    Cut_grpo = np.asarray(Cut_grpo, dtype=np.float64)
    TT_const = np.asarray(TT_const, dtype=np.float64)
    TT_pd = np.asarray(TT_pd, dtype=np.float64)
    TT_grpo = np.asarray(TT_grpo, dtype=np.float64)
    AA_const = np.asarray(AA_const, dtype=np.float64)
    AA_pd = np.asarray(AA_pd, dtype=np.float64)
    AA_grpo = np.asarray(AA_grpo, dtype=np.float64)
    # dqn
    R_dqn_pct = np.asarray(R_dqn_pct, dtype=np.float64)
    Cut_dqn   = np.asarray(Cut_dqn, dtype=np.float64)
    TT_dqn    = np.asarray(TT_dqn, dtype=np.float64)
    AA_dqn    = np.asarray(AA_dqn, dtype=np.float64)

    # ----------------------------- arrays (HT) -----------------------------
    if args.run_ht:
        R_ht_const_pct = np.asarray(R_ht_const_pct, dtype=np.float64)
        R_ht_pd_pct    = np.asarray(R_ht_pd_pct, dtype=np.float64)
        R_ht_dqn_pct   = np.asarray(R_ht_dqn_pct, dtype=np.float64)
        R_ht_grpo_pct  = np.asarray(R_ht_grpo_pct, dtype=np.float64)

        Cut_ht_pd   = np.asarray(Cut_ht_pd, dtype=np.float64)
        Cut_ht_dqn  = np.asarray(Cut_ht_dqn, dtype=np.float64)
        Cut_ht_grpo = np.asarray(Cut_ht_grpo, dtype=np.float64)

        TT_ht_const = np.asarray(TT_ht_const, dtype=np.float64)
        TT_ht_pd    = np.asarray(TT_ht_pd, dtype=np.float64)
        TT_ht_dqn   = np.asarray(TT_ht_dqn, dtype=np.float64)
        TT_ht_grpo  = np.asarray(TT_ht_grpo, dtype=np.float64)

        AA_ht_const = np.asarray(AA_ht_const, dtype=np.float64)
        AA_ht_pd    = np.asarray(AA_ht_pd, dtype=np.float64)
        AA_ht_dqn   = np.asarray(AA_ht_dqn, dtype=np.float64)
        AA_ht_grpo  = np.asarray(AA_ht_grpo, dtype=np.float64)

    time = np.linspace(0, 1, len(R_const_pct))

    # HT time and kHz
    if args.run_ht: 
        time_ht = np.linspace(0, 1, len(R_ht_const_pct))
        R_ht_const_khz = R_ht_const_pct * RATE_SCALE_KHZ
        R_ht_pd_khz    = R_ht_pd_pct    * RATE_SCALE_KHZ
        R_ht_dqn_khz   = R_ht_dqn_pct   * RATE_SCALE_KHZ
        R_ht_grpo_khz  = R_ht_grpo_pct  * RATE_SCALE_KHZ

    # kHz for plots
    R_const_khz = R_const_pct * RATE_SCALE_KHZ
    R_pd_khz = R_pd_pct * RATE_SCALE_KHZ
    R_grpo_khz = R_grpo_pct * RATE_SCALE_KHZ
    R_dqn_khz = R_dqn_pct * RATE_SCALE_KHZ

    target_khz = target * RATE_SCALE_KHZ
    tol_khz = tol * RATE_SCALE_KHZ
    upper_tol_khz = target_khz + tol_khz
    lower_tol_khz = target_khz - tol_khz


    def plot_rel_local(time, const, pid, dqn, grpo, ylabel, title, outpath):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(time, rel_to_t0(const), linestyle="--", linewidth=2.2, label="Constant")
        ax.plot(time, rel_to_t0(pid), linewidth=2.2, label="PID")
        ax.plot(time, rel_to_t0(dqn), linewidth=3.0, linestyle=(0, (8, 2, 2, 2)),
                marker="o", markersize=4, markevery=6, label="DQN")
        ax.plot(time, rel_to_t0(grpo), linewidth=3.2, linestyle=(0, (10, 2, 2, 2)), label="GRPO")
        ax.set_xlabel("Time (Fraction of Run)")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0.5, 2.5)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="best", frameon=True, title=title)
        add_cms_header(fig, run_label=run_label)
        finalize_diag_fig(fig)
        save_png(fig, str(outpath))
        plt.close(fig)

    plot_rel_local(time, TT_const, TT_pd, TT_dqn, TT_grpo,
               ylabel="Relative Efficiency", title="ttbar", outpath=outdir/"L_tt_eff_all_methods")
    plot_rel_local(time, AA_const, AA_pd, AA_dqn, AA_grpo,
               ylabel="Relative Efficiency", title="HToAATo4B", outpath=outdir/"L_aa_eff_all_methods")


    def plot_rel_cum(time, const, pid, dqn, grpo, ylabel, title, outpath):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(time, rel_to_t0(cummean(const)), linestyle="--", linewidth=2.2, label="Constant")
        ax.plot(time, rel_to_t0(cummean(pid)), linewidth=2.2, label="PID")
        ax.plot(time, rel_to_t0(cummean(dqn)), linewidth=3.0, linestyle=(0, (8, 2, 2, 2)),
            marker="o", markersize=4, markevery=6, label="DQN")
        ax.plot(time, rel_to_t0(cummean(grpo)), linewidth=3.2, linestyle=(0, (10, 2, 2, 2)), label="GRPO")
        ax.set_xlabel("Time (Fraction of Run)")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0.5, 2.5)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="best", frameon=True, title=title)
        add_cms_header(fig, run_label=run_label)
        finalize_diag_fig(fig)
        save_png(fig, str(outpath))
        plt.close(fig)

    plot_rel_cum(time, TT_const, TT_pd, TT_dqn, TT_grpo,
             ylabel="Relative Cumulative Efficiency", title="ttbar", outpath=outdir/"C_tt_eff_all_methods")
    plot_rel_cum(time, AA_const, AA_pd, AA_dqn, AA_grpo,
             ylabel="Relative Cumulative Efficiency", title="HToAATo4B", outpath=outdir/"C_aa_eff_all_methods")

    # ----------------------------- core plots -----------------------------
    # ----------------------------- HT core plots -----------------------------
    if args.run_ht:
        # HT rate plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(time_ht, R_ht_const_khz, linestyle="--", linewidth=2.4, label="Constant")
        ax.plot(time_ht, R_ht_pd_khz,    linewidth=2.4, label="PID")
        ax.plot(time_ht, R_ht_dqn_khz,   linewidth=3.0, linestyle=(0, (8, 2, 2, 2)),
            marker="o", markersize=4, markevery=6, label="DQN")
        ax.plot(time_ht, R_ht_grpo_khz,  linewidth=3.0, linestyle=(0, (8, 2, 2, 2)), label="GRPO")

        ax.axhline(upper_tol_khz, linestyle="--", linewidth=1.2)
        ax.axhline(lower_tol_khz, linestyle="--", linewidth=1.2)
        ax.fill_between(time_ht, lower_tol_khz, upper_tol_khz, alpha=0.12, label="Tolerance band")

        ax.set_xlabel("Time (Fraction of Run)")
        ax.set_ylabel("Background rate [kHz]")
        ax.set_ylim(0, 200)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="best", frameon=True, title="HT Trigger")
        add_cms_header(fig, run_label=run_label)
        finalize_diag_fig(fig)
        save_png(fig, str(outdir / "bht_rate_pidData_grpo"))
        plt.close(fig)

        # HT cut plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(time_ht, Cut_ht_pd,   linewidth=2.4, label="PID")
        ax.plot(time_ht, Cut_ht_dqn,  linewidth=2.8, linestyle=(0, (8, 2, 2, 2)), label="DQN")
        ax.plot(time_ht, Cut_ht_grpo, linewidth=3.0, linestyle=(0, (10, 2, 2, 2)), label="GRPO")
        ax.axhline(y=fixed_Ht_cut, color="gray", linestyle="--", linewidth=1.5, label="fixed_Ht_cut")
        ax.set_xlabel("Time (Fraction of Run)")
        ax.set_ylabel("Ht_cut")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="best", frameon=True, title="HT Cut")
        add_cms_header(fig, run_label=run_label)
        finalize_diag_fig(fig)
        save_png(fig, str(outdir / "ht_cut_all_methods"))
        plt.close(fig)

    # rate plot (main)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time, R_const_khz, linestyle="--", linewidth=2.4, label="Constant")
    ax.plot(time, R_pd_khz, linewidth=2.4, label="PID")
    ax.plot(time, R_dqn_khz, linewidth=3.0, linestyle=(0, (8, 2, 2, 2)), marker="o",
            markersize=4, markevery=6, label="DQN")
    ax.plot(time, R_grpo_khz, linewidth=3.0, linestyle=(0, (8, 2, 2, 2)), label="GRPO")

    ax.axhline(upper_tol_khz, linestyle="--", linewidth=1.2)
    ax.axhline(lower_tol_khz, linestyle="--", linewidth=1.2)
    ax.fill_between(time, lower_tol_khz, upper_tol_khz, alpha=0.12, label="Tolerance band")

    ax.set_xlabel("Time (Fraction of Run)")
    ax.set_ylabel("Background rate [kHz]")
    ax.set_ylim(0, 200)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best", frameon=True, title="AD Trigger")
    add_cms_header(fig, run_label=run_label)
    finalize_diag_fig(fig)
    save_png(fig, str(outdir / "bas_rate_pidData_grpo"))
    plt.close(fig)

    # cut evolution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time, Cut_pd, linewidth=2.4, label="PID")
    ax.plot(
        time, Cut_dqn,
        linewidth=2.8,
        label="DQN",
    )
    ax.plot(time, Cut_grpo, linewidth=2.4, linestyle=(0, (8, 2, 2, 2)), label="GRPO")
    ax.axhline(y=fixed_AS_cut, color="gray", linestyle="--", linewidth=1.5, label="fixed_AS_cut")
    ax.set_xlabel("Time (Fraction of Run)")
    ax.set_ylabel("AD_cut")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best", frameon=True, title="AD Cut")
    add_cms_header(fig, run_label=run_label)
    finalize_diag_fig(fig)
    save_png(fig, str(outdir / "as_cut_pidData_grpo"))
    plt.close(fig)

    # reward trace
    if rewards:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(time, np.asarray(rewards, dtype=np.float32), linewidth=1.5)
        ax.set_xlabel("Time (Fraction of Run)")
        ax.set_ylabel("Mean micro reward")
        ax.grid(True, linestyle="--", alpha=0.5)
        add_cms_header(fig, run_label=run_label)
        save_png(fig, str(outdir / "reward_as_pidData_grpo"))
        plt.close(fig)

    # loss trace
    if losses:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(np.arange(len(losses)), losses, linewidth=1.5)
        ax.set_xlabel("Policy update index")
        ax.set_ylabel("Loss")
        ax.grid(True, linestyle="--", alpha=0.5)
        add_cms_header(fig, run_label=run_label)
        save_png(fig, str(outdir / "grpo_loss_as"))
        finalize_diag_fig(fig)
        plt.close(fig)

    # ----------------------------- showcase plots (paper): PD vs DQN vs GRPO -----------------------------
    w = int(args.run_avg_window)

    # ===== AD (AS trigger) =====
    rate_khz_ad = {
        "Constant": R_const_khz,
        "PID":  R_pd_khz,
        "DQN": R_dqn_khz,
        "GRPO": R_grpo_khz,
    }
    inband_ad = {
        "Constant": (np.abs(R_const_pct - target) <= tol),
        "PID":  (np.abs(R_pd_pct  - target) <= tol),
        "DQN": (np.abs(R_dqn_pct - target) <= tol),
        "GRPO":(np.abs(R_grpo_pct - target) <= tol),
    }
    # For Constant cut history, create a flat cut trace of same length
    Cut_const_ad = np.full_like(Cut_pd, fixed_AS_cut, dtype=np.float64)
    cuts_ad = {
        "Constant": Cut_const_ad,
        "PID":  Cut_pd,
        "DQN": Cut_dqn,
        "GRPO": Cut_grpo,
    }

    sum_const_ad = summarize_paper_table(R_const_pct, TT_const, AA_const, Cut_const_ad, target, tol)
    sum_pd_ad    = summarize_paper_table(R_pd_pct,    TT_pd,    AA_pd,    Cut_pd,       target, tol)
    sum_dqn_ad   = summarize_paper_table(R_dqn_pct,   TT_dqn,   AA_dqn,   Cut_dqn,      target, tol)
    sum_gr_ad    = summarize_paper_table(R_grpo_pct,  TT_grpo,  AA_grpo,  Cut_grpo,     target, tol)
    summ_ad = {"Constant": sum_const_ad, "PID": sum_pd_ad, "DQN": sum_dqn_ad, "GRPO": sum_gr_ad}

    plot_cdf_abs_err_multi(
        rate_khz_by_method=rate_khz_ad,
        target_khz=target_khz, tol_khz=tol_khz,
        title="AD Trigger", outpath=plots_dir / "cdf_abs_err_ad_const_pd_dqn_grpo",
        run_label=run_label,
    )
    plot_running_inband_multi(
        time=time, inband_by_method=inband_ad, w=w,
        title="AD Trigger", outpath=plots_dir / "running_inband_ad_const_pd_dqn_grpo",
        run_label=run_label,
    )
    plot_cut_step_hist_multi(
        cut_by_method=cuts_ad,
        xlabel=r"$|\Delta AS\_cut|$",
        title="AD Trigger",
        outpath=plots_dir / "cut_step_hist_ad_const_pd_dqn_grpo",
        run_label=run_label,
    )
    # plot_inband_eff_bars_multi(
    #     summary_by_method=summ_ad,
    #     title="AD Trigger",
    #     outpath=plots_dir / "inband_eff_bars_ad_const_pd_dqn_grpo",
    #     run_label=run_label,
    # )

    # ===== HT trigger (only if enabled) =====
    if args.run_ht:
        rate_khz_ht = {
            "Constant": R_ht_const_khz,
            "PID":  R_ht_pd_khz,
            "DQN": R_ht_dqn_khz,
            "GRPO": R_ht_grpo_khz,
        }
        inband_ht = {
            "Constant": (np.abs(R_ht_const_pct - target) <= tol),
            "PID":  (np.abs(R_ht_pd_pct  - target) <= tol),
            "DQN": (np.abs(R_ht_dqn_pct - target) <= tol),
            "GRPO":(np.abs(R_ht_grpo_pct - target) <= tol),
        }
        Cut_const_ht = np.full_like(Cut_ht_pd, fixed_Ht_cut, dtype=np.float64)
        cuts_ht = {
            "Constant": Cut_const_ht,
            "PID":  Cut_ht_pd,
            "DQN": Cut_ht_dqn,
            "GRPO": Cut_ht_grpo,
        }

        sum_const_ht = summarize_paper_table(R_ht_const_pct, TT_ht_const, AA_ht_const, Cut_const_ht, target, tol)
        sum_pd_ht    = summarize_paper_table(R_ht_pd_pct,    TT_ht_pd,    AA_ht_pd,    Cut_ht_pd,    target, tol)
        sum_dqn_ht   = summarize_paper_table(R_ht_dqn_pct,   TT_ht_dqn,   AA_ht_dqn,   Cut_ht_dqn,   target, tol)
        sum_gr_ht    = summarize_paper_table(R_ht_grpo_pct,  TT_ht_grpo,  AA_ht_grpo,  Cut_ht_grpo,  target, tol)
        summ_ht = {"Constant": sum_const_ht, "PID": sum_pd_ht, "DQN": sum_dqn_ht, "GRPO": sum_gr_ht}


        plot_cdf_abs_err_multi(
            rate_khz_by_method=rate_khz_ht,
            target_khz=target_khz, tol_khz=tol_khz,
            title="HT Trigger", outpath=plots_dir / "cdf_abs_err_ht_const_pd_dqn_grpo",
            run_label=run_label,
        )
        plot_running_inband_multi(
            time=time_ht, inband_by_method=inband_ht, w=w,
            title="HT Trigger", outpath=plots_dir / "running_inband_ht_const_pd_dqn_grpo",
            run_label=run_label,
        )
        plot_cut_step_hist_multi(
            cut_by_method=cuts_ht,
            xlabel=r"$|\Delta Ht\_cut|$",
            title="HT Trigger",
            outpath=plots_dir / "cut_step_hist_ht_const_pd_dqn_grpo",
            run_label=run_label,
        )


    # ----------------------------- paper summary table (CSV + LaTeX) -----------------------------
    rows = []

    def add_row(trigger, method, dct):
        r = {"Trigger": trigger, "Method": method}
        r.update(dct)
        rows.append(r)

    # ---- AD rows ----
    Cut_const_ad = np.full_like(Cut_pd, fixed_AS_cut, dtype=np.float64)

    sum_const_ad = summarize_paper_table(R_const_pct, TT_const, AA_const, Cut_const_ad, target, tol)
    sum_pd_ad    = summarize_paper_table(R_pd_pct,    TT_pd,    AA_pd,    Cut_pd,      target, tol)
    sum_dqn_ad   = summarize_paper_table(R_dqn_pct,   TT_dqn,   AA_dqn,   Cut_dqn,     target, tol)
    sum_gr_ad    = summarize_paper_table(R_grpo_pct,  TT_grpo,  AA_grpo,  Cut_grpo,    target, tol)

    add_row("AD", "Constant", sum_const_ad)
    add_row("AD", "PID",      sum_pd_ad)
    add_row("AD", "DQN",      sum_dqn_ad)
    add_row("AD", "GRPO",     sum_gr_ad)

    # ---- HT rows ----
    if args.run_ht:
        Cut_const_ht = np.full_like(Cut_ht_pd, fixed_Ht_cut, dtype=np.float64)

        sum_const_ht = summarize_paper_table(R_ht_const_pct, TT_ht_const, AA_ht_const, Cut_const_ht, target, tol)
        sum_pd_ht    = summarize_paper_table(R_ht_pd_pct,    TT_ht_pd,    AA_ht_pd,    Cut_ht_pd,    target, tol)
        sum_dqn_ht   = summarize_paper_table(R_ht_dqn_pct,   TT_ht_dqn,   AA_ht_dqn,   Cut_ht_dqn,   target, tol)
        sum_gr_ht    = summarize_paper_table(R_ht_grpo_pct,  TT_ht_grpo,  AA_ht_grpo,  Cut_ht_grpo,  target, tol)

        add_row("HT", "Constant", sum_const_ht)
        add_row("HT", "PID",      sum_pd_ht)
        add_row("HT", "DQN",      sum_dqn_ht)
        add_row("HT", "GRPO",     sum_gr_ht)

    out_csv = tables_dir / "single_trigger_compact_summary.csv"
    out_tex = tables_dir / "single_trigger_compact_summary.tex"
    write_paper_table(rows, out_csv, out_tex, target_pct=target, tol_pct=tol)

    
    
    # ----------------------------- Advantage distribution plots -----------------------------
    # Inspired by: https://arxiv.org/pdf/2504.08837 VL rethinker
    # Note: paper-style normalized advantages use (r - mean)/std. :contentReference[oaicite:1]{index=1}

    # AS (AD trigger)
    adv_raw_as, adv_norm_as, adv_raw_exec_as, adv_norm_exec_as, frac_vanish_as = \
        _group_advantages_from_grpo_samples(grpo_samples, trigger="AS", baseline="mean")

    _plot_adv_hist_and_ecdf(
        adv_raw_as,
        title=f"GRPO Candidate Advantage (AS)  | vanish={frac_vanish_as:.2%}",
        xlabel=r"$A = r - \mathrm{mean}(r)$",
        outpath_prefix=plots_dir / "adv_as_candidate_raw",
        run_label=run_label,
    )
    _plot_adv_hist_and_ecdf(
        adv_norm_as,
        title=f"GRPO Candidate Advantage Norm (AS)  | vanish={frac_vanish_as:.2%}",
        xlabel=r"$\hat A = (r-\mathrm{mean}(r))/\mathrm{std}(r)$",
        outpath_prefix=plots_dir / "adv_as_candidate_norm",
        run_label=run_label,
    )
    _plot_adv_hist_and_ecdf(
        adv_raw_exec_as,
        title="GRPO Executed Advantage (AS) (post-shield)",
        xlabel=r"$A_{\mathrm{exec}} = r_{\mathrm{exec}} - \mathrm{mean}(r_{\mathrm{cand}})$",
        outpath_prefix=plots_dir / "adv_as_executed_raw",
        run_label=run_label,
    )
    _plot_adv_hist_and_ecdf(
        adv_norm_exec_as,
        title="GRPO Executed Advantage Norm (AS) (post-shield)",
        xlabel=r"$\hat A_{\mathrm{exec}}$",
        outpath_prefix=plots_dir / "adv_as_executed_norm",
        run_label=run_label,
    )

    near_occ_as_grpo = np.asarray(near_occ_as_grpo, dtype=np.float32)  # (Tchunk, W)

    if near_occ_as_grpo.size:
        fig, ax = plt.subplots(figsize=(10, 8))
        for k, w0 in enumerate(near_widths_as):
            ax.plot(time, near_occ_as_grpo[:, k], linewidth=2.0,
                label=fr"$|AS-\theta|\leq {w0:g}$")
        style_diag_axes(ax, xlabel="Time (Fraction of Run)", ylabel="Near-cut occupancy (fraction)")
        style_diag_legend(ax, title="GRPO near-cut window")
        finalize_diag_fig(fig)
        add_cms_header(fig, run_label=run_label)
        save_png(fig, str(plots_dir / "near_cut_occupancy_as_chunk_grpo"))
        plt.close(fig)

    # HT (optional)
    if args.run_ht:
        adv_raw_ht, adv_norm_ht, adv_raw_exec_ht, adv_norm_exec_ht, frac_vanish_ht = \
            _group_advantages_from_grpo_samples(grpo_samples, trigger="HT", baseline="mean")

        _plot_adv_hist_and_ecdf(
            adv_raw_ht,
            title=f"GRPO Candidate Advantage (HT)  | vanish={frac_vanish_ht:.2%}",
            xlabel=r"$A = r - \mathrm{mean}(r)$",
            outpath_prefix=plots_dir / "adv_ht_candidate_raw",
            run_label=run_label,
        )
        _plot_adv_hist_and_ecdf(
            adv_norm_ht,
            title=f"GRPO Candidate Advantage Norm (HT)  | vanish={frac_vanish_ht:.2%}",
            xlabel=r"$\hat A = (r-\mathrm{mean}(r))/\mathrm{std}(r)$",
            outpath_prefix=plots_dir / "adv_ht_candidate_norm",
            run_label=run_label,
        )
        _plot_adv_hist_and_ecdf(
            adv_raw_exec_ht,
            title="GRPO Executed Advantage (HT) (post-shield)",
            xlabel=r"$A_{\mathrm{exec}}$",
            outpath_prefix=plots_dir / "adv_ht_executed_raw",
            run_label=run_label,
        )
        _plot_adv_hist_and_ecdf(
            adv_norm_exec_ht,
            title="GRPO Executed Advantage Norm (HT) (post-shield)",
            xlabel=r"$\hat A_{\mathrm{exec}}$",
            outpath_prefix=plots_dir / "adv_ht_executed_norm",
            run_label=run_label,
        )

        # ttbar-only plot: start y from 90
        plot_inband_eff_single_signal_ad_vs_ht(
            summ_ad, summ_ht,
            signal_key="tt",
            signal_label=r"$t\bar{t}$",
            outpath=plots_dir / "inband_eff_tt_ad_vs_ht",
            run_label=run_label,
            ymin=90.0,
            ymax_pad=2.0,
        )

        # h->4b-only plot: start y from 15
        plot_inband_eff_single_signal_ad_vs_ht(
            summ_ad, summ_ht,
            signal_key="h_to_4b",
            signal_label=r"$h\rightarrow 4b$",
            outpath=plots_dir / "inband_eff_h4b_ad_vs_ht",
            run_label=run_label,
            ymin=15.0,
            ymax_pad=2.0,
        )
        near_occ_ht_grpo = np.asarray(near_occ_ht_grpo, dtype=np.float32)  # (Tchunk, W)

        if near_occ_ht_grpo.size:
            fig, ax = plt.subplots(figsize=(10, 8))
            for k, w0 in enumerate(near_widths_ht):
                ax.plot(time_ht, near_occ_ht_grpo[:, k], linewidth=2.0,
                    label=fr"$|HT-\theta|\leq {w0:g}$ GeV")
            style_diag_axes(ax, xlabel="Time (Fraction of Run)", ylabel="Near-cut occupancy (fraction)")
            style_diag_legend(ax, title="GRPO near-cut window")
            finalize_diag_fig(fig)
            add_cms_header(fig, run_label=run_label)
            save_png(fig, str(plots_dir / "near_cut_occupancy_ht_chunk_grpo"))
            plt.close(fig)


    print(f"[OK] Wrote: {out_csv}")
    print(f"[OK] Wrote: {out_tex}")




if __name__ == "__main__":
    main()
