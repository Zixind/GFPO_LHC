#!/usr/bin/env python3
"""
https://arxiv.org/pdf/2312.01488
Implementation of this paper

ADT_Ht_AS_feature.py

ADT baseline (from "ADT: Agent-based Dynamic Thresholding for Anomaly Detection")
adapted to LHC single-trigger threshold control (HT + AD/AS).

Key ADT training behaviors from the paper:
  (1) Action is updated only every l steps; otherwise hold a_t = a_{t-1}.
  (2) Q-network is updated only at the end of each episode (here: end of chunk).

We map:
  - episode  <-> chunk
  - time step t <-> micro-step within chunk (stride over events)

We provide two reward options:
  --reward-mode lhc   : use the existing compute_reward(...) (rate tracking + signal bonus + step penalty)
  --reward-mode paper : use ADT paper-style reward r = α*(TP - FP - FN) + β*TN (computed from current micro-step)
                        (we normalize by counts to keep magnitude stable)

Outputs:
  outdir = RL_outputs/demo_sing_adt_feature
  HT:
    - bht_rate_pidData_adt.png
    - ht_cut_pidData_adt.png
    - sht_rate_pidData2data_adt.png
    - L_sht_rate_pidData2data_adt.png
    - adt_loss_ht.png
    - reward_ht_pidData_adt.png
  AS:
    - bas_rate_pidData_adt.png
    - as_cut_pidData_adt.png
    - sas_rate_pidData2data_adt.png
    - L_sas_rate_pidData2data_adt.png
    - adt_loss_as.png
    - reward_as_pidData_adt.png
"""

from dataclasses import dataclass
from collections import deque
from pathlib import Path
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
import h5py
import hdf5plugin  # noqa: F401

from controllers import PD_controller1, PD_controller2
from triggers import Sing_Trigger
from RL.utils import (
    cummean, rel_to_t0, add_cms_header, plot_rate_with_tolerance,
    save_png, print_h5_tree, read_any_h5, set_paper_style,
    style_diag_axes, style_diag_legend, finalize_diag_fig,
)
from RL.dqn_agent import SeqDQNAgent, DQNConfig, make_event_seq_ht_v0, make_event_seq_as_v0, shield_delta, compute_reward

# ------------------------- Fixing seed for reproducibility -------------------------
SEED = 20251213
random.seed(SEED)
np.random.seed(SEED)

set_paper_style()



@dataclass
class RollingWindow:
    """Sliding window of recent background events for feature construction."""
    max_events: int

    def __post_init__(self):
        self.max_events = int(self.max_events)
        self._bht = deque(maxlen=self.max_events)
        self._bas = deque(maxlen=self.max_events)
        self._bnpv = deque(maxlen=self.max_events)

    def append(self, bht, bas, bnpv):
        self._bht.extend(np.asarray(bht, dtype=np.float32).tolist())
        self._bas.extend(np.asarray(bas, dtype=np.float32).tolist())
        self._bnpv.extend(np.asarray(bnpv, dtype=np.float32).tolist())

    def get(self):
        return (
            np.fromiter(self._bht, dtype=np.float32),
            np.fromiter(self._bas, dtype=np.float32),
            np.fromiter(self._bnpv, dtype=np.float32),
        )


def adt_reward_paper_style(bg_scores, sig1_scores, sig2_scores, cut, alpha=0.7, beta=0.3):
    """
    Original reward from ADT (Eq.(8)): r = α*(TP - FP - FN) + β*TN.
    Here we treat:
      - background events as "normal" (negative class)
      - signal events as "abnormal" (positive class)

    We normalize by total counts to keep reward scale stable across micro-step sizes.
    """
    alpha = float(alpha)
    beta = float(beta)
    s_b = np.asarray(bg_scores, dtype=np.float32)
    s_s1 = np.asarray(sig1_scores, dtype=np.float32)
    s_s2 = np.asarray(sig2_scores, dtype=np.float32)
    s_s = np.concatenate([s_s1, s_s2], axis=0) if (s_s1.size + s_s2.size) > 0 else np.empty(0, np.float32)

    # Predict anomaly if score >= cut (accept)
    fp = int(np.sum(s_b >= cut))                 # normal accepted
    tn = int(np.sum(s_b <  cut))                 # normal rejected
    tp = int(np.sum(s_s >= cut)) if s_s.size else 0
    fn = int(np.sum(s_s <  cut)) if s_s.size else 0

    # Normalize
    nb = max(1, fp + tn)
    ns = max(1, tp + fn)

    tp_n = tp / ns
    fn_n = fn / ns
    fp_n = fp / nb
    tn_n = tn / nb

    r = alpha * (tp_n - fp_n - fn_n) + beta * tn_n
    return float(r)


def moving_avg_nan(x, w=5):
    x = np.asarray(x, dtype=np.float32)
    m = np.isfinite(x).astype(np.float32)
    x0 = np.nan_to_num(x, nan=0.0)
    k = np.ones(w, dtype=np.float32)
    num = np.convolve(x0, k, mode="same")
    den = np.convolve(m,  k, mode="same")
    return num / np.maximum(den, 1e-8)

def _truncate_to_min_len(*arrs):
    """Truncate all 1D arrays/lists to the same minimum length."""
    L = min(len(a) for a in arrs if a is not None)
    out = []
    for a in arrs:
        if a is None:
            out.append(None)
        else:
            out.append(np.asarray(a)[:L])
    return out

def summarize_table_metrics(r_pct, s_tt, s_aa, cut_hist, target_pct, tol_pct):
    """
    Metrics used in your LaTeX table, evaluated in percent units.
    Columns:
      MAE, P95|e|, InBand, UpViol, TV, ttbar(inband), h->4b(inband)
    """
    r_pct = np.asarray(r_pct, dtype=np.float64)
    s_tt  = np.asarray(s_tt,  dtype=np.float64)
    s_aa  = np.asarray(s_aa,  dtype=np.float64)
    cut   = np.asarray(cut_hist, dtype=np.float64)

    # Align lengths safely 
    r_pct, s_tt, s_aa, cut = _truncate_to_min_len(r_pct, s_tt, s_aa, cut)

    err = r_pct - float(target_pct)
    abs_err = np.abs(err)
    inband = abs_err <= float(tol_pct)

    upper = float(target_pct) + float(tol_pct)
    lower = float(target_pct) - float(tol_pct)

    out = {}
    out["MAE"]      = float(np.mean(abs_err)) if abs_err.size else np.nan
    out["P95"]      = float(np.percentile(abs_err, 95)) if abs_err.size else np.nan
    out["InBand"]   = float(np.mean(inband)) if inband.size else np.nan
    out["UpFrac"]   = float(np.mean(r_pct > upper)) if r_pct.size else np.nan
    out["DownFrac"] = float(np.mean(r_pct < lower)) if r_pct.size else np.nan


    dc = np.diff(cut) if cut.size >= 2 else np.array([], dtype=np.float64)

    def safe_mean(x, m):
        return float(np.mean(x[m])) if np.any(m) else np.nan

    out["ttbar"] = safe_mean(s_tt, inband)
    out["h4b"]   = safe_mean(s_aa, inband)
    return out

def write_adt_table(rows, tex_path, caption, label):
    """
    Write a LaTeX table matching your screenshot style (two blocks: HT trigger, AD trigger).
    rows: list of dicts with keys:
      Trigger, Method, MAE, P95, InBand, UpViol, TV, ttbar, h4b
    """
    def fmt(x):
        if x is None:
            return "xx"
        if isinstance(x, (float, np.floating)):
            if not np.isfinite(x):
                return "xx"
            # table-like compact formatting
            return f"{x:.3g}"
        return str(x)

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{5pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.08}")
    lines.append(r"\begin{tabular}{llrrrrrrr}")
    lines.append(r"\hline")
    lines.append(r"Trigger & Method & MAE$\downarrow$ & P95$|e|\,\downarrow$ & InBand$\uparrow$ & UpViol$\downarrow$ & UpFrac$\downarrow$ & DownFrac$\downarrow$ & $\bar{t}\bar{t}\uparrow$ & $h\to4b\uparrow$ \\")
    lines.append(r"\hline")

    def emit_block(block_title, trig_key):
        lines.append(rf"\multicolumn{{9}}{{l}}{{\textbf{{{block_title}}}}} \\")
        for r in rows:
            if r["Trigger"] != trig_key:
                continue
            lines.append(
                f"{fmt(r['Trigger'])} & {fmt(r['Method'])} & "
                f"{fmt(r['MAE'])} & {fmt(r['P95'])} & {fmt(r['InBand'])} & {fmt(r['UpFrac'])} & "
                f"{fmt(r['DownFrac'])} & {fmt(r['ttbar'])} & {fmt(r['h4b'])} \\\\"
            )
        lines.append(r"\hline")

    emit_block("HT trigger", "HT")
    emit_block("AD trigger", "AD")

    lines.append(r"\end{tabular}")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\end{table}")

    Path(tex_path).write_text("\n".join(lines) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="Data/Trigger_food_MC.h5",
                    choices=["Data/Trigger_food_MC.h5", "Data/Matched_data_2016_dim2.h5", "Data/Trigger_food_MC_ablation_4.h5", "Data/Trigger_food_MC_ablation_6.h5", "Data/Trigger_food_MC_ablation_8.h5", "Data/Trigger_food_MC_ablation_10.h5", "Data/Trigger_food_MC_ablation_12.h5", "Data/Trigger_food_MC_ablation_14.h5", "Data/Trigger_food_MC_ablation_16.h5"])
    ap.add_argument("--outdir", default="outputs/demo_sing_adt_feature", help="output dir")
    ap.add_argument("--control", default="MC", choices=["MC", "RealData"],
                    help="Control type: MC or RealData")
    ap.add_argument("--score-dim-hint", type=int, default=2,
                    help="If file has only scoreXX, use this dim (e.g. 2 -> score02).")
    ap.add_argument("--as-dim", type=int, default=2, choices=[1, 2, 4, 6, 8, 10, 12, 14, 16],
                    help="Which AS dimension to use (1->score01, 2->score02, 4->score04).")

    ap.add_argument("--print-keys", action="store_true")
    ap.add_argument("--print-keys-max", type=int, default=None)

    # micro-step / sequence features
    ap.add_argument("--window-events-chunk-size", type=int, default=3,
                    help="How many chunks worth of background events to keep in the rolling feature window.")
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--inner-stride", type=int, default=10000)
    ap.add_argument("--train-steps-per-episode", type=int, default=50,
                    help="ADT: number of gradient updates done ONLY at end of chunk (episode).")

    # ADT action-hold
    ap.add_argument("--adt-l", type=int, default=10,
                    help="ADT: update action every l micro-steps; otherwise hold a_t=a_{t-1}.")

    # action sets
    ap.add_argument("--ht-deltas", type=str, default="-2,-1,0,1,2")
    ap.add_argument("--as-deltas", type=str, default="-3,-1.5,0,1.5,3")
    ap.add_argument("--as-step", type=float, default=0.5)

    # reward setup
    ap.add_argument("--reward-mode", default="paper", choices=["lhc", "paper"],
                    help="lhc: use compute_reward; paper: use ADT paper-style TP/TN/FP/FN reward.")
    ap.add_argument("--alpha", type=float, default=0.4,
                    help="LHC reward: signal bonus weight. Paper reward: α in Eq.(8).")
    ap.add_argument("--beta", type=float, default=0.2,
                    help="LHC reward: move penalty weight. Paper reward: β in Eq.(8) (recommended α+β≈1).")
    ap.add_argument("--gamma-stab", type=float, default=0.3,
                    help="LHC reward: stability weight.")
    ap.add_argument("--target", type=float, default=0.25,
                    help="Target background acceptance rate in percent units.")
    ap.add_argument("--tol", type=float, default=0.02,
                    help="Tolerance band in percent units (|bg-target|<=tol).")

    args = ap.parse_args()

    run_label = "MC" if args.control == "MC" else "283408"

    if args.print_keys:
        print_h5_tree(args.input, max_items=args.print_keys_max)
        raise SystemExit(0)

    outdir = Path(args.outdir+"_"+args.control)
    outdir.mkdir(parents=True, exist_ok=True)

    d = read_any_h5(args.input, score_dim_hint=args.score_dim_hint)
    matched_by_index = bool(d["meta"].get("matched_by_index", False))

    Bht, Bnpv = d["Bht"], d["Bnpv"]
    Tht, Tnpv = d["Tht"], d["Tnpv"]
    Aht, Anpv = d["Aht"], d["Anpv"]

    # Pick AS dim (expects keys like Bas2/Tas2/Aas2, Bas4/Tas4/Aas4, ...)
    dim = int(args.as_dim)

    Bas = d.get(f"Bas{dim}")
    Tas = d.get(f"Tas{dim}")
    Aas = d.get(f"Aas{dim}")

    if Bas is None or Tas is None or Aas is None:
        raise SystemExit("AS arrays missing for requested --as-dim. Check your input file keys.")

    N = len(Bht)
    chunk_size = 50000 if args.control == "MC" else 20000
    start_event = max(0, (chunk_size * 10 // chunk_size) * chunk_size)
    if start_event + chunk_size > N:
        start_event = max(0, ((N - chunk_size) // chunk_size) * chunk_size)

    # fixed cuts from calibration window
    win_lo = min(start_event, N - 1)
    win_hi = min(start_event + (100000 if args.control == "MC" else 10000), N)

    fixed_Ht_cut = float(np.percentile(Bht[win_lo:win_hi], 99.75))
    fixed_AS_cut = float(np.percentile(Bas[win_lo:win_hi], 99.75))

    # clip ranges
    ht_lo = float(np.percentile(Bht[start_event:], 95.0))
    ht_hi = float(np.percentile(Bht[start_event:], 99.99))
    ht_mid = 0.5 * (ht_lo + ht_hi)
    ht_span = max(1.0, ht_hi - ht_lo)

    ref_as = Bas[win_lo:win_hi]
    as_lo = float(np.min(ref_as))
    as_hi = float(np.max(ref_as))
    as_mid = 0.5 * (as_lo + as_hi)
    as_span = max(1e-6, as_hi - as_lo)

    print(f"[INFO] matched_by_index={matched_by_index} N={N} chunk={chunk_size} start_event={start_event}")
    print(f"[HT] fixed={fixed_Ht_cut:.3f} clip=({ht_lo:.3f},{ht_hi:.3f}) window=[{win_lo}:{win_hi}]")
    print(f"[AS dim={args.as_dim}] fixed={fixed_AS_cut:.6f} clip=({as_lo:.6f},{as_hi:.6f}) as_step={args.as_step}")

    # Baselines init
    Ht_cut_pd = fixed_Ht_cut
    AS_cut_pd = fixed_AS_cut
    pre_ht_err = 0.0
    pre_as_err = 0.0

    # ADT init
    Ht_cut_adt = fixed_Ht_cut
    AS_cut_adt = fixed_AS_cut
    prev_bg_ht_adt = None
    prev_bg_as_adt = None
    last_dht_adt = 0.0
    last_das_adt = 0.0
    prev_act_ht_adt = 0
    prev_act_as_adt = 0

    # action grids
    HT_DELTAS = np.array([float(x) for x in args.ht_deltas.split(",")], dtype=np.float32)
    AS_DELTAS = np.array([float(x) for x in args.as_deltas.split(",")], dtype=np.float32)
    AS_STEP = float(args.as_step)

    MAX_DELTA_HT = float(np.max(np.abs(HT_DELTAS)))
    MAX_DELTA_AS = float(np.max(np.abs(AS_DELTAS))) * AS_STEP

    # agent configs
    cfg_ht = DQNConfig(lr=5e-4, gamma=0.95, batch_size=32, target_update=200)
    cfg_as = DQNConfig(lr=1e-4, gamma=0.95, batch_size=32, target_update=200)

    K = int(args.seq_len)
    near_widths_ht = (5.0, 10.0, 20.0)
    near_widths_as = (0.25, 0.5, 1.0)

    feat_dim_ht = 10 + len(near_widths_ht)
    feat_dim_as = 10 + len(near_widths_as)

    agent_ht = SeqDQNAgent(seq_len=K, feat_dim=feat_dim_ht, n_actions=len(HT_DELTAS), cfg=cfg_ht, seed=SEED)
    agent_as = SeqDQNAgent(seq_len=K, feat_dim=feat_dim_as, n_actions=len(AS_DELTAS), cfg=cfg_as, seed=SEED)

    roll = RollingWindow(max_events=int(args.window_events_chunk_size * chunk_size))

    # logs
    R_const_ht, R_pd_ht, R_adt_ht = [], [], []
    Ht_pd_hist, Ht_adt_hist = [], []
    L_tt_ht_const, L_tt_ht_pd, L_tt_ht_adt = [], [], []
    L_aa_ht_const, L_aa_ht_pd, L_aa_ht_adt = [], [], []

    R_const_as, R_pd_as, R_adt_as = [], [], []
    As_pd_hist, As_adt_hist = [], []
    L_tt_as_const, L_tt_as_pd, L_tt_as_adt = [], [], []
    L_aa_as_const, L_aa_as_pd, L_aa_as_adt = [], [], []

    losses_ht, losses_as = [], []
    rewards_ht, rewards_as = [], []

    batch_starts = list(range(start_event, N, chunk_size))

    target = float(args.target)
    tol = float(args.tol)
    alpha = float(args.alpha)
    beta = float(args.beta)

    stride = max(500, int(args.inner_stride))
    l_hold = max(1, int(args.adt_l))

    for t, I in enumerate(batch_starts):
        end = min(I + chunk_size, N, len(Bnpv), len(Bas))
        if end <= I:
            break

        idx = np.arange(I, end)
        bht = Bht[idx]
        bas = Bas[idx]
        bnpv = Bnpv[idx]

        chunk_len = end - I
        n_micro = max(1, int(np.ceil(chunk_len / stride)))

        micro_rewards_ht = []
        micro_rewards_as = []

        # -------------------------
        # micro loop (episode steps)
        # -------------------------
        for j in range(n_micro):
            j_lo = I + j * stride
            j_hi = min(I + (j + 1) * stride, end)
            if j_hi <= j_lo:
                continue
            idxj = np.arange(j_lo, j_hi)

            bht_j = Bht[idxj]
            bas_j = Bas[idxj]
            bnpv_j = Bnpv[idxj]

            roll.append(bht_j, bas_j, bnpv_j)
            bht_w, bas_w, bnpv_w = roll.get()

            # signal subset for this micro-step
            if matched_by_index:
                end_sig_j = min(j_hi, len(Tht), len(Aht), len(Tas), len(Aas))
                if j_lo >= end_sig_j:
                    sht_tt_j = np.empty(0, np.float32); sas_tt_j = np.empty(0, np.float32)
                    sht_aa_j = np.empty(0, np.float32); sas_aa_j = np.empty(0, np.float32)
                else:
                    idx_sig_j = np.arange(j_lo, end_sig_j)
                    sht_tt_j = Tht[idx_sig_j];  sas_tt_j = Tas[idx_sig_j]
                    sht_aa_j = Aht[idx_sig_j];  sas_aa_j = Aas[idx_sig_j]
            else:
                npv_min = float(np.min(bnpv_j))
                npv_max = float(np.max(bnpv_j))
                mask_tt = (Tnpv >= npv_min) & (Tnpv <= npv_max)
                mask_aa = (Anpv >= npv_min) & (Anpv <= npv_max)
                sht_tt_j = Tht[mask_tt];  sas_tt_j = Tas[mask_tt]
                sht_aa_j = Aht[mask_aa];  sas_aa_j = Aas[mask_aa]

            step = t * n_micro + j
            eps = max(0.05, 1.0 * (0.98 ** step))

            # =========================================================
            # HT ADT step
            # =========================================================
            bg_before_ht = Sing_Trigger(bht_j, Ht_cut_adt)
            if prev_bg_ht_adt is None:
                prev_bg_ht_adt = bg_before_ht

            obs_ht = make_event_seq_ht_v0(
                bht=bht_w, bnpv=bnpv_w,
                bg_rate=bg_before_ht,
                prev_bg_rate=prev_bg_ht_adt,
                cut=Ht_cut_adt,
                ht_mid=ht_mid, ht_span=ht_span,
                target=target, K=K,
                last_delta=last_dht_adt, max_delta=MAX_DELTA_HT,
                near_widths=near_widths_ht,
            )

            # ADT: update action only every l steps
            if (j % l_hold) == 0:
                act_ht = agent_ht.act(obs_ht, eps=eps)
                prev_act_ht_adt = int(act_ht)
            else:
                act_ht = int(prev_act_ht_adt)

            dht = float(HT_DELTAS[int(act_ht)])

            # safety shield
            sd = shield_delta(bg_before_ht, target, tol, MAX_DELTA_HT)
            if sd is not None:
                dht = float(sd)

            Ht_cut_next = float(np.clip(Ht_cut_adt + dht, ht_lo, ht_hi))

            bg_after_ht = Sing_Trigger(bht_j, Ht_cut_next)
            tt_after_ht = Sing_Trigger(sht_tt_j, Ht_cut_next)
            aa_after_ht = Sing_Trigger(sht_aa_j, Ht_cut_next)

            obs_ht_next = make_event_seq_ht_v0(
                bht=bht_w, bnpv=bnpv_w,
                bg_rate=bg_after_ht,
                prev_bg_rate=bg_before_ht,
                cut=Ht_cut_next,
                ht_mid=ht_mid, ht_span=ht_span,
                target=target, K=K,
                last_delta=dht, max_delta=MAX_DELTA_HT,
                near_widths=near_widths_ht,
            )

            if args.reward_mode == "lhc":
                r_ht = compute_reward(
                    bg_rate=bg_after_ht,
                    target=target, tol=tol,
                    sig_rate_1=tt_after_ht, sig_rate_2=aa_after_ht,
                    delta_applied=dht, max_delta=MAX_DELTA_HT,
                    alpha=alpha, beta=beta,
                    prev_bg_rate=bg_before_ht,
                    gamma_stab=float(args.gamma_stab),
                )
            else:
                r_ht = adt_reward_paper_style(
                    bg_scores=bht_j,
                    sig1_scores=sht_tt_j,
                    sig2_scores=sht_aa_j,
                    cut=Ht_cut_next,
                    alpha=alpha, beta=beta,
                )

            agent_ht.buf.push(obs_ht, int(act_ht), float(r_ht), obs_ht_next, done=False)
            micro_rewards_ht.append(float(r_ht))

            # advance
            Ht_cut_adt = Ht_cut_next
            prev_bg_ht_adt = bg_after_ht
            last_dht_adt = dht

            # =========================================================
            # AS ADT step
            # =========================================================
            bg_before_as = Sing_Trigger(bas_j, AS_cut_adt)
            if prev_bg_as_adt is None:
                prev_bg_as_adt = bg_before_as

            obs_as = make_event_seq_as_v0(
                bas=bas_w, bnpv=bnpv_w,
                bg_rate=bg_before_as,
                prev_bg_rate=prev_bg_as_adt,
                cut=AS_cut_adt,
                as_mid=as_mid, as_span=as_span,
                target=target, K=K,
                last_delta=last_das_adt, max_delta=MAX_DELTA_AS,
                near_widths=near_widths_as,
            )

            if (j % l_hold) == 0:
                act_as = agent_as.act(obs_as, eps=eps)
                prev_act_as_adt = int(act_as)
            else:
                act_as = int(prev_act_as_adt)

            das = float(AS_DELTAS[int(act_as)] * AS_STEP)

            sd = shield_delta(bg_before_as, target, tol, MAX_DELTA_AS)
            if sd is not None:
                das = float(sd)

            AS_cut_next = float(np.clip(AS_cut_adt + das, as_lo, as_hi))

            bg_after_as = Sing_Trigger(bas_j, AS_cut_next)
            tt_after_as = Sing_Trigger(sas_tt_j, AS_cut_next)
            aa_after_as = Sing_Trigger(sas_aa_j, AS_cut_next)

            obs_as_next = make_event_seq_as_v0(
                bas=bas_w, bnpv=bnpv_w,
                bg_rate=bg_after_as,
                prev_bg_rate=bg_before_as,
                cut=AS_cut_next,
                as_mid=as_mid, as_span=as_span,
                target=target, K=K,
                last_delta=das, max_delta=MAX_DELTA_AS,
                near_widths=near_widths_as,
            )

            if args.reward_mode == "lhc":
                r_as = compute_reward(
                    bg_rate=bg_after_as,
                    target=target, tol=tol,
                    sig_rate_1=tt_after_as, sig_rate_2=aa_after_as,
                    delta_applied=das, max_delta=MAX_DELTA_AS,
                    alpha=alpha, beta=beta,
                    prev_bg_rate=bg_before_as,
                    gamma_stab=float(args.gamma_stab),
                )
            else:
                r_as = adt_reward_paper_style(
                    bg_scores=bas_j,
                    sig1_scores=sas_tt_j,
                    sig2_scores=sas_aa_j,
                    cut=AS_cut_next,
                    alpha=alpha, beta=beta,
                )

            agent_as.buf.push(obs_as, int(act_as), float(r_as), obs_as_next, done=False)
            micro_rewards_as.append(float(r_as))

            AS_cut_adt = AS_cut_next
            prev_bg_as_adt = bg_after_as
            last_das_adt = das

        # -------------------------
        # End of episode (chunk): ADT-style training ONLY HERE
        # -------------------------
        for _ in range(int(args.train_steps_per_episode)):
            lht = agent_ht.train_step()
            if lht is not None:
                losses_ht.append(lht)
            las = agent_as.train_step()
            if las is not None:
                losses_as.append(las)

        rewards_ht.append(float(np.mean(micro_rewards_ht)) if micro_rewards_ht else np.nan)
        rewards_as.append(float(np.mean(micro_rewards_as)) if micro_rewards_as else np.nan)

        # -------------------------
        # chunk-level signals for logging/plots
        # -------------------------
        if matched_by_index:
            end_sig = min(end, len(Tht), len(Aht), len(Tas), len(Aas))
            idx_sig = np.arange(I, end_sig)
            sht_tt = Tht[idx_sig];  sas_tt = Tas[idx_sig]
            sht_aa = Aht[idx_sig];  sas_aa = Aas[idx_sig]
        else:
            npv_min = float(np.min(bnpv))
            npv_max = float(np.max(bnpv))
            mask_tt = (Tnpv >= npv_min) & (Tnpv <= npv_max)
            mask_aa = (Anpv >= npv_min) & (Anpv <= npv_max)
            sht_tt = Tht[mask_tt];  sas_tt = Tas[mask_tt]
            sht_aa = Aht[mask_aa];  sas_aa = Aas[mask_aa]

        # Constant
        bg_const_ht = Sing_Trigger(bht, fixed_Ht_cut)
        bg_const_as = Sing_Trigger(bas, fixed_AS_cut)

        tt_const_ht = Sing_Trigger(sht_tt, fixed_Ht_cut)
        aa_const_ht = Sing_Trigger(sht_aa, fixed_Ht_cut)
        tt_const_as = Sing_Trigger(sas_tt, fixed_AS_cut)
        aa_const_as = Sing_Trigger(sas_aa, fixed_AS_cut)

        # PD (updated once per chunk)
        bg_pd_ht = Sing_Trigger(bht, Ht_cut_pd)
        bg_pd_as = Sing_Trigger(bas, AS_cut_pd)

        tt_pd_ht = Sing_Trigger(sht_tt, Ht_cut_pd)
        aa_pd_ht = Sing_Trigger(sht_aa, Ht_cut_pd)
        tt_pd_as = Sing_Trigger(sas_tt, AS_cut_pd)
        aa_pd_as = Sing_Trigger(sas_aa, AS_cut_pd)

        Ht_cut_pd, pre_ht_err = PD_controller1(bg_pd_ht, pre_ht_err, Ht_cut_pd)
        AS_cut_pd, pre_as_err = PD_controller2(bg_pd_as, pre_as_err, AS_cut_pd)
        Ht_cut_pd = float(np.clip(Ht_cut_pd, ht_lo, ht_hi))
        AS_cut_pd = float(np.clip(AS_cut_pd, as_lo, as_hi))

        # ADT (current cut)
        bg_adt_ht = Sing_Trigger(bht, Ht_cut_adt)
        bg_adt_as = Sing_Trigger(bas, AS_cut_adt)

        tt_adt_ht = Sing_Trigger(sht_tt, Ht_cut_adt)
        aa_adt_ht = Sing_Trigger(sht_aa, Ht_cut_adt)
        tt_adt_as = Sing_Trigger(sas_tt, AS_cut_adt)
        aa_adt_as = Sing_Trigger(sas_aa, AS_cut_adt)

        # log
        R_const_ht.append(bg_const_ht); R_pd_ht.append(bg_pd_ht); R_adt_ht.append(bg_adt_ht)
        Ht_pd_hist.append(Ht_cut_pd);   Ht_adt_hist.append(Ht_cut_adt)
        L_tt_ht_const.append(tt_const_ht); L_tt_ht_pd.append(tt_pd_ht); L_tt_ht_adt.append(tt_adt_ht)
        L_aa_ht_const.append(aa_const_ht); L_aa_ht_pd.append(aa_pd_ht); L_aa_ht_adt.append(aa_adt_ht)

        R_const_as.append(bg_const_as); R_pd_as.append(bg_pd_as); R_adt_as.append(bg_adt_as)
        As_pd_hist.append(AS_cut_pd);   As_adt_hist.append(AS_cut_adt)
        L_tt_as_const.append(tt_const_as); L_tt_as_pd.append(tt_pd_as); L_tt_as_adt.append(tt_adt_as)
        L_aa_as_const.append(aa_const_as); L_aa_as_pd.append(aa_pd_as); L_aa_as_adt.append(aa_adt_as)

        if t % 5 == 0:
            lh = losses_ht[-1] if losses_ht else None
            la = losses_as[-1] if losses_as else None
            print(f"[chunk {t:4d}] "
                  f"HT bg% const={bg_const_ht:.3f} pd={bg_pd_ht:.3f} adt={bg_adt_ht:.3f} | "
                  f"ht_cut pd={Ht_cut_pd:.1f} adt={Ht_cut_adt:.1f} loss={lh} | "
                  f"AS bg% const={bg_const_as:.3f} pd={bg_pd_as:.3f} adt={bg_adt_as:.3f} | "
                  f"as_cut pd={AS_cut_pd:.4f} adt={AS_cut_adt:.4f} loss={la}")


    # =========================================================
    # Summary metrics + LaTeX/CSV table output (percent units)
    # =========================================================
    tables_dir = outdir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # rates are in percent units here (since Sing_Trigger returns percent)
    R_const_ht_pct = np.asarray(R_const_ht, dtype=np.float64)
    R_pd_ht_pct    = np.asarray(R_pd_ht,    dtype=np.float64)
    R_adt_ht_pct   = np.asarray(R_adt_ht,   dtype=np.float64)

    R_const_ad_pct = np.asarray(R_const_as, dtype=np.float64)  # AD == AS
    R_pd_ad_pct    = np.asarray(R_pd_as,    dtype=np.float64)
    R_adt_ad_pct   = np.asarray(R_adt_as,   dtype=np.float64)

    # efficiency logs (also in percent)
    L_tt_ht_const_np = np.asarray(L_tt_ht_const, dtype=np.float64)
    L_tt_ht_pd_np    = np.asarray(L_tt_ht_pd,    dtype=np.float64)
    L_tt_ht_adt_np   = np.asarray(L_tt_ht_adt,   dtype=np.float64)
    L_aa_ht_const_np = np.asarray(L_aa_ht_const, dtype=np.float64)
    L_aa_ht_pd_np    = np.asarray(L_aa_ht_pd,    dtype=np.float64)
    L_aa_ht_adt_np   = np.asarray(L_aa_ht_adt,   dtype=np.float64)

    L_tt_ad_const_np = np.asarray(L_tt_as_const, dtype=np.float64)
    L_tt_ad_pd_np    = np.asarray(L_tt_as_pd,    dtype=np.float64)
    L_tt_ad_adt_np   = np.asarray(L_tt_as_adt,   dtype=np.float64)
    L_aa_ad_const_np = np.asarray(L_aa_as_const, dtype=np.float64)
    L_aa_ad_pd_np    = np.asarray(L_aa_as_pd,    dtype=np.float64)
    L_aa_ad_adt_np   = np.asarray(L_aa_as_adt,   dtype=np.float64)

    # cut histories
    Ht_pd_np  = np.asarray(Ht_pd_hist,  dtype=np.float64)
    Ht_adt_np = np.asarray(Ht_adt_hist, dtype=np.float64)
    As_pd_np  = np.asarray(As_pd_hist,  dtype=np.float64)
    As_adt_np = np.asarray(As_adt_hist, dtype=np.float64)

    # constant cut histories (TV=0)
    Ht_const_np = np.full_like(Ht_pd_np, fixed_Ht_cut, dtype=np.float64)
    As_const_np = np.full_like(As_pd_np, fixed_AS_cut, dtype=np.float64)

    target_pct = float(target)
    tol_pct    = float(tol)

    rows = []

    def add_row(trigger, method, metrics):
        r = {"Trigger": trigger, "Method": method}
        r.update(metrics)
        rows.append(r)

    # HT block
    add_row("HT", "Constant",
            summarize_table_metrics(R_const_ht_pct, L_tt_ht_const_np, L_aa_ht_const_np, Ht_const_np, target_pct, tol_pct))
    add_row("HT", "PD",
            summarize_table_metrics(R_pd_ht_pct,    L_tt_ht_pd_np,    L_aa_ht_pd_np,    Ht_pd_np,    target_pct, tol_pct))
    add_row("HT", "ADT Yang et al. (2024)",
            summarize_table_metrics(R_adt_ht_pct,   L_tt_ht_adt_np,   L_aa_ht_adt_np,   Ht_adt_np,   target_pct, tol_pct))

    # AD (AS) block
    add_row("AD", "Constant",
            summarize_table_metrics(R_const_ad_pct, L_tt_ad_const_np, L_aa_ad_const_np, As_const_np, target_pct, tol_pct))
    add_row("AD", "PD",
            summarize_table_metrics(R_pd_ad_pct,    L_tt_ad_pd_np,    L_aa_ad_pd_np,    As_pd_np,    target_pct, tol_pct))
    add_row("AD", "ADT Yang et al. (2024)",
            summarize_table_metrics(R_adt_ad_pct,   L_tt_ad_adt_np,   L_aa_ad_adt_np,   As_adt_np,   target_pct, tol_pct))

    # CSV
    import csv
    csv_path = tables_dir / "adt_summary.csv"
    with open(csv_path, "w", newline="") as f:
        fieldnames = ["Trigger","Method","MAE","P95","InBand","UpFrac","DownFrac","ttbar","h4b"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    # LaTeX
    tex_path = tables_dir / "adt_summary.tex"
    caption = (
    f"ADT baseline summary. Rates evaluated in percent units with target $r^*={target_pct:.3g}\\%$ "
    f"and tolerance $\\pm{tol_pct:.3g}\\%$. "
    "MAE and P95$|e|$ summarize typical and tail absolute rate errors; "
    "InBand, UpFrac, and DownFrac are fractions of chunks within band, above the upper tolerance, "
    "and below the lower tolerance. "
    "$\\bar{t}\\bar{t}$ and $h\\to4b$ report mean signal efficiencies restricted to in-band chunks."
    )
    write_adt_table(rows, tex_path, caption=caption, label="tab:adt_summary")

    print(f"[OK] wrote {csv_path}")
    print(f"[OK] wrote {tex_path}")

    # ------------------------- Convert to arrays + scale -------------------------
    RATE_SCALE_KHZ = 400.0
    upper_tol_khz = (target + tol) * RATE_SCALE_KHZ
    lower_tol_khz = (target - tol) * RATE_SCALE_KHZ

    time = np.linspace(0, 1, len(R_const_ht))
    time_as = np.linspace(0, 1, len(R_const_as))

    R_const_ht = np.asarray(R_const_ht) * RATE_SCALE_KHZ
    R_pd_ht    = np.asarray(R_pd_ht)    * RATE_SCALE_KHZ
    R_adt_ht   = np.asarray(R_adt_ht)   * RATE_SCALE_KHZ

    R_const_as = np.asarray(R_const_as) * RATE_SCALE_KHZ
    R_pd_as    = np.asarray(R_pd_as)    * RATE_SCALE_KHZ
    R_adt_as   = np.asarray(R_adt_as)   * RATE_SCALE_KHZ

    Ht_pd_hist  = np.asarray(Ht_pd_hist)
    Ht_adt_hist = np.asarray(Ht_adt_hist)
    As_pd_hist  = np.asarray(As_pd_hist)
    As_adt_hist = np.asarray(As_adt_hist)

    L_tt_ht_const = np.asarray(L_tt_ht_const)
    L_tt_ht_pd    = np.asarray(L_tt_ht_pd)
    L_tt_ht_adt   = np.asarray(L_tt_ht_adt)
    L_aa_ht_const = np.asarray(L_aa_ht_const)
    L_aa_ht_pd    = np.asarray(L_aa_ht_pd)
    L_aa_ht_adt   = np.asarray(L_aa_ht_adt)

    L_tt_as_const = np.asarray(L_tt_as_const)
    L_tt_as_pd    = np.asarray(L_tt_as_pd)
    L_tt_as_adt   = np.asarray(L_tt_as_adt)
    L_aa_as_const = np.asarray(L_aa_as_const)
    L_aa_as_pd    = np.asarray(L_aa_as_pd)
    L_aa_as_adt   = np.asarray(L_aa_as_adt)

    rewards_ht = np.asarray(rewards_ht, dtype=np.float32)
    rewards_as = np.asarray(rewards_as, dtype=np.float32)

    # ------------------------- Styles -------------------------
    CONST_STYLE = dict(linestyle="--", linewidth=2.8, alpha=0.85, zorder=2)
    PD_STYLE    = dict(linestyle="-",  linewidth=2.4, alpha=0.90, zorder=3)
    ADT_STYLE   = dict(linestyle="-.", linewidth=2.8, alpha=0.95, zorder=4)

    # ------------------------- Reward plots -------------------------
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time, rewards_ht, linewidth=1.2, alpha=0.35, label="HT reward (per chunk)")
    ax.plot(time, moving_avg_nan(rewards_ht, w=5), linewidth=2.2, label="HT reward (moving avg)")
    ax.set_xlabel("Time (Fraction of Run)")
    ax.set_ylabel("Reward")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best", frameon=True)
    add_cms_header(fig, run_label=run_label)
    save_png(fig, str(outdir / "reward_ht_pidData_adt"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time_as, rewards_as, linewidth=1.2, alpha=0.35, label="AS reward (per chunk)")
    ax.plot(time_as, moving_avg_nan(rewards_as, w=5), linewidth=2.2, label="AS reward (moving avg)")
    ax.set_xlabel("Time (Fraction of Run)")
    ax.set_ylabel("Reward")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best", frameon=True)
    add_cms_header(fig, run_label=run_label)
    save_png(fig, str(outdir / "reward_as_pidData_adt"))
    plt.close(fig)

    # ------------------------- Rate plots -------------------------
    plot_rate_with_tolerance(
        time, R_const_ht, R_pd_ht, R_adt_ht,
        outbase=outdir / "bht_rate_pidData_adt",
        run_label=run_label,
        legend_title="HT Trigger",
        ylim=(0, 200),
        tol_upper=upper_tol_khz,
        tol_lower=lower_tol_khz,
        const_style=dict(color="tab:blue", **CONST_STYLE),
        pd_style=dict(color="mediumblue", **PD_STYLE),
        dqn_style=dict(color="tab:purple", **ADT_STYLE),  # reuse arg name in util,
        dqn_label="ADT",
        add_cms_header=add_cms_header,
        save_pdf_png=save_png,
    )

    plot_rate_with_tolerance(
        time_as, R_const_as, R_pd_as, R_adt_as,
        outbase=outdir / "bas_rate_pidData_adt",
        run_label=run_label,
        legend_title="AD Trigger",
        ylim=(0, 200),
        tol_upper=upper_tol_khz,
        tol_lower=lower_tol_khz,
        const_style=dict(color="tab:blue", **CONST_STYLE),
        pd_style=dict(color="mediumblue", **PD_STYLE),
        dqn_style=dict(color="tab:purple", **ADT_STYLE),
        dqn_label="ADT",
        add_cms_header=add_cms_header,
        save_pdf_png=save_png,
    )

    # ------------------------- Cut evolution -------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time, Ht_pd_hist,  color="mediumblue", linewidth=2.0, label="PD Controller")
    ax.plot(time, Ht_adt_hist, color="tab:purple", **ADT_STYLE, label="ADT")
    ax.axhline(y=fixed_Ht_cut, color="gray", linestyle="--", linewidth=1.5, label="fixed_Ht_cut")
    ax.set_xlabel("Time (Fraction of Run)")
    ax.set_ylabel("Ht_cut [GeV]")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(title="HT Cut", fontsize=14, frameon=True, loc="best")
    add_cms_header(fig, run_label=run_label)
    save_png(fig, str(outdir / "ht_cut_pidData_adt"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time_as, As_pd_hist,  color="mediumblue", linewidth=2.0, label="PD Controller")
    ax.plot(time_as, As_adt_hist, color="tab:purple", **ADT_STYLE, label="ADT")
    ax.axhline(y=fixed_AS_cut, color="gray", linestyle="--", linewidth=1.5, label="fixed_AS_cut")
    ax.set_xlabel("Time (Fraction of Run)")
    ax.set_ylabel("Anomaly Score Cut")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(title="AD Cut", fontsize=14, frameon=True, loc="best")
    add_cms_header(fig, run_label=run_label)
    save_png(fig, str(outdir / "as_cut_pidData_adt"))
    plt.close(fig)

    # ------------------------- Efficiencies -------------------------
    # cumulative (relative to t0)
    tt_c_const_ht = cummean(L_tt_ht_const)
    tt_c_pd_ht    = cummean(L_tt_ht_pd)
    tt_c_adt_ht   = cummean(L_tt_ht_adt)
    aa_c_const_ht = cummean(L_aa_ht_const)
    aa_c_pd_ht    = cummean(L_aa_ht_pd)
    aa_c_adt_ht   = cummean(L_aa_ht_adt)

    colors_ht = {"ttbar": "goldenrod", "HToAATo4B": "seagreen"}
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time, rel_to_t0(tt_c_const_ht), color=colors_ht["ttbar"], **CONST_STYLE, label="Const, ttbar")
    ax.plot(time, rel_to_t0(aa_c_const_ht), color=colors_ht["HToAATo4B"], **CONST_STYLE, label="Const, HToAATo4B")
    ax.plot(time, rel_to_t0(tt_c_pd_ht), color=colors_ht["ttbar"], **PD_STYLE, label="PD, ttbar")
    ax.plot(time, rel_to_t0(aa_c_pd_ht), color=colors_ht["HToAATo4B"], **PD_STYLE, label="PD, HToAATo4B")
    ax.plot(time, rel_to_t0(tt_c_adt_ht), color=colors_ht["ttbar"], **ADT_STYLE, label="ADT, ttbar")
    ax.plot(time, rel_to_t0(aa_c_adt_ht), color=colors_ht["HToAATo4B"], **ADT_STYLE, label="ADT, HToAATo4B")
    style_diag_axes(ax, "Time (Fraction of Run)", "Relative Cumulative Efficiency", ylim=(0.5, 2.5))
    style_diag_legend(ax, title="HT Trigger")
    finalize_diag_fig(fig)
    add_cms_header(fig, run_label=run_label)
    save_png(fig, str(outdir / "sht_rate_pidData2data_adt"))
    plt.close(fig)

    # local (relative to t0)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time, rel_to_t0(L_tt_ht_const), color=colors_ht["ttbar"], **CONST_STYLE, label="Const, ttbar")
    ax.plot(time, rel_to_t0(L_aa_ht_const), color=colors_ht["HToAATo4B"], **CONST_STYLE, label="Const, HToAATo4B")
    ax.plot(time, rel_to_t0(L_tt_ht_pd), color=colors_ht["ttbar"], **PD_STYLE, label="PD, ttbar")
    ax.plot(time, rel_to_t0(L_aa_ht_pd), color=colors_ht["HToAATo4B"], **PD_STYLE, label="PD, HToAATo4B")
    ax.plot(time, rel_to_t0(L_tt_ht_adt), color=colors_ht["ttbar"], **ADT_STYLE, label="ADT, ttbar")
    ax.plot(time, rel_to_t0(L_aa_ht_adt), color=colors_ht["HToAATo4B"], **ADT_STYLE, label="ADT, HToAATo4B")
    style_diag_axes(ax, "Time (Fraction of Run)", "Relative Efficiency", ylim=(0.5, 2.5))
    style_diag_legend(ax, title="HT Trigger")
    finalize_diag_fig(fig)
    add_cms_header(fig, run_label=run_label)
    save_png(fig, str(outdir / "L_sht_rate_pidData2data_adt"))
    plt.close(fig)

    # AS cumulative/local
    tt_c_const_as = cummean(L_tt_as_const)
    tt_c_pd_as    = cummean(L_tt_as_pd)
    tt_c_adt_as   = cummean(L_tt_as_adt)
    aa_c_const_as = cummean(L_aa_as_const)
    aa_c_pd_as    = cummean(L_aa_as_pd)
    aa_c_adt_as   = cummean(L_aa_as_adt)

    colors_ad = {"ttbar": "goldenrod", "HToAATo4B": "limegreen"}
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time_as, rel_to_t0(tt_c_const_as), color=colors_ad["ttbar"], **CONST_STYLE, label="Const, ttbar")
    ax.plot(time_as, rel_to_t0(aa_c_const_as), color=colors_ad["HToAATo4B"], **CONST_STYLE, label="Const, HToAATo4B")
    ax.plot(time_as, rel_to_t0(tt_c_pd_as), color=colors_ad["ttbar"], **PD_STYLE, label="PD, ttbar")
    ax.plot(time_as, rel_to_t0(aa_c_pd_as), color=colors_ad["HToAATo4B"], **PD_STYLE, label="PD, HToAATo4B")
    ax.plot(time_as, rel_to_t0(tt_c_adt_as), color=colors_ad["ttbar"], **ADT_STYLE, label="ADT, ttbar")
    ax.plot(time_as, rel_to_t0(aa_c_adt_as), color=colors_ad["HToAATo4B"], **ADT_STYLE, label="ADT, HToAATo4B")
    style_diag_axes(ax, "Time (Fraction of Run)", "Relative Cumulative Efficiency", ylim=(0.5, 2.5))
    style_diag_legend(ax, title="AD Trigger")
    finalize_diag_fig(fig)
    add_cms_header(fig, run_label=run_label)
    save_png(fig, str(outdir / "sas_rate_pidData2data_adt"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time_as, rel_to_t0(L_tt_as_const), color=colors_ad["ttbar"], **CONST_STYLE, label="Const, ttbar")
    ax.plot(time_as, rel_to_t0(L_aa_as_const), color=colors_ad["HToAATo4B"], **CONST_STYLE, label="Const, HToAATo4B")
    ax.plot(time_as, rel_to_t0(L_tt_as_pd), color=colors_ad["ttbar"], **PD_STYLE, label="PD, ttbar")
    ax.plot(time_as, rel_to_t0(L_aa_as_pd), color=colors_ad["HToAATo4B"], **PD_STYLE, label="PD, HToAATo4B")
    ax.plot(time_as, rel_to_t0(L_tt_as_adt), color=colors_ad["ttbar"], **ADT_STYLE, label="ADT, ttbar")
    ax.plot(time_as, rel_to_t0(L_aa_as_adt), color=colors_ad["HToAATo4B"], **ADT_STYLE, label="ADT, HToAATo4B")
    style_diag_axes(ax, "Time (Fraction of Run)", "Relative Efficiency", ylim=(0.5, 2.5))
    style_diag_legend(ax, title="AD Trigger")
    finalize_diag_fig(fig)
    add_cms_header(fig, run_label=run_label)
    save_png(fig, str(outdir / "L_sas_rate_pidData2data_adt"))
    plt.close(fig)

    # ------------------------- Loss plots -------------------------
    if losses_ht:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(np.arange(len(losses_ht)), losses_ht, linewidth=1.5)
        ax.set_title("HT ADT training loss")
        ax.set_xlabel("Gradient step (end-of-episode updates)")
        ax.set_ylabel("Loss")
        ax.grid(True, linestyle="--", alpha=0.5)
        add_cms_header(fig, run_label=run_label)
        save_png(fig, str(outdir / "adt_loss_ht"))
        plt.close(fig)

    if losses_as:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(np.arange(len(losses_as)), losses_as, linewidth=1.5)
        ax.set_title("AD ADT training loss")
        ax.set_xlabel("Gradient step (end-of-episode updates)")
        ax.set_ylabel("Loss")
        ax.grid(True, linestyle="--", alpha=0.5)
        add_cms_header(fig, run_label=run_label)
        save_png(fig, str(outdir / "adt_loss_as"))
        plt.close(fig)

    print("\nSaved to:", outdir)
    for p in sorted(outdir.glob("*.png")):
        print(" -", p.name)


if __name__ == "__main__":
    main()
