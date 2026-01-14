#!/usr/bin/env python3
"""
DQN_Ht_AS.py

SingleTrigger: 
Constant vs PD vs DQN
- HT trigger: accept = (HT >= Ht_cut)
- AS trigger: accept = (AS >= AS_cut)

We train two independent DQNs:
  (1) DQN_HT controls Ht_cut using HT-only rates
  (2) DQN_AS controls AS_cut using AS-only rates For AS only, we use binned steps to ensure stability.

Outputs:

HT trigger outputs:
  - bht_rate_pidData_dqn.png          (HT background rate [kHz])
  - ht_cut_pidData_dqn.png            (Ht_cut evolution)
  - sht_rate_pidData2data_dqn.png     (cumulative signal eff, relative to t0)
  - L_sht_rate_pidData2data_dqn.png   (local signal eff, relative to t0)
  - dqn_loss_ht.png                   (HT DQN loss)

AS trigger outputs:
  - bas_rate_pidData_dqn.png          (AS background rate [kHz])
  - as_cut_pidData_dqn.png            (AS_cut evolution)
  - sas_rate_pidData2data_dqn.png     (cumulative signal eff, relative to t0)
  - L_sas_rate_pidData2data_dqn.png   (local signal eff, relative to t0)
  - dqn_loss_as.png                   (AS DQN loss)

"""

import random
import argparse

import numpy as np
import matplotlib.pyplot as plt
import h5py
import hdf5plugin  # noqa: F401

from pathlib import Path
from controllers import PD_controller1, PD_controller2
from triggers import Sing_Trigger
from RL.utils import cummean, rel_to_t0, add_cms_header, plot_rate_with_tolerance, save_png, print_h5_tree, read_any_h5, compute_auroc_windows #save_pdf_png,
from RL.dqn_agent import DQNAgent, make_obs, shield_delta, compute_reward, DQNConfig, SeqDQNAgent, make_event_seq_as, make_event_seq_ht

# ------------------------- Fixing seed for reproducibility -------------------------
SEED = 20251213
random.seed(SEED)
np.random.seed(SEED)



# ------------------------- main -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="Data/Matched_data_2016_dim2.h5",
                    help="Matched_data_*.h5 (data) or Trigger_food_*.h5 (MC)")
    ap.add_argument("--outdir", default="RL_outputs/demo_sing_dqn_separate", help="output dir")
    ap.add_argument("--control", default="MC", choices=["MC", "RealData"],
                    help="Control type: MC or RealData")
    ap.add_argument("--score-dim-hint", type=int, default=2,
                    help="If file has only scoreXX, use this dim (e.g. 2 -> score02).")
    ap.add_argument("--as-dim", type=int, default=2, choices=[1, 4, 2],
                    help="Which AS dimension to use (1->score01, 4->score04).")

    ap.add_argument("--ht-deltas", type=str, default="-2,-1,0,1,2",
                    help="HT DQN deltas (in HT cut units, like your HT script).")
    ap.add_argument("--as-deltas", type=str, default="-3,-1.5,0,1.5,3",
                    help="AS DQN delta multipliers.")
    ap.add_argument("--as-step", type=float, default=0.5,
                    help="AS step: final delta or step we make for AS trigger = as_delta * as_step (tune the AS scale).")
    ap.add_argument("--print-keys", action="store_true",
                help="Print all HDF5 groups/datasets (with shapes/dtypes) and exit.")
    ap.add_argument("--print-keys-max", type=int, default=None,
                help="Optional cap on number of printed items.")
    # Let's predefine AS bins to ensure better dqn stability
    ap.add_argument("--as-bins", type=int, default=20, choices=[10, 20, 30, 40, 50],
                help="Number of bins used to define AS step a in the cut-range.")
    ap.add_argument("--as-p-lo", type=float, default=99.0,
                help="Low percentile for AS cut range.")
    ap.add_argument("--as-p-hi", type=float, default=99.995,
                help="High percentile for AS cut range.")

    args = ap.parse_args()
    if args.control == "MC":
        run_label = "MC"
    else:
        run_label = "283408"
    if args.print_keys:
        print_h5_tree(args.input, max_items=args.print_keys_max)
        raise SystemExit(0)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    d = read_any_h5(args.input, score_dim_hint=args.score_dim_hint)
    matched_by_index = bool(d["meta"].get("matched_by_index", False))

    Bht, Bnpv = d["Bht"], d["Bnpv"]
    Tht, Tnpv = d["Tht"], d["Tnpv"]
    Aht, Anpv = d["Aht"], d["Anpv"]

    # Pick AS dim
    # For now only support dim=2 (score02) in this script
    if args.as_dim == 2:
        Bas, Tas, Aas = d["Bas2"], d["Tas2"], d["Aas2"]

    if Bas is None or Tas is None or Aas is None:
        raise SystemExit("AS arrays missing for requested --as-dim. Check your input file.")

    N = len(Bht)
    if args.control == "MC":
        chunk_size = 50000
    else:
        #real data
        chunk_size = 20000
    start_event = chunk_size * 10

    # Align start_event to chunk boundary
    start_event = max(0, (start_event // chunk_size) * chunk_size)
    if start_event + chunk_size > N:
        start_event = max(0, ((N - chunk_size) // chunk_size) * chunk_size)

    # Fixed cuts from a reference window (mimic Data_SingleTrigger.py)
    win_lo = min(start_event, N - 1)
    if args.control == "MC":
        # Run_SingleTrigger.py uses 100k for MC DEBUG 
        win_hi = min(start_event + 100000, N)
    else:
        #Data_SingleTrigger.py uses 10k for RealData DEBUG
        win_hi = min(start_event + 10000, N)


    fixed_Ht_cut = float(np.percentile(Bht[win_lo:win_hi], 99.75))
    fixed_AS_cut = float(np.percentile(Bas[win_lo:win_hi], 99.75))

    # Clip ranges
    ht_lo = float(np.percentile(Bht[start_event:], 95.0))
    ht_hi = float(np.percentile(Bht[start_event:], 99.99))
    ht_mid = 0.5 * (ht_lo + ht_hi)
    ht_span = max(1.0, ht_hi - ht_lo)

    ref_as = Bas[win_lo:win_hi]

    as_lo = float(np.min(ref_as))
    as_hi = float(np.max(ref_as))
    as_mid  = 0.5 * (as_lo + as_hi)
    as_span = max(1e-6, as_hi - as_lo)


    print(f"[INFO] matched_by_index={matched_by_index} N={N} chunk={chunk_size} start_event={start_event}")
    print(f"[HT] fixed={fixed_Ht_cut:.3f} clip=({ht_lo:.3f},{ht_hi:.3f}) window=[{win_lo}:{win_hi}]")
    print(f"[AS dim={args.as_dim}] fixed={fixed_AS_cut:.6f} clip=({as_lo:.6f},{as_hi:.6f}) as_step={args.as_step}")

    # ------------------------- init cuts -------------------------
    # HT
    Ht_cut_pd  = fixed_Ht_cut
    Ht_cut_dqn = fixed_Ht_cut
    pre_ht_err = 0.0

    # AS
    AS_cut_pd  = fixed_AS_cut
    AS_cut_dqn = fixed_AS_cut
    pre_as_err = 0.0

    # ------------------------- DQN configs -------------------------
    target = 0.25  # %
    tol = 0.02     # background - target/tolerance for reward?
    alpha = 0.4    # signal bonus
    beta  = 0.1   # move penalty

    HT_DELTAS = np.array([float(x) for x in args.ht_deltas.split(",")], dtype=np.float32)
    HT_STEP = 1.0
    MAX_DELTA_HT = float(np.max(np.abs(HT_DELTAS))) * HT_STEP

    AS_DELTAS = np.array([float(x) for x in args.as_deltas.split(",")], dtype=np.float32)
    AS_STEP = float(args.as_step)
    MAX_DELTA_AS = float(np.max(np.abs(AS_DELTAS))) * AS_STEP
    print("MAX_DELTA_AS=", MAX_DELTA_AS)

    cfg = DQNConfig(lr=5e-4, gamma=0.95, batch_size=128, target_update=200)
    agent_ht = DQNAgent(obs_dim=3, n_actions=len(HT_DELTAS), cfg=cfg, seed = SEED)
    # Make AS agent larger learning rate for faster adaptation
    cfg_as = DQNConfig(lr=1e-4, gamma=0.95, batch_size=128, target_update=200)
    agent_as = DQNAgent(obs_dim=3, n_actions=len(AS_DELTAS), cfg=cfg_as, seed = SEED)

    # state trackers (HT)
    prev_obs_ht = None
    prev_act_ht = None
    prev_bg_ht = None
    last_dht = 0.0
    losses_ht = []
    rewards_ht = []   # HT reward per chunk (t)

    # state trackers (AS)
    prev_obs_as = None
    prev_act_as = None
    prev_bg_as = None
    last_das = 0.0
    losses_as = []
    rewards_as = []   # AS reward per chunk (t)

    # ------------------------- logs (HT) -------------------------
    R1_ht, R2_ht, R3_ht = [], [], []                  # background % (const, PID, DQN)
    Ht_pd_hist, Ht_dqn_hist = [], []
    L_tt_ht_const, L_tt_ht_pd, L_tt_ht_dqn = [], [], []
    L_aa_ht_const, L_aa_ht_pd, L_aa_ht_dqn = [], [], []

    # ------------------------- logs (AS) -------------------------
    R1_as, R2_as, R3_as = [], [], []                  # background % (const, PID, DQN)
    As_pd_hist, As_dqn_hist = [], []
    L_tt_as_const, L_tt_as_pd, L_tt_as_dqn = [], [], []
    L_aa_as_const, L_aa_as_pd, L_aa_as_dqn = [], [], []

    # ------------------------- batching loop -------------------------
    batch_starts = list(range(start_event, N, chunk_size))

    for t, I in enumerate(batch_starts):
        # clip chunk end to the smallest available array length
        end = min(I + chunk_size, N, len(Bnpv), len(Bas))
        if end <= I:
            break
        idx = np.arange(I, end)

        bht  = Bht[idx]
        bas  = Bas[idx]
        bnpv = Bnpv[idx]

        # ---- signals per chunk ----
        if matched_by_index:
            end_sig = min(end, len(Tht), len(Aht), len(Tas), len(Aas))
            idx_sig = np.arange(I, end_sig)

            sht_tt = Tht[idx_sig]
            sas_tt = Tas[idx_sig]
            sht_aa = Aht[idx_sig]
            sas_aa = Aas[idx_sig]
        else:
            npv_min = float(np.min(bnpv))
            npv_max = float(np.max(bnpv))
            mask_tt = (Tnpv >= npv_min) & (Tnpv <= npv_max)
            mask_aa = (Anpv >= npv_min) & (Anpv <= npv_max)

            sht_tt = Tht[mask_tt]
            sas_tt = Tas[mask_tt]
            sht_aa = Aht[mask_aa]
            sas_aa = Aas[mask_aa]

        # =========================================================
        # HT trigger (separate)
        # =========================================================
        bg_const_ht = Sing_Trigger(bht, fixed_Ht_cut)
        bg_pd_ht    = Sing_Trigger(bht, Ht_cut_pd)
        bg_dqn_ht   = Sing_Trigger(bht, Ht_cut_dqn)

        tt_const_ht = Sing_Trigger(sht_tt, fixed_Ht_cut)
        tt_pd_ht    = Sing_Trigger(sht_tt, Ht_cut_pd)
        tt_dqn_ht   = Sing_Trigger(sht_tt, Ht_cut_dqn)

        aa_const_ht = Sing_Trigger(sht_aa, fixed_Ht_cut)
        aa_pd_ht    = Sing_Trigger(sht_aa, Ht_cut_pd)
        aa_dqn_ht   = Sing_Trigger(sht_aa, Ht_cut_dqn)

        # PID update HT
        Ht_cut_pd, pre_ht_err = PD_controller1(bg_pd_ht, pre_ht_err, Ht_cut_pd)
        Ht_cut_pd = float(np.clip(Ht_cut_pd, ht_lo, ht_hi))

        # DQN HT update (train on previous transition, choose next delta)
        if prev_bg_ht is None:
            prev_bg_ht = bg_dqn_ht
        obs_ht = make_obs(bg_dqn_ht, prev_bg_ht, Ht_cut_dqn, ht_mid, ht_span, target)

        reward_ht_t = np.nan
        if (prev_obs_ht is not None) and (prev_act_ht is not None):
            reward_ht_t = compute_reward(
                bg_rate=bg_dqn_ht,
                target=target,
                tol=tol,
                sig_rate_1=tt_dqn_ht,
                sig_rate_2=aa_dqn_ht,
                delta_applied=last_dht,
                max_delta=MAX_DELTA_HT,
                alpha=alpha,
                beta=beta,
                prev_bg_rate=prev_bg_ht,
                gamma_stab=0.3,
            )

            agent_ht.buf.push(prev_obs_ht, prev_act_ht, reward_ht_t, obs_ht, done=False)
            loss = agent_ht.train_step()
            if loss is not None:
                losses_ht.append(loss)

        rewards_ht.append(reward_ht_t)

        eps = max(0.05, 1.0 * (0.98 ** t))
        act_ht = agent_ht.act(obs_ht, eps=eps) 
        dht = float(HT_DELTAS[act_ht])

        sd = shield_delta(bg_dqn_ht, target, tol, MAX_DELTA_HT)
        if sd is not None:
            dht = float(sd)

        prev_obs_ht = obs_ht
        prev_act_ht = act_ht
        prev_bg_ht = bg_dqn_ht
        last_dht = dht

        Ht_cut_dqn = float(np.clip(Ht_cut_dqn + dht, ht_lo, ht_hi))

        # record HT logs
        R1_ht.append(bg_const_ht)
        R2_ht.append(bg_pd_ht)
        R3_ht.append(bg_dqn_ht)
        Ht_pd_hist.append(Ht_cut_pd)
        Ht_dqn_hist.append(Ht_cut_dqn)
        L_tt_ht_const.append(tt_const_ht)
        L_tt_ht_pd.append(tt_pd_ht)
        L_tt_ht_dqn.append(tt_dqn_ht)
        L_aa_ht_const.append(aa_const_ht)
        L_aa_ht_pd.append(aa_pd_ht)
        L_aa_ht_dqn.append(aa_dqn_ht)

        # =========================================================
        # AD trigger (separate)
        # =========================================================
        bg_const_as = Sing_Trigger(bas, fixed_AS_cut)
        bg_pd_as    = Sing_Trigger(bas, AS_cut_pd)
        bg_dqn_as   = Sing_Trigger(bas, AS_cut_dqn)

        tt_const_as = Sing_Trigger(sas_tt, fixed_AS_cut)
        tt_pd_as    = Sing_Trigger(sas_tt, AS_cut_pd)
        tt_dqn_as   = Sing_Trigger(sas_tt, AS_cut_dqn)

        aa_const_as = Sing_Trigger(sas_aa, fixed_AS_cut)
        aa_pd_as    = Sing_Trigger(sas_aa, AS_cut_pd)
        aa_dqn_as   = Sing_Trigger(sas_aa, AS_cut_dqn)

        # PID update AS
        AS_cut_pd, pre_as_err = PD_controller2(bg_pd_as, pre_as_err, AS_cut_pd)
        AS_cut_pd = float(np.clip(AS_cut_pd, as_lo, as_hi))

        # DQN AS update
        if prev_bg_as is None:
            prev_bg_as = bg_dqn_as
        obs_as = make_obs(bg_dqn_as, prev_bg_as, AS_cut_dqn, as_mid, as_span, target)
        reward_as_t = np.nan
        if (prev_obs_as is not None) and (prev_act_as is not None):
            reward_as_t = compute_reward(
                bg_rate=bg_dqn_as,
                target=target,
                tol=tol,
                sig_rate_1=tt_dqn_as,
                sig_rate_2=aa_dqn_as,
                delta_applied=last_das,
                max_delta=MAX_DELTA_AS,
                alpha=alpha,
                beta=beta,
                prev_bg_rate=prev_bg_as,
                gamma_stab=0.3,
            )

            agent_as.buf.push(prev_obs_as, prev_act_as, reward_as_t, obs_as, done=False)
            loss = agent_as.train_step()
            if loss is not None:
                losses_as.append(loss)

        rewards_as.append(reward_as_t)
        act_as = agent_as.act(obs_as, eps=eps)
        das = float(AS_DELTAS[act_as] * AS_STEP)

        sd = shield_delta(bg_dqn_as, target, tol, MAX_DELTA_AS)
        if sd is not None:
            das = float(sd)

        prev_obs_as = obs_as
        prev_act_as = act_as
        prev_bg_as = bg_dqn_as
        last_das = das

        if t % 5 == 0:
            print(f"[DBG AS] act={act_as} delta_mult={AS_DELTAS[act_as]} das={das:.6f} cut_before={AS_cut_dqn:.6f}")
            # print(f"[DBG SHIELD] sd={sd} bg={bg_dqn_as:.3f} target={target} tol={tol}")
        AS_cut_dqn = float(np.clip(AS_cut_dqn + das, as_lo, as_hi))

        # record AS logs
        R1_as.append(bg_const_as)
        R2_as.append(bg_pd_as)
        R3_as.append(bg_dqn_as)
        As_pd_hist.append(AS_cut_pd)
        As_dqn_hist.append(AS_cut_dqn)
        L_tt_as_const.append(tt_const_as)
        L_tt_as_pd.append(tt_pd_as)
        L_tt_as_dqn.append(tt_dqn_as)
        L_aa_as_const.append(aa_const_as)
        L_aa_as_pd.append(aa_pd_as)
        L_aa_as_dqn.append(aa_dqn_as)

        if t % 5 == 0:
            lh = losses_ht[-1] if losses_ht else None
            la = losses_as[-1] if losses_as else None
            print(f"[batch {t:4d}] eps={eps:.3f} "
                  f"HT bg% c={bg_const_ht:.3f} pd={bg_pd_ht:.3f} dqn={bg_dqn_ht:.3f} "
                  f"| ht_cut pd={Ht_cut_pd:.1f} dqn={Ht_cut_dqn:.1f} loss={lh} "
                  f"|| AS bg% c={bg_const_as:.3f} pd={bg_pd_as:.3f} dqn={bg_dqn_as:.3f} "
                  f"| as_cut pd={AS_cut_pd:.4f} dqn={AS_cut_dqn:.4f} loss={la}")

    # ------------------------- convert + scale -------------------------
    RATE_SCALE_KHZ = 400.0
    upper_tol_khz = 0.275 * RATE_SCALE_KHZ
    lower_tol_khz = 0.225 * RATE_SCALE_KHZ

    # HT
    R1_ht = np.array(R1_ht) * RATE_SCALE_KHZ
    R2_ht = np.array(R2_ht) * RATE_SCALE_KHZ
    R3_ht = np.array(R3_ht) * RATE_SCALE_KHZ
    Ht_pd_hist = np.array(Ht_pd_hist)
    Ht_dqn_hist = np.array(Ht_dqn_hist)
    L_tt_ht_const = np.array(L_tt_ht_const)
    L_tt_ht_pd    = np.array(L_tt_ht_pd)
    L_tt_ht_dqn   = np.array(L_tt_ht_dqn)
    L_aa_ht_const = np.array(L_aa_ht_const)
    L_aa_ht_pd    = np.array(L_aa_ht_pd)
    L_aa_ht_dqn   = np.array(L_aa_ht_dqn)

    # AS
    R1_as = np.array(R1_as) * RATE_SCALE_KHZ
    R2_as = np.array(R2_as) * RATE_SCALE_KHZ
    R3_as = np.array(R3_as) * RATE_SCALE_KHZ
    As_pd_hist = np.array(As_pd_hist)
    As_dqn_hist = np.array(As_dqn_hist)
    L_tt_as_const = np.array(L_tt_as_const)
    L_tt_as_pd    = np.array(L_tt_as_pd)
    L_tt_as_dqn   = np.array(L_tt_as_dqn)
    L_aa_as_const = np.array(L_aa_as_const)
    L_aa_as_pd    = np.array(L_aa_as_pd)
    L_aa_as_dqn   = np.array(L_aa_as_dqn)

    CONST_STYLE = dict(linestyle="--", linewidth=2.8, alpha=0.85, zorder=2)
    PD_STYLE    = dict(linestyle="-",  linewidth=2.4, alpha=0.90, zorder=3)

    # DQN: thick + custom dash pattern + markers
    DQN_STYLE   = dict(
        linestyle=(0, (8, 2, 2, 2)),   # long dash, gap, short dash, gap
        linewidth=3.2,
        marker="o",
        markersize=4,
        markevery=6,                  # marker every ~6 points
        alpha=0.95,
        zorder=5,
    )
    time = np.linspace(0, 1, len(R1_ht))
    rewards_ht = np.asarray(rewards_ht, dtype=np.float32)
    rewards_as = np.asarray(rewards_as, dtype=np.float32)

    def moving_avg_nan(x, w=5):
        x = np.asarray(x, dtype=np.float32)
        m = np.isfinite(x).astype(np.float32)
        x0 = np.nan_to_num(x, nan=0.0)
        k = np.ones(w, dtype=np.float32)
        num = np.convolve(x0, k, mode="same")
        den = np.convolve(m,  k, mode="same")
        return num / np.maximum(den, 1e-8)

    # HT reward vs time
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time, rewards_ht, linewidth=1.2, alpha=0.35, label="HT reward (per chunk)")
    ax.plot(time, moving_avg_nan(rewards_ht, w=5), linewidth=2.2, label="HT reward (moving avg)")
    ax.set_xlabel("Time (Fraction of Run)", loc="center")
    ax.set_ylabel("Reward", loc="center")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best", frameon=True)
    add_cms_header(fig, run_label=run_label)
    save_png(fig, str(outdir / "reward_ht_pidData_dqn"))
    plt.close(fig)

    # AS reward vs time
    time_as = np.linspace(0, 1, len(R1_as))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time_as, rewards_as, linewidth=1.2, alpha=0.35, label="AS reward (per chunk)")
    ax.plot(time_as, moving_avg_nan(rewards_as, w=5), linewidth=2.2, label="AS reward (moving avg)")
    ax.set_xlabel("Time (Fraction of Run)", loc="center")
    ax.set_ylabel("Reward", loc="center")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best", frameon=True)
    add_cms_header(fig, run_label=run_label)
    save_png(fig, str(outdir / "reward_as_pidData_dqn"))
    plt.close(fig)
    
    plot_rate_with_tolerance(
        time, R1_ht, R2_ht, R3_ht,
        outbase=outdir / "bht_rate_pidData_dqn",
        run_label=run_label,
        legend_title="HT Trigger",
        ylim=(0, 200),
        tol_upper=upper_tol_khz,
        tol_lower=lower_tol_khz,
        const_style=dict(color="tab:blue", **CONST_STYLE),
        pd_style=dict(color="mediumblue", **PD_STYLE),
        dqn_style=dict(color="tab:purple", **DQN_STYLE),
        # pass your functions from utils import
        add_cms_header=add_cms_header,
        save_pdf_png=save_png,
    )

    # ------------------------- common styles -------------------------
    styles = {
        "Constant": CONST_STYLE,
        "PD":       PD_STYLE,
        "DQN":      DQN_STYLE,
    }


    # =========================================================
    # HT plots
    # =========================================================
    # (2) HT cut evolution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time, Ht_pd_hist,  color="mediumblue", linewidth=2.0, label="PID Controller")
    ax.plot(time, Ht_dqn_hist, color="tab:purple", label="DQN", **DQN_STYLE)
    ax.axhline(y=fixed_Ht_cut, color="gray", linestyle="--", linewidth=1.5, label="fixed_Ht_cut")
    ax.set_xlabel("Time (Fraction of Run)", loc="center")
    ax.set_ylabel("Ht_cut [GeV]", loc="center")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(title="HT Cut", fontsize=14, frameon=True, loc="best")
    add_cms_header(fig, run_label=run_label)
    save_png(fig, str(outdir / "ht_cut_pidData_dqn"))
    plt.close(fig)

    # (3) HT cumulative eff (relative to t0)
    tt_c_const = cummean(L_tt_ht_const)
    tt_c_pd    = cummean(L_tt_ht_pd)
    tt_c_dqn   = cummean(L_tt_ht_dqn)
    aa_c_const = cummean(L_aa_ht_const)
    aa_c_pd    = cummean(L_aa_ht_pd)
    aa_c_dqn   = cummean(L_aa_ht_dqn)

    colors_ht = {"ttbar": "goldenrod", "HToAATo4B": "seagreen"}
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time, rel_to_t0(tt_c_const), color=colors_ht["ttbar"], **styles["Constant"],
            label=fr"Constant Menu, ttbar ($\epsilon[t_0]={tt_c_const[0]:.2f}\%$)")
    ax.plot(time, rel_to_t0(aa_c_const), color=colors_ht["HToAATo4B"], **styles["Constant"],
            label=fr"Constant Menu, HToAATo4B ($\epsilon[t_0]={aa_c_const[0]:.2f}\%$)")
    ax.plot(time, rel_to_t0(tt_c_pd), color=colors_ht["ttbar"], **styles["PD"],
            label=fr"PID Controller, ttbar ($\epsilon[t_0]={tt_c_pd[0]:.2f}\%$)")
    ax.plot(time, rel_to_t0(aa_c_pd), color=colors_ht["HToAATo4B"], **styles["PD"],
            label=fr"PID Controller, HToAATo4B ($\epsilon[t_0]={aa_c_pd[0]:.2f}\%$)")
    ax.plot(time, rel_to_t0(tt_c_dqn), color=colors_ht["ttbar"],
            label=fr"DQN, ttbar ($\epsilon[t_0]={tt_c_dqn[0]:.2f}\%$)", **DQN_STYLE)
    ax.plot(time, rel_to_t0(aa_c_dqn), color=colors_ht["HToAATo4B"],
            label=fr"DQN, HToAATo4B ($\epsilon[t_0]={aa_c_dqn[0]:.2f}\%$)", **DQN_STYLE)
    ax.set_xlabel("Time (Fraction of Run)", loc="center")
    ax.set_ylabel("Relative Cumulative Efficiency", loc="center")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_ylim(0.5, 2.5)
    ax.legend(title="HT Trigger", fontsize=14, frameon=True, loc="best")
    add_cms_header(fig, run_label=run_label)
    save_png(fig, str(outdir / "sht_rate_pidData2data_dqn"))
    plt.close(fig)

    # (4) HT local eff (relative to t0)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time, rel_to_t0(L_tt_ht_const), color=colors_ht["ttbar"], **styles["Constant"],
            label=fr"Constant Menu, ttbar ($\epsilon[t_0]={L_tt_ht_const[0]:.2f}\%$)")
    ax.plot(time, rel_to_t0(L_aa_ht_const), color=colors_ht["HToAATo4B"], **styles["Constant"],
            label=fr"Constant Menu, HToAATo4B ($\epsilon[t_0]={L_aa_ht_const[0]:.2f}\%$)")
    ax.plot(time, rel_to_t0(L_tt_ht_pd), color=colors_ht["ttbar"], **styles["PD"],
            label=fr"PID Controller, ttbar ($\epsilon[t_0]={L_tt_ht_pd[0]:.2f}\%$)")
    ax.plot(time, rel_to_t0(L_aa_ht_pd), color=colors_ht["HToAATo4B"], **styles["PD"],
            label=fr"PID Controller, HToAATo4B ($\epsilon[t_0]={L_aa_ht_pd[0]:.2f}\%$)")
    ax.plot(time, rel_to_t0(L_tt_ht_dqn), color=colors_ht["ttbar"], 
            label=fr"DQN, ttbar ($\epsilon[t_0]={L_tt_ht_dqn[0]:.2f}\%$)", **DQN_STYLE)
    ax.plot(time, rel_to_t0(L_aa_ht_dqn), color=colors_ht["HToAATo4B"], 
            label=fr"DQN, HToAATo4B ($\epsilon[t_0]={L_aa_ht_dqn[0]:.2f}\%$)", **DQN_STYLE)
    ax.set_xlabel("Time (Fraction of Run)", loc="center")
    ax.set_ylabel("Relative Efficiency", loc="center")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_ylim(0.0, 2.5)
    ax.legend(title="HT Trigger", fontsize=14, frameon=True, loc="best")
    add_cms_header(fig, run_label=run_label)
    save_png(fig, str(outdir / "L_sht_rate_pidData2data_dqn"))
    plt.close(fig)

    # (5) HT loss
    if losses_ht:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(np.arange(len(losses_ht)), losses_ht, linewidth=1.5)
        ax.set_title("HT DQN training loss")
        ax.set_xlabel("Gradient step")
        ax.set_ylabel("Loss")
        ax.grid(True, linestyle="--", alpha=0.5)
        add_cms_header(fig, run_label=run_label)
        save_png(fig, str(outdir / "dqn_loss_ht"))
        plt.close(fig)

    # =========================================================
    # AD plots
    # =========================================================
    time_as = np.linspace(0, 1, len(R1_as))
    plot_rate_with_tolerance(
        time_as, R1_as, R2_as, R3_as,
        outbase=outdir / "bas_rate_pidData_dqn",
        run_label=run_label,
        legend_title="AD Trigger",
        ylim=(0, 200),
        tol_upper=upper_tol_khz,
        tol_lower=lower_tol_khz,
        const_style=dict(color="tab:blue", **CONST_STYLE),
        pd_style=dict(color="mediumblue", **PD_STYLE),
        dqn_style=dict(color="tab:purple", **DQN_STYLE),
        add_cms_header=add_cms_header,
        save_pdf_png=save_png,
    )

    # (A2) AS cut evolution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time_as, As_pd_hist,  color="mediumblue", linewidth=2.0, label="PD Controller")
    ax.plot(time_as, As_dqn_hist, color="tab:purple", linewidth=2.0, label="DQN")
    ax.axhline(y=fixed_AS_cut, color="gray", linestyle="--", linewidth=1.5, label="fixed_AS_cut")
    ax.set_xlabel("Time (Fraction of Run)", loc="center")
    ax.set_ylabel("AS_cut", loc="center")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(title="AD Cut", fontsize=14, frameon=True, loc="best")
    add_cms_header(fig, run_label=run_label)
    save_png(fig, str(outdir / "as_cut_pidData_dqn"))
    plt.close(fig)

    # (A3) AD cumulative eff (relative to t0)
    tt_c_const = cummean(L_tt_as_const)
    tt_c_pd    = cummean(L_tt_as_pd)
    tt_c_dqn   = cummean(L_tt_as_dqn)
    aa_c_const = cummean(L_aa_as_const)
    aa_c_pd    = cummean(L_aa_as_pd)
    aa_c_dqn   = cummean(L_aa_as_dqn)

    colors_ad = {"ttbar": "goldenrod", "HToAATo4B": "limegreen"}

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time_as, rel_to_t0(tt_c_const), color=colors_ad["ttbar"], **styles["Constant"],
            label=fr"Constant Menu, ttbar ($\epsilon[t_0]={tt_c_const[0]:.2f}\%$)")
    ax.plot(time_as, rel_to_t0(aa_c_const), color=colors_ad["HToAATo4B"], **styles["Constant"],
            label=fr"Constant Menu, HToAATo4B ($\epsilon[t_0]={aa_c_const[0]:.2f}\%$)")
    ax.plot(time_as, rel_to_t0(tt_c_pd), color=colors_ad["ttbar"], **styles["PD"],
            label=fr"PID Controller, ttbar ($\epsilon[t_0]={tt_c_pd[0]:.2f}\%$)")
    ax.plot(time_as, rel_to_t0(aa_c_pd), color=colors_ad["HToAATo4B"], **styles["PD"],
            label=fr"PID Controller, HToAATo4B ($\epsilon[t_0]={aa_c_pd[0]:.2f}\%$)")
    ax.plot(time_as, rel_to_t0(tt_c_dqn), color=colors_ad["ttbar"],
            label=fr"DQN, ttbar ($\epsilon[t_0]={tt_c_dqn[0]:.2f}\%$)", **DQN_STYLE)
    ax.plot(time_as, rel_to_t0(aa_c_dqn), color=colors_ad["HToAATo4B"], 
            label=fr"DQN, HToAATo4B ($\epsilon[t_0]={aa_c_dqn[0]:.2f}\%$)", **DQN_STYLE)

    ax.set_xlabel("Time (Fraction of Run)", loc="center")
    ax.set_ylabel("Relative Cumulative Efficiency", loc="center")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_ylim(0.5, 2.5)
    ax.legend(title="AD Trigger", fontsize=14, frameon=True, loc="best")
    add_cms_header(fig, run_label=run_label)
    save_png(fig, str(outdir / "sas_rate_pidData2data_dqn"))
    plt.close(fig)

    # (A4) AD local eff (relative to t0)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time_as, rel_to_t0(L_tt_as_const), color=colors_ad["ttbar"], **styles["Constant"],
            label=fr"Constant Menu, ttbar ($\epsilon[t_0]={L_tt_as_const[0]:.2f}\%$)")
    ax.plot(time_as, rel_to_t0(L_aa_as_const), color=colors_ad["HToAATo4B"], **styles["Constant"],
            label=fr"Constant Menu, HToAATo4B ($\epsilon[t_0]={L_aa_as_const[0]:.2f}\%$)")
    ax.plot(time_as, rel_to_t0(L_tt_as_pd), color=colors_ad["ttbar"], **styles["PD"],
            label=fr"PID Controller, ttbar ($\epsilon[t_0]={L_tt_as_pd[0]:.2f}\%$)")
    ax.plot(time_as, rel_to_t0(L_aa_as_pd), color=colors_ad["HToAATo4B"], **styles["PD"],
            label=fr"PID Controller, HToAATo4B ($\epsilon[t_0]={L_aa_as_pd[0]:.2f}\%$)")
    ax.plot(time_as, rel_to_t0(L_tt_as_dqn), color=colors_ad["ttbar"], linewidth=2.2, linestyle="dashdot",
            label=fr"DQN, ttbar ($\epsilon[t_0]={L_tt_as_dqn[0]:.2f}\%$)")
    ax.plot(time_as, rel_to_t0(L_aa_as_dqn), color=colors_ad["HToAATo4B"], linewidth=2.2, linestyle="dashdot",
            label=fr"DQN, HToAATo4B ($\epsilon[t_0]={L_aa_as_dqn[0]:.2f}\%$)")

    ax.set_xlabel("Time (Fraction of Run)", loc="center")
    ax.set_ylabel("Relative Efficiency", loc="center")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_ylim(0.5, 2.5)
    ax.legend(title="AD Trigger", fontsize=14, frameon=True, loc="best")
    add_cms_header(fig, run_label=run_label)
    save_png(fig, str(outdir / "L_sas_rate_pidData2data_dqn"))
    plt.close(fig)

    # (A5) AS loss
    if losses_as:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(np.arange(len(losses_as)), losses_as, linewidth=1.5)
        ax.set_title("AD DQN training loss")
        ax.set_xlabel("Gradient step")
        ax.set_ylabel("Loss")
        ax.grid(True, linestyle="--", alpha=0.5)
        add_cms_header(fig, run_label=run_label)
        save_png(fig, str(outdir / "dqn_loss_as"))
        plt.close(fig)

    print("\nSaved to:", outdir)
    for p in sorted(outdir.glob("*.pdf")):
        print(" -", p.name)

    

    # =========================================================
    # AUROC vs time in FIXED 50k-event windows (margin = x - cut)
    # =========================================================
    auroc_window = chunk_size
    plots_dir = outdir / "extra_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # HT AUROC windows
    t_auc_ht, auc_ht_pd, auc_ht_dqn = compute_auroc_windows(
        start_event=start_event,
        window_events=auroc_window,
        update_chunk_size=chunk_size,   # PD updates once per training chunk, DQN logged per chunk too
        matched_by_index=matched_by_index,
        Bnpv=Bnpv, Tnpv=Tnpv, Anpv=Anpv,
        Bx=Bht, Tx=Tht, Ax=Aht,
        cut_hist_pd=Ht_pd_hist,
        cut_hist_dqn=Ht_dqn_hist,
        max_n=200_000,
        seed=SEED,
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t_auc_ht, auc_ht_pd,  linewidth=2.2, label=f"HT PD (AUROC / {auroc_window} evts)")
    ax.plot(t_auc_ht, auc_ht_dqn, linewidth=2.2, label=f"HT DQN (AUROC / {auroc_window} evts)")
    ax.set_xlabel("Time (Fraction of Run)")
    ax.set_ylabel("AUROC")
    ax.set_title(f"HT: AUROC vs time (score = HT - cut, window={auroc_window} events)")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best", frameon=True)
    add_cms_header(fig, run_label=run_label)
    save_png(fig, str(plots_dir / f"auroc_window{auroc_window}_ht_pd_vs_dqn"))
    plt.close(fig)

    # AS AUROC windows
    t_auc_as, auc_as_pd, auc_as_dqn = compute_auroc_windows(
        start_event=start_event,
        window_events=auroc_window,
        update_chunk_size=chunk_size,
        matched_by_index=matched_by_index,
        Bnpv=Bnpv, Tnpv=Tnpv, Anpv=Anpv,
        Bx=Bas, Tx=Tas, Ax=Aas,
        cut_hist_pd=As_pd_hist,
        cut_hist_dqn=As_dqn_hist,
        max_n=200_000,
        seed=SEED + 999,
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t_auc_as, auc_as_pd,  linewidth=2.2, label=f"AS PD (AUROC / {auroc_window} evts)")
    ax.plot(t_auc_as, auc_as_dqn, linewidth=2.2, label=f"AS DQN (AUROC / {auroc_window} evts)")
    ax.set_xlabel("Time (Fraction of Run)")
    ax.set_ylabel("AUROC")
    ax.set_title(f"AS: AUROC vs time (score = AS - cut, window={auroc_window} events)")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best", frameon=True)
    add_cms_header(fig, run_label=run_label)
    save_png(fig, str(plots_dir / f"auroc_window{auroc_window}_as_pd_vs_dqn"))
    plt.close(fig)

    print(f"[OK] AUROC window plots saved under {plots_dir}/")

if __name__ == "__main__":
    main()