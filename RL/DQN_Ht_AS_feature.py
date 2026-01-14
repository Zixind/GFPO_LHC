#!/usr/bin/env python3
"""
DQN_Ht_AS_feature.py

SingleTrigger: 
Constant vs PD vs DQN
- HT trigger: accept = (HT >= Ht_cut)
- AS trigger: accept = (AS >= AS_cut)

We train two independent DQNs:
  (1) DQN_HT controls Ht_cut using HT-only rates
  (2) DQN_AS controls AS_cut using AS-only rates For AS only, we use binned steps to ensure stability.

Outputs:
outdir = RL_outputs/demo_sing_dqn_separate_feature
HT trigger outputs:
  - bht_rate_pidData_dqn_feature.png          (HT background rate [kHz])
  - ht_cut_pidData_dqn_feature.png            (Ht_cut evolution)
  - sht_rate_pidData2data_dqn_feature.png     (cumulative signal eff, relative to t0)
  - L_sht_rate_pidData2data_dqn_feature.png   (local signal eff, relative to t0)
  - dqn_loss_ht_feature.png                   (HT DQN loss)

AS trigger outputs:
  - bas_rate_pidData_dqn_feature.png          (AS background rate [kHz])
  - as_cut_pidData_dqn_feature.png            (AS_cut evolution)
  - sas_rate_pidData2data_dqn_feature.png     (cumulative signal eff, relative to t0)
  - L_sas_rate_pidData2data_dqn_feature.png   (local signal eff, relative to t0)
  - dqn_loss_as_feature.png                   (AS DQN loss)

"""
from dataclasses import dataclass
import random
import argparse
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import h5py
import hdf5plugin  # noqa: F401
import csv
from pathlib import Path
from controllers import PD_controller1, PD_controller2
from triggers import Sing_Trigger
from RL.utils import cummean, rel_to_t0, add_cms_header, plot_rate_with_tolerance, plot_rate_with_tolerance_4, save_png, print_h5_tree, read_any_h5, compute_auroc_windows_separate, compute_operating_point_windows_separate, style_diag_axes, style_diag_legend, finalize_diag_fig, set_paper_style  #save_pdf_png,
from RL.dqn_agent import DQNAgent, make_obs, shield_delta, compute_reward, DQNConfig, SeqDQNAgent, make_event_seq_as_v0, make_event_seq_ht_v0

# ------------------------- Fixing seed for reproducibility -------------------------
SEED = 20251213
random.seed(SEED)
np.random.seed(SEED)

set_paper_style()

def near_occupancy(x, cut, widths):
    x = np.asarray(x, dtype=np.float32)
    out = []
    for w in widths:
        out.append(float(np.mean(np.abs(x - cut) <= float(w))))
    return np.array(out, dtype=np.float32)
@dataclass
class RollingWindow: #sliding window for event-level features
    def __init__(self, max_events: int):
        self.max_events = int(max_events)
        self._bht  = deque(maxlen=self.max_events)
        self._bas  = deque(maxlen=self.max_events)
        self._bnpv = deque(maxlen=self.max_events)

    def append(self, bht, bas, bnpv):
        self._bht.extend(np.asarray(bht,  dtype=np.float32).tolist())
        self._bas.extend(np.asarray(bas,  dtype=np.float32).tolist())
        self._bnpv.extend(np.asarray(bnpv, dtype=np.float32).tolist())

    def get(self):
        return (
            np.fromiter(self._bht,  dtype=np.float32),
            np.fromiter(self._bas,  dtype=np.float32),
            np.fromiter(self._bnpv, dtype=np.float32),
        )

from pathlib import Path
import csv
import numpy as np

RATE_SCALE_KHZ = 400.0

def summarize_paper_table(r_pct, s_tt, s_aa, cut_hist, target_pct, tol_pct):
    r = np.asarray(r_pct, dtype=np.float64)
    s_tt = np.asarray(s_tt, dtype=np.float64)
    s_aa = np.asarray(s_aa, dtype=np.float64)
    c = np.asarray(cut_hist, dtype=np.float64)

    err = r - float(target_pct)
    abs_err = np.abs(err)
    inband = abs_err <= float(tol_pct)

    def safe_mean(x, m):
        return float(np.mean(x[m])) if np.any(m) else np.nan

    out = {}
    out["MAE"] = float(np.mean(abs_err)) if r.size else np.nan
    out["P95_abs_err"] = float(np.percentile(abs_err, 95)) if r.size else np.nan
    out["InBand"] = float(np.mean(inband)) if r.size else np.nan
    out["UpFrac"]   = float(np.mean(err >  float(tol_pct))) if r.size else np.nan
    out["DownFrac"] = float(np.mean(err < -float(tol_pct))) if r.size else np.nan
    out["tt"] = safe_mean(s_tt, inband)
    out["h_to_4b"] = safe_mean(s_aa, inband)
    return out

def write_paper_table(rows, out_csv: Path, out_tex: Path, target_pct, tol_pct):
    if not rows:
        return

    # CSV
    fieldnames = ["Trigger", "Method", "MAE", "P95_abs_err", "InBand", "UpFrac", "DownFrac", "tt", "h_to_4b"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, None) for k in fieldnames})

    # Forced order
    trigger_order = ["HT", "AD"]
    method_order  = ["Constant", "PID", "ADT", "DQN", "GRPO", "GFPO-F", "GFPO-FR"]
    trig_rank = {t: i for i, t in enumerate(trigger_order)}
    meth_rank = {m: i for i, m in enumerate(method_order)}
    rows = sorted(rows, key=lambda r: (trig_rank.get(r["Trigger"], 10**9), meth_rank.get(r["Method"], 10**9)))

    # bold best per trigger
    higher_better = {"InBand", "tt", "h_to_4b"}
    lower_better  = {"MAE", "P95_abs_err", "UpFrac", "DownFrac"}

    triggers = [t for t in trigger_order if any(rr["Trigger"] == t for rr in rows)]
    triggers += [t for t in sorted(set(rr["Trigger"] for rr in rows)) if t not in triggers]

    best = {tr: {} for tr in triggers}
    for tr in triggers:
        sub = [r for r in rows if r["Trigger"] == tr]
        for k in higher_better:
            vals = np.array([float(x[k]) for x in sub], dtype=np.float64)
            best[tr][k] = sub[int(np.nanargmax(vals))]["Method"] if np.any(np.isfinite(vals)) else sub[0]["Method"]
        for k in lower_better:
            vals = np.array([float(x[k]) for x in sub], dtype=np.float64)
            best[tr][k] = sub[int(np.nanargmin(vals))]["Method"] if np.any(np.isfinite(vals)) else sub[0]["Method"]

    def fmt(v, nd=3):
        if v is None:
            return "nan"
        if isinstance(v, (float, np.floating)):
            if not np.isfinite(v):
                return "nan"
            return f"{v:.{nd}f}"
        return str(v)

    def cell(tr, method, key, val):
        s = fmt(val, 3)
        return r"\textbf{" + s + "}" if best.get(tr, {}).get(key) == method else s

    # LaTeX
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{6pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.10}")
    lines.append(r"\begin{tabular}{llrrrrrrr}")
    lines.append(r"\hline")
    lines.append(
        r"Trigger & Method & MAE$\downarrow$ & P95$|e|$$\downarrow$ & InBand$\uparrow$ & "
        r"UpFrac$\downarrow$ & DownFrac$\downarrow$ & $t\bar{t}\uparrow$ & $h\rightarrow 4b\uparrow$ \\"
    )
    lines.append(r"\hline")

    for tr in triggers:
        sub = [r for r in rows if r["Trigger"] == tr]
        lines.append(rf"\multicolumn{{9}}{{l}}{{\textbf{{{tr} trigger}}}} \\")
        for r in sub:
            m = r["Method"]
            lines.append(
                f"{tr} & {m} & "
                f"{cell(tr,m,'MAE',r['MAE'])} & "
                f"{cell(tr,m,'P95_abs_err',r['P95_abs_err'])} & "
                f"{cell(tr,m,'InBand',r['InBand'])} & "
                f"{cell(tr,m,'UpFrac',r['UpFrac'])} & "
                f"{cell(tr,m,'DownFrac',r['DownFrac'])} & "
                f"{cell(tr,m,'tt',r['tt'])} & "
                f"{cell(tr,m,'h_to_4b',r['h_to_4b'])} \\\\"
            )
        lines.append(r"\hline")

    lines.append(r"\end{tabular}")
    lines.append(
        rf"\caption{{Summary of single-trigger control. Rates are in percent units with target "
        rf"$r^*={target_pct:.3f}\%$ and tolerance $\pm {tol_pct:.3f}\%$. "
        rf"InBand is the fraction of chunks within $|r-r^*|\le\tau$. "
        rf"UpFrac and DownFrac measure upward/downward band violations. "
        rf"$t\bar t$ and $h\rightarrow 4b$ are mean signal efficiencies conditioned on in-band chunks.}}"
    )
    lines.append(r"\label{tab:single_trigger_summary_paper}")
    lines.append(r"\end{table}")

    out_tex.write_text("\n".join(lines) + "\n")


# ------------------------- main -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="Data/Matched_data_2016_dim2.h5",
                    help="Matched_data_*.h5 (data) or Trigger_food_*.h5 (MC)")
    ap.add_argument("--outdir", default="RL_outputs/demo_sing_dqn_separate_feature", help="output dir")
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
    ap.add_argument("--window-events-chunk-size", type=int, default=3,
                help="How many most-recent background events of chunk size to keep for features (sliding window).")
    ap.add_argument("--seq-len", type=int, default=128,
                help="Sequence length K passed into make_event_seq_* (downsample/pad to this).")
    ap.add_argument("--inner-stride", type=int, default=10000,
        help="Micro-step size inside each chunk (events). 50k chunk with 10k stride -> 5 transitions per chunk.")
    ap.add_argument("--train-steps-per-micro", type=int, default=1,
        help="How many gradient updates to do per micro-step (usually 1).")
    # --- Frozen DQN baseline (DQN-F) ---
    ap.add_argument("--dqn-f-train-chunks", type=int, default=1,
                    help="Train DQN-F only on the first N chunks, then freeze weights and rollout.")
    ap.add_argument("--dqn-f-eps", type=float, default=0.0,
                    help="Epsilon used for DQN-F rollout AFTER freezing (default: greedy).")
    
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
        # Data_SingleTrigger.py uses 10k for RealData DEBUG
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
    tol = 0.025     # background - target/tolerance for reward?
    alpha = 0.4    # signal bonus
    beta  = 0.2   # move penalty

    HT_DELTAS = np.array([float(x) for x in args.ht_deltas.split(",")], dtype=np.float32)
    HT_STEP = 1.0
    MAX_DELTA_HT = float(np.max(np.abs(HT_DELTAS))) * HT_STEP

    AS_DELTAS = np.array([float(x) for x in args.as_deltas.split(",")], dtype=np.float32)
    AS_STEP = float(args.as_step)
    MAX_DELTA_AS = float(np.max(np.abs(AS_DELTAS))) * AS_STEP
    print("MAX_DELTA_AS=", MAX_DELTA_AS)

    cfg = DQNConfig(lr=5e-4, gamma=0.95, batch_size=32, target_update=200)
    
    # Make AS agent slower learning rate for faster adaptation
    cfg_as = DQNConfig(lr=1e-4, gamma=0.95, batch_size=32, target_update=200)
    K = int(args.seq_len)
    near_widths_ht = (5.0, 10.0, 20.0)
    feat_dim_ht = 10 + len(near_widths_ht)   # 13       

    near_widths_as = (0.25, 0.5, 1.0)
    feat_dim_as = 10 + len(near_widths_as)   # 13

    agent_ht = SeqDQNAgent(seq_len=K, feat_dim=feat_dim_ht, n_actions=len(HT_DELTAS), cfg=cfg, seed=SEED)
    agent_as = SeqDQNAgent(seq_len=K, feat_dim=feat_dim_as, n_actions=len(AS_DELTAS), cfg=cfg_as, seed=SEED)

    # ---- DQN-F agents (separate weights; train only early then freeze) ----
    agent_ht_f = SeqDQNAgent(seq_len=K, feat_dim=feat_dim_ht, n_actions=len(HT_DELTAS), cfg=cfg, seed=SEED + 7)
    agent_as_f = SeqDQNAgent(seq_len=K, feat_dim=feat_dim_as, n_actions=len(AS_DELTAS), cfg=cfg_as, seed=SEED + 7)

    roll = RollingWindow(max_events=int(args.window_events_chunk_size * chunk_size))

    # state trackers (HT)
    prev_obs_ht = None
    prev_act_ht = None
    prev_bg_ht = None
    last_dht = 0.0
    losses_ht = []

    # state trackers (AS)
    prev_obs_as = None
    prev_act_as = None
    prev_bg_as = None
    last_das = 0.0
    losses_as = []

    rewards_ht = []   # reward used to train HT agent per chunk
    rewards_as = []   # reward used to train AS agent per chunk

    # --- near-cut occupancy logs ---
    near_occ_ht = []   # list of shape (W_ht,)
    near_occ_as = []   # list of shape (W_as,)
    near_t = []        # micro-step time in [0,1]

    # --- sensitivity proxy logs  ---
    sens_ht = []       # |Δrate| / |Δcut|
    sens_as = []
    occ_mid_ht = []    # pick one width for scatter, e.g. 10 GeV
    occ_mid_as = []    # e.g. 0.02

    # ------------------------- logs (HT) -------------------------
    R1_ht, R2_ht, R3_ht = [], [], []                  # background % (const, PD, DQN)
    Ht_pd_hist, Ht_dqn_hist = [], []
    L_tt_ht_const, L_tt_ht_pd, L_tt_ht_dqn = [], [], []
    L_aa_ht_const, L_aa_ht_pd, L_aa_ht_dqn = [], [], []
    L_tt_ht_dqnf, L_aa_ht_dqnf = [], []

    # ------------------------- logs (AS) -------------------------
    R1_as, R2_as, R3_as = [], [], []                  # background % (const, PD, DQN)
    As_pd_hist, As_dqn_hist = [], []
    L_tt_as_const, L_tt_as_pd, L_tt_as_dqn = [], [], []
    L_aa_as_const, L_aa_as_pd, L_aa_as_dqn = [], [], []
    L_tt_as_dqnf, L_aa_as_dqnf = [], []

    # --- Counterfactual (CF) action landscape logs (per chunk) ---
    cf_r_ht = []          # list of shape (n_actions_ht,)
    cf_r_as = []          # list of shape (n_actions_as,)
    dqn_act_ht_chunk = [] # last micro-step action index for HT within each chunk
    dqn_act_as_chunk = [] # last micro-step action index for AS within each chunk


    # DQN-F (frozen baseline)
    Ht_cut_dqnf = fixed_Ht_cut
    AS_cut_dqnf = fixed_AS_cut

    prev_bg_ht_f = None
    last_dht_f = 0.0
    losses_ht_f = []

    prev_bg_as_f = None
    last_das_f = 0.0
    losses_as_f = []

    # ------------------------- logs (HT) DQN-F -------------------------
    R4_ht = []   # add R4_ht = DQN-F
    Ht_dqnf_hist = []

    # ------------------------- logs (AS) -------------------------
    R4_as = []   # add R4_as = DQN-F
    As_dqnf_hist = []

    # ------------------------- batching loop -------------------------
    batch_starts = list(range(start_event, N, chunk_size))

    for t, I in enumerate(batch_starts):
        end = min(I + chunk_size, N, len(Bnpv), len(Bas))
        if end <= I:
            break
        idx = np.arange(I, end)

        # chunk background arrays
        bht  = Bht[idx]
        bas  = Bas[idx]
        bnpv = Bnpv[idx]

        # micro-step setup
        stride = max(500, int(args.inner_stride))
        chunk_len = end - I
        n_micro = max(1, int(np.ceil(chunk_len / stride)))

        micro_rewards_ht = []
        micro_rewards_as = []

        train_f = (t < int(args.dqn_f_train_chunks))
        # -------------------------
        # micro loop: rl transitions
        # -------------------------
        bg_acc_ht = 0; bg_tot_ht = 0
        bg_acc_as = 0; bg_tot_as = 0
        for j in range(n_micro):
            j_lo = I + j * stride
            j_hi = min(I + (j + 1) * stride, end)
            if j_hi <= j_lo:
                continue

            idxj = np.arange(j_lo, j_hi)

            bht_j  = Bht[idxj]
            bas_j  = Bas[idxj]
            bnpv_j = Bnpv[idxj]

            # sliding-window update
            roll.append(bht_j, bas_j, bnpv_j)
            bht_w, bas_w, bnpv_w = roll.get()

            # signal subsets for THIS micro-step
            if matched_by_index:
                end_sig_j = min(j_hi, len(Tht), len(Aht), len(Tas), len(Aas))
                if j_lo >= end_sig_j:
                    # no signal available for this micro-step
                    sht_tt_j = np.empty(0, dtype=np.float32); sas_tt_j = np.empty(0, dtype=np.float32)
                    sht_aa_j = np.empty(0, dtype=np.float32); sas_aa_j = np.empty(0, dtype=np.float32)
                else:
                    idx_sig_j = np.arange(j_lo, end_sig_j)
                    sht_tt_j = Tht[idx_sig_j];  sas_tt_j = Tas[idx_sig_j]
                    sht_aa_j = Aht[idx_sig_j];  sas_aa_j = Aas[idx_sig_j]
            else:
                # Pick signal events with similar NPV range as the current micro background window
                npv_min = float(np.min(bnpv_j))
                npv_max = float(np.max(bnpv_j))

                mask_tt = (Tnpv >= npv_min) & (Tnpv <= npv_max)
                mask_aa = (Anpv >= npv_min) & (Anpv <= npv_max)

                sht_tt_j = Tht[mask_tt];  sas_tt_j = Tas[mask_tt]
                sht_aa_j = Aht[mask_aa];  sas_aa_j = Aas[mask_aa]
            # global micro-step index for epsilon
            step = t * n_micro + j
            eps = max(0.05, 1.0 * (0.98 ** step))

            # =========================================================
            # HT micro-step: (s, a, r, s')
            # =========================================================
            bg_before_ht = Sing_Trigger(bht_j, Ht_cut_dqn)
            if prev_bg_ht is None:
                prev_bg_ht = bg_before_ht

            obs_ht = make_event_seq_ht_v0(
                bht=bht_w, bnpv=bnpv_w,
                bg_rate=bg_before_ht,
                prev_bg_rate=prev_bg_ht,
                cut=Ht_cut_dqn,
                ht_mid=ht_mid, ht_span=ht_span,
                target=target,
                K=K,
                last_delta=last_dht,
                max_delta=MAX_DELTA_HT,
                near_widths=near_widths_ht,
            )

            act_ht = agent_ht.act(obs_ht, eps=eps)
            last_act_ht_in_chunk = act_ht
            dht = float(HT_DELTAS[act_ht])

            # shield based on micro-step bg
            sd = shield_delta(bg_before_ht, target, tol, MAX_DELTA_HT)
            if sd is not None:
                dht = float(sd)

            Ht_cut_next = float(np.clip(Ht_cut_dqn + dht, ht_lo, ht_hi))

            bg_after_ht = Sing_Trigger(bht_j, Ht_cut_next)
            tt_after_ht = Sing_Trigger(sht_tt_j, Ht_cut_next)
            aa_after_ht = Sing_Trigger(sht_aa_j, Ht_cut_next)

            obs_ht_next = make_event_seq_ht_v0(
                bht=bht_w, bnpv=bnpv_w,
                bg_rate=bg_after_ht,
                prev_bg_rate=bg_before_ht,
                cut=Ht_cut_next,
                ht_mid=ht_mid, ht_span=ht_span,
                target=target,
                K=K,
                last_delta=dht,
                max_delta=MAX_DELTA_HT,
                near_widths=near_widths_ht,
            )   

            r_ht = compute_reward(
                bg_rate=bg_after_ht,
                target=target,
                tol=tol,
                sig_rate_1=tt_after_ht,
                sig_rate_2=aa_after_ht,
                delta_applied=dht,
                max_delta=MAX_DELTA_HT,
                alpha=alpha,
                beta=beta,
                prev_bg_rate=bg_before_ht,
                gamma_stab=0.3,
            )

            agent_ht.buf.push(obs_ht, act_ht, r_ht, obs_ht_next, done=False)
            for _ in range(int(args.train_steps_per_micro)):
                loss = agent_ht.train_step()
                if loss is not None:
                    losses_ht.append(loss)

            micro_rewards_ht.append(r_ht)

            # advance HT state
            Ht_cut_dqn = Ht_cut_next
            prev_bg_ht = bg_after_ht
            last_dht = dht

            # =========================================================
            # AS micro-step: (s, a, r, s')
            # =========================================================
            bg_before_as = Sing_Trigger(bas_j, AS_cut_dqn)
            if prev_bg_as is None:
                prev_bg_as = bg_before_as

            obs_as = make_event_seq_as_v0(
                bas=bas_w, bnpv=bnpv_w,
                bg_rate=bg_before_as,
                prev_bg_rate=prev_bg_as,
                cut=AS_cut_dqn,
                as_mid=as_mid, as_span=as_span,
                target=target,
                K=K,
                last_delta=last_das,
                max_delta=MAX_DELTA_AS,
                near_widths=near_widths_as,
            )

            act_as = agent_as.act(obs_as, eps=eps)
            last_act_as_in_chunk = act_as
            das = float(AS_DELTAS[act_as] * AS_STEP)

            sd = shield_delta(bg_before_as, target, tol, MAX_DELTA_AS)
            if sd is not None:
                das = float(sd)

            AS_cut_next = float(np.clip(AS_cut_dqn + das, as_lo, as_hi))

            bg_after_as = Sing_Trigger(bas_j, AS_cut_next)
            tt_after_as = Sing_Trigger(sas_tt_j, AS_cut_next)
            aa_after_as = Sing_Trigger(sas_aa_j, AS_cut_next)

            obs_as_next = make_event_seq_as_v0(
                bas=bas_w, bnpv=bnpv_w,
                bg_rate=bg_after_as,
                prev_bg_rate=bg_before_as,
                cut=AS_cut_next,
                as_mid=as_mid, as_span=as_span,
                target=target,
                K=K,
                last_delta=das,
                max_delta=MAX_DELTA_AS,
                near_widths=near_widths_as,
            )

            r_as = compute_reward(
                bg_rate=bg_after_as,
                target=target,
                tol=tol,
                sig_rate_1=tt_after_as,
                sig_rate_2=aa_after_as,
                delta_applied=das,
                max_delta=MAX_DELTA_AS,
                alpha=alpha,
                beta=beta,
                prev_bg_rate=bg_before_as,
                gamma_stab=0.3,
            )

            agent_as.buf.push(obs_as, act_as, r_as, obs_as_next, done=False)
            for _ in range(int(args.train_steps_per_micro)):
                loss = agent_as.train_step()
                if loss is not None:
                    losses_as.append(loss)

            micro_rewards_as.append(r_as)

            # advance AS state
            AS_cut_dqn = AS_cut_next
            prev_bg_as = bg_after_as
            last_das = das


            # =========================================================
            # HT micro-step: DQN-F (train early, then frozen rollout)
            # =========================================================
            bg_before_ht_f = Sing_Trigger(bht_j, Ht_cut_dqnf)
            if prev_bg_ht_f is None:
                prev_bg_ht_f = bg_before_ht_f

            obs_ht_f = make_event_seq_ht_v0(
                bht=bht_w, bnpv=bnpv_w,
                bg_rate=bg_before_ht_f,
                prev_bg_rate=prev_bg_ht_f,
                cut=Ht_cut_dqnf,
                ht_mid=ht_mid, ht_span=ht_span,
                target=target,
                K=K,
                last_delta=last_dht_f,
                max_delta=MAX_DELTA_HT,
                near_widths=near_widths_ht,
            )

            eps_f = eps if train_f else float(args.dqn_f_eps)
            act_ht_f = agent_ht_f.act(obs_ht_f, eps=eps_f)
            dht_f = float(HT_DELTAS[act_ht_f])

            sd_f = shield_delta(bg_before_ht_f, target, tol, MAX_DELTA_HT)
            if sd_f is not None:
                dht_f = float(sd_f)

            Ht_cut_next_f = float(np.clip(Ht_cut_dqnf + dht_f, ht_lo, ht_hi))

            bg_after_ht_f = Sing_Trigger(bht_j, Ht_cut_next_f)
            tt_after_ht_f = Sing_Trigger(sht_tt_j, Ht_cut_next_f)
            aa_after_ht_f = Sing_Trigger(sht_aa_j, Ht_cut_next_f)

            obs_ht_next_f = make_event_seq_ht_v0(
                bht=bht_w, bnpv=bnpv_w,
                bg_rate=bg_after_ht_f,
                prev_bg_rate=bg_before_ht_f,
                cut=Ht_cut_next_f,
                ht_mid=ht_mid, ht_span=ht_span,
                target=target,
                K=K,
                last_delta=dht_f,
                max_delta=MAX_DELTA_HT,
                near_widths=near_widths_ht,
            )

            r_ht_f = compute_reward(
                bg_rate=bg_after_ht_f,
                target=target,
                tol=tol,
                sig_rate_1=tt_after_ht_f,
                sig_rate_2=aa_after_ht_f,
                delta_applied=dht_f,
                max_delta=MAX_DELTA_HT,
                alpha=alpha,
                beta=beta,
                prev_bg_rate=bg_before_ht_f,
                gamma_stab=0.3,
            )

            if train_f:
                agent_ht_f.buf.push(obs_ht_f, act_ht_f, r_ht_f, obs_ht_next_f, done=False)
                for _ in range(int(args.train_steps_per_micro)):
                    lf = agent_ht_f.train_step()
                    if lf is not None:
                        losses_ht_f.append(lf)

            # advance DQN-F HT state
            Ht_cut_dqnf = Ht_cut_next_f
            prev_bg_ht_f = bg_after_ht_f
            last_dht_f = dht_f


            # =========================================================
            # AS micro-step: DQN-F (train early, then frozen rollout)
            # =========================================================
            bg_before_as_f = Sing_Trigger(bas_j, AS_cut_dqnf)
            if prev_bg_as_f is None:
                prev_bg_as_f = bg_before_as_f

            obs_as_f = make_event_seq_as_v0(
                bas=bas_w, bnpv=bnpv_w,
                bg_rate=bg_before_as_f,
                prev_bg_rate=prev_bg_as_f,
                cut=AS_cut_dqnf,
                as_mid=as_mid, as_span=as_span,
                target=target,
                K=K,
                last_delta=last_das_f,
                max_delta=MAX_DELTA_AS,
                near_widths=near_widths_as,
            )

            eps_f = eps if train_f else float(args.dqn_f_eps)
            act_as_f = agent_as_f.act(obs_as_f, eps=eps_f)
            das_f = float(AS_DELTAS[act_as_f] * AS_STEP)

            sd_f = shield_delta(bg_before_as_f, target, tol, MAX_DELTA_AS)
            if sd_f is not None:
                das_f = float(sd_f)

            AS_cut_next_f = float(np.clip(AS_cut_dqnf + das_f, as_lo, as_hi))

            bg_after_as_f = Sing_Trigger(bas_j, AS_cut_next_f)
            tt_after_as_f = Sing_Trigger(sas_tt_j, AS_cut_next_f)
            aa_after_as_f = Sing_Trigger(sas_aa_j, AS_cut_next_f)

            obs_as_next_f = make_event_seq_as_v0(
                bas=bas_w, bnpv=bnpv_w,
                bg_rate=bg_after_as_f,
                prev_bg_rate=bg_before_as_f,
                cut=AS_cut_next_f,
                as_mid=as_mid, as_span=as_span,
                target=target,
                K=K,
                last_delta=das_f,
                max_delta=MAX_DELTA_AS,
                near_widths=near_widths_as,
            )

            r_as_f = compute_reward(
                bg_rate=bg_after_as_f,
                target=target,
                tol=tol,
                sig_rate_1=tt_after_as_f,
                sig_rate_2=aa_after_as_f,
                delta_applied=das_f,
                max_delta=MAX_DELTA_AS,
                alpha=alpha,
                beta=beta,
                prev_bg_rate=bg_before_as_f,
                gamma_stab=0.3,
            )

            if train_f:
                agent_as_f.buf.push(obs_as_f, act_as_f, r_as_f, obs_as_next_f, done=False)
                for _ in range(int(args.train_steps_per_micro)):
                    lf = agent_as_f.train_step()
                    if lf is not None:
                        losses_as_f.append(lf)

            # advance DQN-F AS state
            AS_cut_dqnf = AS_cut_next_f
            prev_bg_as_f = bg_after_as_f
            last_das_f = das_f
        # -------------------------
        # AFTER MICRO LOOP (once per chunk): build chunk-level signals
        # -------------------------
        if matched_by_index:
            end_sig = min(end, len(Tht), len(Aht), len(Tas), len(Aas))
            idx_sig = np.arange(I, end_sig)
            sht_tt_j = Tht[idx_sig];  sas_tt_j = Tas[idx_sig]
            sht_aa_j = Aht[idx_sig];  sas_aa_j = Aas[idx_sig]
        else:
            npv_min = float(np.min(bnpv))
            npv_max = float(np.max(bnpv))
            mask_tt = (Tnpv >= npv_min) & (Tnpv <= npv_max)
            mask_aa = (Anpv >= npv_min) & (Anpv <= npv_max)
            sht_tt_j = Tht[mask_tt];  sas_tt_j = Tas[mask_tt]
            sht_aa_j = Aht[mask_aa];  sas_aa_j = Aas[mask_aa]

        # -------------------------
        # CHUNK RATES for plots/logging
        # -------------------------
        # HT
        bg_const_ht = Sing_Trigger(bht, fixed_Ht_cut)
        bg_pd_ht = Sing_Trigger(bht, Ht_cut_pd)
        bg_dqn_ht = Sing_Trigger(bht, Ht_cut_dqn)   
        # HT DQN-F chunk rate
        bg_dqnf_ht = Sing_Trigger(bht, Ht_cut_dqnf)

        tt_const_ht = Sing_Trigger(sht_tt_j, fixed_Ht_cut)
        tt_pd_ht = Sing_Trigger(sht_tt_j, Ht_cut_pd)
        tt_dqn_ht = Sing_Trigger(sht_tt_j, Ht_cut_dqn)
        tt_dqnf_ht = Sing_Trigger(sht_tt_j, Ht_cut_dqnf)

        aa_const_ht = Sing_Trigger(sht_aa_j, fixed_Ht_cut)
        aa_pd_ht = Sing_Trigger(sht_aa_j, Ht_cut_pd)
        aa_dqn_ht = Sing_Trigger(sht_aa_j, Ht_cut_dqn)
        aa_dqnf_ht = Sing_Trigger(sht_aa_j, Ht_cut_dqnf)

        # PD update (once per chunk)
        Ht_cut_pd, pre_ht_err = PD_controller1(bg_pd_ht, pre_ht_err, Ht_cut_pd)
        Ht_cut_pd = float(np.clip(Ht_cut_pd, ht_lo, ht_hi))

        # chunk reward = mean of micro rewards
        reward_ht_t = float(np.mean(micro_rewards_ht)) if micro_rewards_ht else np.nan
        rewards_ht.append(reward_ht_t)

        # log once per chunk
        R1_ht.append(bg_const_ht); R2_ht.append(bg_pd_ht); R3_ht.append(bg_dqn_ht)
        # log once per chunk for DQN-F
        R4_ht.append(bg_dqnf_ht)

        Ht_pd_hist.append(Ht_cut_pd); Ht_dqn_hist.append(Ht_cut_dqn); Ht_dqnf_hist.append(Ht_cut_dqnf)
        L_tt_ht_const.append(tt_const_ht); L_tt_ht_pd.append(tt_pd_ht); L_tt_ht_dqn.append(tt_dqn_ht); L_tt_ht_dqnf.append(tt_dqnf_ht)
        L_aa_ht_const.append(aa_const_ht); L_aa_ht_pd.append(aa_pd_ht); L_aa_ht_dqn.append(aa_dqn_ht); L_aa_ht_dqnf.append(aa_dqnf_ht)

        # AS
        bg_const_as = Sing_Trigger(bas, fixed_AS_cut)
        bg_pd_as    = Sing_Trigger(bas, AS_cut_pd)
        bg_dqn_as   = Sing_Trigger(bas, AS_cut_dqn)   
        # AS DQN-F chunk rate
        bg_dqnf_as = Sing_Trigger(bas, AS_cut_dqnf)

        tt_const_as = Sing_Trigger(sas_tt_j, fixed_AS_cut)
        tt_pd_as    = Sing_Trigger(sas_tt_j, AS_cut_pd)
        tt_dqn_as   = Sing_Trigger(sas_tt_j, AS_cut_dqn)
        tt_dqnf_as = Sing_Trigger(sas_tt_j, AS_cut_dqnf)

        aa_const_as = Sing_Trigger(sas_aa_j, fixed_AS_cut)
        aa_pd_as    = Sing_Trigger(sas_aa_j, AS_cut_pd)
        aa_dqn_as   = Sing_Trigger(sas_aa_j, AS_cut_dqn)
        aa_dqnf_as = Sing_Trigger(sas_aa_j, AS_cut_dqnf)

        # PD update (once per chunk)
        AS_cut_pd, pre_as_err = PD_controller2(bg_pd_as, pre_as_err, AS_cut_pd)
        AS_cut_pd = float(np.clip(AS_cut_pd, as_lo, as_hi))

        reward_as_t = float(np.mean(micro_rewards_as)) if micro_rewards_as else np.nan
        rewards_as.append(reward_as_t)

        R1_as.append(bg_const_as); R2_as.append(bg_pd_as); R3_as.append(bg_dqn_as); R4_as.append(bg_dqnf_as)
        As_pd_hist.append(AS_cut_pd); As_dqn_hist.append(AS_cut_dqn); As_dqnf_hist.append(AS_cut_dqnf)
        L_tt_as_const.append(tt_const_as); L_tt_as_pd.append(tt_pd_as); L_tt_as_dqn.append(tt_dqn_as); L_tt_as_dqnf.append(tt_dqnf_as)
        L_aa_as_const.append(aa_const_as); L_aa_as_pd.append(aa_pd_as); L_aa_as_dqn.append(aa_dqn_as); L_aa_as_dqnf.append(aa_dqnf_as)

        # =========================================================
        # Counterfactual reward landscape (per chunk)
        # Evaluate r(delta) for all actions on the SAME chunk
        # =========================================================

        # --- HT counterfactuals ---
        bg_before_ht_cf = Sing_Trigger(bht, Ht_cut_dqn)  # percent units
        cf_ht = np.zeros(len(HT_DELTAS), dtype=np.float32)

        for a, d in enumerate(HT_DELTAS):
            cut_next = float(np.clip(Ht_cut_dqn + float(d), ht_lo, ht_hi))
            bg_after = Sing_Trigger(bht, cut_next)
            tt_after = Sing_Trigger(sht_tt_j, cut_next)
            aa_after = Sing_Trigger(sht_aa_j, cut_next)

            cf_ht[a] = compute_reward(
                bg_rate=bg_after,
                target=target,
                tol=tol,
                sig_rate_1=tt_after,
                sig_rate_2=aa_after,
                delta_applied=float(d),
                max_delta=MAX_DELTA_HT,
                alpha=alpha,
                beta=beta,
                prev_bg_rate=bg_before_ht_cf,   # stability term is meaningful here
                gamma_stab=0.3,
            )

        cf_r_ht.append(cf_ht)
        dqn_act_ht_chunk.append(-1 if last_act_ht_in_chunk is None else int(last_act_ht_in_chunk))

        # --- AS counterfactuals ---
        bg_before_as_cf = Sing_Trigger(bas, AS_cut_dqn)
        cf_as = np.zeros(len(AS_DELTAS), dtype=np.float32)

        for a, dm in enumerate(AS_DELTAS):
            d = float(dm) * AS_STEP
            cut_next = float(np.clip(AS_cut_dqn + d, as_lo, as_hi))
            bg_after = Sing_Trigger(bas, cut_next)
            tt_after = Sing_Trigger(sas_tt_j, cut_next)
            aa_after = Sing_Trigger(sas_aa_j, cut_next)

            cf_as[a] = compute_reward(
                bg_rate=bg_after,
                target=target,
                tol=tol,
                sig_rate_1=tt_after,
                sig_rate_2=aa_after,
                delta_applied=d,
                max_delta=MAX_DELTA_AS,
                alpha=alpha,
                beta=beta,
                prev_bg_rate=bg_before_as_cf,
                gamma_stab=0.3,
            )

        cf_r_as.append(cf_as)
        dqn_act_as_chunk.append(-1 if last_act_as_in_chunk is None else int(last_act_as_in_chunk))

        
        # ===========================
        # Per-chunk feature diagnostics
        # ===========================
        # (i) near-cut occupancy (use chunk background arrays, final DQN cut)
        occ_ht = near_occupancy(bht, Ht_cut_dqn, near_widths_ht)  # (3,)
        occ_as = near_occupancy(bas, AS_cut_dqn, near_widths_as)  # (3,)
        near_occ_ht.append(occ_ht)
        near_occ_as.append(occ_as)

        # pick mid width index: (5,10,20)->1 and (0.01,0.02,0.05)->1
        occ_mid_ht.append(float(occ_ht[1]))
        occ_mid_as.append(float(occ_as[1]))
        
        EPS_NORM = 1e-3  # in normalized cut units (tune 1e-4~1e-2)

        # (ii) sensitivity proxy based on chunk-to-chunk changes (DQN only)
        # Use *percent* units to make HT/AS comparable (optional but recommended)
        if len(R3_ht) >= 2:
            dr_ht = float(R3_ht[-1] - R3_ht[-2])          # background percent change
            dtheta_ht = float(Ht_dqn_hist[-1] - Ht_dqn_hist[-2]) / ht_span
            sens_ht.append(abs(dr_ht) / (abs(dtheta_ht) + EPS_NORM))
        else:
            sens_ht.append(np.nan)

        if len(R3_as) >= 2:
            dr_as = float(R3_as[-1] - R3_as[-2])          # background percent change
            dtheta_as = float(As_dqn_hist[-1] - As_dqn_hist[-2]) / as_span
            sens_as.append(abs(dr_as)/ (abs(dtheta_as) + EPS_NORM))
        else:
            sens_as.append(np.nan)  

        # DEBUG print per 5 chunks
        if t % 5 == 0:
            lh = losses_ht[-1] if losses_ht else None
            lhf = losses_ht_f[-1] if losses_ht_f else None
            la = losses_as[-1] if losses_as else None
            laf = losses_as_f[-1] if losses_as_f else None
            mode = "train" if (t < int(args.dqn_f_train_chunks)) else "frozen"
            msg = (f"[batch {t:4d}] "
                f"HT bg% c={bg_const_ht:.3f} pd={bg_pd_ht:.3f} dqn={bg_dqn_ht:.3f} "
                f"| ht_cut pd={Ht_cut_pd:.1f} dqn={Ht_cut_dqn:.1f} loss={lh} "
                f"|| AS bg% c={bg_const_as:.3f} pd={bg_pd_as:.3f} dqn={bg_dqn_as:.3f} "
                f"| as_cut pd={AS_cut_pd:.4f} dqn={AS_cut_dqn:.4f} loss={la} "
                f"| reward_ht={reward_ht_t} reward_as={reward_as_t}")
            msg += (
                    f" || DQN-F[{mode}] "
                    f"HT bg%={bg_dqnf_ht:.3f} ht_cut={Ht_cut_dqnf:.1f} loss={lhf} "
                    f"|| AS bg%={bg_dqnf_as:.3f} as_cut={AS_cut_dqnf:.4f} loss={laf} "
                    f"(eps_f={'same' if (t < int(args.dqn_f_train_chunks)) else args.dqn_f_eps})"
                )
            print(msg)


    # ------------------------- convert + scale -------------------------
    RATE_SCALE_KHZ = 400.0
    upper_tol_khz = 0.275 * RATE_SCALE_KHZ
    lower_tol_khz = 0.225 * RATE_SCALE_KHZ

    # HT
    R1_ht = np.array(R1_ht) * RATE_SCALE_KHZ
    R2_ht = np.array(R2_ht) * RATE_SCALE_KHZ
    R3_ht = np.array(R3_ht) * RATE_SCALE_KHZ
    R4_ht = np.array(R4_ht) * RATE_SCALE_KHZ
    Ht_pd_hist = np.array(Ht_pd_hist)
    Ht_dqn_hist = np.array(Ht_dqn_hist)
    Ht_dqnf_hist = np.array(Ht_dqnf_hist)
    L_tt_ht_const = np.array(L_tt_ht_const)
    L_tt_ht_pd    = np.array(L_tt_ht_pd)
    L_tt_ht_dqn   = np.array(L_tt_ht_dqn)
    L_tt_ht_dqnf = np.array(L_tt_ht_dqnf)
    L_aa_ht_const = np.array(L_aa_ht_const)
    L_aa_ht_pd    = np.array(L_aa_ht_pd)
    L_aa_ht_dqn   = np.array(L_aa_ht_dqn)
    L_aa_ht_dqnf = np.array(L_aa_ht_dqnf)


    # AS
    R1_as = np.array(R1_as) * RATE_SCALE_KHZ
    R2_as = np.array(R2_as) * RATE_SCALE_KHZ
    R3_as = np.array(R3_as) * RATE_SCALE_KHZ
    R4_as = np.array(R4_as) * RATE_SCALE_KHZ
    As_pd_hist = np.array(As_pd_hist)
    As_dqn_hist = np.array(As_dqn_hist)
    As_dqnf_hist = np.array(As_dqnf_hist)
    L_tt_as_const = np.array(L_tt_as_const)
    L_tt_as_pd    = np.array(L_tt_as_pd)
    L_tt_as_dqn   = np.array(L_tt_as_dqn)
    L_tt_as_dqnf = np.array(L_tt_as_dqnf)
    L_aa_as_const = np.array(L_aa_as_const)
    L_aa_as_pd    = np.array(L_aa_as_pd)
    L_aa_as_dqn   = np.array(L_aa_as_dqn)
    L_aa_as_dqnf = np.array(L_aa_as_dqnf)

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
    # DQN-F: Explicitly make it identical to DQN, only color differs
    DQNF_STYLE = dict(DQN_STYLE)  

    DQNF_SIG_STYLE = dict(DQN_STYLE)
    DQNF_SIG_STYLE.update(
        linestyle=":",
        marker="D",              # filled diamond
        markersize=5,
        markeredgewidth=1.2,
    )       

    # ------------------------- common styles -------------------------
    styles = {
        "Constant": CONST_STYLE,
        "PD":       PD_STYLE,
        "DQN":      DQN_STYLE,
        "DQN-F":   DQNF_STYLE
    }

    # --- consistent paper fonts ---
    AX_LABEL_FS = 22
    TICK_FS     = 18
    LEGEND_FS   = 14
    LEGEND_TITLE_FS = 16

    def apply_axes_style(ax, xlabel, ylabel, ylim=None):
        ax.set_xlabel(xlabel, loc="center", fontsize=AX_LABEL_FS)
        ax.set_ylabel(ylabel, loc="center", fontsize=AX_LABEL_FS)
        ax.tick_params(axis="both", which="major", labelsize=TICK_FS)
        if ylim is not None:
            ax.set_ylim(*ylim)
        

    
    # ------------------------- diagnostic plot styling -------------------------

    # DIAG_AX_LABEL_FS = AX_LABEL_FS
    # DIAG_TICK_FS     = TICK_FS
    # DIAG_LEGEND_FS   = LEGEND_FS
    # DIAG_LEGEND_TITLE_FS = LEGEND_TITLE_FS

    # def style_diag_axes(ax, xlabel, ylabel, ylim=None):
    #     ax.set_xlabel(xlabel, fontsize=DIAG_AX_LABEL_FS)
    #     ax.set_ylabel(ylabel, fontsize=DIAG_AX_LABEL_FS)
    #     ax.tick_params(axis="both", which="major", labelsize=DIAG_TICK_FS)
    #     if ylim is not None:
    #         ax.set_ylim(*ylim)
    #     ax.grid(True, linestyle="--", alpha=0.5)

    # def style_diag_legend(ax, title=None, loc="best"):
    #     leg = ax.legend(loc=loc, frameon=True, fontsize=DIAG_LEGEND_FS, title=title)
    #     if title is not None and leg is not None:
    #         leg.get_title().set_fontsize(DIAG_LEGEND_TITLE_FS)
    #     return leg

    # def finalize_diag_fig(fig, top=0.86):
    #     # Reserve space for CMS header so it doesn’t collide with ticks/title
    #     fig.tight_layout()
    #     fig.subplots_adjust(top=top)

    time = np.linspace(0, 1, len(R1_ht))

    # ------------------------- reward plots -------------------------
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


    # =========================================================
    # Extra paper plots + summary tables (PD vs DQN baseline)
    # =========================================================
    plots_dir = outdir / "extra_plots"
    tables_dir = outdir / "tables"
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    cf_r_ht = np.asarray(cf_r_ht, dtype=np.float32)  # (T, Aht)
    cf_r_as = np.asarray(cf_r_as, dtype=np.float32)  # (T, Aas)
    dqn_act_ht_chunk = np.asarray(dqn_act_ht_chunk, dtype=np.int32)
    dqn_act_as_chunk = np.asarray(dqn_act_as_chunk, dtype=np.int32)

    def plot_cf_landscape(cf_r, deltas, act_idx, outpath, title, xlabel):
        # show a few representative times: early/mid/late
        T = cf_r.shape[0]
        picks = [0, T//2, T-1] if T >= 3 else list(range(T))
        fig, ax = plt.subplots(figsize=(10, 6))
        for t0 in picks:
            ax.plot(deltas, cf_r[t0], linewidth=2.0, label=f"chunk {t0}")
            a = act_idx[t0]
            if 0 <= a < len(deltas):
                ax.scatter([deltas[a]], [cf_r[t0, a]], s=60)

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Counterfactual reward  r(Δθ)")
        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="best", frameon=True)
        add_cms_header(fig, run_label=run_label)
        save_png(fig, str(outpath))
        plt.close(fig)

    def plot_regret_gap(cf_r, act_idx, outpath, title):
        # regret gap = best achievable CF reward - reward of chosen action
        best = np.max(cf_r, axis=1)
        chosen = np.full(cf_r.shape[0], np.nan, dtype=np.float32)
        for t in range(cf_r.shape[0]):
            a = act_idx[t]
            if 0 <= a < cf_r.shape[1]:
                chosen[t] = cf_r[t, a]
        gap = best - chosen

        tt = np.linspace(0, 1, len(gap))
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(tt, gap, linewidth=2.0)
        ax.set_xlabel("Time (Fraction of Run)")
        ax.set_ylabel("Regret gap")
        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.5)
        add_cms_header(fig, run_label=run_label)
        save_png(fig, str(outpath))
        plt.close(fig)

    # HT plots
    plot_cf_landscape(
        cf_r=cf_r_ht,
        deltas=HT_DELTAS,
        act_idx=dqn_act_ht_chunk,
        outpath=plots_dir / "cf_reward_landscape_ht",
        title="HT: counterfactual reward landscape r(ΔHt_cut) on the same chunk",
        xlabel=r"$\Delta Ht\_cut$ [GeV]",
    )
    plot_regret_gap(
        cf_r=cf_r_ht,
        act_idx=dqn_act_ht_chunk,
        outpath=plots_dir / "regret_gap_ht",
        title="HT: regret gap = max_a r(a) - r(a_DQN) (per chunk)",
    )

    # AS plots
    plot_cf_landscape(
        cf_r=cf_r_as,
        deltas=AS_DELTAS * AS_STEP,  # actual delta in cut units
        act_idx=dqn_act_as_chunk,
        outpath=plots_dir / "cf_reward_landscape_as",
        title="AS: counterfactual reward landscape r(ΔAS_cut) on the same chunk",
        xlabel=r"$\Delta AS\_cut$",
    )
    plot_regret_gap(
        cf_r=cf_r_as,
        act_idx=dqn_act_as_chunk,
        outpath=plots_dir / "regret_gap_as",
        title="AS: regret gap = max_a r(a) - r(a_DQN) (per chunk)",
    )


    plot_rate_with_tolerance_4(
        time, R1_ht, R2_ht, R3_ht, R4_ht,
        outbase=outdir / "bht_rate_pidData_dqn_dqnf",
        run_label=run_label,
        legend_title="HT Trigger",
        ylim=(0, 200),
        tol_upper=upper_tol_khz,
        tol_lower=lower_tol_khz,
        const_style=dict(color="tab:blue", **CONST_STYLE),
        pd_style=dict(color="mediumblue", **PD_STYLE),
        dqn_style=dict(color="tab:purple", **DQN_STYLE),
        dqnf_style=dict(color="tab:red", **DQNF_STYLE),
        dqnf_label=f"DQN-F (train {args.dqn_f_train_chunks} chunks)" if args.dqn_f_train_chunks > 1 else f"DQN-F (train {args.dqn_f_train_chunks} chunk)",
        add_cms_header=add_cms_header,
        save_pdf_png=save_png,
    )


    # -------------------------
    # Chunk-level diagnostics plots
    # -------------------------
    near_occ_ht = np.asarray(near_occ_ht, dtype=np.float32)  # (Tchunk, 3)
    near_occ_as = np.asarray(near_occ_as, dtype=np.float32)
    sens_ht = np.asarray(sens_ht, dtype=np.float32)
    sens_as = np.asarray(sens_as, dtype=np.float32)
    occ_mid_ht = np.asarray(occ_mid_ht, dtype=np.float32)
    occ_mid_as = np.asarray(occ_mid_as, dtype=np.float32)

    # Plot 1a: HT near-cut occupancy vs time (per chunk)
    fig, ax = plt.subplots(figsize=(10, 8))
    for k, w in enumerate(near_widths_ht):
        ax.plot(time, near_occ_ht[:, k], linewidth=2.0, label=fr"$|HT-\theta|\leq {w:g}$ GeV")
    style_diag_axes(
        ax,
        xlabel="Time (Fraction of Run)",
        ylabel="Near-cut occupancy (fraction)",
    )
    style_diag_legend(ax, title="Near-cut window")

    finalize_diag_fig(fig)
    add_cms_header(fig, run_label=run_label)
    save_png(fig, str(plots_dir / "near_cut_occupancy_ht_chunk"))
    plt.close(fig)

    # Plot 2a: HT sensitivity vs occupancy scatter (per chunk)
    m = np.isfinite(sens_ht)
    # fig, ax = plt.subplots(figsize=(6.5, 5.0))
    fig, ax = plt.subplots(figsize=(10, 8)) 
    ax.scatter(occ_mid_ht[m], sens_ht[m], s=18, alpha=0.50, label = "Per chunk")
    style_diag_axes(
        ax,
        xlabel=r"Near-cut occupancy ($w=10$ GeV)",
        ylabel=r"$|\,\Delta r\,| / (|\,\Delta \theta\,|+\epsilon)$  [pct/GeV]",
    )
    style_diag_legend(ax, title="Sensitivity scatter", loc="upper right")
    finalize_diag_fig(fig)
    add_cms_header(fig, run_label=run_label)
    save_png(fig, str(plots_dir / "sensitivity_vs_occupancy_ht_chunk"))
    plt.close(fig)

    # HT reward vs time
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time, rewards_ht, linewidth=1.2, alpha=0.35, label="HT reward (per chunk)")
    ax.plot(time, moving_avg_nan(rewards_ht, w=5), linewidth=2.2, label="HT reward (moving avg)")
    ax.set_xlabel("Time (Fraction of Run)", loc="center")
    ax.set_ylabel("Reward", loc="center")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best", frameon=True)
    add_cms_header(fig, run_label=run_label)
    save_png(fig, str(outdir / "reward_ht_pidData_dqn_feature"))
    plt.close(fig)
    
    time_as = np.linspace(0, 1, len(R1_as))
    # AS reward vs time
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time_as, rewards_as, linewidth=1.2, alpha=0.35, label="AS reward (per chunk)")
    ax.plot(time_as, moving_avg_nan(rewards_as, w=5), linewidth=2.2, label="AS reward (moving avg)")
    ax.set_xlabel("Time (Fraction of Run)", loc="center")
    ax.set_ylabel("Reward", loc="center")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best", frameon=True)
    add_cms_header(fig, run_label=run_label)
    save_png(fig, str(outdir / "reward_as_pidData_dqn_feature"))
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

    
    # =========================================================
    # HT plots
    # =========================================================
    # (2) HT cut evolution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time, Ht_pd_hist,  color="mediumblue", linewidth=2.0, label="PD Controller")
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
    tt_c_dqnf = cummean(L_tt_ht_dqnf)
    aa_c_const = cummean(L_aa_ht_const)
    aa_c_pd    = cummean(L_aa_ht_pd)
    aa_c_dqn   = cummean(L_aa_ht_dqn)
    aa_c_dqnf = cummean(L_aa_ht_dqnf)

    colors_ht = {"ttbar": "goldenrod", "HToAATo4B": "seagreen"}
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time, rel_to_t0(tt_c_const), color=colors_ht["ttbar"], **styles["Constant"],
            label=fr"Constant Menu, ttbar ($\epsilon[t_0]={tt_c_const[0]:.2f}\%$)")
    ax.plot(time, rel_to_t0(aa_c_const), color=colors_ht["HToAATo4B"], **styles["Constant"],
            label=fr"Constant Menu, HToAATo4B ($\epsilon[t_0]={aa_c_const[0]:.2f}\%$)")
    ax.plot(time, rel_to_t0(tt_c_pd), color=colors_ht["ttbar"], **styles["PD"],
            label=fr"PD Controller, ttbar ($\epsilon[t_0]={tt_c_pd[0]:.2f}\%$)")
    ax.plot(time, rel_to_t0(aa_c_pd), color=colors_ht["HToAATo4B"], **styles["PD"],
            label=fr"PD Controller, HToAATo4B ($\epsilon[t_0]={aa_c_pd[0]:.2f}\%$)")
    ax.plot(time, rel_to_t0(tt_c_dqn), color=colors_ht["ttbar"],
            label=fr"DQN, ttbar ($\epsilon[t_0]={tt_c_dqn[0]:.2f}\%$)", **DQN_STYLE)
    ax.plot(time, rel_to_t0(aa_c_dqn), color=colors_ht["HToAATo4B"],
            label=fr"DQN, HToAATo4B ($\epsilon[t_0]={aa_c_dqn[0]:.2f}\%$)", **DQN_STYLE)
    ax.plot(time, rel_to_t0(tt_c_dqnf), color=colors_ht["ttbar"],
        label=fr"DQN-F, ttbar ($\epsilon[t_0]={tt_c_dqnf[0]:.2f}\%$)", **DQNF_SIG_STYLE)
    ax.plot(time, rel_to_t0(aa_c_dqnf), color=colors_ht["HToAATo4B"],
        label=fr"DQN-F, HToAATo4B ($\epsilon[t_0]={aa_c_dqnf[0]:.2f}\%$)", **DQNF_SIG_STYLE)


    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_ylim(0.5, 2.5)
    # ax.legend(title="HT Trigger", fontsize=14, frameon=True, loc="best")
    style_diag_axes(ax, "Time (Fraction of Run)", "Relative Cumulative Efficiency", ylim=(0.5, 2.5))
    style_diag_legend(ax, title="HT Trigger")
    finalize_diag_fig(fig)
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
            label=fr"PD Controller, ttbar ($\epsilon[t_0]={L_tt_ht_pd[0]:.2f}\%$)")
    ax.plot(time, rel_to_t0(L_aa_ht_pd), color=colors_ht["HToAATo4B"], **styles["PD"],
            label=fr"PD Controller, HToAATo4B ($\epsilon[t_0]={L_aa_ht_pd[0]:.2f}\%$)")
    ax.plot(time, rel_to_t0(L_tt_ht_dqn), color=colors_ht["ttbar"], 
            label=fr"DQN, ttbar ($\epsilon[t_0]={L_tt_ht_dqn[0]:.2f}\%$)", **DQN_STYLE)
    ax.plot(time, rel_to_t0(L_aa_ht_dqn), color=colors_ht["HToAATo4B"], 
            label=fr"DQN, HToAATo4B ($\epsilon[t_0]={L_aa_ht_dqn[0]:.2f}\%$)", **DQN_STYLE)
    ax.plot(time, rel_to_t0(L_tt_ht_dqnf), color=colors_ht["ttbar"],
        label=fr"DQN-F, ttbar ($\epsilon[t_0]={L_tt_ht_dqnf[0]:.2f}\%$)", **DQNF_SIG_STYLE)
    ax.plot(time, rel_to_t0(L_aa_ht_dqnf), color=colors_ht["HToAATo4B"],
        label=fr"DQN-F, HToAATo4B ($\epsilon[t_0]={L_aa_ht_dqnf[0]:.2f}\%$)", **DQNF_SIG_STYLE)


    ax.grid(True, linestyle="--", alpha=0.6)

    # leg = ax.legend(title="HT Trigger", fontsize=LEGEND_FS, frameon=True, loc="best")
    style_diag_axes(ax, "Time (Fraction of Run)", "Relative Efficiency", ylim=(0.5, 2.5))
    style_diag_legend(ax, title="HT Trigger")
    finalize_diag_fig(fig)
    # leg.get_title().set_fontsize(LEGEND_TITLE_FS)

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
    # Plot 1b: AS near-cut occupancy vs time (per chunk)
    fig, ax = plt.subplots(figsize=(10, 8))
    for k, w in enumerate(near_widths_as):
        ax.plot(time_as, near_occ_as[:, k], linewidth=2.0, label=fr"$|AS-\theta|\leq {w:g}$")
    style_diag_axes(
        ax,
        xlabel="Time (Fraction of Run)",
        ylabel="Near-cut occupancy (fraction)",
    )
    style_diag_legend(ax, title="Near-cut window")

    finalize_diag_fig(fig)
    add_cms_header(fig, run_label=run_label)
    save_png(fig, str(plots_dir / "near_cut_occupancy_as_chunk"))
    plt.close(fig)
    
    # Plot 2b: AS sensitivity vs occupancy scatter (per chunk)
    m = np.isfinite(sens_as) & np.isfinite(occ_mid_as)
    # fig, ax = plt.subplots(figsize=(6.5, 5.0))
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(
        occ_mid_as[m],
        sens_as[m],
        s=22,
        alpha=0.50,
        label="Per chunk",
    )

    style_diag_axes(
        ax,
        xlabel=r"Near-cut occupancy ($w=0.5$)",
        ylabel=r"$|\,\Delta r\,| / (|\,\Delta \theta\,|+\epsilon)$  [pct/unit]",
    )
    style_diag_legend(ax, title=None)

    finalize_diag_fig(fig)
    add_cms_header(fig, run_label=run_label)
    save_png(fig, str(plots_dir / "sensitivity_vs_occupancy_as_chunk"))
    plt.close(fig)

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

    plot_rate_with_tolerance_4(
        time_as, R1_as, R2_as, R3_as, R4_as,
        outbase=outdir / "bas_rate_pidData_dqn_dqnf",
        run_label=run_label,
        legend_title="AD Trigger",
        ylim=(0, 200),
        tol_upper=upper_tol_khz,
        tol_lower=lower_tol_khz,
        const_style=dict(color="tab:blue", **CONST_STYLE),
        pd_style=dict(color="mediumblue", **PD_STYLE),
        dqn_style=dict(color="tab:purple", **DQN_STYLE),
        dqnf_style=dict(color="tab:red", **DQNF_STYLE),
        dqnf_label=f"DQN-F (train {args.dqn_f_train_chunks} chunks)",
        add_cms_header=add_cms_header,
        save_pdf_png=save_png,
    )

    # (A2) AS cut evolution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time_as, As_pd_hist,  color="mediumblue", linewidth=2.0, label="PD Controller")
    ax.plot(time_as, As_dqn_hist, color="tab:purple", linewidth=2.0, label="DQN")
    ax.axhline(y=fixed_AS_cut, color="gray", linestyle="--", linewidth=1.5, label="fixed_AS_cut")
    ax.set_xlabel("Time (Fraction of Run)", loc="center")
    ax.set_ylabel("Anomaly Score Cut", loc="center")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(title="AD Cut", fontsize=14, frameon=True, loc="best")
    add_cms_header(fig, run_label=run_label)
    save_png(fig, str(outdir / "as_cut_pidData_dqn"))
    plt.close(fig)

    # (A3) AD cumulative eff (relative to t0)
    tt_c_const = cummean(L_tt_as_const)
    tt_c_pd    = cummean(L_tt_as_pd)
    tt_c_dqn   = cummean(L_tt_as_dqn)
    tt_c_dqnf = cummean(L_tt_as_dqnf)
    aa_c_const = cummean(L_aa_as_const)
    aa_c_pd    = cummean(L_aa_as_pd)
    aa_c_dqn   = cummean(L_aa_as_dqn)
    aa_c_dqnf = cummean(L_aa_as_dqnf)

    colors_ad = {"ttbar": "goldenrod", "HToAATo4B": "limegreen"}

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time_as, rel_to_t0(tt_c_const), color=colors_ad["ttbar"], **styles["Constant"],
            label=fr"Constant Menu, ttbar ($\epsilon[t_0]={tt_c_const[0]:.2f}\%$)")
    ax.plot(time_as, rel_to_t0(aa_c_const), color=colors_ad["HToAATo4B"], **styles["Constant"],
            label=fr"Constant Menu, HToAATo4B ($\epsilon[t_0]={aa_c_const[0]:.2f}\%$)")
    ax.plot(time_as, rel_to_t0(tt_c_pd), color=colors_ad["ttbar"], **styles["PD"],
            label=fr"PD Controller, ttbar ($\epsilon[t_0]={tt_c_pd[0]:.2f}\%$)")
    ax.plot(time_as, rel_to_t0(aa_c_pd), color=colors_ad["HToAATo4B"], **styles["PD"],
            label=fr"PD Controller, HToAATo4B ($\epsilon[t_0]={aa_c_pd[0]:.2f}\%$)")
    ax.plot(time_as, rel_to_t0(tt_c_dqn), color=colors_ad["ttbar"],
            label=fr"DQN, ttbar ($\epsilon[t_0]={tt_c_dqn[0]:.2f}\%$)", **DQN_STYLE)
    ax.plot(time_as, rel_to_t0(aa_c_dqn), color=colors_ad["HToAATo4B"], 
            label=fr"DQN, HToAATo4B ($\epsilon[t_0]={aa_c_dqn[0]:.2f}\%$)", **DQN_STYLE)
    ax.plot(time_as, rel_to_t0(tt_c_dqnf), color=colors_ad["ttbar"],
        label=fr"DQN-F, ttbar ($\epsilon[t_0]={tt_c_dqnf[0]:.2f}\%$)", **DQNF_STYLE)
    ax.plot(time_as, rel_to_t0(aa_c_dqnf), color=colors_ad["HToAATo4B"],
        label=fr"DQN-F, HToAATo4B ($\epsilon[t_0]={aa_c_dqnf[0]:.2f}\%$)", **DQNF_STYLE)

    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_ylim(0.5, 2.5)
    # leg=ax.legend(title="AD Trigger", fontsize=14, frameon=True, loc="best")
    style_diag_axes(ax, "Time (Fraction of Run)", "Relative Cumulative Efficiency", ylim=(0.5, 2.5))
    style_diag_legend(ax, title="AD Trigger")
    finalize_diag_fig(fig)
    # leg.get_title().set_fontsize(LEGEND_TITLE_FS)
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
            label=fr"PD Controller, ttbar ($\epsilon[t_0]={L_tt_as_pd[0]:.2f}\%$)")
    ax.plot(time_as, rel_to_t0(L_aa_as_pd), color=colors_ad["HToAATo4B"], **styles["PD"],
            label=fr"PD Controller, HToAATo4B ($\epsilon[t_0]={L_aa_as_pd[0]:.2f}\%$)")
    ax.plot(time_as, rel_to_t0(L_tt_as_dqn), color=colors_ad["ttbar"], linewidth=2.2, linestyle="dashdot",
            label=fr"DQN, ttbar ($\epsilon[t_0]={L_tt_as_dqn[0]:.2f}\%$)")
    ax.plot(time_as, rel_to_t0(L_aa_as_dqn), color=colors_ad["HToAATo4B"], linewidth=2.2, linestyle="dashdot",
            label=fr"DQN, HToAATo4B ($\epsilon[t_0]={L_aa_as_dqn[0]:.2f}\%$)")
    ax.plot(time_as, rel_to_t0(L_tt_as_dqnf), color=colors_ad["ttbar"],
        label=fr"DQN-F, ttbar ($\epsilon[t_0]={L_tt_as_dqnf[0]:.2f}\%$)", **DQNF_SIG_STYLE)
    ax.plot(time_as, rel_to_t0(L_aa_as_dqnf), color=colors_ad["HToAATo4B"],
        label=fr"DQN-F, HToAATo4B ($\epsilon[t_0]={L_aa_as_dqnf[0]:.2f}\%$)", **DQNF_SIG_STYLE)


    ax.grid(True, linestyle="--", alpha=0.6)
    # ax.legend(title="AD Trigger", fontsize=14, frameon=True, loc="best")
    style_diag_axes(ax, "Time (Fraction of Run)", "Relative Efficiency", ylim=(0.5, 2.5))
    style_diag_legend(ax, title="AD Trigger")
    finalize_diag_fig(fig)
    add_cms_header(fig, run_label=run_label)
    save_png(fig, str(outdir / "L_sas_rate_pidData2data_dqn"))
    plt.close(fig)

    # AS loss
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



    TARGET_PCT = float(target)               # 0.25 (percent)
    TOL_PCT = float(tol)                     # 0.025 (percent)
    TARGET_KHZ = TARGET_PCT * RATE_SCALE_KHZ
    TOL_KHZ = TOL_PCT * RATE_SCALE_KHZ

    # ---------------------------------------------------------
    # Build percent-rate arrays (since R*_ht/as are in kHz now)
    # ---------------------------------------------------------
    r_const_ht_pct = R1_ht / RATE_SCALE_KHZ
    r_pd_ht_pct    = R2_ht / RATE_SCALE_KHZ
    r_dqn_ht_pct   = R3_ht / RATE_SCALE_KHZ
    r_dqnf_ht_pct = R4_ht / RATE_SCALE_KHZ
    

    r_const_as_pct = R1_as / RATE_SCALE_KHZ
    r_pd_as_pct    = R2_as / RATE_SCALE_KHZ
    r_dqn_as_pct   = R3_as / RATE_SCALE_KHZ
    r_dqnf_as_pct = R4_as / RATE_SCALE_KHZ

    # Constant cut arrays (for jitter metrics)
    Ht_const_hist = np.full_like(Ht_pd_hist, fixed_Ht_cut, dtype=np.float64)
    As_const_hist = np.full_like(As_pd_hist, fixed_AS_cut, dtype=np.float64)

    # ---------------- helpers ----------------
    def ecdf(x):
        x = np.asarray(x, dtype=np.float64)
        x = x[np.isfinite(x)]
        if x.size == 0:
            return np.array([]), np.array([])
        x = np.sort(x)
        y = (np.arange(1, x.size + 1) / x.size)
        return x, y
    
    def summarize_metrics(r_pct, s_tt, s_aa, cut, target_pct=0.25, tol_pct=0.02):
        r = np.asarray(r_pct, dtype=np.float64)
        inband = np.abs(r - target_pct) <= tol_pct

        def safe_mean(x, m):
            x = np.asarray(x, dtype=np.float64)
            return float(np.mean(x[m])) if np.any(m) else np.nan

        err = r - target_pct
        out = {}
        out["mae"] = float(np.mean(np.abs(err)))
        out["rmse"] = float(np.sqrt(np.mean(err**2)))
        out["p95_abs_err"] = float(np.percentile(np.abs(err), 95))
        out["inband_frac"] = float(np.mean(inband))
        out["upper_viol_frac"] = float(np.mean(r > (target_pct + tol_pct)))
        out["lower_viol_frac"] = float(np.mean(r < (target_pct - tol_pct)))
        out["viol_mag"] = float(np.mean(np.maximum(0.0, np.abs(err) - tol_pct)))

        c = np.asarray(cut, dtype=np.float64)
        dc = np.diff(c) if c.size >= 2 else np.array([], dtype=np.float64)
        out["cut_TV"] = float(np.sum(np.abs(dc))) if dc.size else 0.0
        out["cut_step_rms"] = float(np.sqrt(np.mean(dc**2))) if dc.size else 0.0
        out["cut_step_max"] = float(np.max(np.abs(dc))) if dc.size else 0.0

        out["tt_inband"] = safe_mean(s_tt, inband)
        out["aa_inband"] = safe_mean(s_aa, inband)
        out["score_50_50"] = safe_mean(0.5*np.asarray(s_tt) + 0.5*np.asarray(s_aa), inband)
        out["score_80_AA"] = safe_mean(0.2*np.asarray(s_tt) + 0.8*np.asarray(s_aa), inband)
        return out


    def _save(fig, outbase: Path):
        add_cms_header(fig, run_label=run_label)
        save_png(fig, str(outbase))
        plt.close(fig)

    # ---------------- build rows ----------------
    sum_const_ht = summarize_metrics(r_const_ht_pct, L_tt_ht_const, L_aa_ht_const, Ht_const_hist, TARGET_PCT, TOL_PCT)
    sum_pd_ht    = summarize_metrics(r_pd_ht_pct,    L_tt_ht_pd,    L_aa_ht_pd,    Ht_pd_hist,   TARGET_PCT, TOL_PCT)
    sum_dqn_ht   = summarize_metrics(r_dqn_ht_pct,   L_tt_ht_dqn,   L_aa_ht_dqn,   Ht_dqn_hist,  TARGET_PCT, TOL_PCT)
    sum_dqnf_ht = summarize_metrics(r_dqnf_ht_pct, L_tt_ht_dqnf, L_aa_ht_dqnf, Ht_dqnf_hist, TARGET_PCT, TOL_PCT)


    sum_const_as = summarize_metrics(r_const_as_pct, L_tt_as_const, L_aa_as_const, As_const_hist, TARGET_PCT, TOL_PCT)
    sum_pd_as    = summarize_metrics(r_pd_as_pct,    L_tt_as_pd,    L_aa_as_pd,    As_pd_hist,   TARGET_PCT, TOL_PCT)
    sum_dqn_as   = summarize_metrics(r_dqn_as_pct,   L_tt_as_dqn,   L_aa_as_dqn,   As_dqn_hist,  TARGET_PCT, TOL_PCT)
    sum_dqnf_as = summarize_metrics(r_dqnf_as_pct, L_tt_as_dqnf, L_aa_as_dqnf, As_dqnf_hist, TARGET_PCT, TOL_PCT)


    # Save tables here:
    #   outdir/tables/pd_vs_dqn_summary.csv
    #   outdir/tables/pd_vs_dqn_summary.tex
    rows_paper = []

    def add_paper_row(trigger, method, r_pct, tt_eff, aa_eff, cut_hist):
        s = summarize_paper_table(r_pct, tt_eff, aa_eff, cut_hist, target, tol)
        rows_paper.append({"Trigger": trigger, "Method": method, **s})

    # HT
    add_paper_row("HT", "Constant", r_const_ht_pct, L_tt_ht_const, L_aa_ht_const, Ht_const_hist)
    add_paper_row("HT", "PD",       r_pd_ht_pct,    L_tt_ht_pd,    L_aa_ht_pd,    Ht_pd_hist)
    add_paper_row("HT", "DQN",      r_dqn_ht_pct,   L_tt_ht_dqn,   L_aa_ht_dqn,   Ht_dqn_hist)
    add_paper_row("HT", "DQN-F",    r_dqnf_ht_pct,  L_tt_ht_dqnf,  L_aa_ht_dqnf,  Ht_dqnf_hist)

    # AD/AS
    add_paper_row("AD", "Constant", r_const_as_pct, L_tt_as_const, L_aa_as_const, As_const_hist)
    add_paper_row("AD", "PD",       r_pd_as_pct,    L_tt_as_pd,    L_aa_as_pd,    As_pd_hist)
    add_paper_row("AD", "DQN",      r_dqn_as_pct,   L_tt_as_dqn,   L_aa_as_dqn,   As_dqn_hist)
    add_paper_row("AD", "DQN-F",    r_dqnf_as_pct,  L_tt_as_dqnf,  L_aa_as_dqnf,  As_dqnf_hist)

    write_paper_table(
        rows_paper,
        out_csv=outdir / "tables" / "summary_paper.csv",
        out_tex=outdir / "tables" / "summary_paper.tex",
        target_pct=target,
        tol_pct=tol,
    )
    print(f"[OK] wrote {outdir/'tables'/'summary_paper.csv'}")
    print(f"[OK] wrote {outdir/'tables'/'summary_paper.tex'}")

    # ---------------------------------------------------------
    # Extra Plot 1: CDF of |rate error| (kHz)  (HT + AS)
    # ---------------------------------------------------------
    def plot_cdf_abs_err(r_khz_pd, r_khz_dqn, outpath: Path, title: str):
        e_pd  = np.abs(np.asarray(r_khz_pd)  - TARGET_KHZ)
        e_dqn = np.abs(np.asarray(r_khz_dqn) - TARGET_KHZ)
        x1, y1 = ecdf(e_pd)
        x2, y2 = ecdf(e_dqn)

        fig, ax = plt.subplots(figsize=(7.5, 5.0))
        ax.plot(x1, y1, linewidth=2.2, label="PD")
        ax.plot(x2, y2, linewidth=2.2, label="DQN")
        ax.axvline(TOL_KHZ, linestyle="--", linewidth=1.6, label=f"Tolerance = {TOL_KHZ:.1f} kHz")
        ax.set_xlabel(r"$|r - r^*|$  [kHz]")
        ax.set_ylabel("CDF")
        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="best", frameon=True)
        _save(fig, outpath)

    plot_cdf_abs_err(R2_ht, R3_ht, plots_dir / "cdf_abs_err_ht", "HT: CDF of absolute background-rate error")
    plot_cdf_abs_err(R2_as, R3_as, plots_dir / "cdf_abs_err_as", "AS: CDF of absolute background-rate error")

    # ---------------------------------------------------------
    # Extra Plot 2: In-band efficiency bars (PD vs DQN)  (HT + AS)
    # ---------------------------------------------------------
    def plot_inband_bars(sum_pd, sum_dqn, outpath: Path, title: str):
        labels = ["ttbar", "HToAATo4B"]
        pd_vals = [sum_pd["tt_inband"], sum_pd["aa_inband"]]
        dqn_vals = [sum_dqn["tt_inband"], sum_dqn["aa_inband"]]

        x = np.arange(len(labels))
        w = 0.35

        fig, ax = plt.subplots(figsize=(7.5, 5.0))
        ax.bar(x - w/2, pd_vals, width=w, label="PD")
        ax.bar(x + w/2, dqn_vals, width=w, label="DQN")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Signal efficiency (mean, in-band)")
        ax.set_title(title)
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)
        ax.legend(loc="best", frameon=True)
        _save(fig, outpath)

    plot_inband_bars(sum_pd_ht, sum_dqn_ht, plots_dir / "inband_eff_bars_ht", "HT: in-band mean efficiency (PD vs DQN)")
    plot_inband_bars(sum_pd_as, sum_dqn_as, plots_dir / "inband_eff_bars_as", "AS: in-band mean efficiency (PD vs DQN)")

    # ---------------------------------------------------------
    # Extra Plot 3: Cut-step magnitude histogram (jitter) (PD vs DQN)
    # ---------------------------------------------------------
    def plot_cut_step_hist(cut_pd, cut_dqn, outbase: Path, title: str, xlabel: str):
        dp = np.diff(np.asarray(cut_pd, dtype=np.float64))
        dd = np.diff(np.asarray(cut_dqn, dtype=np.float64))
        fig, ax = plt.subplots(figsize=(7.5, 5.0))
        ax.hist(np.abs(dp), bins=30, alpha=0.55, label="PD")
        ax.hist(np.abs(dd), bins=30, alpha=0.55, label="DQN")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="best", frameon=True)
        _save(fig, outbase)

    plot_cut_step_hist(Ht_pd_hist, Ht_dqn_hist, plots_dir / "cut_step_hist_ht",
                       "HT: |Δ cut| distribution (jitter)", xlabel=r"$|\Delta Ht\_cut|$ [GeV]")
    plot_cut_step_hist(As_pd_hist, As_dqn_hist, plots_dir / "cut_step_hist_as",
                       "AS: |Δ cut| distribution (jitter)", xlabel=r"$|\Delta AS\_cut|$")

    # ---------------------------------------------------------
    # Extra Plot 4: Running in-band fraction over time (PD vs DQN)
    # ---------------------------------------------------------
    def running_mean_bool(mask, w=5):
        m = np.asarray(mask, dtype=np.float64)
        k = np.ones(w, dtype=np.float64)
        return np.convolve(m, k, mode="same") / np.convolve(np.ones_like(m), k, mode="same")

    def plot_running_inband(r_khz_pd, r_khz_dqn, outbase: Path, title: str, w=5):
        rpd = np.asarray(r_khz_pd, dtype=np.float64) / RATE_SCALE_KHZ  
        rdq = np.asarray(r_khz_dqn, dtype=np.float64) / RATE_SCALE_KHZ
        in_pd = np.abs(rpd - TARGET_PCT) <= TOL_PCT
        in_dq = np.abs(rdq - TARGET_PCT) <= TOL_PCT

        tgrid = np.linspace(0, 1, len(rpd))
        fig, ax = plt.subplots(figsize=(7.5, 5.0))
        ax.plot(tgrid, running_mean_bool(in_pd, w=w), linewidth=2.2, label=f"PD (w={w})")
        ax.plot(tgrid, running_mean_bool(in_dq, w=w), linewidth=2.2, label=f"DQN (w={w})")
        ax.set_xlabel("Time (Fraction of Run)")
        ax.set_ylabel("Running in-band fraction")
        ax.set_title(title)
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="best", frameon=True)
        _save(fig, outbase)


    plot_running_inband(R2_ht, R3_ht, plots_dir / "running_inband_ht", "HT: running in-band fraction (PD vs DQN)", w = 5)
    plot_running_inband(R2_as, R3_as, plots_dir / "running_inband_as", "AS: running in-band fraction (PD vs DQN)", w = 5)

    print("\nSaved to:", outdir)
    for p in sorted(outdir.glob("*.png")):
        print(" -", p.name)
    

    # Example usage for HT (replace with AS by passing Bas/Tas/Aas and cut hists)
    t_mid, auc_tt_pd, auc_tt_dqn, auc_aa_pd, auc_aa_dqn = compute_auroc_windows_separate(
        start_event=start_event,
        window_events=chunk_size,              # chunk size
        update_chunk_size=chunk_size,     # controller update interval (your big chunk)
        matched_by_index=matched_by_index,
        Bnpv=Bnpv, Tnpv=Tnpv, Anpv=Anpv,
        Bx=Bht, Tx=Tht, Ax=Aht,
        cut_hist_pd=Ht_pd_hist,
        cut_hist_dqn=Ht_dqn_hist,
        max_n=200000,
        seed=SEED,
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t_mid, auc_tt_pd,  label="AUROC TT vs BKG (PD)",  linewidth=2.0)
    ax.plot(t_mid, auc_tt_dqn, label="AUROC TT vs BKG (DQN)", linewidth=2.0)
    ax.plot(t_mid, auc_aa_pd,  label="AUROC AA vs BKG (PD)",  linewidth=2.0)
    ax.plot(t_mid, auc_aa_dqn, label="AUROC AA vs BKG (DQN)", linewidth=2.0)
    ax.set_xlabel("Time (Fraction of Run)")
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best", frameon=True)
    add_cms_header(fig, run_label=run_label)
    save_png(fig, str(outdir / "auroc_tt_aa_vs_time_ht"))
    plt.close(fig)


    # Operating point (accept if x > cut) — the anomaly decision
    t_mid2, fpr_pd, fpr_dqn, tpr_tt_pd, tpr_tt_dqn, tpr_aa_pd, tpr_aa_dqn = compute_operating_point_windows_separate(
        start_event=start_event,
        window_events=50000,
        update_chunk_size=chunk_size,
        matched_by_index=matched_by_index,
        Bnpv=Bnpv, Tnpv=Tnpv, Anpv=Anpv,
        Bx=Bht, Tx=Tht, Ax=Aht,
        cut_hist_pd=Ht_pd_hist,
        cut_hist_dqn=Ht_dqn_hist,
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t_mid2, fpr_pd,  label="BKG accept fraction (PD)",  linewidth=2.0)
    ax.plot(t_mid2, fpr_dqn, label="BKG accept fraction (DQN)", linewidth=2.0)
    ax.plot(t_mid2, tpr_tt_pd,  label="TT accept fraction (PD)",  linewidth=2.0)
    ax.plot(t_mid2, tpr_tt_dqn, label="TT accept fraction (DQN)", linewidth=2.0)
    ax.plot(t_mid2, tpr_aa_pd,  label="AA accept fraction (PD)",  linewidth=2.0)
    ax.plot(t_mid2, tpr_aa_dqn, label="AA accept fraction (DQN)", linewidth=2.0)
    ax.set_xlabel("Time (Fraction of Run)")
    ax.set_ylabel("Accept fraction at margin > 0")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best", frameon=True)
    add_cms_header(fig, run_label=run_label)
    save_png(fig, str(outdir / "operating_point_accept_frac_vs_time_ht"))
    plt.close(fig)

if __name__ == "__main__":
    main()