#!/usr/bin/env python3
"""
demo_single_trigger_grpo_as_feature_all.py

Single-trigger threshold control with event-sequence features with sliding window.

Main focus: AD/AS trigger control (AS_cut) comparing:
  - Constant menu threshold (fixed from calibration window)
  - PD baseline (uses PD_controller2 for AS)
  - DQN baseline (sequence DQN; epsilon-greedy)
  - GRPO (bandit-style group sampling + policy update)
  - GFPO-F : abs_err_topk
  - GFPO-FR: feasible_first_sig

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
- If the tolerance band is [90,110] kHz around 100 kHz, use tol=0.025 (%).
- Controller mapping (IMPORTANT):
    * AS_cut (AD trigger) PD baseline  -> PD_controller2
    * Ht_cut (HT trigger) PD baseline  -> PD_controller1
"""

import argparse
import random
import csv
import numpy as np
from collections import deque, defaultdict
from dataclasses import dataclass
from pathlib import Path
import matplotlib.pyplot as plt
from controllers import PD_controller1, PD_controller2
from triggers import Sing_Trigger
from RL.utils import add_cms_header, save_png, print_h5_tree, read_any_h5, cummean, rel_to_t0, near_occupancy, style_diag_axes, style_diag_legend, finalize_diag_fig, apply_paper_style, plot_inband_eff_single_signal_ad_vs_ht
from RL.grpo_agent import GRPOAgent, GRPOConfig, GRPORewardCfg #GRPO agent
from RL.gfpo_agent import GFPOAgent, GFPOConfig
from RL.dqn_agent import SeqDQNAgent, DQNConfig  # DQN agent
# from RL.dqn_agent import make_event_seq_as_v0, make_event_seq_ht_v0
from RL.dqn_agent import make_event_seq_as, make_event_seq_ht, shield_delta
from RL.ppo_agent import SeqPPOAgent, SeqPPOConfig

SEED = 20251221
random.seed(SEED)
np.random.seed(SEED)

RATE_SCALE_KHZ = 400.0

import mplhep as hep
hep.style.use("CMS")

apply_paper_style()
from cycler import cycler
plt.rcParams["axes.prop_cycle"] = cycler(color=plt.get_cmap("tab10").colors)

# ----------------------------- plot method order -----------------------------
def compute_micro_action_entropy(samples, *, trigger, method, target, tol, kept_only=False):
    """
    Per-micro stats from grpo_samples:
      - normalized entropy of sampled actions (0..1)
      - feasible ratio among considered candidates
      - reward std among considered candidates (proxy for advantage signal)
    """
    from collections import defaultdict
    by_micro = defaultdict(list)

    for r in samples:
        if r.get("trigger") != trigger: 
            continue
        if r.get("method") != method:
            continue
        if r.get("phase") != "candidate":
            continue
        if kept_only and int(r.get("kept", 0)) != 1:
            continue
        # require action + reward
        a = r.get("a", None)
        rw = r.get("reward_raw", None)
        bg = r.get("bg_after", None)
        if a is None or rw is None or bg is None:
            continue
        by_micro[int(r["micro"])].append((int(a), float(rw), float(bg)))

    micros = np.array(sorted(by_micro.keys()), dtype=np.int64)
    if micros.size == 0:
        return None

    ent = np.zeros_like(micros, dtype=np.float64)
    feas = np.zeros_like(micros, dtype=np.float64)
    rstd = np.zeros_like(micros, dtype=np.float64)

    for i, m in enumerate(micros):
        rows = by_micro[int(m)]
        acts = np.array([x[0] for x in rows], dtype=np.int64)
        rews = np.array([x[1] for x in rows], dtype=np.float64)
        bgs  = np.array([x[2] for x in rows], dtype=np.float64)

        # feasible ratio
        feas[i] = float(np.mean(np.abs(bgs - float(target)) <= float(tol))) if bgs.size else np.nan

        # reward std (if ~0, advantages vanish)
        rstd[i] = float(np.std(rews)) if rews.size else np.nan

        # normalized entropy over actions
        # p(a) from empirical counts
        if acts.size:
            K = int(np.max(acts)) + 1  # safe upper bound (only used for logK)
            # count unique
            uniq, cnt = np.unique(acts, return_counts=True)
            p = cnt.astype(np.float64) / float(np.sum(cnt))
            H = -np.sum(p * np.log(p + 1e-12))
            Hmax = np.log(max(2, len(uniq)))  # normalize to support actually sampled support
            ent[i] = float(H / (Hmax + 1e-12))
        else:
            ent[i] = np.nan

    return {"micros": micros, "entropy": ent, "feasible_ratio": feas, "reward_std": rstd}


def plot_entropy_timeseries(stats_by_label, *, title, outpath, run_label):
    fig, ax = plt.subplots(figsize=(9, 5.4))
    for label, st in stats_by_label.items():
        if st is None:
            continue
        ax.plot(st["micros"], st["entropy"], linewidth=2.2, drawstyle="steps-post", label=label)
    ax.set_xlabel("Micro-step")
    ax.set_ylabel("Normalized action entropy (empirical)")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle="--", alpha=0.5)
    small_legend(ax, loc="best", title=title)
    add_cms_header(fig, run_label=run_label)
    finalize_diag_fig(fig)
    save_png(fig, str(outpath))
    plt.close(fig)


def collect_candidate_abs_err_window(samples, *, trigger, method, target, micro_max, kept_only=False):
    ae = []
    for r in samples:
        if r.get("trigger") != trigger:
            continue
        if r.get("method") != method:
            continue
        if r.get("phase") != "candidate":
            continue
        if int(r.get("micro", 10**9)) > int(micro_max):
            continue
        if kept_only and int(r.get("kept", 0)) != 1:
            continue
        bg = r.get("bg_after", None)
        if bg is None:
            continue
        ae.append(abs(float(bg) - float(target)))
    return np.asarray(ae, dtype=np.float64)


def plot_early_abs_err_hist(grpo_ae, gfpo_ae, *, title, outpath, run_label):
    grpo_ae = np.asarray(grpo_ae, dtype=np.float64); grpo_ae = grpo_ae[np.isfinite(grpo_ae)]
    gfpo_ae = np.asarray(gfpo_ae, dtype=np.float64); gfpo_ae = gfpo_ae[np.isfinite(gfpo_ae)]
    if grpo_ae.size == 0 and gfpo_ae.size == 0:
        return

    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    if grpo_ae.size:
        ax.hist(grpo_ae, bins=60, density=True, alpha=0.55, label="GRPO candidates")
    if gfpo_ae.size:
        ax.hist(gfpo_ae, bins=60, density=True, alpha=0.55, label="GFPO kept candidates")

    ax.set_xlabel(r"$|bg-target|$  (percent units)")
    ax.set_ylabel("Density")
    ax.grid(True, linestyle="--", alpha=0.4)
    small_legend(ax, loc="best", title=title)
    add_cms_header(fig, run_label=run_label)
    finalize_diag_fig(fig)
    save_png(fig, str(outpath))
    plt.close(fig)

PLOT_METHODS = ["Constant", "PID", "ADT", "DQN", "DQN-F", "PPO", "GRPO", "GFPO-F", "GFPO-FR"]

def select_plot_methods(d):
    """
    Filter + order a dict keyed by method name.
    Keeps ONLY methods in PLOT_METHODS, in that order.
    """
    if not d:
        return {}
    return {m: d[m] for m in PLOT_METHODS if m in d}

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

# for HT
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

def confusion_counts_at_cut(bg_scores, sig1_scores, sig2_scores, cut):
    """
    Treat "positive" = (score >= cut), i.e., accepted by the trigger.

      FP = bg accepted
      TN = bg rejected
      TP = signal accepted (sig1 + sig2)
      FN = signal rejected (sig1 + sig2)

    Returns counts + common rates.
    """
    s_b  = np.asarray(bg_scores,  dtype=np.float32)
    s_s1 = np.asarray(sig1_scores, dtype=np.float32)
    s_s2 = np.asarray(sig2_scores, dtype=np.float32)

    if (s_s1.size + s_s2.size) > 0:
        s_s = np.concatenate([s_s1, s_s2], axis=0)
    else:
        s_s = np.empty(0, np.float32)

    fp = int(np.sum(s_b >= cut))
    tn = int(np.sum(s_b <  cut))

    tp = int(np.sum(s_s >= cut)) if s_s.size else 0
    fn = int(np.sum(s_s <  cut)) if s_s.size else 0

    nb = int(fp + tn)
    ns = int(tp + fn)

    # rates (safe)
    tpr = float(tp / ns) if ns > 0 else np.nan   # recall / signal acceptance
    fnr = float(fn / ns) if ns > 0 else np.nan
    fpr = float(fp / nb) if nb > 0 else np.nan   # background acceptance
    tnr = float(tn / nb) if nb > 0 else np.nan

    precision = tp/(tp+fp) if (tp+fp)>0 else 0.0
    recall    = tp/(tp+fn) if (tp+fn)>0 else 0.0
    f1        = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0.0

    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "nb": nb, "ns": ns,
        "tpr": tpr, "fnr": fnr, "fpr": fpr, "tnr": tnr,
        "precision": precision, "recall": recall, "f1": f1,
    }


# ----------------------------- metrics helpers -----------------------------
def adt_reward_paper_style(bg_scores, sig1_scores, sig2_scores, cut, alpha=0.7, beta=0.3):
    cm = confusion_counts_at_cut(bg_scores, sig1_scores, sig2_scores, cut)

    nb = max(1, cm["fp"] + cm["tn"])
    ns = max(1, cm["tp"] + cm["fn"])

    tp_n = cm["tp"] / ns
    fn_n = cm["fn"] / ns
    fp_n = cm["fp"] / nb
    tn_n = cm["tn"] / nb
    return float(alpha * (tp_n - fp_n - fn_n) + beta * tn_n)

@dataclass
class CtrlOut:
    micro_global: int

class BaseCtrl:
    name: str
    def step_micro(self, *, chunk: int, micro_global: int, **kwargs) -> CtrlOut:
        return CtrlOut(micro_global=micro_global)

    def end_chunk(self, *, chunk: int, **kwargs):
        """Optional hook called once per chunk (episode)."""
        return

class ConstantCtrl(BaseCtrl):
    def __init__(self, name, fixed_cut):
        self.name = name
        self.cut = float(fixed_cut)
    def cut_value(self): return self.cut

class PIDCtrl(BaseCtrl):
    """
    PID/PD baseline that updates ONCE per chunk (time chunk).
    Holds cut constant within chunk; updates cut at end_chunk for next chunk.
    Uses PD_controller2 for AD/AS.
    """
    def __init__(self, name, init_cut, lo, hi):
        self.name = name
        self.cut = float(init_cut)
        self.err = 0.0
        self.lo, self.hi = float(lo), float(hi)

    def cut_value(self):
        return self.cut

    def step_micro(self, *, micro_global: int, **kwargs) -> CtrlOut:
        # No-op inside chunk (cut held constant)
        return CtrlOut(micro_global=micro_global)

    def end_chunk(self, *, chunk: int, bas_j=None, bas_chunk=None, **kwargs):
        # Accept either bas_j (per-chunk array) or bas_chunk (older name)
        scores = bas_j if bas_j is not None else bas_chunk
        if scores is None:
            return

        bg = float(Sing_Trigger(scores, self.cut))
        cut_next, err_next = PD_controller2(bg, self.err, self.cut)  # AD/AS uses PD_controller2
        self.cut = float(cut_next) #float(np.clip(cut_next, self.lo, self.hi))
        self.err = float(err_next)


class PIDCtrlHT(BaseCtrl):
    """
    Chunk-updated PID/PD baseline for HT.
    Uses PD_controller1 for HT.
    """
    def __init__(self, name, init_cut, lo, hi):
        self.name = name
        self.cut = float(init_cut)
        self.err = 0.0
        self.lo, self.hi = float(lo), float(hi)

    def cut_value(self):
        return self.cut

    def step_micro(self, *, micro_global: int, **kwargs) -> CtrlOut:
        return CtrlOut(micro_global=micro_global)

    def end_chunk(self, *, chunk: int, bht_j=None, bht_chunk=None, **kwargs):
        scores = bht_j if bht_j is not None else bht_chunk
        if scores is None:
            return

        bg = float(Sing_Trigger(scores, self.cut))
        cut_next, err_next = PD_controller1(bg, self.err, self.cut)  # HT uses PD_controller1
        self.cut = float(cut_next) #float(np.clip(cut_next, self.lo, self.hi))
        self.err = float(err_next)
    
class DQNCtrl(BaseCtrl):
    def __init__(self, name, init_cut, lo, hi, *, agent, deltas, step, max_delta, as_mid, as_span,
                 near_widths, K, target, tol, eps_min, eps_decay, train_steps_per_micro, alpha, beta):
        self.name = name
        self.cut = float(init_cut)
        self.lo, self.hi = float(lo), float(hi)
        self.agent = agent
        self.deltas = np.asarray(deltas, np.float32)
        self.step = float(step)
        self.max_delta = float(max_delta)
        self.as_mid, self.as_span = float(as_mid), float(as_span)
        self.near_widths = near_widths
        self.K = int(K)
        self.target, self.tol = float(target), float(tol)
        self.eps_min, self.eps_decay = float(eps_min), float(eps_decay)
        self.train_steps_per_micro = int(train_steps_per_micro)
        self.alpha, self.beta = float(alpha), float(beta)

        self.prev_bg = None
        self.last_delta = 0.0
        self.err_i = 0.0
        self.step_count = 0

    def cut_value(self): return self.cut

    def step_micro(self, *, bas_w, bnpv_w, bas_j, sas_tt, sas_aa, micro_global,
                   chunk=None, grpo_samples=None, **kwargs):
        bg_before = float(Sing_Trigger(bas_j, self.cut))
        if self.prev_bg is None:
            self.prev_bg = bg_before

        self.err_i = update_err_i(self.err_i, bg_before, self.target)
        dbgcut = d_bg_d_cut_norm(bas_j, self.cut, self.step, self.target)

        obs = make_event_seq_as(
            bas=bas_w, bnpv=bnpv_w,
            bg_rate=bg_before, prev_bg_rate=self.prev_bg,
            cut=self.cut,
            as_mid=self.as_mid, as_span=self.as_span,
            target=self.target, K=self.K,
            last_delta=self.last_delta, max_delta=self.max_delta,
            near_widths=self.near_widths,
            step=self.step, tol=self.tol,
            err_i=self.err_i, d_bg_d_cut=dbgcut
        )

        eps = max(self.eps_min, 1.0 * (self.eps_decay ** self.step_count))
        a = int(self.agent.act(obs, eps=eps))
        dlt = float(self.deltas[a] * self.step)

        sd = shield_delta(bg_before, self.target, self.tol, self.max_delta)
        shielded = (sd is not None)
        if sd is not None:
            dlt = float(sd)

        cut_next = float(np.clip(self.cut + dlt, self.lo, self.hi))
        bg_after = float(Sing_Trigger(bas_j, cut_next))
        tt_after = float(Sing_Trigger(sas_tt, cut_next))
        aa_after = float(Sing_Trigger(sas_aa, cut_next))

        dbgcut_next = d_bg_d_cut_norm(bas_j, cut_next, self.step, self.target)
        obs_next = make_event_seq_as(
            bas=bas_w, bnpv=bnpv_w,
            bg_rate=bg_after, prev_bg_rate=bg_before,
            cut=cut_next,
            as_mid=self.as_mid, as_span=self.as_span,
            target=self.target, K=self.K,
            last_delta=dlt, max_delta=self.max_delta,
            near_widths=self.near_widths,
            step=self.step, tol=self.tol,
            err_i=update_err_i(self.err_i, bg_after, self.target),
            d_bg_d_cut=dbgcut_next,
        )

        r = float(SeqDQNAgent.compute_reward(
            bg_rate=bg_after, target=self.target, tol=self.tol,
            sig_rate_1=tt_after, sig_rate_2=aa_after,
            delta_applied=dlt, max_delta=self.max_delta,
            alpha=self.alpha, beta=self.beta,
            prev_bg_rate=bg_before, gamma_stab=0.3
        ))

        self.agent.buf.push(obs, a, r, obs_next, done=False)
        for _ in range(self.train_steps_per_micro):
            _ = self.agent.train_step()

        self.cut = cut_next
        self.prev_bg = bg_after
        self.last_delta = dlt
        self.step_count += 1

        return CtrlOut(micro_global=micro_global)  # DQN doesn't use micro_global for logging here


class DQNCtrlHT(BaseCtrl):
    def __init__(self, name, init_cut, lo, hi, *, agent, deltas, step, max_delta,
                 ht_mid, ht_span, near_widths, K, target, tol, eps_min, eps_decay,
                 train_steps_per_micro, alpha, beta):
        self.name = name
        self.cut = float(init_cut)
        self.lo, self.hi = float(lo), float(hi)
        self.agent = agent
        self.deltas = np.asarray(deltas, np.float32)
        self.step = float(step)
        self.max_delta = float(max_delta)
        self.ht_mid, self.ht_span = float(ht_mid), float(ht_span)
        self.near_widths = near_widths
        self.K = int(K)
        self.target, self.tol = float(target), float(tol)
        self.eps_min, self.eps_decay = float(eps_min), float(eps_decay)
        self.train_steps_per_micro = int(train_steps_per_micro)
        self.alpha, self.beta = float(alpha), float(beta)

        self.prev_bg = None
        self.last_delta = 0.0
        self.err_i = 0.0
        self.step_count = 0

    def cut_value(self): 
        return self.cut

    def step_micro(self, *, bht_w, bnpv_w, bht_j, sht_tt, sht_aa, micro_global,
                   chunk=None, grpo_samples=None, **kwargs):
        bg_before = float(Sing_Trigger(bht_j, self.cut))
        if self.prev_bg is None:
            self.prev_bg = bg_before

        self.err_i = update_err_i(self.err_i, bg_before, self.target)
        dbgcut = d_bg_d_cut_norm(bht_j, self.cut, self.step, self.target)

        obs = make_event_seq_ht(
            bht=bht_w, bnpv=bnpv_w,
            bg_rate=bg_before, prev_bg_rate=self.prev_bg,
            cut=self.cut,
            ht_mid=self.ht_mid, ht_span=self.ht_span,
            target=self.target, K=self.K,
            last_delta=self.last_delta, max_delta=self.max_delta,
            near_widths=self.near_widths,
            step=self.step, tol=self.tol,
            err_i=self.err_i, d_bg_d_cut=dbgcut
        )

        eps = max(self.eps_min, 1.0 * (self.eps_decay ** self.step_count))
        a = int(self.agent.act(obs, eps=eps))
        dlt = float(self.deltas[a] * self.step)

        sd = shield_delta(bg_before, self.target, self.tol, self.max_delta)
        if sd is not None:
            dlt = float(sd)

        cut_next = float(np.clip(self.cut + dlt, self.lo, self.hi))
        bg_after = float(Sing_Trigger(bht_j, cut_next))
        tt_after = float(Sing_Trigger(sht_tt, cut_next))
        aa_after = float(Sing_Trigger(sht_aa, cut_next))

        dbgcut_next = d_bg_d_cut_norm(bht_j, cut_next, self.step, self.target)
        obs_next = make_event_seq_ht(
            bht=bht_w, bnpv=bnpv_w,
            bg_rate=bg_after, prev_bg_rate=bg_before,
            cut=cut_next,
            ht_mid=self.ht_mid, ht_span=self.ht_span,
            target=self.target, K=self.K,
            last_delta=dlt, max_delta=self.max_delta,
            near_widths=self.near_widths,
            step=self.step, tol=self.tol,
            err_i=update_err_i(self.err_i, bg_after, self.target),
            d_bg_d_cut=dbgcut_next,
        )

        r = float(SeqDQNAgent.compute_reward(
            bg_rate=bg_after, target=self.target, tol=self.tol,
            sig_rate_1=tt_after, sig_rate_2=aa_after,
            delta_applied=dlt, max_delta=self.max_delta,
            alpha=self.alpha, beta=self.beta,
            prev_bg_rate=bg_before, gamma_stab=0.3
        ))

        self.agent.buf.push(obs, a, r, obs_next, done=False)
        for _ in range(self.train_steps_per_micro):
            _ = self.agent.train_step()

        self.cut = cut_next
        self.prev_bg = bg_after
        self.last_delta = dlt
        self.step_count += 1
        return CtrlOut(micro_global=micro_global)

class DQNFrozenCtrl(DQNCtrl):
    """
    DQN-F: train only on first N chunks; after that, stop pushing to replay + stop optimizer steps.
    Rollout epsilon after freeze is args.dqn_f_eps (default greedy).
    """
    def __init__(self, *args, train_chunks: int, eps_after_freeze: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_chunks = int(train_chunks)
        self.eps_after_freeze = float(eps_after_freeze)

    def step_micro(self, *, bas_w, bnpv_w, bas_j, sas_tt, sas_aa, micro_global,
                   chunk=None, grpo_samples=None, **kwargs):
        ch = 0 if chunk is None else int(chunk)
        train_mode = (ch < self.train_chunks)

        bg_before = float(Sing_Trigger(bas_j, self.cut))
        if self.prev_bg is None:
            self.prev_bg = bg_before

        self.err_i = update_err_i(self.err_i, bg_before, self.target)
        dbgcut = d_bg_d_cut_norm(bas_j, self.cut, self.step, self.target)

        obs = make_event_seq_as(
            bas=bas_w, bnpv=bnpv_w,
            bg_rate=bg_before, prev_bg_rate=self.prev_bg,
            cut=self.cut,
            as_mid=self.as_mid, as_span=self.as_span,
            target=self.target, K=self.K,
            last_delta=self.last_delta, max_delta=self.max_delta,
            near_widths=self.near_widths,
            step=self.step, tol=self.tol,
            err_i=self.err_i, d_bg_d_cut=dbgcut
        )

        # eps schedule: normal DQN during training, fixed eps after freeze
        if train_mode:
            eps = max(self.eps_min, 1.0 * (self.eps_decay ** self.step_count))
        else:
            eps = self.eps_after_freeze

        a = int(self.agent.act(obs, eps=eps))
        dlt = float(self.deltas[a] * self.step)

        sd = shield_delta(bg_before, self.target, self.tol, self.max_delta)
        if sd is not None:
            dlt = float(sd)

        cut_next = float(np.clip(self.cut + dlt, self.lo, self.hi))
        bg_after = float(Sing_Trigger(bas_j, cut_next))
        tt_after = float(Sing_Trigger(sas_tt, cut_next))
        aa_after = float(Sing_Trigger(sas_aa, cut_next))

        dbgcut_next = d_bg_d_cut_norm(bas_j, cut_next, self.step, self.target)
        obs_next = make_event_seq_as(
            bas=bas_w, bnpv=bnpv_w,
            bg_rate=bg_after, prev_bg_rate=bg_before,
            cut=cut_next,
            as_mid=self.as_mid, as_span=self.as_span,
            target=self.target, K=self.K,
            last_delta=dlt, max_delta=self.max_delta,
            near_widths=self.near_widths,
            step=self.step, tol=self.tol,
            err_i=update_err_i(self.err_i, bg_after, self.target),
            d_bg_d_cut=dbgcut_next,
        )

        r = float(SeqDQNAgent.compute_reward(
            bg_rate=bg_after, target=self.target, tol=self.tol,
            sig_rate_1=tt_after, sig_rate_2=aa_after,
            delta_applied=dlt, max_delta=self.max_delta,
            alpha=self.alpha, beta=self.beta,
            prev_bg_rate=bg_before, gamma_stab=0.3
        ))

        # ONLY train during early chunks
        if train_mode:
            self.agent.buf.push(obs, a, r, obs_next, done=False)
            for _ in range(self.train_steps_per_micro):
                _ = self.agent.train_step()

        # advance state always
        self.cut = cut_next
        self.prev_bg = bg_after
        self.last_delta = dlt
        self.step_count += 1

        return CtrlOut(micro_global=micro_global)


class DQNFrozenCtrlHT(DQNCtrlHT):
    """
    HT version of DQN-F.
    """
    def __init__(self, *args, train_chunks: int, eps_after_freeze: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_chunks = int(train_chunks)
        self.eps_after_freeze = float(eps_after_freeze)

    def step_micro(self, *, bht_w, bnpv_w, bht_j, sht_tt, sht_aa, micro_global,
                   chunk=None, grpo_samples=None, **kwargs):
        ch = 0 if chunk is None else int(chunk)
        train_mode = (ch < self.train_chunks)

        bg_before = float(Sing_Trigger(bht_j, self.cut))
        if self.prev_bg is None:
            self.prev_bg = bg_before

        self.err_i = update_err_i(self.err_i, bg_before, self.target)
        dbgcut = d_bg_d_cut_norm(bht_j, self.cut, self.step, self.target)

        obs = make_event_seq_ht(
            bht=bht_w, bnpv=bnpv_w,
            bg_rate=bg_before, prev_bg_rate=self.prev_bg,
            cut=self.cut,
            ht_mid=self.ht_mid, ht_span=self.ht_span,
            target=self.target, K=self.K,
            last_delta=self.last_delta, max_delta=self.max_delta,
            near_widths=self.near_widths,
            step=self.step, tol=self.tol,
            err_i=self.err_i, d_bg_d_cut=dbgcut
        )

        if train_mode:
            eps = max(self.eps_min, 1.0 * (self.eps_decay ** self.step_count))
        else:
            eps = self.eps_after_freeze

        a = int(self.agent.act(obs, eps=eps))
        dlt = float(self.deltas[a] * self.step)

        sd = shield_delta(bg_before, self.target, self.tol, self.max_delta)
        if sd is not None:
            dlt = float(sd)

        cut_next = float(np.clip(self.cut + dlt, self.lo, self.hi))
        bg_after = float(Sing_Trigger(bht_j, cut_next))
        tt_after = float(Sing_Trigger(sht_tt, cut_next))
        aa_after = float(Sing_Trigger(sht_aa, cut_next))

        dbgcut_next = d_bg_d_cut_norm(bht_j, cut_next, self.step, self.target)
        obs_next = make_event_seq_ht(
            bht=bht_w, bnpv=bnpv_w,
            bg_rate=bg_after, prev_bg_rate=bg_before,
            cut=cut_next,
            ht_mid=self.ht_mid, ht_span=self.ht_span,
            target=self.target, K=self.K,
            last_delta=dlt, max_delta=self.max_delta,
            near_widths=self.near_widths,
            step=self.step, tol=self.tol,
            err_i=update_err_i(self.err_i, bg_after, self.target),
            d_bg_d_cut=dbgcut_next,
        )

        r = float(SeqDQNAgent.compute_reward(
            bg_rate=bg_after, target=self.target, tol=self.tol,
            sig_rate_1=tt_after, sig_rate_2=aa_after,
            delta_applied=dlt, max_delta=self.max_delta,
            alpha=self.alpha, beta=self.beta,
            prev_bg_rate=bg_before, gamma_stab=0.3
        ))

        if train_mode:
            self.agent.buf.push(obs, a, r, obs_next, done=False)
            for _ in range(self.train_steps_per_micro):
                _ = self.agent.train_step()

        self.cut = cut_next
        self.prev_bg = bg_after
        self.last_delta = dlt
        self.step_count += 1

        return CtrlOut(micro_global=micro_global)



class ADTCtrl(DQNCtrl):
    def __init__(self, *args, adt_l=10, train_steps_per_episode=50,
                 reward_mode="lhc", adt_alpha=0.7, adt_beta=0.3, **kwargs):
        super().__init__(*args, **kwargs)
        self.adt_l = max(1, int(adt_l))
        self.train_steps_per_episode = int(train_steps_per_episode)
        self.reward_mode = str(reward_mode)
        self.adt_alpha = float(adt_alpha)
        self.adt_beta = float(adt_beta)

        self._micro_in_chunk = 0
        self._prev_action = 0  # held action index

    def step_micro(self, *, bas_w, bnpv_w, bas_j, sas_tt, sas_aa, micro_global,
                   chunk=None, grpo_samples=None, **kwargs):

        bg_before = float(Sing_Trigger(bas_j, self.cut))
        if self.prev_bg is None:
            self.prev_bg = bg_before

        self.err_i = update_err_i(self.err_i, bg_before, self.target)
        dbgcut = d_bg_d_cut_norm(bas_j, self.cut, self.step, self.target)

        obs = make_event_seq_as(
            bas=bas_w, bnpv=bnpv_w,
            bg_rate=bg_before, prev_bg_rate=self.prev_bg,
            cut=self.cut,
            as_mid=self.as_mid, as_span=self.as_span,
            target=self.target, K=self.K,
            last_delta=self.last_delta, max_delta=self.max_delta,
            near_widths=self.near_widths,
            step=self.step, tol=self.tol,
            err_i=self.err_i, d_bg_d_cut=dbgcut
        )

        # --- ADT action-hold ---
        eps = max(self.eps_min, 1.0 * (self.eps_decay ** self.step_count))
        if (self._micro_in_chunk % self.adt_l) == 0:
            self._prev_action = int(self.agent.act(obs, eps=eps))
        a = int(self._prev_action)

        dlt = float(self.deltas[a] * self.step)

        # shield -> also remap action index to match executed delta (avoid label mismatch)
        sd = shield_delta(bg_before, self.target, self.tol, self.max_delta)
        if sd is not None:
            dlt = float(sd)
            a = int(np.argmin(np.abs(self.deltas * self.step - dlt)))

        cut_next = float(np.clip(self.cut + dlt, self.lo, self.hi))

        bg_after = float(Sing_Trigger(bas_j, cut_next))
        tt_after = float(Sing_Trigger(sas_tt, cut_next))
        aa_after = float(Sing_Trigger(sas_aa, cut_next))

        dbgcut_next = d_bg_d_cut_norm(bas_j, cut_next, self.step, self.target)
        obs_next = make_event_seq_as(
            bas=bas_w, bnpv=bnpv_w,
            bg_rate=bg_after, prev_bg_rate=bg_before,
            cut=cut_next,
            as_mid=self.as_mid, as_span=self.as_span,
            target=self.target, K=self.K,
            last_delta=dlt, max_delta=self.max_delta,
            near_widths=self.near_widths,
            step=self.step, tol=self.tol,
            err_i=update_err_i(self.err_i, bg_after, self.target),
            d_bg_d_cut=dbgcut_next,
        )

        # --- reward mode ---
        if self.reward_mode == "paper":
            r = adt_reward_paper_style(
                bg_scores=bas_j, sig1_scores=sas_tt, sig2_scores=sas_aa,
                cut=cut_next, alpha=self.adt_alpha, beta=self.adt_beta
            )
        else:
            r = float(SeqDQNAgent.compute_reward(
                bg_rate=bg_after, target=self.target, tol=self.tol,
                sig_rate_1=tt_after, sig_rate_2=aa_after,
                delta_applied=dlt, max_delta=self.max_delta,
                alpha=self.alpha, beta=self.beta,
                prev_bg_rate=bg_before, gamma_stab=0.3
            ))

        # store transition (NO training here)
        self.agent.buf.push(obs, a, float(r), obs_next, done=False)

        # advance state
        self.cut = cut_next
        self.prev_bg = bg_after
        self.last_delta = dlt
        self.step_count += 1
        self._micro_in_chunk += 1

        return CtrlOut(micro_global=micro_global)

    def end_chunk(self, *, chunk: int, **kwargs):
        # ADT-style: update ONLY at end of episode (chunk)
        for _ in range(self.train_steps_per_episode):
            _ = self.agent.train_step()
        self._micro_in_chunk = 0
class ADTCtrlHT(DQNCtrlHT):
    def __init__(self, *args, adt_l=10, train_steps_per_episode=50,
                 reward_mode="lhc", adt_alpha=0.7, adt_beta=0.3, **kwargs):
        super().__init__(*args, **kwargs)
        self.adt_l = max(1, int(adt_l))
        self.train_steps_per_episode = int(train_steps_per_episode)
        self.reward_mode = str(reward_mode)
        self.adt_alpha = float(adt_alpha)
        self.adt_beta = float(adt_beta)

        self._micro_in_chunk = 0
        self._prev_action = 0

    def step_micro(self, *, bht_w, bnpv_w, bht_j, sht_tt, sht_aa, micro_global,
                   chunk=None, grpo_samples=None, **kwargs):

        bg_before = float(Sing_Trigger(bht_j, self.cut))
        if self.prev_bg is None:
            self.prev_bg = bg_before

        self.err_i = update_err_i(self.err_i, bg_before, self.target)
        dbgcut = d_bg_d_cut_norm(bht_j, self.cut, self.step, self.target)

        obs = make_event_seq_ht(
            bht=bht_w, bnpv=bnpv_w,
            bg_rate=bg_before, prev_bg_rate=self.prev_bg,
            cut=self.cut,
            ht_mid=self.ht_mid, ht_span=self.ht_span,
            target=self.target, K=self.K,
            last_delta=self.last_delta, max_delta=self.max_delta,
            near_widths=self.near_widths,
            step=self.step, tol=self.tol,
            err_i=self.err_i, d_bg_d_cut=dbgcut
        )

        eps = max(self.eps_min, 1.0 * (self.eps_decay ** self.step_count))
        if (self._micro_in_chunk % self.adt_l) == 0:
            self._prev_action = int(self.agent.act(obs, eps=eps))
        a = int(self._prev_action)

        dlt = float(self.deltas[a] * self.step)

        sd = shield_delta(bg_before, self.target, self.tol, self.max_delta)
        if sd is not None:
            dlt = float(sd)
            a = int(np.argmin(np.abs(self.deltas * self.step - dlt)))

        cut_next = float(np.clip(self.cut + dlt, self.lo, self.hi))

        bg_after = float(Sing_Trigger(bht_j, cut_next))
        tt_after = float(Sing_Trigger(sht_tt, cut_next))
        aa_after = float(Sing_Trigger(sht_aa, cut_next))

        dbgcut_next = d_bg_d_cut_norm(bht_j, cut_next, self.step, self.target)
        obs_next = make_event_seq_ht(
            bht=bht_w, bnpv=bnpv_w,
            bg_rate=bg_after, prev_bg_rate=bg_before,
            cut=cut_next,
            ht_mid=self.ht_mid, ht_span=self.ht_span,
            target=self.target, K=self.K,
            last_delta=dlt, max_delta=self.max_delta,
            near_widths=self.near_widths,
            step=self.step, tol=self.tol,
            err_i=update_err_i(self.err_i, bg_after, self.target),
            d_bg_d_cut=dbgcut_next,
        )

        if self.reward_mode == "paper":
            r = adt_reward_paper_style(
                bg_scores=bht_j, sig1_scores=sht_tt, sig2_scores=sht_aa,
                cut=cut_next, alpha=self.adt_alpha, beta=self.adt_beta
            )
        else:
            r = float(SeqDQNAgent.compute_reward(
                bg_rate=bg_after, target=self.target, tol=self.tol,
                sig_rate_1=tt_after, sig_rate_2=aa_after,
                delta_applied=dlt, max_delta=self.max_delta,
                alpha=self.alpha, beta=self.beta,
                prev_bg_rate=bg_before, gamma_stab=0.3
            ))

        self.agent.buf.push(obs, a, float(r), obs_next, done=False)

        self.cut = cut_next
        self.prev_bg = bg_after
        self.last_delta = dlt
        self.step_count += 1
        self._micro_in_chunk += 1

        return CtrlOut(micro_global=micro_global)

    def end_chunk(self, *, chunk: int, **kwargs):
        for _ in range(self.train_steps_per_episode):
            _ = self.agent.train_step()
        self._micro_in_chunk = 0

# ----------------------------- PPO controllers -----------------------------
from typing import Optional, Tuple, Any

def _ppo_unpack_act(out: Any) -> Tuple[int, float, float, Optional[np.ndarray]]:
    """
    SeqPPOAgent.act(...) compatibility:
      returns either (a, logp, v) or (a, logp, v, extra)
    """
    if isinstance(out, (tuple, list)):
        if len(out) == 3:
            a, logp, v = out
            return int(a), float(logp), float(v), None
        if len(out) >= 4:
            a, logp, v, extra = out[0], out[1], out[2], out[3]
            extra_np = None if extra is None else np.asarray(extra)
            return int(a), float(logp), float(v), extra_np
    raise TypeError(f"Unexpected SeqPPOAgent.act return: {type(out)} / {out}")

def _ppo_eval_logp_v(agent: "SeqPPOAgent", obs: np.ndarray, act: int) -> Tuple[float, float]:
    """
    Prefer agent.eval_logp_v(obs, act). Fallbacks for slightly different agent APIs.
    """
    if hasattr(agent, "eval_logp_v"):
        lp, v = agent.eval_logp_v(obs, act)
        return float(lp), float(v)
    if hasattr(agent, "evaluate"):
        # common naming in PPO implementations
        lp, v = agent.evaluate(obs, act)
        return float(lp), float(v)
    raise AttributeError("SeqPPOAgent must implement eval_logp_v(obs, act) or evaluate(obs, act).")

def _ppo_value(agent: "SeqPPOAgent", obs: np.ndarray) -> float:
    """
    Prefer agent.value(obs). Fallback to eval_logp_v with a dummy act if needed.
    """
    if hasattr(agent, "value"):
        return float(agent.value(obs))
    # fallback: some agents only expose value together with logp
    lp, v = _ppo_eval_logp_v(agent, obs, act=0)
    return float(v)

class PPOCtrl(BaseCtrl):
    """
    On-policy SeqPPOAgent controller for AD/AS cut.
    Stores (obs, a, logp, v, r, done) per micro-step, then PPO update per chunk.
    """
    def __init__(
        self,
        name,
        init_cut,
        lo,
        hi,
        *,
        agent: "SeqPPOAgent",
        deltas,
        step,
        max_delta,
        as_mid,
        as_span,
        near_widths,
        K,
        target,
        tol,
        alpha,
        beta,
        ppo_temperature=1.0,
        use_shield=True,
    ):
        self.name = name
        self.cut = float(init_cut)
        self.lo, self.hi = float(lo), float(hi)

        self.agent: SeqPPOAgent = agent
        self.deltas = np.asarray(deltas, np.float32)
        self.step = float(step)
        self.max_delta = float(max_delta)

        self.as_mid, self.as_span = float(as_mid), float(as_span)
        self.near_widths = near_widths
        self.K = int(K)
        self.target, self.tol = float(target), float(tol)
        self.alpha, self.beta = float(alpha), float(beta)

        self.prev_bg = None
        self.last_delta = 0.0
        self.err_i = 0.0

        self.ppo_temperature = float(ppo_temperature)
        self.use_shield = bool(use_shield)

        self._last_obs_next: Optional[np.ndarray] = None  # bootstrap value at end_chunk

    def cut_value(self):
        return self.cut

    def _obs(self, bas_w, bnpv_w, bas_j, bg_before):
        self.err_i = update_err_i(self.err_i, bg_before, self.target)
        dbgcut = d_bg_d_cut_norm(bas_j, self.cut, self.step, self.target)
        return make_event_seq_as(
            bas=bas_w,
            bnpv=bnpv_w,
            bg_rate=bg_before,
            prev_bg_rate=self.prev_bg,
            cut=self.cut,
            as_mid=self.as_mid,
            as_span=self.as_span,
            target=self.target,
            K=self.K,
            last_delta=self.last_delta,
            max_delta=self.max_delta,
            near_widths=self.near_widths,
            step=self.step,
            tol=self.tol,
            err_i=self.err_i,
            d_bg_d_cut=dbgcut,
        )

    def step_micro(
        self,
        *,
        bas_w,
        bnpv_w,
        bas_j,
        sas_tt,
        sas_aa,
        micro_global,
        chunk=None,
        grpo_samples=None,
        **kwargs,
    ):
        bg_before = float(Sing_Trigger(bas_j, self.cut))
        if self.prev_bg is None:
            self.prev_bg = bg_before

        obs = self._obs(bas_w, bnpv_w, bas_j, bg_before)

        # sample action from PPO policy
        a, logp, v, _ = _ppo_unpack_act(self.agent.act(obs, temperature=self.ppo_temperature))
        dlt = float(self.deltas[a] * self.step)

        # optional shielding (map to executed action index to avoid label mismatch)
        if self.use_shield:
            sd = shield_delta(bg_before, self.target, self.tol, self.max_delta)
            if sd is not None:
                dlt = float(sd)
                a_exec = int(np.argmin(np.abs(self.deltas * self.step - dlt)))
                logp, v = _ppo_eval_logp_v(self.agent, obs, a_exec)
                a = a_exec

        cut_next = float(np.clip(self.cut + dlt, self.lo, self.hi))

        bg_after = float(Sing_Trigger(bas_j, cut_next))
        tt_after = float(Sing_Trigger(sas_tt, cut_next))
        aa_after = float(Sing_Trigger(sas_aa, cut_next))

        # next obs for bootstrap
        dbgcut_next = d_bg_d_cut_norm(bas_j, cut_next, self.step, self.target)
        obs_next = make_event_seq_as(
            bas=bas_w,
            bnpv=bnpv_w,
            bg_rate=bg_after,
            prev_bg_rate=bg_before,
            cut=cut_next,
            as_mid=self.as_mid,
            as_span=self.as_span,
            target=self.target,
            K=self.K,
            last_delta=dlt,
            max_delta=self.max_delta,
            near_widths=self.near_widths,
            step=self.step,
            tol=self.tol,
            err_i=update_err_i(self.err_i, bg_after, self.target),
            d_bg_d_cut=dbgcut_next,
        )

        r = float(
            SeqDQNAgent.compute_reward(
                bg_rate=bg_after,
                target=self.target,
                tol=self.tol,
                sig_rate_1=tt_after,
                sig_rate_2=aa_after,
                delta_applied=dlt,
                max_delta=self.max_delta,
                alpha=self.alpha,
                beta=self.beta,
                prev_bg_rate=bg_before,
                gamma_stab=0.3,
            )
        )

        # store on-policy transition
        self.agent.store(obs, a, logp, v, r, done=False)
        self._last_obs_next = obs_next

        # advance
        self.cut = cut_next
        self.prev_bg = bg_after
        self.last_delta = dlt

        return CtrlOut(micro_global=micro_global)

    def end_chunk(self, *, chunk: int, **kwargs):
        # bootstrap at end-of-chunk
        last_v = 0.0
        if self._last_obs_next is not None:
            last_v = _ppo_value(self.agent, self._last_obs_next)

        # finish path + update
        self.agent.finish_path(last_value=float(last_v))
        _ = self.agent.update()
        self._last_obs_next = None


class PPOCtrlHT(BaseCtrl):
    """
    On-policy SeqPPOAgent controller for HT cut.
    """
    def __init__(
        self,
        name,
        init_cut,
        lo,
        hi,
        *,
        agent: "SeqPPOAgent",
        deltas,
        step,
        max_delta,
        ht_mid,
        ht_span,
        near_widths,
        K,
        target,
        tol,
        alpha,
        beta,
        ppo_temperature=1.0,
        use_shield=True,
    ):
        self.name = name
        self.cut = float(init_cut)
        self.lo, self.hi = float(lo), float(hi)

        self.agent: SeqPPOAgent = agent
        self.deltas = np.asarray(deltas, np.float32)
        self.step = float(step)
        self.max_delta = float(max_delta)

        self.ht_mid, self.ht_span = float(ht_mid), float(ht_span)
        self.near_widths = near_widths
        self.K = int(K)
        self.target, self.tol = float(target), float(tol)
        self.alpha, self.beta = float(alpha), float(beta)

        self.prev_bg = None
        self.last_delta = 0.0
        self.err_i = 0.0

        self.ppo_temperature = float(ppo_temperature)
        self.use_shield = bool(use_shield)

        self._last_obs_next: Optional[np.ndarray] = None

    def cut_value(self):
        return self.cut

    def _obs(self, bht_w, bnpv_w, bht_j, bg_before):
        self.err_i = update_err_i(self.err_i, bg_before, self.target)
        dbgcut = d_bg_d_cut_norm(bht_j, self.cut, self.step, self.target)
        return make_event_seq_ht(
            bht=bht_w,
            bnpv=bnpv_w,
            bg_rate=bg_before,
            prev_bg_rate=self.prev_bg,
            cut=self.cut,
            ht_mid=self.ht_mid,
            ht_span=self.ht_span,
            target=self.target,
            K=self.K,
            last_delta=self.last_delta,
            max_delta=self.max_delta,
            near_widths=self.near_widths,
            step=self.step,
            tol=self.tol,
            err_i=self.err_i,
            d_bg_d_cut=dbgcut,
        )

    def step_micro(
        self,
        *,
        bht_w,
        bnpv_w,
        bht_j,
        sht_tt,
        sht_aa,
        micro_global,
        chunk=None,
        grpo_samples=None,
        **kwargs,
    ):
        bg_before = float(Sing_Trigger(bht_j, self.cut))
        if self.prev_bg is None:
            self.prev_bg = bg_before

        obs = self._obs(bht_w, bnpv_w, bht_j, bg_before)

        a, logp, v, _ = _ppo_unpack_act(self.agent.act(obs, temperature=self.ppo_temperature))
        dlt = float(self.deltas[a] * self.step)

        if self.use_shield:
            sd = shield_delta(bg_before, self.target, self.tol, self.max_delta)
            if sd is not None:
                dlt = float(sd)
                a_exec = int(np.argmin(np.abs(self.deltas * self.step - dlt)))
                logp, v = _ppo_eval_logp_v(self.agent, obs, a_exec)
                a = a_exec

        cut_next = float(np.clip(self.cut + dlt, self.lo, self.hi))

        bg_after = float(Sing_Trigger(bht_j, cut_next))
        tt_after = float(Sing_Trigger(sht_tt, cut_next))
        aa_after = float(Sing_Trigger(sht_aa, cut_next))

        dbgcut_next = d_bg_d_cut_norm(bht_j, cut_next, self.step, self.target)
        obs_next = make_event_seq_ht(
            bht=bht_w,
            bnpv=bnpv_w,
            bg_rate=bg_after,
            prev_bg_rate=bg_before,
            cut=cut_next,
            ht_mid=self.ht_mid,
            ht_span=self.ht_span,
            target=self.target,
            K=self.K,
            last_delta=dlt,
            max_delta=self.max_delta,
            near_widths=self.near_widths,
            step=self.step,
            tol=self.tol,
            err_i=update_err_i(self.err_i, bg_after, self.target),
            d_bg_d_cut=dbgcut_next,
        )

        r = float(
            SeqDQNAgent.compute_reward(
                bg_rate=bg_after,
                target=self.target,
                tol=self.tol,
                sig_rate_1=tt_after,
                sig_rate_2=aa_after,
                delta_applied=dlt,
                max_delta=self.max_delta,
                alpha=self.alpha,
                beta=self.beta,
                prev_bg_rate=bg_before,
                gamma_stab=0.3,
            )
        )

        self.agent.store(obs, a, logp, v, r, done=False)
        self._last_obs_next = obs_next

        self.cut = cut_next
        self.prev_bg = bg_after
        self.last_delta = dlt

        return CtrlOut(micro_global=micro_global)

    def end_chunk(self, *, chunk: int, **kwargs):
        last_v = 0.0
        if self._last_obs_next is not None:
            last_v = _ppo_value(self.agent, self._last_obs_next)

        self.agent.finish_path(last_value=float(last_v))
        _ = self.agent.update()
        self._last_obs_next = None


class GRPOCtrl(BaseCtrl):
    """
    Plain GRPO: sample G actions, train on all, execute best reward.
    """
    def __init__(self, name, init_cut, lo, hi, *, agent, deltas, step, max_delta, as_mid, as_span,
                 near_widths, K, target, tol, train_every, temperature, group_size_keep: int):
        self.name = name
        self.cut = float(init_cut)
        self.lo, self.hi = float(lo), float(hi)
        self.agent = agent
        self.deltas = np.asarray(deltas, np.float32)
        self.step = float(step)
        self.max_delta = float(max_delta)
        self.as_mid, self.as_span = float(as_mid), float(as_span)
        self.near_widths = near_widths
        self.K = int(K)
        self.target, self.tol = float(target), float(tol)
        self.train_every = int(train_every)
        self.temperature = float(temperature)

        self.prev_bg = None
        self.last_delta = 0.0
        self.err_i = 0.0
        self.micro_counter = 0
        self.group_size_keep = int(group_size_keep)

    def cut_value(self): return self.cut

    def _obs(self, bas_w, bnpv_w, bas_j, bg_before):
        self.err_i = update_err_i(self.err_i, bg_before, self.target)
        dbgcut = d_bg_d_cut_norm(bas_j, self.cut, self.step, self.target)
        return make_event_seq_as(
            bas=bas_w, bnpv=bnpv_w,
            bg_rate=bg_before, prev_bg_rate=self.prev_bg,
            cut=self.cut,
            as_mid=self.as_mid, as_span=self.as_span,
            target=self.target, K=self.K,
            last_delta=self.last_delta, max_delta=self.max_delta,
            near_widths=self.near_widths,
            step=self.step, tol=self.tol,
            err_i=self.err_i, d_bg_d_cut=dbgcut
        )

    def step_micro(self, *, chunk: int, bas_w, bnpv_w, bas_j, sas_tt, sas_aa, micro_global, grpo_samples=None):
        bg_before = float(Sing_Trigger(bas_j, self.cut))
        if self.prev_bg is None:
            self.prev_bg = bg_before

        obs = self._obs(bas_w, bnpv_w, bas_j, bg_before)
        G = self.group_size_keep  
        acts, old_logps = self.agent.sample_group_actions(obs, group_size=G, temperature=self.temperature)

        occ_mid = float(near_occupancy(bas_j, self.cut, self.near_widths)[1])
        cand_r = np.zeros(G, np.float32)

        for k in range(G):
            a = int(acts[k])
            dlt = float(self.deltas[a] * self.step)
            cut_next = float(np.clip(self.cut + dlt, self.lo, self.hi))
            bg_after = float(Sing_Trigger(bas_j, cut_next))
            tt_after = float(Sing_Trigger(sas_tt, cut_next))
            aa_after = float(Sing_Trigger(sas_aa, cut_next))
            r = float(self.agent.compute_reward(
                bg_after=bg_after, tt_after=tt_after, aa_after=aa_after,
                delta_applied=dlt, max_delta=self.max_delta,
                prev_bg=bg_before, occ_mid=occ_mid, update_dual=False
            ))
            cand_r[k] = r

            if grpo_samples is not None:
                log_grpo_row(
                    grpo_samples, method=self.name, trigger="AD",
                    chunk=chunk, micro=self.micro_counter, micro_global=micro_global,
                    phase="candidate", k=k, a=a, delta=dlt, step=self.step,
                    cut_before=self.cut, cut_next=cut_next, cut_lo=self.lo, cut_hi=self.hi,
                    bg_before=bg_before, bg_after=bg_after,
                    tt_after=tt_after, aa_after=aa_after,
                    occ_mid=occ_mid,
                    reward_raw=r, executed=0, shielded=0, kept=1
                )
            micro_global += 1

        # train on all
        self.agent.store_group(obs=obs, actions=acts, logp=old_logps, rewards=cand_r, baseline="mean")

        k_best = int(np.argmax(cand_r))
        a_exec = int(acts[k_best])
        d_exec = float(self.deltas[a_exec] * self.step)

        sd = shield_delta(bg_before, self.target, self.tol, self.max_delta)
        shielded = (sd is not None)
        if sd is not None:
            d_exec = float(sd)

        cut_next = float(np.clip(self.cut + d_exec, self.lo, self.hi))
        bg_after = float(Sing_Trigger(bas_j, cut_next))
        tt_after = float(Sing_Trigger(sas_tt, cut_next))
        aa_after = float(Sing_Trigger(sas_aa, cut_next))

        r_exec = float(self.agent.compute_reward(
            bg_after=bg_after, tt_after=tt_after, aa_after=aa_after,
            delta_applied=d_exec, max_delta=self.max_delta,
            prev_bg=bg_before, occ_mid=occ_mid, update_dual=True
        ))

        if grpo_samples is not None:
            log_grpo_row(
                grpo_samples, method=self.name, trigger="AD",
                chunk=chunk, micro=self.micro_counter, micro_global=micro_global,
                phase="executed", k=k_best, a=a_exec, delta=d_exec, step=self.step,
                cut_before=self.cut, cut_next=cut_next, cut_lo=self.lo, cut_hi=self.hi,
                bg_before=bg_before, bg_after=bg_after,
                tt_after=tt_after, aa_after=aa_after,
                occ_mid=occ_mid,
                reward_best_sample=float(cand_r[k_best]),
                reward_exec=r_exec,
                executed=1, shielded=int(shielded), kept=1
            )
        micro_global += 1

        self.cut = cut_next
        self.prev_bg = bg_after
        self.last_delta = d_exec
        self.micro_counter += 1

        if (self.micro_counter % self.train_every) == 0:
            _ = self.agent.update()

        return CtrlOut(micro_global=micro_global)


class GRPOFilterCtrl(GRPOCtrl):
    """
    GFPO-F: sample G_sample, KEEP only those within |bg-target| <= band_mult*tol (fallback: closest),
            train only on kept, execute best among kept by reward.
    """
    def __init__(self, *args, band_mult=1.0, group_size_sample=32, group_size_keep=16, **kwargs):
        super().__init__(*args, group_size_keep=group_size_keep, **kwargs)
        self.band_mult = float(band_mult)
        self.G_sample = int(group_size_sample)
        self.G_keep = int(group_size_keep)

    def step_micro(self, *, chunk: int, bas_w, bnpv_w, bas_j, sas_tt, sas_aa, micro_global, grpo_samples=None):
        bg_before = float(Sing_Trigger(bas_j, self.cut))
        if self.prev_bg is None:
            self.prev_bg = bg_before

        obs = self._obs(bas_w, bnpv_w, bas_j, bg_before)
        acts, old_logps = self.agent.sample_group_actions(obs, group_size=self.G_sample, temperature=self.temperature)

        occ_mid = float(near_occupancy(bas_j, self.cut, self.near_widths)[1])

        cand_r = np.zeros(self.G_sample, np.float32)
        cand_bg = np.zeros(self.G_sample, np.float32)
        cand_tt = np.zeros(self.G_sample, np.float32)
        cand_aa = np.zeros(self.G_sample, np.float32)
        cand_cut = np.zeros(self.G_sample, np.float32)

        for k in range(self.G_sample):
            a = int(acts[k])
            dlt = float(self.deltas[a] * self.step)
            cut_next = float(np.clip(self.cut + dlt, self.lo, self.hi))
            bg_after = float(Sing_Trigger(bas_j, cut_next))
            tt_after = float(Sing_Trigger(sas_tt, cut_next))
            aa_after = float(Sing_Trigger(sas_aa, cut_next))
            r = float(self.agent.compute_reward(
                bg_after=bg_after, tt_after=tt_after, aa_after=aa_after,
                delta_applied=dlt, max_delta=self.max_delta,
                prev_bg=bg_before, occ_mid=occ_mid, update_dual=False
            ))
            cand_r[k] = r
            cand_bg[k] = bg_after
            cand_tt[k] = tt_after
            cand_aa[k] = aa_after
            cand_cut[k] = cut_next

        # keep-set = in-band; fallback to closest abs_err if none
        abs_err = np.abs(cand_bg - self.target)
        inband = abs_err <= (self.band_mult * self.tol)
        keep = np.where(inband)[0]
        if keep.size == 0:
            keep = np.array([int(np.argmin(abs_err))], dtype=np.int64)

        # cap keep to G_keep (best rewards among kept)
        if keep.size > self.G_keep:
            keep = keep[np.argsort(-cand_r[keep])[:self.G_keep]]

        keep = keep.astype(np.int64)
        k_best = int(keep[np.argmax(cand_r[keep])])

        # log candidates + kept mask (optional)
        if grpo_samples is not None:
            keep_set = set(int(x) for x in keep.tolist())
            for k in range(self.G_sample):
                a = int(acts[k])
                dlt = float(self.deltas[a] * self.step)
                log_grpo_row(
                    grpo_samples, method=self.name, trigger="AD",
                    chunk=chunk, micro=self.micro_counter, micro_global=micro_global,
                    phase="candidate", k=k, a=a, delta=dlt, step=self.step,
                    cut_before=self.cut, cut_next=float(cand_cut[k]),
                    cut_lo=self.lo, cut_hi=self.hi,
                    bg_before=bg_before, bg_after=float(cand_bg[k]),
                    tt_after=float(cand_tt[k]), aa_after=float(cand_aa[k]),
                    occ_mid=occ_mid,
                    reward_raw=float(cand_r[k]), executed=0, shielded=0,
                    kept=int(k in keep_set)
                )
                micro_global += 1

        # train ONLY on kept
        self.agent.store_group(
            obs=obs,
            actions=acts[keep],
            logp=old_logps[keep],
            rewards=cand_r[keep],
            baseline="mean"
        )

        # execute best kept (with shield)
        a_exec = int(acts[k_best])
        d_exec = float(self.deltas[a_exec] * self.step)
        sd = shield_delta(bg_before, self.target, self.tol, self.max_delta)
        shielded = (sd is not None)
        if sd is not None:
            d_exec = float(sd)

        cut_next = float(np.clip(self.cut + d_exec, self.lo, self.hi))
        bg_after = float(Sing_Trigger(bas_j, cut_next))
        tt_after = float(Sing_Trigger(sas_tt, cut_next))
        aa_after = float(Sing_Trigger(sas_aa, cut_next))

        r_exec = float(self.agent.compute_reward(
            bg_after=bg_after, tt_after=tt_after, aa_after=aa_after,
            delta_applied=d_exec, max_delta=self.max_delta,
            prev_bg=bg_before, occ_mid=occ_mid, update_dual=True
        ))

        if grpo_samples is not None:
            log_grpo_row(
                grpo_samples, method=self.name, trigger="AD",
                chunk=chunk, micro=self.micro_counter, micro_global=micro_global,
                phase="executed", k=int(k_best), a=a_exec, delta=d_exec, step=self.step,
                cut_before=self.cut, cut_next=cut_next,
                cut_lo=self.lo, cut_hi=self.hi,
                bg_before=bg_before, bg_after=bg_after,
                tt_after=tt_after, aa_after=aa_after,
                occ_mid=occ_mid,
                reward_best_sample=float(cand_r[k_best]),
                reward_exec=r_exec,
                executed=1, shielded=int(shielded), kept=1
            )
        micro_global += 1

        self.cut = cut_next
        self.prev_bg = bg_after
        self.last_delta = d_exec
        self.micro_counter += 1

        if (self.micro_counter % self.train_every) == 0:
            _ = self.agent.update()

        return CtrlOut(micro_global=micro_global)


class GFPOCtrl(GRPOCtrl):
    """
    GFPO: sample G_sample, keep G_keep by filter, train on kept (reward_train optional),
          execute best kept (k_best = keep[0] for feasible_first_sig).
    """
    def __init__(self, *args, gfpo_filter="abs_err_topk", group_size_sample=32, group_size_keep=16,
                 feas_mult=1.0, mix=0.80, band_mult=1.0, sig_bonus=1.0, **kwargs):
        super().__init__(*args, group_size_keep=group_size_keep, **kwargs)
        self.gfpo_filter = str(gfpo_filter)
        self.G_sample = int(group_size_sample)
        self.G_keep = int(group_size_keep)
        self.feas_mult = float(feas_mult)
        self.mix = float(mix)
        self.band_mult = float(band_mult)
        self.sig_bonus = float(sig_bonus)

    def step_micro(self, *, chunk: int, bas_w, bnpv_w, bas_j, sas_tt, sas_aa, micro_global, grpo_samples=None):
        bg_before = float(Sing_Trigger(bas_j, self.cut))
        if self.prev_bg is None:
            self.prev_bg = bg_before

        obs = self._obs(bas_w, bnpv_w, bas_j, bg_before)
        acts, old_logps = self.agent.sample_group_actions(obs, group_size=self.G_sample, temperature=self.temperature)

        occ_mid = float(near_occupancy(bas_j, self.cut, self.near_widths)[1])

        cand_bg = np.zeros(self.G_sample, np.float32)
        cand_tt = np.zeros(self.G_sample, np.float32)
        cand_aa = np.zeros(self.G_sample, np.float32)
        cand_cut = np.zeros(self.G_sample, np.float32)
        cand_a   = np.zeros(self.G_sample, np.int32)
        cand_d   = np.zeros(self.G_sample, np.float32)
        cand_r_raw = np.zeros(self.G_sample, np.float32)
        cand_r_train = np.zeros(self.G_sample, np.float32)
        cand_abs_err = np.zeros(self.G_sample, np.float32)
        cand_sig = np.zeros(self.G_sample, np.float32)

        for k in range(self.G_sample):
            a = int(acts[k])
            dlt = float(self.deltas[a] * self.step)
            cut_next = float(np.clip(self.cut + dlt, self.lo, self.hi))
            bg_after = float(Sing_Trigger(bas_j, cut_next))
            tt_after = float(Sing_Trigger(sas_tt, cut_next))
            aa_after = float(Sing_Trigger(sas_aa, cut_next))

            r_raw = float(self.agent.compute_reward(
                bg_after=bg_after, tt_after=tt_after, aa_after=aa_after,
                delta_applied=dlt, max_delta=self.max_delta,
                prev_bg=bg_before, occ_mid=occ_mid, update_dual=False
            ))

            abs_err = abs(bg_after - self.target)
            sig_score = self.mix * tt_after + (1.0 - self.mix) * aa_after
            inband = (abs_err <= self.band_mult * self.tol)
            r_train = r_raw + self.sig_bonus * sig_score * (1.0 if inband else 0.0)

            cand_a[k] = a; cand_d[k] = dlt; cand_cut[k] = cut_next
            cand_bg[k] = bg_after; cand_tt[k] = tt_after; cand_aa[k] = aa_after
            cand_r_raw[k] = r_raw; cand_r_train[k] = r_train
            cand_abs_err[k] = abs_err; cand_sig[k] = sig_score

        # keep-set + executed
        if self.gfpo_filter == "abs_err_topk":
            keep = np.argsort(cand_abs_err)[:min(self.G_keep, self.G_sample)]
            # execute: smallest abs_err, tie-break by larger signal score
            k_best = int(keep[np.lexsort((-cand_sig[keep], cand_abs_err[keep]))][0])
        elif self.gfpo_filter == "feasible_first_sig":
            keep, _, _ = gfpo_topk_keep_indices(
                bg_after=cand_bg, tt_after=cand_tt, aa_after=cand_aa, rewards=cand_r_raw,
                target=self.target, tol=self.tol, feas_mult=self.feas_mult,
                mix=self.mix, k_keep=min(self.G_keep, self.G_sample)
            )
            k_best = int(keep[0])
        else:
            raise ValueError(f"Unknown GFPO filter {self.gfpo_filter}")

        keep = keep.astype(np.int64)
        keep_set = set(int(x) for x in keep.tolist())

        if grpo_samples is not None:
            for k in range(self.G_sample):
                log_grpo_row(
                    grpo_samples, method=self.name, trigger="AD",
                    chunk=chunk, micro=self.micro_counter, micro_global=micro_global,
                    phase="candidate", k=k,
                    a=int(cand_a[k]), delta=float(cand_d[k]), step=self.step,
                    cut_before=self.cut, cut_next=float(cand_cut[k]),
                    cut_lo=self.lo, cut_hi=self.hi,
                    bg_before=bg_before, bg_after=float(cand_bg[k]),
                    tt_after=float(cand_tt[k]), aa_after=float(cand_aa[k]),
                    occ_mid=occ_mid,
                    reward_raw=float(cand_r_raw[k]), reward_train=float(cand_r_train[k]),
                    executed=0, shielded=0, kept=int(k in keep_set)
                )
                micro_global += 1

        # train kept-only (use reward_train)
        self.agent.store_group(
            obs=obs,
            actions=acts[keep],
            logp=old_logps[keep],
            rewards=cand_r_train[keep],
            baseline="mean"
        )

        # execute best (shield)
        a_exec = int(acts[k_best])
        d_exec = float(self.deltas[a_exec] * self.step)
        sd = shield_delta(bg_before, self.target, self.tol, self.max_delta)
        shielded = (sd is not None)
        if sd is not None:
            d_exec = float(sd)

        cut_next = float(np.clip(self.cut + d_exec, self.lo, self.hi))
        bg_after = float(Sing_Trigger(bas_j, cut_next))
        tt_after = float(Sing_Trigger(sas_tt, cut_next))
        aa_after = float(Sing_Trigger(sas_aa, cut_next))

        r_exec = float(self.agent.compute_reward(
            bg_after=bg_after, tt_after=tt_after, aa_after=aa_after,
            delta_applied=d_exec, max_delta=self.max_delta,
            prev_bg=bg_before, occ_mid=occ_mid, update_dual=True
        ))

        if grpo_samples is not None:
            log_grpo_row(
                grpo_samples, method=self.name, trigger="AD",
                chunk=chunk, micro=self.micro_counter, micro_global=micro_global,
                phase="executed", k=int(k_best),
                a=int(a_exec), delta=float(d_exec), step=self.step,
                cut_before=self.cut, cut_next=cut_next,
                cut_lo=self.lo, cut_hi=self.hi,
                bg_before=bg_before, bg_after=bg_after,
                tt_after=tt_after, aa_after=aa_after,
                occ_mid=occ_mid,
                reward_best_sample=float(cand_r_raw[int(k_best)]),
                reward_exec=float(r_exec),
                executed=1, shielded=int(shielded), kept=1
            )
        micro_global += 1

        self.cut = cut_next
        self.prev_bg = bg_after
        self.last_delta = d_exec
        self.micro_counter += 1

        if (self.micro_counter % self.train_every) == 0:
            _ = self.agent.update()

        return CtrlOut(micro_global=micro_global)


class GRPOCtrlHT(BaseCtrl):
    """
    HT version of GRPOCtrl (same logic; uses make_event_seq_ht and HT arrays).
    """
    def __init__(self, name, init_cut, lo, hi, *, agent, deltas, step, max_delta,
                 ht_mid, ht_span, near_widths, K, target, tol, train_every, temperature, group_size_keep: int):
        self.name = name
        self.cut = float(init_cut)
        self.lo, self.hi = float(lo), float(hi)
        self.agent = agent
        self.deltas = np.asarray(deltas, np.float32)
        self.step = float(step)
        self.max_delta = float(max_delta)
        self.ht_mid, self.ht_span = float(ht_mid), float(ht_span)
        self.near_widths = near_widths
        self.K = int(K)
        self.target, self.tol = float(target), float(tol)
        self.train_every = int(train_every)
        self.temperature = float(temperature)

        self.prev_bg = None
        self.last_delta = 0.0
        self.err_i = 0.0
        self.micro_counter = 0
        self.group_size_keep = int(group_size_keep)

    def cut_value(self): return self.cut

    def _obs(self, bht_w, bnpv_w, bht_j, bg_before):
        self.err_i = update_err_i(self.err_i, bg_before, self.target)
        dbgcut = d_bg_d_cut_norm(bht_j, self.cut, self.step, self.target)
        return make_event_seq_ht(
            bht=bht_w, bnpv=bnpv_w,
            bg_rate=bg_before, prev_bg_rate=self.prev_bg,
            cut=self.cut,
            ht_mid=self.ht_mid, ht_span=self.ht_span,
            target=self.target, K=self.K,
            last_delta=self.last_delta, max_delta=self.max_delta,
            near_widths=self.near_widths,
            step=self.step, tol=self.tol,
            err_i=self.err_i, d_bg_d_cut=dbgcut
        )

    def step_micro(self, *, chunk: int, bht_w, bnpv_w, bht_j, sht_tt, sht_aa, micro_global, grpo_samples=None):
        bg_before = float(Sing_Trigger(bht_j, self.cut))
        if self.prev_bg is None:
            self.prev_bg = bg_before

        obs = self._obs(bht_w, bnpv_w, bht_j, bg_before)
        G = self.group_size_keep
        acts, old_logps = self.agent.sample_group_actions(obs, group_size=G, temperature=self.temperature)

        occ_mid = float(near_occupancy(bht_j, self.cut, self.near_widths)[1])
        cand_r = np.zeros(G, np.float32)

        for k in range(G):
            a = int(acts[k])
            dlt = float(self.deltas[a] * self.step)
            cut_next = float(np.clip(self.cut + dlt, self.lo, self.hi))
            bg_after = float(Sing_Trigger(bht_j, cut_next))
            tt_after = float(Sing_Trigger(sht_tt, cut_next))
            aa_after = float(Sing_Trigger(sht_aa, cut_next))
            r = float(self.agent.compute_reward(
                bg_after=bg_after, tt_after=tt_after, aa_after=aa_after,
                delta_applied=dlt, max_delta=self.max_delta,
                prev_bg=bg_before, occ_mid=occ_mid, update_dual=False
            ))
            cand_r[k] = r

            if grpo_samples is not None:
                log_grpo_row(
                    grpo_samples, method=self.name, trigger="HT",
                    chunk=chunk, micro=self.micro_counter, micro_global=micro_global,
                    phase="candidate", k=k, a=a, delta=dlt, step=self.step,
                    cut_before=self.cut, cut_next=cut_next, cut_lo=self.lo, cut_hi=self.hi,
                    bg_before=bg_before, bg_after=bg_after,
                    tt_after=tt_after, aa_after=aa_after,
                    occ_mid=occ_mid,
                    reward_raw=r, executed=0, shielded=0, kept=1
                )
            micro_global += 1

        # train on all candidates
        self.agent.store_group(obs=obs, actions=acts, logp=old_logps, rewards=cand_r, baseline="mean")

        k_best = int(np.argmax(cand_r))
        a_exec = int(acts[k_best])
        d_exec = float(self.deltas[a_exec] * self.step)

        sd = shield_delta(bg_before, self.target, self.tol, self.max_delta)
        shielded = (sd is not None)
        if sd is not None:
            d_exec = float(sd)

        cut_next = float(np.clip(self.cut + d_exec, self.lo, self.hi))
        bg_after = float(Sing_Trigger(bht_j, cut_next))
        tt_after = float(Sing_Trigger(sht_tt, cut_next))
        aa_after = float(Sing_Trigger(sht_aa, cut_next))

        r_exec = float(self.agent.compute_reward(
            bg_after=bg_after, tt_after=tt_after, aa_after=aa_after,
            delta_applied=d_exec, max_delta=self.max_delta,
            prev_bg=bg_before, occ_mid=occ_mid, update_dual=True
        ))

        if grpo_samples is not None:
            log_grpo_row(
                grpo_samples, method=self.name, trigger="HT",
                chunk=chunk, micro=self.micro_counter, micro_global=micro_global,
                phase="executed", k=k_best, a=a_exec, delta=d_exec, step=self.step,
                cut_before=self.cut, cut_next=cut_next, cut_lo=self.lo, cut_hi=self.hi,
                bg_before=bg_before, bg_after=bg_after,
                tt_after=tt_after, aa_after=aa_after,
                occ_mid=occ_mid,
                reward_best_sample=float(cand_r[k_best]),
                reward_exec=r_exec,
                executed=1, shielded=int(shielded), kept=1
            )
        micro_global += 1

        self.cut = cut_next
        self.prev_bg = bg_after
        self.last_delta = d_exec
        self.micro_counter += 1

        if (self.micro_counter % self.train_every) == 0:
            _ = self.agent.update()

        return CtrlOut(micro_global=micro_global)


class GRPOFilterCtrlHT(GRPOCtrlHT):
    """
    HT version of GRPOFilterCtrl.
    """
    def __init__(self, *args, band_mult=1.0, group_size_sample=32, group_size_keep=16, **kwargs):
        super().__init__(*args, **kwargs)
        self.band_mult = float(band_mult)
        self.G_sample = int(group_size_sample)
        self.G_keep = int(group_size_keep)

    def step_micro(self, *, chunk: int, bht_w, bnpv_w, bht_j, sht_tt, sht_aa, micro_global, grpo_samples=None):
        bg_before = float(Sing_Trigger(bht_j, self.cut))
        if self.prev_bg is None:
            self.prev_bg = bg_before

        obs = self._obs(bht_w, bnpv_w, bht_j, bg_before)
        acts, old_logps = self.agent.sample_group_actions(obs, group_size=self.G_sample, temperature=self.temperature)

        occ_mid = float(near_occupancy(bht_j, self.cut, self.near_widths)[1])

        cand_r = np.zeros(self.G_sample, np.float32)
        cand_bg = np.zeros(self.G_sample, np.float32)
        cand_tt = np.zeros(self.G_sample, np.float32)
        cand_aa = np.zeros(self.G_sample, np.float32)
        cand_cut = np.zeros(self.G_sample, np.float32)

        for k in range(self.G_sample):
            a = int(acts[k])
            dlt = float(self.deltas[a] * self.step)
            cut_next = float(np.clip(self.cut + dlt, self.lo, self.hi))
            bg_after = float(Sing_Trigger(bht_j, cut_next))
            tt_after = float(Sing_Trigger(sht_tt, cut_next))
            aa_after = float(Sing_Trigger(sht_aa, cut_next))
            r = float(self.agent.compute_reward(
                bg_after=bg_after, tt_after=tt_after, aa_after=aa_after,
                delta_applied=dlt, max_delta=self.max_delta,
                prev_bg=bg_before, occ_mid=occ_mid, update_dual=False
            ))
            cand_r[k] = r
            cand_bg[k] = bg_after
            cand_tt[k] = tt_after
            cand_aa[k] = aa_after
            cand_cut[k] = cut_next

        abs_err = np.abs(cand_bg - self.target)
        inband = abs_err <= (self.band_mult * self.tol)
        keep = np.where(inband)[0]
        if keep.size == 0:
            keep = np.array([int(np.argmin(abs_err))], dtype=np.int64)

        if keep.size > self.G_keep:
            keep = keep[np.argsort(-cand_r[keep])[:self.G_keep]]

        keep = keep.astype(np.int64)
        k_best = int(keep[np.argmax(cand_r[keep])])

        if grpo_samples is not None:
            keep_set = set(int(x) for x in keep.tolist())
            for k in range(self.G_sample):
                a = int(acts[k])
                dlt = float(self.deltas[a] * self.step)
                log_grpo_row(
                    grpo_samples, method=self.name, trigger="HT",
                    chunk=chunk, micro=self.micro_counter, micro_global=micro_global,
                    phase="candidate", k=k, a=a, delta=dlt, step=self.step,
                    cut_before=self.cut, cut_next=float(cand_cut[k]),
                    cut_lo=self.lo, cut_hi=self.hi,
                    bg_before=bg_before, bg_after=float(cand_bg[k]),
                    tt_after=float(cand_tt[k]), aa_after=float(cand_aa[k]),
                    occ_mid=occ_mid,
                    reward_raw=float(cand_r[k]), executed=0, shielded=0,
                    kept=int(k in keep_set)
                )
                micro_global += 1

        self.agent.store_group(
            obs=obs,
            actions=acts[keep],
            logp=old_logps[keep],
            rewards=cand_r[keep],
            baseline="mean"
        )

        a_exec = int(acts[k_best])
        d_exec = float(self.deltas[a_exec] * self.step)
        sd = shield_delta(bg_before, self.target, self.tol, self.max_delta)
        shielded = (sd is not None)
        if sd is not None:
            d_exec = float(sd)

        cut_next = float(np.clip(self.cut + d_exec, self.lo, self.hi))
        bg_after = float(Sing_Trigger(bht_j, cut_next))
        tt_after = float(Sing_Trigger(sht_tt, cut_next))
        aa_after = float(Sing_Trigger(sht_aa, cut_next))

        r_exec = float(self.agent.compute_reward(
            bg_after=bg_after, tt_after=tt_after, aa_after=aa_after,
            delta_applied=d_exec, max_delta=self.max_delta,
            prev_bg=bg_before, occ_mid=occ_mid, update_dual=True
        ))

        if grpo_samples is not None:
            log_grpo_row(
                grpo_samples, method=self.name, trigger="HT",
                chunk=chunk, micro=self.micro_counter, micro_global=micro_global,
                phase="executed", k=int(k_best), a=a_exec, delta=d_exec, step=self.step,
                cut_before=self.cut, cut_next=cut_next,
                cut_lo=self.lo, cut_hi=self.hi,
                bg_before=bg_before, bg_after=bg_after,
                tt_after=tt_after, aa_after=aa_after,
                occ_mid=occ_mid,
                reward_best_sample=float(cand_r[k_best]),
                reward_exec=r_exec,
                executed=1, shielded=int(shielded), kept=1
            )
        micro_global += 1

        self.cut = cut_next
        self.prev_bg = bg_after
        self.last_delta = d_exec
        self.micro_counter += 1

        if (self.micro_counter % self.train_every) == 0:
            _ = self.agent.update()

        return CtrlOut(micro_global=micro_global)


class GFPOCtrlHT(GRPOCtrlHT):
    """
    HT version of GFPOCtrl.
    """
    def __init__(self, *args, gfpo_filter="abs_err_topk", group_size_sample=32, group_size_keep=16,
                 feas_mult=1.0, mix=0.80, band_mult=1.0, sig_bonus=1.0, **kwargs):
        super().__init__(*args, group_size_keep=group_size_keep, **kwargs)
        self.gfpo_filter = str(gfpo_filter)
        self.G_sample = int(group_size_sample)
        self.G_keep = int(group_size_keep)
        self.feas_mult = float(feas_mult)
        self.mix = float(mix)
        self.band_mult = float(band_mult)
        self.sig_bonus = float(sig_bonus)

    def step_micro(self, *, chunk: int, bht_w, bnpv_w, bht_j, sht_tt, sht_aa, micro_global, grpo_samples=None):
        bg_before = float(Sing_Trigger(bht_j, self.cut))
        if self.prev_bg is None:
            self.prev_bg = bg_before

        obs = self._obs(bht_w, bnpv_w, bht_j, bg_before)
        acts, old_logps = self.agent.sample_group_actions(obs, group_size=self.G_sample, temperature=self.temperature)

        occ_mid = float(near_occupancy(bht_j, self.cut, self.near_widths)[1])

        cand_bg = np.zeros(self.G_sample, np.float32)
        cand_tt = np.zeros(self.G_sample, np.float32)
        cand_aa = np.zeros(self.G_sample, np.float32)
        cand_cut = np.zeros(self.G_sample, np.float32)
        cand_a   = np.zeros(self.G_sample, np.int32)
        cand_d   = np.zeros(self.G_sample, np.float32)
        cand_r_raw = np.zeros(self.G_sample, np.float32)
        cand_r_train = np.zeros(self.G_sample, np.float32)
        cand_abs_err = np.zeros(self.G_sample, np.float32)
        cand_sig = np.zeros(self.G_sample, np.float32)

        for k in range(self.G_sample):
            a = int(acts[k])
            dlt = float(self.deltas[a] * self.step)
            cut_next = float(np.clip(self.cut + dlt, self.lo, self.hi))
            bg_after = float(Sing_Trigger(bht_j, cut_next))
            tt_after = float(Sing_Trigger(sht_tt, cut_next))
            aa_after = float(Sing_Trigger(sht_aa, cut_next))

            r_raw = float(self.agent.compute_reward(
                bg_after=bg_after, tt_after=tt_after, aa_after=aa_after,
                delta_applied=dlt, max_delta=self.max_delta,
                prev_bg=bg_before, occ_mid=occ_mid, update_dual=False
            ))

            abs_err = abs(bg_after - self.target)
            sig_score = self.mix * tt_after + (1.0 - self.mix) * aa_after
            inband = (abs_err <= self.band_mult * self.tol)
            r_train = r_raw + self.sig_bonus * sig_score * (1.0 if inband else 0.0)

            cand_a[k] = a; cand_d[k] = dlt; cand_cut[k] = cut_next
            cand_bg[k] = bg_after; cand_tt[k] = tt_after; cand_aa[k] = aa_after
            cand_r_raw[k] = r_raw; cand_r_train[k] = r_train
            cand_abs_err[k] = abs_err; cand_sig[k] = sig_score

        if self.gfpo_filter == "abs_err_topk":
            keep = np.argsort(cand_abs_err)[:min(self.G_keep, self.G_sample)]
            k_best = int(keep[np.lexsort((-cand_sig[keep], cand_abs_err[keep]))][0])
        elif self.gfpo_filter == "feasible_first_sig":
            keep, _, _ = gfpo_topk_keep_indices(
                bg_after=cand_bg, tt_after=cand_tt, aa_after=cand_aa, rewards=cand_r_raw,
                target=self.target, tol=self.tol, feas_mult=self.feas_mult,
                mix=self.mix, k_keep=min(self.G_keep, self.G_sample)
            )
            k_best = int(keep[0])
        else:
            raise ValueError(f"Unknown GFPO filter {self.gfpo_filter}")

        keep = keep.astype(np.int64)
        keep_set = set(int(x) for x in keep.tolist())

        if grpo_samples is not None:
            for k in range(self.G_sample):
                log_grpo_row(
                    grpo_samples, method=self.name, trigger="HT",
                    chunk=chunk, micro=self.micro_counter, micro_global=micro_global,
                    phase="candidate", k=k,
                    a=int(cand_a[k]), delta=float(cand_d[k]), step=self.step,
                    cut_before=self.cut, cut_next=float(cand_cut[k]),
                    cut_lo=self.lo, cut_hi=self.hi,
                    bg_before=bg_before, bg_after=float(cand_bg[k]),
                    tt_after=float(cand_tt[k]), aa_after=float(cand_aa[k]),
                    occ_mid=occ_mid,
                    reward_raw=float(cand_r_raw[k]), reward_train=float(cand_r_train[k]),
                    executed=0, shielded=0, kept=int(k in keep_set)
                )
                micro_global += 1

        self.agent.store_group(
            obs=obs,
            actions=acts[keep],
            logp=old_logps[keep],
            rewards=cand_r_train[keep],
            baseline="mean"
        )

        a_exec = int(acts[k_best])
        d_exec = float(self.deltas[a_exec] * self.step)
        sd = shield_delta(bg_before, self.target, self.tol, self.max_delta)
        shielded = (sd is not None)
        if sd is not None:
            d_exec = float(sd)

        cut_next = float(np.clip(self.cut + d_exec, self.lo, self.hi))
        bg_after = float(Sing_Trigger(bht_j, cut_next))
        tt_after = float(Sing_Trigger(sht_tt, cut_next))
        aa_after = float(Sing_Trigger(sht_aa, cut_next))

        r_exec = float(self.agent.compute_reward(
            bg_after=bg_after, tt_after=tt_after, aa_after=aa_after,
            delta_applied=d_exec, max_delta=self.max_delta,
            prev_bg=bg_before, occ_mid=occ_mid, update_dual=True
        ))

        if grpo_samples is not None:
            log_grpo_row(
                grpo_samples, method=self.name, trigger="HT",
                chunk=chunk, micro=self.micro_counter, micro_global=micro_global,
                phase="executed", k=int(k_best),
                a=int(a_exec), delta=float(d_exec), step=self.step,
                cut_before=self.cut, cut_next=cut_next,
                cut_lo=self.lo, cut_hi=self.hi,
                bg_before=bg_before, bg_after=bg_after,
                tt_after=tt_after, aa_after=aa_after,
                occ_mid=occ_mid,
                reward_best_sample=float(cand_r_raw[int(k_best)]),
                reward_exec=float(r_exec),
                executed=1, shielded=int(shielded), kept=1
            )
        micro_global += 1

        self.cut = cut_next
        self.prev_bg = bg_after
        self.last_delta = d_exec
        self.micro_counter += 1

        if (self.micro_counter % self.train_every) == 0:
            _ = self.agent.update()

        return CtrlOut(micro_global=micro_global)

def update_err_i(err_i, bg_rate, target, lam=0.95):
    e = (float(bg_rate) - float(target)) / max(float(target), 1e-6)
    return float(lam) * float(err_i) + (1.0 - float(lam)) * float(e)

# ---- GLOBAL chunk-level log store (used by log_chunk_stats / write_chunk_stats_csv) ----
chunk_rows = []

def inband_eff_by_method(chunk_rows, trigger):
    """
    Returns dict: method -> {"tt": mean_tt_inband, "h_to_4b": mean_aa_inband}
    Robust to trigger labels: AD may appear as "AD" or "AS".
    """
    if trigger == "AD":
        trig_ok = {"AD", "AS"}
    else:
        trig_ok = {trigger}

    acc = defaultdict(lambda: {"tt": [], "h_to_4b": []})

    for r in chunk_rows:
        tr = str(r.get("trigger", ""))
        if tr not in trig_ok:
            continue
        if int(r.get("inband", 0)) != 1:
            continue

        m = str(r.get("method", "UNK"))
        acc[m]["tt"].append(float(r.get("tt", np.nan)))
        acc[m]["h_to_4b"].append(float(r.get("aa", np.nan)))  # aa == h→4b

    out = {}
    for m, d in acc.items():
        tt = np.asarray(d["tt"], dtype=np.float64)
        aa = np.asarray(d["h_to_4b"], dtype=np.float64)

        tt = tt[np.isfinite(tt)]
        aa = aa[np.isfinite(aa)]

        out[m] = {
            "tt": float(np.mean(tt)) if tt.size else np.nan,
            "h_to_4b": float(np.mean(aa)) if aa.size else np.nan,
        }
    return out
def d_bg_d_cut_norm(scores, cut, step, target):
    # normalized derivative: (d bg_rate / d cut) / target
    step = float(step)
    if step <= 0:
        return 0.0
    p_plus  = float(Sing_Trigger(scores, float(cut) + step))
    p_minus = float(Sing_Trigger(scores, float(cut) - step))
    dp_dcut = (p_plus - p_minus) / (2.0 * step)  # typically negative
    return float(dp_dcut) / max(float(target), 1e-6)
def _group_advantages_from_samples(samples, *, trigger, method,
                                  baseline="mean",
                                  reward_key="reward_raw",
                                  kept_only=False,
                                  eps=1e-8):
    """
    Reconstruct per-micro-step group-relative advantages.

    For GRPO: kept_only=False, reward_key="reward_raw" (matches store_group)
    For GFPO: kept_only=True,  reward_key="reward_train" (matches store_group)
    """
    # Filter rows for this trigger
    rows = [
        r for r in samples
        if r.get("trigger") == trigger and r.get("method", "GRPO") == method
    ]
    # Group candidate rewards by micro-step
    cand_by_micro = {}
    exec_by_micro = {}  # store executed row (reward_exec)
    for r in rows:
        micro = int(r["micro"])
        if r["phase"] == "candidate":
            if kept_only and int(r.get("kept", 0)) != 1:
                continue
            rr = r.get(reward_key, None)
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

        b = float(np.median(rs)) if baseline == "median" else float(np.mean(rs))
        s = float(np.std(rs))

        total_groups += 1
        if s < 1e-12:
            vanish_groups += 1
            s = 0.0

        adv = rs - b
        adv_raw_all.extend(adv.tolist())
        adv_norm_all.extend((adv / (s + eps)).tolist() if s > 0 else np.zeros_like(adv).tolist())

        # Executed advantage (compare executed reward to candidate baseline)
        if micro in exec_by_micro:
            re = float(exec_by_micro[micro])
            ae = re - b
            adv_raw_exec.append(ae)
            adv_norm_exec.append(ae / (s + eps) if s > 0 else 0.0)

    frac_vanish = (vanish_groups / max(1, total_groups))
    return adv_raw_all, adv_norm_all, adv_raw_exec, adv_norm_exec, frac_vanish

def _make_edges(x, lo_q=0.5, hi_q=99.5, nbins=80):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    lo = float(np.percentile(x, lo_q))
    hi = float(np.percentile(x, hi_q))
    if not (hi > lo):
        hi = lo + 1.0
    return np.linspace(lo, hi, int(nbins) + 1)

def _score_chunk_stats(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return dict(mean=np.nan, p05=np.nan, p50=np.nan, p95=np.nan)
    return dict(
        mean=float(np.mean(x)),
        p05=float(np.percentile(x, 5)),
        p50=float(np.percentile(x, 50)),
        p95=float(np.percentile(x, 95)),
    )

def plot_inband_eff_grouped_by_trigger(eff_ad, eff_ht, *, signal_key, signal_label,
                                       outpath, run_label,
                                       trigger_order=("HT", "AD")):
    """
    Grouped bars like the CMS figure:
      x-axis: triggers (AD Trigger, HT Trigger)
      bars within each group: methods (Constant, PID, ADT, DQN, GRPO, GFPO-F, GFPO-FR)

    eff_ad/eff_ht: dict method -> {"tt": val, "h_to_4b": val}
    signal_key: "tt" or "h_to_4b"
    """
    # which methods exist in either trigger
    methods = [m for m in PLOT_METHODS if (m in eff_ad) or (m in eff_ht)]
    if not methods:
        return

    # trigger groups
    trig_map = {"AD": eff_ad, "HT": eff_ht}
    triggers = [t for t in trigger_order if t in trig_map]
    if not triggers:
        return

    # values: shape (T, M)
    vals = np.zeros((len(triggers), len(methods)), dtype=np.float64)
    for ti, tr in enumerate(triggers):
        eff = trig_map[tr]
        for mi, m in enumerate(methods):
            vals[ti, mi] = float(eff.get(m, {}).get(signal_key, np.nan))

    x = np.arange(len(triggers), dtype=np.float64)
    bw = 0.80 / max(1, len(methods))  # fill 80% of group width

    fig, ax = plt.subplots(figsize=(10, 5.6))

    # bars (one legend entry per method)
    for mi, m in enumerate(methods):
        ax.bar(
            x - 0.40 + (mi + 0.5) * bw,
            vals[:, mi],
            width=bw,
            label=m,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{tr} Trigger" for tr in triggers])
    ax.set_ylabel(f"In-band efficiency ({signal_label})")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    # start y-axis at 80 for ttbar
    if signal_key == "tt":
        ax.set_ylim(bottom=85)          # keep top auto
        # or: ax.set_ylim(80, 100)       # if want fixed top
    else:
        ax.set_ylim(bottom=15)           # keep top auto
    # legend is methods 
    small_legend(ax, loc="best", ncol=1)

    add_cms_header(fig, run_label=run_label)
    finalize_diag_fig(fig)
    save_png(fig, str(outpath))
    plt.close(fig)


def _plot_score_density_heatmap(time, hists, edges, *, title, outpath, run_label):
    """
    hists: shape (T, nbins) where nbins = len(edges)-1, density per chunk
    edges: bin edges (len = nbins+1)
    """
    H = np.asarray(hists, dtype=np.float64)
    if H.size == 0:
        return

    # transpose so y-axis is score
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    im = ax.imshow(
        H.T,
        origin="lower",
        aspect="auto",
        extent=[float(time[0]), float(time[-1]), float(edges[0]), float(edges[-1])],
        interpolation="nearest",
    )
    ax.set_xlabel("Time (Fraction of Run)")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="Density")
    add_cms_header(fig, run_label=run_label)
    finalize_diag_fig(fig)
    save_png(fig, str(outpath))
    plt.close(fig)

def _plot_score_summary(time, stats_list, *, title, outpath, run_label):
    """
    stats_list: list of dicts with keys mean, p05, p50, p95 (one per chunk)
    """
    if not stats_list:
        return
    mean = np.array([s["mean"] for s in stats_list], dtype=np.float64)
    p05  = np.array([s["p05"]  for s in stats_list], dtype=np.float64)
    p50  = np.array([s["p50"]  for s in stats_list], dtype=np.float64)
    p95  = np.array([s["p95"]  for s in stats_list], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.plot(time, mean, linewidth=2.2, label="Mean")
    ax.plot(time, p50,  linewidth=2.2, linestyle="--", label="Median (p50)")
    ax.fill_between(time, p05, p95, alpha=0.15, label="p05–p95 band")
    ax.set_xlabel("Time (Fraction of Run)")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.5)
    small_legend(ax, loc="best")
    add_cms_header(fig, run_label=run_label)
    finalize_diag_fig(fig)
    save_png(fig, str(outpath))
    plt.close(fig)

def _plot_adv_compare_ecdf(x_grpo, x_gfpo, *, title, outpath, run_label):
    x_grpo = np.asarray(x_grpo, dtype=np.float64); x_grpo = x_grpo[np.isfinite(x_grpo)]
    x_gfpo = np.asarray(x_gfpo, dtype=np.float64); x_gfpo = x_gfpo[np.isfinite(x_gfpo)]
    if x_grpo.size == 0 and x_gfpo.size == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 5.2))
    if x_grpo.size:
        xs, ys = ecdf(x_grpo)
        ax.plot(xs, ys, linewidth=2.2, label="GRPO (candidates)")
    if x_gfpo.size:
        xs, ys = ecdf(x_gfpo)
        ax.plot(xs, ys, linewidth=2.2, linestyle=(0, (4, 2)), label="GFPO (kept candidates)")

    ax.set_xlabel(r"Normalized advantage  $\hat A$")
    ax.set_ylabel("CDF")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_title(title)
    small_legend(ax, loc="best")
    add_cms_header(fig, run_label=run_label)
    finalize_diag_fig(fig)
    save_png(fig, str(outpath))
    plt.close(fig)


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

from collections import defaultdict

def _run_tag(args, target, tol):
    # safe filename tag (no dots)
    def f(x): return str(x).replace(".", "p")
    return f"{args.control}_asdim{args.as_dim}_t{f(target)}_tol{f(tol)}"

def build_series_from_chunk_rows(chunk_rows, trigger):
    """
    Returns dict: method -> dict of arrays (sorted by chunk)
    """
    from collections import defaultdict
    by_method = defaultdict(list)
    for r in chunk_rows:
        if r.get("trigger") == trigger:
            by_method[r.get("method")].append(r)

    out = {}
    for m, rows in by_method.items():
        rows = sorted(rows, key=lambda x: int(x["chunk"]))
        out[m] = dict(
            chunk=np.array([rr["chunk"] for rr in rows], dtype=np.int64),
            bg_pct=np.array([rr["bg_pct"] for rr in rows], dtype=np.float64),
            bg_khz=np.array([rr["bg_khz"] for rr in rows], dtype=np.float64),
            cut=np.array([rr["cut"] for rr in rows], dtype=np.float64),
            tt=np.array([rr["tt"] for rr in rows], dtype=np.float64),
            aa=np.array([rr["aa"] for rr in rows], dtype=np.float64),
            occ_mid=np.array([rr["occ_mid"] for rr in rows], dtype=np.float64),
            inband=np.array([rr["inband"] for rr in rows], dtype=bool),
        )
    return out

# ----------------------------- legend styling -----------------------------
LEGEND_FONTSIZE = 13
LEGEND_TITLE_FONTSIZE = 13

def small_legend(ax, *, title=None, loc="best", ncol=1, **kwargs):
    """
    Consistent compact legend across all plots.
    """
    if "fontsize" not in kwargs:
        kwargs["fontsize"] = LEGEND_FONTSIZE
    if title and ("title_fontsize" not in kwargs):
        kwargs["title_fontsize"] = LEGEND_TITLE_FONTSIZE
    return ax.legend(
        loc=loc,
        frameon=True,
        title=title,
        ncol=ncol,
        handlelength=1.6,
        handletextpad=0.4,
        labelspacing=0.25,
        borderpad=0.30,
        columnspacing=0.8,
        markerscale=0.9,
        **kwargs,
    )

def make_original_plots_for_trigger(series, *, trigger_name, fixed_cut, target, tol, plots_dir, run_label, w=3):
    if not series:
        return
    # Keep ONLY: PID, DQN, GRPO, GFPO-F, GFPO-FR (in this order)
    series = select_plot_methods(series)
    if not series:
        return
    # 1) CDF of |rate error| (kHz)
    target_khz = float(target) * RATE_SCALE_KHZ
    tol_khz    = float(tol)    * RATE_SCALE_KHZ
    rate_khz_by_method = {m: s["bg_khz"] for m, s in series.items()}
    plot_cdf_abs_err_multi(
        rate_khz_by_method=rate_khz_by_method,
        target_khz=target_khz, tol_khz=tol_khz,
        title=f"{trigger_name} Trigger",
        outpath=plots_dir / f"cdf_abs_err_{trigger_name.lower()}",
        run_label=run_label,
    )

    # 2) Running in-band fraction vs time
    inband_by_method = {m: s["inband"] for m, s in series.items()}
    max_len = max(len(s["inband"]) for s in series.values())
    time_ref = np.linspace(0.0, 1.0, max_len)
    plot_running_inband_multi(
        time=time_ref,
        inband_by_method=inband_by_method,
        w=int(w),
        title=f"{trigger_name} Trigger",
        outpath=plots_dir / f"running_inband_{trigger_name.lower()}",
        run_label=run_label,
    )

    # 3) Cut-step magnitude histogram |Δcut|
    cut_by_method = {m: s["cut"] for m, s in series.items()}
    plot_cut_step_hist_multi(
        cut_by_method=cut_by_method,
        xlabel=r"$|\Delta \mathrm{cut}|$",
        title=f"{trigger_name} Trigger",
        outpath=plots_dir / f"cut_step_hist_{trigger_name.lower()}",
        run_label=run_label,
        raw=True,
        use_abs=False,
        max_points=8000,
    )

    # 5) Rate + cut time-series (“core plots”)
    plot_rate_from_series(
        series,
        target=target, tol=tol,
        title=f"{trigger_name} Trigger",
        outpath=plots_dir / f"rate_{trigger_name.lower()}",
        run_label=run_label,
    )
    plot_cut_from_series(
        series,
        fixed_cut=fixed_cut,
        ylabel=f"{trigger_name}_cut",
        title=f"{trigger_name} Cut",
        outpath=plots_dir / f"cut_{trigger_name.lower()}",
        run_label=run_label,
    )

def plot_rate_from_series(series_by_method, *, target, tol, title, outpath, run_label):
    if not series_by_method:
        return

    # choose a reference length for x-axis
    max_len = max(len(v["bg_khz"]) for v in series_by_method.values())
    time_ref = np.linspace(0.0, 1.0, max_len)

    target_khz = float(target) * RATE_SCALE_KHZ
    tol_khz    = float(tol)    * RATE_SCALE_KHZ

    fig, ax = plt.subplots(figsize=(10, 6))
    for method, s in select_plot_methods(series_by_method).items():
        y = s["bg_khz"]
        t = np.linspace(0.0, 1.0, len(y))
        ax.plot(t, y, linewidth=2.4, label=method)

    ax.axhline(target_khz + tol_khz, linestyle="--", linewidth=1.2)
    ax.axhline(target_khz - tol_khz, linestyle="--", linewidth=1.2)
    ax.fill_between(time_ref, target_khz - tol_khz, target_khz + tol_khz, alpha=0.12, label="Tolerance band")

    ax.set_xlabel("Time (Fraction of Run)")
    ax.set_ylabel("Background rate [kHz]")
    ax.set_ylim(0, 200)
    ax.grid(True, linestyle="--", alpha=0.5)
    small_legend(ax, loc="best", title=title)
    add_cms_header(fig, run_label=run_label)
    finalize_diag_fig(fig)
    save_png(fig, str(outpath))
    plt.close(fig)

def plot_cut_from_series(series_by_method, *, fixed_cut, ylabel, title, outpath, run_label):
    if not series_by_method:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    for method, s in select_plot_methods(series_by_method).items():
        y = s["cut"]
        t = np.linspace(0.0, 1.0, len(y))
        ax.plot(t, y, linewidth=2.4, label=method)

    ax.axhline(float(fixed_cut), color="gray", linestyle="--", linewidth=1.5, label="fixed")
    ax.set_xlabel("Time (Fraction of Run)")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.5)
    small_legend(ax, loc="best", title=title)
    add_cms_header(fig, run_label=run_label)
    finalize_diag_fig(fig)
    save_png(fig, str(outpath))
    plt.close(fig)

def log_grpo_row(rows, *, method="GRPO",
                trigger, chunk, micro, micro_global, phase,
                k, a, delta, step,
                cut_before, cut_next, cut_lo, cut_hi,
                bg_before, bg_after,
                tt_after, aa_after,
                occ_mid,
                reward_raw=None, reward_train=None, reward_best_sample=None, reward_exec=None,
                executed=0, shielded=0,
                kept=0):
    rows.append({
        "method": str(method),            # "GRPO" or "GFPO"
        "trigger": str(trigger),          # "AS" or "HT"
        "chunk": int(chunk),
        "micro": int(micro),
        "micro_global": int(micro_global),
        "phase": str(phase),              # "candidate" or "executed"

        "k": int(k),
        "a": int(a),
        "delta": float(delta),
        "step": float(step),

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
        "reward_train": (None if reward_train is None else float(reward_train)),
        "reward_best_sample": (None if reward_best_sample is None else float(reward_best_sample)),
        "reward_exec": (None if reward_exec is None else float(reward_exec)),

        "executed": int(executed),
        "shielded": int(shielded),
        "kept": int(kept),                # only meaningful for GFPO candidates
    })


def log_chunk_stats(*, chunk, trigger, method, cut, bg_pct, tt, aa, occ_mid, target, tol,
                    tp=None, fp=None, tn=None, fn=None,
                    tpr=None, fpr=None, precision=None, f1=None,
                    tp_tt = None, fn_tt = None,
                    tp_h4b = None, fn_h4b = None,
                    tpr_tt=None, precision_tt=None, f1_tt=None,
                    tpr_h4b=None, precision_h4b=None, f1_h4b=None):
    # we only care about in band rates for signal efficiency
    bg_khz = float(bg_pct) * RATE_SCALE_KHZ
    target_khz = float(target) * RATE_SCALE_KHZ
    tol_khz = float(tol) * RATE_SCALE_KHZ
    abs_err_khz = abs(bg_khz - target_khz)
    inband = int(abs(float(bg_pct) - float(target)) <= float(tol))
    
    # mask signal efficiencies by inband
    # tt_inband = (None if tt is None else (float(tt) if inband else None))
    # aa_inband = (None if aa is None else (float(aa) if inband else None))

    def mask_if_outband(x, cast=float):
        """Return cast(x) if inband else None. Preserves 0.0 correctly."""
        if x is None:
            return None
        return cast(x) if inband else None

    chunk_rows.append(dict(
        # always log control / rate stats
        chunk=int(chunk),
        trigger=str(trigger),
        method=str(method),
        cut=float(cut),
        bg_pct=float(bg_pct),
        bg_khz=float(bg_khz),
        abs_err_khz=float(abs_err_khz),
        inband=int(inband),
        occ_mid=float(occ_mid),

        # ONLY log signal stats if inband
        tt=mask_if_outband(tt, float),
        aa=mask_if_outband(aa, float),

        tp=mask_if_outband(tp, int),
        fp=mask_if_outband(fp, int),
        tn=mask_if_outband(tn, int),
        fn=mask_if_outband(fn, int),

        tpr=mask_if_outband(tpr, float),
        fpr=mask_if_outband(fpr, float),
        precision=mask_if_outband(precision, float),
        f1=mask_if_outband(f1, float),

        tp_tt=mask_if_outband(tp_tt, int),
        fn_tt=mask_if_outband(fn_tt, int),
        tp_h4b=mask_if_outband(tp_h4b, int),
        fn_h4b=mask_if_outband(fn_h4b, int),

        tpr_tt=mask_if_outband(tpr_tt, float),
        precision_tt=mask_if_outband(precision_tt, float),
        f1_tt=mask_if_outband(f1_tt, float),

        tpr_h4b=mask_if_outband(tpr_h4b, float),
        precision_h4b=mask_if_outband(precision_h4b, float),
        f1_h4b=mask_if_outband(f1_h4b, float),
    ))
    # chunk_rows.append(dict(
    #     chunk=int(chunk),
    #     trigger=str(trigger),     # "AS" or "HT"
    #     method=str(method),
    #     cut=float(cut),
    #     bg_pct=float(bg_pct),
    #     bg_khz=float(bg_khz),
    #     abs_err_khz=float(abs_err_khz),
    #     inband=int(inband),
    #     tt=float(tt_inband) if tt_inband else None, #make it inband ttbar
    #     aa=float(aa_inband) if aa_inband else None, #make it inband aa
    #     occ_mid=float(occ_mid),

    #     tp=(None if tp is None else int(tp)),
    #     fp=(None if fp is None else int(fp)),
    #     tn=(None if tn is None else int(tn)),
    #     fn=(None if fn is None else int(fn)),
    #     tpr=(None if tpr is None else float(tpr)),
    #     fpr=(None if fpr is None else float(fpr)),
    #     precision=(None if precision is None else float(precision)),
    #     f1=(None if f1 is None else float(f1)),
    #     tp_tt = (None if tp_tt is None else int(tp_tt)),
    #     fn_tt = (None if fn_tt is None else int(fn_tt)),
    #     tp_h4b = (None if tp_h4b is None else int(tp_h4b)),
    #     fn_h4b = (None if fn_h4b is None else int(fn_h4b)),

    #     tpr_tt=(None if tpr_tt is None else float(tpr_tt)),
    #     precision_tt=(None if precision_tt is None else float(precision_tt)),
    #     f1_tt=(None if f1_tt is None else float(f1_tt)),

    #     tpr_h4b=(None if tpr_h4b is None else float(tpr_h4b)),
    #     precision_h4b=(None if precision_h4b is None else float(precision_h4b)),
    #     f1_h4b=(None if f1_h4b is None else float(f1_h4b)),
    # ))

def write_chunk_stats_csv(path: Path):
    if not chunk_rows:
        return
    cols = ["chunk","trigger","method","cut","bg_pct","bg_khz","abs_err_khz","inband","tt","aa","occ_mid",
            "tp","fp","tn","fn","tpr","fpr","precision","f1","tp_tt","fn_tt","tp_h4b","fn_h4b",
            "tpr_tt","precision_tt","f1_tt",
            "tpr_h4b","precision_h4b","f1_h4b"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(chunk_rows)

        
def _safe_mean(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if x.size else np.nan

def _safe_pctl(x, p):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(np.percentile(x, p)) if x.size else np.nan

def _window_rows(chunk_rows, *, trigger, c_lo, c_hi):
    return [
        r for r in chunk_rows
        if (r.get("trigger") == trigger)
        and (c_lo <= int(r.get("chunk", -1)) <= c_hi)
    ]

def _summarize_window(rows, *, target_pct, tol_pct):
    """
    rows: list of chunk_rows dicts for a single (trigger, method) within a chunk window
    Returns dict of window-level aggregated stats.
    """
    if not rows:
        return None

    bg_pct = np.array([r["bg_pct"] for r in rows], dtype=np.float64)
    bg_khz = np.array([r["bg_khz"] for r in rows], dtype=np.float64)
    cut    = np.array([r["cut"]    for r in rows], dtype=np.float64)
    tt     = np.array([r["tt"]     for r in rows], dtype=np.float64)
    aa     = np.array([r["aa"]     for r in rows], dtype=np.float64)
    occ    = np.array([r["occ_mid"] for r in rows], dtype=np.float64)

    err_pct = bg_pct - float(target_pct)
    abs_err_pct = np.abs(err_pct)

    inband_mask = abs_err_pct <= float(tol_pct)

    tt_inband = tt[inband_mask]
    aa_inband = aa[inband_mask]

    # in kHz space
    target_khz = float(target_pct) * RATE_SCALE_KHZ
    abs_err_khz = np.abs(bg_khz - target_khz)

    # violation magnitudes in kHz (how far beyond band, not just whether)
    tol_khz = float(tol_pct) * RATE_SCALE_KHZ
    viol_up_khz = np.maximum(0.0, (bg_khz - (target_khz + tol_khz)))
    viol_dn_khz = np.maximum(0.0, ((target_khz - tol_khz) - bg_khz))

    dc = np.diff(cut) if cut.size >= 2 else np.array([], dtype=np.float64)
    step_rms = float(np.sqrt(np.mean(dc * dc))) if dc.size else 0.0


    def _sum_int(key):
        vals = [r.get(key, None) for r in rows]
        vals = [int(v) for v in vals if v is not None]
        return int(np.sum(vals)) if vals else None

    tp_sum = _sum_int("tp")
    fp_sum = _sum_int("fp")
    tn_sum = _sum_int("tn")
    fn_sum = _sum_int("fn")

    nb = (fp_sum or 0) + (tn_sum or 0)
    ns = (tp_sum or 0) + (fn_sum or 0)
    tpr = (tp_sum / ns) if (tp_sum is not None and ns > 0) else np.nan
    fpr = (fp_sum / nb) if (fp_sum is not None and nb > 0) else np.nan
    
    prec = (tp_sum / (tp_sum + fp_sum)) if (tp_sum is not None and fp_sum is not None and (tp_sum + fp_sum) > 0) else np.nan

    f1_macro = _safe_mean([r.get("f1", np.nan) for r in rows])

    return dict(
        n=int(len(rows)),
        bg_khz_mean=_safe_mean(bg_khz),
        mae_khz=_safe_mean(abs_err_khz),
        p95_abs_err_khz=_safe_pctl(abs_err_khz, 95),
        inband=float(np.mean(inband_mask)) if bg_pct.size else np.nan,
        upfrac=float(np.mean(err_pct >  float(tol_pct))) if bg_pct.size else np.nan,
        downfrac=float(np.mean(err_pct < -float(tol_pct))) if bg_pct.size else np.nan,
        violmag_khz=_safe_mean(viol_up_khz + viol_dn_khz),
        step_rms=float(step_rms),
        cut_mean=_safe_mean(cut),
        occ_mean=_safe_mean(occ),
        tt_inband=_safe_mean(tt[inband_mask]) if np.any(inband_mask) else np.nan,
        aa_inband=_safe_mean(aa[inband_mask]) if np.any(inband_mask) else np.nan,
        TP=tp_sum, FP=fp_sum, TN=tn_sum, FN=fn_sum,
        TPR=float(tpr) if np.isfinite(tpr) else np.nan,
        FPR=float(fpr) if np.isfinite(fpr) else np.nan,
        Precision=float(prec) if np.isfinite(prec) else np.nan,
        F1=float(f1_macro) if np.isfinite(f1_macro) else np.nan,  
    )

def print_every_k_chunk_stats(chunk_rows, *, trigger, c_hi, k, target_pct, tol_pct):
    c_lo = max(0, int(c_hi) - int(k) + 1)
    rows = _window_rows(chunk_rows, trigger=trigger, c_lo=c_lo, c_hi=c_hi)
    if not rows:
        return

    # group by method
    by_method = {}
    for r in rows:
        m = r.get("method", "UNK")
        by_method.setdefault(m, []).append(r)

    # enforce paper plot order
    ordered = [m for m in PLOT_METHODS if m in by_method]

    print(f"\n[{trigger}] Window chunks {c_lo}..{c_hi} (K={k})")
    print("  Method    | InBand  MAE(kHz)  P95|e|(kHz)  UpFrac  DownFrac  ViolMag(kHz)  StepRMS  tt(inband)  aa(inband)  bg_mean(kHz)  cut_mean  occ_mean | TPR FPR Precision F1")
    print("  ----------+-------------------------------------------------------------------------------------------------------------------------------------------------")

    for m in ordered:
        s = _summarize_window(by_method[m], target_pct=target_pct, tol_pct=tol_pct)
        
        rows.append({
            "Trigger": trigger,
            "Method":  m,
            "MAE": s["mae_khz"],
            "P95_abs_err": s["p95_abs_err_khz"],
            "InBand": s["inband"],
            "UpFrac": s["upfrac"],
            "DownFrac": s["downfrac"],
            "tt": s["tt_inband"],
            "h_to_4b": s["aa_inband"],
            "TP": s["TP"], "FP": s["FP"], "TN": s["TN"], "FN": s["FN"], "TPR": s["TPR"], "FPR": s["FPR"], "Precision": s["Precision"], "F1": s["F1"],
        })
        if s is None:
            continue

        def f(x, w=8, nd=3):
            if x is None or (isinstance(x, float) and not np.isfinite(x)):
                return " " * (w - 3) + "nan"
            return f"{x:{w}.{nd}f}"

        print(
            f"  {m:<9} |"
            f" (n={s['n']}) |"
            f"{f(s['inband'], w=7, nd=3)}"
            f"{f(s['mae_khz'], w=10, nd=2)}"
            f"{f(s['p95_abs_err_khz'], w=13, nd=2)}"
            f"{f(s['upfrac'], w=8, nd=3)}"
            f"{f(s['downfrac'], w=10, nd=3)}"
            f"{f(s['violmag_khz'], w=13, nd=2)}"
            f"{f(s['step_rms'], w=9, nd=3)}"
            f"{f(s['tt_inband'], w=11, nd=3)}"
            f"{f(s['aa_inband'], w=11, nd=3)}"
            f"{f(s['bg_khz_mean'], w=13, nd=1)}"
            f"{f(s['cut_mean'], w=9, nd=3)}"
            f"{f(s['occ_mean'], w=9, nd=3)}",
            f"{f(s['TPR'], w=9, nd=3)}",
            f"{f(s['FPR'], w=9, nd=3)}",
            f"{f(s['Precision'], w=11, nd=3)}",
            f"{f(s['F1'], w=8, nd=3)}",     
        )

def ecdf(x):
    """Creating error cdf"""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.array([]), np.array([])
    x = np.sort(x)
    y = (np.arange(1, x.size + 1) / x.size)
    return x, y

def _sig_score(tt, aa, mix):
    return float(mix) * float(tt) + (1.0 - float(mix)) * float(aa)

def collect_kept_candidate_arrays(samples, *, trigger, method, target, tol, mix):
    """
    Collect arrays over *kept candidates* for a given (trigger, method).
    Returns dict with abs_err, sig_score, feasible (bool).
    """
    abs_err = []
    sig = []
    feas = []
    for r in samples:
        if r.get("trigger") != trigger: 
            continue
        if r.get("method") != method:
            continue
        if r.get("phase") != "candidate":
            continue
        if int(r.get("kept", 0)) != 1:
            continue

        bg = float(r.get("bg_after", np.nan))
        tt = float(r.get("tt_after", np.nan))
        aa = float(r.get("aa_after", np.nan))
        if not np.isfinite(bg) or not np.isfinite(tt) or not np.isfinite(aa):
            continue

        ae = abs(bg - float(target))
        abs_err.append(ae)
        sig.append(_sig_score(tt, aa, mix))
        feas.append(ae <= float(tol))

    return {
        "abs_err": np.asarray(abs_err, dtype=np.float64),
        "sig": np.asarray(sig, dtype=np.float64),
        "feas": np.asarray(feas, dtype=bool),
    }

def collect_executed_arrays(samples, *, trigger, method, target, tol, mix):
    """
    Collect arrays over *executed* steps for a given (trigger, method).
    Returns dict with abs_err, sig_score, feasible (bool), shielded (0/1).
    """
    abs_err = []
    sig = []
    feas = []
    shielded = []
    for r in samples:
        if r.get("trigger") != trigger:
            continue
        if r.get("method") != method:
            continue
        if r.get("phase") != "executed":
            continue

        bg = float(r.get("bg_after", np.nan))
        tt = float(r.get("tt_after", np.nan))
        aa = float(r.get("aa_after", np.nan))
        if not np.isfinite(bg) or not np.isfinite(tt) or not np.isfinite(aa):
            continue

        ae = abs(bg - float(target))
        abs_err.append(ae)
        sig.append(_sig_score(tt, aa, mix))
        feas.append(ae <= float(tol))
        shielded.append(int(r.get("shielded", 0)))

    return {
        "abs_err": np.asarray(abs_err, dtype=np.float64),
        "sig": np.asarray(sig, dtype=np.float64),
        "feas": np.asarray(feas, dtype=bool),
        "shielded": np.asarray(shielded, dtype=np.int32),
    }

def _plot_two_hists(x1, x2, *, label1, label2, title, xlabel, outpath, run_label):
    x1 = np.asarray(x1, dtype=np.float64); x1 = x1[np.isfinite(x1)]
    x2 = np.asarray(x2, dtype=np.float64); x2 = x2[np.isfinite(x2)]
    if x1.size == 0 and x2.size == 0:
        return
    fig, ax = plt.subplots(figsize=(8, 5.2))
    if x1.size:
        ax.hist(x1, bins=60, density=True, alpha=0.55, label=label1)
    if x2.size:
        ax.hist(x2, bins=60, density=True, alpha=0.55, label=label2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.grid(True, linestyle="--", alpha=0.4)
    # ax.set_title(title)
    small_legend(ax, loc="best")
    add_cms_header(fig, run_label=run_label)
    finalize_diag_fig(fig)
    save_png(fig, str(outpath))
    plt.close(fig)

def _plot_exec_tradeoff(exec_f, exec_fr, *, title, outpath, run_label):
    """
    Scatter: abs_err (x) vs sig_score (y), executed-only. Feasible points are filled, infeasible hollow.
    """
    def scatter_one(ax, d, label):
        x = d["abs_err"]; y = d["sig"]; feas = d["feas"]
        if x.size == 0:
            return
        ax.scatter(x[~feas], y[~feas], s=18, alpha=0.55, facecolors="none", label=f"{label} (infeas)")
        ax.scatter(x[feas],  y[feas],  s=18, alpha=0.55, label=f"{label} (feas)")

    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    scatter_one(ax, exec_f,  "GFPO-F")
    scatter_one(ax, exec_fr, "GFPO-FR")
    ax.set_xlabel(r"$|bg-target|$  (percent units)")
    ax.set_ylabel(r"Signal score  $mix\cdot t\bar t + (1-mix)\cdot h\to4b$")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_title(title)
    small_legend(ax, loc="best", ncol=1)
    add_cms_header(fig, run_label=run_label)
    finalize_diag_fig(fig)
    save_png(fig, str(outpath))
    plt.close(fig)

def make_gfpo_f_vs_fr_diagnostics(grpo_samples, *, trigger, target, tol, mix, group_size_keep,
                                 plots_dir, run_label, tag):
    """
    Creates intermediate plots that *separate* GFPO-F vs GFPO-FR.
    """
    # 1) feasibility time series (candidates + kept + pad + shield)
    st_f  = compute_feasibility_micro_stats(
        grpo_samples, trigger=trigger, method="GFPO-F",
        target=target, tol=tol, group_size_keep=group_size_keep,
        requires_feasible_pad=False,   # abs_err_topk
    )
    st_fr = compute_feasibility_micro_stats(
        grpo_samples, trigger=trigger, method="GFPO-FR",
        target=target, tol=tol, group_size_keep=group_size_keep,
        requires_feasible_pad=True,    # feasible_first_sig
    )

    # plot candidate feasibility ratio over micro-steps
    if st_f is not None or st_fr is not None:
        plot_feasible_ratio_timeseries(
            stats_grpo=st_f, stats_gfpo=st_fr,
            title=f"{trigger} GFPO-F vs GFPO-FR: candidate feasibility",
            outpath=plots_dir / f"gfpoF_vs_FR_feas_ratio_{tag}_{trigger.lower()}",
            run_label=run_label
        )

        # bar summary: feasible_ratio_mean, kept_feasible_ratio_mean, pad_rate, shield_rate
        plot_feasibility_bar(
            stats_grpo=st_f, stats_gfpo=st_fr,
            title="",#f"{trigger} GFPO-F vs GFPO-FR: feasibility summary", # we don't want to have title for feasibility part.
            outpath=plots_dir / f"gfpoF_vs_FR_feas_bar_{tag}_{trigger.lower()}",
            run_label=run_label
        )

    # 2) kept-candidate distributions (abs_err and sig_score)
    kept_f  = collect_kept_candidate_arrays(grpo_samples, trigger=trigger, method="GFPO-F",
                                           target=target, tol=tol, mix=mix)
    kept_fr = collect_kept_candidate_arrays(grpo_samples, trigger=trigger, method="GFPO-FR",
                                            target=target, tol=tol, mix=mix)

    _plot_two_hists(
        kept_f["abs_err"], kept_fr["abs_err"],
        label1="GFPO-F kept", label2="GFPO-FR kept",
        title=f"{trigger}: kept candidates |bg-target|",
        xlabel=r"$|bg-target|$  (percent units)",
        outpath=plots_dir / f"gfpoF_vs_FR_kept_abs_err_{tag}_{trigger.lower()}",
        run_label=run_label
    )
    _plot_two_hists(
        kept_f["sig"], kept_fr["sig"],
        label1="GFPO-F kept", label2="GFPO-FR kept",
        title=f"{trigger}: kept candidates signal score",
        xlabel=r"$mix\cdot t\bar t + (1-mix)\cdot h\to4b$",
        outpath=plots_dir / f"gfpoF_vs_FR_kept_sig_{tag}_{trigger.lower()}",
        run_label=run_label
    )

    # 3) executed tradeoff scatter
    exec_f  = collect_executed_arrays(grpo_samples, trigger=trigger, method="GFPO-F",
                                      target=target, tol=tol, mix=mix)
    exec_fr = collect_executed_arrays(grpo_samples, trigger=trigger, method="GFPO-FR",
                                      target=target, tol=tol, mix=mix)

    _plot_exec_tradeoff(
        exec_f, exec_fr,
        title=f"{trigger}: executed tradeoff (closeness vs signal)",
        outpath=plots_dir / f"gfpoF_vs_FR_exec_tradeoff_{tag}_{trigger.lower()}",
        run_label=run_label
    )

def confusion_counts_at_cut_split(bg_scores, tt_scores, h4b_scores, cut):
    s_b   = np.asarray(bg_scores,  dtype=np.float32)
    s_tt  = np.asarray(tt_scores,  dtype=np.float32)
    s_h4b = np.asarray(h4b_scores, dtype=np.float32)

    # background counts
    fp = int(np.sum(s_b  >= cut))
    tn = int(np.sum(s_b  <  cut))
    nb = fp + tn

    # per-signal counts
    tp_tt = int(np.sum(s_tt  >= cut)) if s_tt.size else 0
    fn_tt = int(np.sum(s_tt  <  cut)) if s_tt.size else 0
    ns_tt = tp_tt + fn_tt

    tp_h4b = int(np.sum(s_h4b >= cut)) if s_h4b.size else 0
    fn_h4b = int(np.sum(s_h4b <  cut)) if s_h4b.size else 0
    ns_h4b = tp_h4b + fn_h4b

    # combined signal counts
    tp = tp_tt + tp_h4b
    fn = fn_tt + fn_h4b
    ns = ns_tt + ns_h4b

    def _safe_div(a, b):
        return (float(a) / float(b)) if (b is not None and b > 0) else np.nan

    def _f1(p, r):
        return (2.0 * p * r / (p + r)) if np.isfinite(p) and np.isfinite(r) and (p + r) > 0 else np.nan


    # rates (safe)
    tpr = float(tp / ns) if ns > 0 else np.nan
    fnr = float(fn / ns) if ns > 0 else np.nan
    fpr = float(fp / nb) if nb > 0 else np.nan
    tnr = float(tn / nb) if nb > 0 else np.nan
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0
    recall    = tp/(tp+fn) if (tp+fn)>0 else 0.0

    tpr_tt  = float(tp_tt / ns_tt)  if ns_tt  > 0 else np.nan
    tpr_h4b = float(tp_h4b / ns_h4b) if ns_h4b > 0 else np.nan

    # per-signal precision using the SAME FP(background) you already use
    # (binary view: signal vs background)
    precision_tt  = _safe_div(tp_tt,  tp_tt  + fp)
    precision_h4b = _safe_div(tp_h4b, tp_h4b + fp)

    # per-signal F1
    f1_tt  = _f1(precision_tt,  tpr_tt)
    f1_h4b = _f1(precision_h4b, tpr_h4b)

    # f1 = _f1(precision, tpr)
    f1        = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0.0

    return {
        # counts
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "tp_tt": tp_tt, "fn_tt": fn_tt,
        "tp_h4b": tp_h4b, "fn_h4b": fn_h4b,
        "nb": nb, "ns": ns, "ns_tt": ns_tt, "ns_h4b": ns_h4b,

        # rates
        "tpr": tpr, "fnr": fnr, "fpr": fpr, "tnr": tnr,
        "precision": precision, "f1": f1,

        # per-signal rates
        "tpr_tt": tpr_tt,
        "tpr_h4b": tpr_h4b,
        "precision_tt": precision_tt,
        "precision_h4b": precision_h4b,
        "f1_tt": f1_tt,
        "f1_h4b": f1_h4b,
    }



def summarize_confusion_from_chunk_rows(chunk_rows, *, trigger, method):
    """
    Return MICRO-averaged rates computed from summed counts across chunks: This is for overall rates for paper_table.tex
      TPR = sum(tp) / sum(tp+fn)
      FPR = sum(fp) / sum(fp+tn)
      TNR = sum(tn) / sum(fp+tn)
      FNR = sum(fn) / sum(tp+fn)
      Precision = sum(tp) / sum(tp+fp)
    """
    rows = [r for r in chunk_rows if r.get("trigger") == trigger and r.get("method") == method]
    if not rows:
        return {}

    tp = np.array([r.get("tp") for r in rows], dtype=np.float64)
    fp = np.array([r.get("fp") for r in rows], dtype=np.float64)
    tn = np.array([r.get("tn") for r in rows], dtype=np.float64)
    fn = np.array([r.get("fn") for r in rows], dtype=np.float64)

    ok = np.isfinite(tp) & np.isfinite(fp) & np.isfinite(tn) & np.isfinite(fn)
    if not np.any(ok):
        # fallback to any pre-logged rates if counts are missing
        tpr = np.array([r.get("tpr") for r in rows], dtype=np.float64)
        fpr = np.array([r.get("fpr") for r in rows], dtype=np.float64)
        prec = np.array([r.get("precision") for r in rows], dtype=np.float64)
        return {
            "TPR": float(np.nanmean(tpr)) if np.any(np.isfinite(tpr)) else np.nan,
            "FPR": float(np.nanmean(fpr)) if np.any(np.isfinite(fpr)) else np.nan,
            "TNR": np.nan,
            "FNR": np.nan,
            "Precision": float(np.nanmean(prec)) if np.any(np.isfinite(prec)) else np.nan,
        }

    TP = float(np.nansum(tp[ok]))
    FP = float(np.nansum(fp[ok]))
    TN = float(np.nansum(tn[ok]))
    FN = float(np.nansum(fn[ok]))

    ns = TP + FN
    nb = FP + TN

    with np.errstate(divide="ignore", invalid="ignore"):
        tpr = TP / ns if ns > 0 else np.nan
        fnr = FN / ns if ns > 0 else np.nan
        fpr = FP / nb if nb > 0 else np.nan
        tnr = TN / nb if nb > 0 else np.nan
        prec = TP / (TP + FP) if (TP + FP) > 0 else np.nan
        denom = 2 * TP + FP + FN
        f1 = (2 * TP / denom) if denom > 0 else np.nan

    return {"TPR": tpr, "FPR": fpr, "TNR": tnr, "FNR": fnr, "Precision": prec, "F1": f1}


def summarize_confusion_from_chunk_rows_split(chunk_rows, *, trigger, method):
    """
    MICRO-averaged per-signal metrics from summed counts across chunks. This is for signal break down. Two signals for confusion_tt/h4b.tex

    Returns:
      FP, TN (background)
      TP_tt, FN_tt, TPR_tt, Precision_tt Include breaking down signal
      TP_h4b, FN_h4b, TPR_h4b, Precision_h4b
      plus FPR (background acceptance)
    """
    rows = [r for r in chunk_rows if r.get("trigger") == trigger and r.get("method") == method]
    if not rows:
        return {}

    def _sum_int(key):
        vals = [rr.get(key, None) for rr in rows]
        vals = [int(v) for v in vals if v is not None]
        return int(np.sum(vals)) if vals else None

    FP = _sum_int("fp")
    TN = _sum_int("tn")
    TP_tt  = _sum_int("tp_tt")
    FN_tt  = _sum_int("fn_tt")
    TP_h4b = _sum_int("tp_h4b")
    FN_h4b = _sum_int("fn_h4b")

    nb = (FP or 0) + (TN or 0)

    def _safe_div(a, b):
        return (float(a) / float(b)) if (a is not None and b is not None and b > 0) else np.nan
    def _f1(p, r):
        return (2.0 * p * r / (p + r)) if np.isfinite(p) and np.isfinite(r) and (p + r) > 0 else np.nan

    # background rate
    FPR = _safe_div(FP, nb)

    # per-signal recall (TPR) and precision (TP/(TP+FP))
    ns_tt  = (TP_tt or 0)  + (FN_tt or 0)
    ns_h4b = (TP_h4b or 0) + (FN_h4b or 0)

    TPR_tt  = _safe_div(TP_tt,  ns_tt)
    TPR_h4b = _safe_div(TP_h4b, ns_h4b)

    Precision_tt  = _safe_div(TP_tt,  (TP_tt  or 0) + (FP or 0))
    Precision_h4b = _safe_div(TP_h4b, (TP_h4b or 0) + (FP or 0))

    F1_tt  = _f1(Precision_tt,  TPR_tt)
    F1_h4b = _f1(Precision_h4b, TPR_h4b)

    return {
        "FP": FP, "TN": TN, "FPR": FPR,
        "TP_tt": TP_tt, "FN_tt": FN_tt, "TPR_tt": TPR_tt, "Precision_tt": Precision_tt, "F1_tt": F1_tt,
        "TP_h4b": TP_h4b, "FN_h4b": FN_h4b, "TPR_h4b": TPR_h4b, "Precision_h4b": Precision_h4b, "F1_h4b": F1_h4b,
    }


def summarize_paper_table(r_pct, s_tt, s_aa, cut_hist, target_pct, tol_pct):
    """
    Paper-table metrics (matching screenshot):

      MAE↓      = mean(|r - r*|)
      P95|e|↓   = 95th percentile of |r - r*|
      InBand↑   = mean( |r-r*| <= tol )
      UpFrac↓   = mean(err >  tol)   # fraction of upward violations
      DownFrac↓ = mean(err < -tol)   # fraction of downward violations
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


    # Fractions (these relate to 1-InBand)
    out["UpFrac"]   = float(np.mean(err >  float(tol_pct))) if r.size else np.nan
    out["DownFrac"] = float(np.mean(err < -float(tol_pct))) if r.size else np.nan


    # Signal efficiencies conditioned on being in-band
    out["tt"] = safe_mean(s_tt, inband)
    out["h_to_4b"] = safe_mean(s_aa, inband)
    return out


def write_paper_table(rows, out_csv: Path, out_tex: Path, target_pct, tol_pct):
    """
    Writes:
      - CSV with columns:
        Trigger, Method, MAE, P95_abs_err, InBand, UpFrac, DownFrac, TP, FP, TN, FN, Prec., F1, tt, h_to_4b
      - LaTeX table with the same columns (and bold best-per-trigger)
    """
    if not rows:
        return

    # ---------------- CSV ----------------
    fieldnames = [
        "Trigger", "Method",
        "MAE", "P95_abs_err", "InBand", "UpFrac", "DownFrac",
        "TPR", "FPR", "TNR", "FNR", "Precision", "F1",
        "tt", "h_to_4b",
    ]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, None) for k in fieldnames})

    # ---------------- best-per-trigger ----------------
    higher_better = {"InBand", "tt", "h_to_4b", "TPR", "TNR", "Precision", "F1"}
    lower_better  = {"MAE", "P95_abs_err", "UpFrac", "DownFrac", "FPR", "FNR"}

    # Force output order
    trigger_order = ["HT", "AD"]
    method_order  = ["Constant", "PID", "ADT", "DQN", "PPO", "GRPO", "GFPO-F", "GFPO-FR"]
    trig_rank = {t: i for i, t in enumerate(trigger_order)}
    meth_rank = {m: i for i, m in enumerate(method_order)}
    def _trig_key(t): return trig_rank.get(t, 10**9)
    def _meth_key(m): return meth_rank.get(m, 10**9)

    rows = sorted(rows, key=lambda r: (_trig_key(r["Trigger"]), _meth_key(r["Method"])))

    triggers = [t for t in trigger_order if any(rr["Trigger"] == t for rr in rows)]
    triggers += [t for t in sorted(set(rr["Trigger"] for rr in rows)) if t not in triggers]

    def _as_float(v):
        if v is None:
            return np.nan
        try:
            return float(v)
        except Exception:
            return np.nan

    best = {tr: {} for tr in triggers}
    for tr in triggers:
        sub = [r for r in rows if r["Trigger"] == tr]

        for k in higher_better:
            vals = np.array([_as_float(x.get(k, None)) for x in sub], dtype=np.float64)
            i = int(np.nanargmax(vals)) if np.any(np.isfinite(vals)) else 0
            best[tr][k] = sub[i]["Method"]

        for k in lower_better:
            vals = np.array([_as_float(x.get(k, None)) for x in sub], dtype=np.float64)
            i = int(np.nanargmin(vals)) if np.any(np.isfinite(vals)) else 0
            best[tr][k] = sub[i]["Method"]

    # ---------------- formatting helpers ----------------
    def fmt_key(key, val, nd=3):
        if val is None:
            return "nan"
        try:
            v = float(val)
        except Exception:
            return "nan"
        if not np.isfinite(v):
            return "nan"
        return f"{v:.{nd}f}"

    def maybe_bold(tr, method, key, s):
        # only bold if this method is best for this trigger+metric
        if best.get(tr, {}).get(key, None) == method:
            return r"\textbf{" + s + "}"
        return s

    # ---------------- LaTeX ----------------
    # NOTE: this is wide; reduce spacing + font size.
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{2.5pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.05}")
    lines.append(r"\begin{tabular}{llccccccrrrrcc}")
    lines.append(r"\toprule")
    lines.append(
        r"Trigger & Method & "
        r"MAE$\downarrow$ & P95$|e|$\downarrow$ & InBand$\uparrow$ & UpFrac$\downarrow$ & DownFrac$\downarrow$ & "
        r"TPR/Recall$\uparrow$ & FPR$\downarrow$ & TNR$\uparrow$ & FNR$\downarrow$ & Prec.$\uparrow$ & F1$\uparrow$ & "
        r"$t\bar t\,\uparrow$ & $h\to4b\,\uparrow$ \\"
    )
    lines.append(r"\midrule")

    # Optional: group by trigger with a midrule
    cur_tr = None
    for r in rows:
        tr = r["Trigger"]
        m  = r["Method"]
        if cur_tr is None:
            cur_tr = tr
        elif tr != cur_tr:
            lines.append(r"\midrule")
            cur_tr = tr

        row = []
        row.append(tr)
        row.append(m)

        for key, nd in [
            ("MAE", 3),
            ("P95_abs_err", 3),
            ("InBand", 3),
            ("UpFrac", 3),
            ("DownFrac", 3),
            ("TPR", 3), ("FPR", 3), ("TNR", 3), ("FNR", 3), ("Precision", 3), ("F1", 3),
            ("tt", 3),
            ("h_to_4b", 3),
        ]:
            s = fmt_key(key, r.get(key, None), nd=nd)
            s = maybe_bold(tr, m, key, s)
            row.append(s)

        # build line
        # columns: Trigger, Method, MAE, P95, InBand, UpFrac, DownFrac, TPR, FPR, TNR, FNR, Precision, F1, tt, h_to_4b
        lines.append(
            f"{row[0]} & {row[1]} & "
            f"{row[2]} & {row[3]} & {row[4]} & {row[5]} & {row[6]} & "
            f"{row[7]} & {row[8]} & {row[9]} & {row[10]} & "
            f"{row[11]} & {row[12]} & {row[13]} & {row[14]}  \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(
        rf"\caption{{Summary over evaluation window. Target={float(target_pct)*RATE_SCALE_KHZ:.1f} kHz, "
        rf"tolerance={float(tol_pct)*RATE_SCALE_KHZ:.1f} kHz.}}"
    )
    lines.append(r"\label{tab:trigger_summary}")
    lines.append(r"\end{table}")

    out_tex.write_text("\n".join(lines) + "\n")

def write_confusion_split_tables_tex(chunk_rows, out_tt_tex: Path, out_h4b_tex: Path):
    """
    Writes two LaTeX tables (micro-averaged over chunks by summing counts):
      - out_tt_tex  : ttbar-as-signal vs background
      - out_h4b_tex : h->4b-as-signal vs background

    Uses summarize_confusion_from_chunk_rows_split(...).
    """
    trigger_order = ["HT", "AD"]
    method_order  = ["Constant", "PID", "ADT", "DQN", "PPO", "GRPO", "GFPO-F", "GFPO-FR"]

    def _fmt(x, nd=3):
        if x is None:
            return "nan"
        try:
            v = float(x)
        except Exception:
            return "nan"
        if not np.isfinite(v):
            return "nan"
        return f"{v:.{nd}f}"

    def _write_one(path: Path, *, which: str):
        # which in {"tt", "h4b"}
        lines = []
        lines.append(r"\begin{table}[t]")
        lines.append(r"\centering")
        lines.append(r"\scriptsize")
        lines.append(r"\setlength{\tabcolsep}{3.0pt}")
        lines.append(r"\renewcommand{\arraystretch}{1.05}")

        if which == "tt":
            caption = r"Confusion / classification metrics treating $t\bar t$ as signal and background as negative."
            colhdr  = r"TP$_{t\bar t}$ & FN$_{t\bar t}$ & FP & TN & TPR$_{t\bar t}$ & Prec.$_{t\bar t}$ & F1$_{t\bar t}$ & FPR"
        else:
            caption = r"Confusion / classification metrics treating $h\to4b$ as signal and background as negative."
            colhdr  = r"TP$_{h\to4b}$ & FN$_{h\to4b}$ & FP & TN & TPR$_{h\to4b}$ & Prec.$_{h\to4b}$ & F1$_{h\to4b}$ & FPR"

        lines.append(r"\begin{tabular}{llrrrrrrrr}")
        lines.append(r"\toprule")
        lines.append(r"Trigger & Method & " + colhdr + r" \\")
        lines.append(r"\midrule")

        cur_tr = None
        for tr in trigger_order:
            for m in method_order:
                s = summarize_confusion_from_chunk_rows_split(chunk_rows, trigger=tr, method=m)
                if not s:
                    continue

                if cur_tr is None:
                    cur_tr = tr
                elif tr != cur_tr:
                    lines.append(r"\midrule")
                    cur_tr = tr

                if which == "tt":
                    TP = s.get("TP_tt", None); FN = s.get("FN_tt", None)
                    TPR = s.get("TPR_tt", np.nan)
                    PRE = s.get("Precision_tt", np.nan)
                    F1  = s.get("F1_tt", np.nan)
                else:
                    TP = s.get("TP_h4b", None); FN = s.get("FN_h4b", None)
                    TPR = s.get("TPR_h4b", np.nan)
                    PRE = s.get("Precision_h4b", np.nan)
                    F1  = s.get("F1_h4b", np.nan)

                FP = s.get("FP", None)
                TN = s.get("TN", None)
                FPR = s.get("FPR", np.nan)

                # counts as ints, rates as floats
                def _int_or_nan(v):
                    return ("nan" if v is None else str(int(v)))

                row = [
                    tr, m,
                    _int_or_nan(TP),
                    _int_or_nan(FN),
                    _int_or_nan(FP),
                    _int_or_nan(TN),
                    _fmt(TPR, 3),
                    _fmt(PRE, 3),
                    _fmt(F1,  3),
                    _fmt(FPR, 3),
                ]
                lines.append(" & ".join(row) + r" \\")

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\caption{" + caption + r"}")
        lines.append(r"\end{table}")
        path.write_text("\n".join(lines) + "\n")

    _write_one(out_tt_tex, which="tt")
    _write_one(out_h4b_tex, which="h4b")



def build_paper_rows_from_chunk_rows(chunk_rows, *, target_pct, tol_pct):
    rows_out = []

    # NOTE: build_series_from_chunk_rows expects trigger labels that match chunk_rows.
    # triggers in chunk_rows are typically "HT" and "AS" (AD trigger).
    for trig in ["HT", "AD"]:
        series = build_series_from_chunk_rows(chunk_rows, trigger=trig)
        if not series:
            continue
        
        for method, s in series.items():
            metrics = summarize_paper_table(
                r_pct=s["bg_pct"],
                s_tt=s["tt"],
                s_aa=s["aa"],
                cut_hist=s["cut"],
                target_pct=target_pct,
                tol_pct=tol_pct,
            )

            row = {"Trigger": ("AD" if trig == "AD" else trig), "Method": method, **metrics}
            row.update(summarize_confusion_from_chunk_rows(
                chunk_rows, trigger=trig, method=method
            ))
            rows_out.append(row)
    return rows_out


def running_mean_bool(mask, w=3):
    m = np.asarray(mask, dtype=np.float64)
    k = np.ones(int(w), dtype=np.float64)
    return np.convolve(m, k, mode="same") / np.convolve(np.ones_like(m), k, mode="same")

def plot_cdf_abs_err_multi(rate_khz_by_method, target_khz, tol_khz, title, outpath, run_label):
    """
    rate_khz_by_method: dict(name -> 1D array of rates in kHz)
    """
    rate_khz_by_method = select_plot_methods(rate_khz_by_method)
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
    small_legend(ax, loc="best", title=title)
    add_cms_header(fig, run_label=run_label)
    finalize_diag_fig(fig)
    save_png(fig, str(outpath))
    plt.close(fig)

def _feasible(bg_after, target, tol):
    return (abs(float(bg_after) - float(target)) <= float(tol))

def compute_feasibility_micro_stats(samples, *, trigger, method, target, tol, group_size_keep=16,
                                    requires_feasible_pad: bool = True):
    """
    Returns per-micro-step series + overall metrics for:
      - feasible_ratio:  feas_cand / n_cand
      - kept_feasible_ratio: feas_kept / n_kept   GFPO-F and GFPO-FR
      - pad_flag: 1 if feas_cand < n_kept (GFPO-F and GFPO-FR proxy for "feasible < G_keep")
      - shield_rate: fraction executed steps shielded
    """
    # group by micro
    by_micro = {}
    exec_shield = []
    exec_feas = []
    for r in samples:
        if r.get("trigger") != trigger:
            continue
        if r.get("method", "GRPO") != method:
            continue

        micro = int(r["micro"])
        by_micro.setdefault(micro, {
            "n_cand": 0,
            "feas_cand": 0,
            "n_kept": 0,
            "feas_kept": 0,
        })

        if r["phase"] == "candidate":
            by_micro[micro]["n_cand"] += 1
            if _feasible(r["bg_after"], target, tol):
                by_micro[micro]["feas_cand"] += 1

            if int(r.get("kept", 0)) == 1:
                by_micro[micro]["n_kept"] += 1
                if _feasible(r["bg_after"], target, tol):
                    by_micro[micro]["feas_kept"] += 1

        elif r["phase"] == "executed":
            exec_shield.append(int(r.get("shielded", 0)))
            exec_feas.append(1 if _feasible(r["bg_after"], target, tol) else 0)

    micros = np.array(sorted(by_micro.keys()), dtype=np.int64)
    if micros.size == 0:
        return None

    feas_ratio = []
    kept_feas_ratio = []
    pad_flag = []

    for m in micros:
        d = by_micro[int(m)]
        n_c = max(1, int(d["n_cand"]))
        feas_ratio.append(float(d["feas_cand"]) / n_c)

        n_k = int(d["n_kept"])
        if n_k > 0:
            kept_feas_ratio.append(float(d["feas_kept"]) / max(1, n_k))
            # pad_flag.append(1.0 if int(d["feas_cand"]) < group_size_keep else 0.0)
            if requires_feasible_pad:
                pad_flag.append(1.0 if int(d["feas_cand"]) < group_size_keep else 0.0)
            else:
                pad_flag.append(0.0)   # not a “pad” algorithm
        else:
            kept_feas_ratio.append(np.nan)
            pad_flag.append(np.nan)

    shield_rate = float(np.mean(exec_shield)) if len(exec_shield) else np.nan

    out = {
        "micros": micros,
        "feasible_ratio": np.asarray(feas_ratio, dtype=np.float64),
        "kept_feasible_ratio": np.asarray(kept_feas_ratio, dtype=np.float64),
        "pad_rate": float(np.nanmean(pad_flag)) if np.any(np.isfinite(pad_flag)) else np.nan,
        "shield_rate": shield_rate,
        "feasible_ratio_mean": float(np.mean(feas_ratio)) if len(feas_ratio) else np.nan,
        "kept_feasible_ratio_mean": float(np.nanmean(kept_feas_ratio)) if np.any(np.isfinite(kept_feas_ratio)) else np.nan,
    }
    out["shield_rate"] = np.mean(exec_shield) if exec_shield else np.nan
    out["exec_feasible_rate"] = np.mean(exec_feas) if exec_feas else np.nan
    return out


def plot_feasible_ratio_timeseries(stats_grpo, stats_gfpo, *, title, outpath, run_label):
    fig, ax = plt.subplots(figsize=(9, 5.4))

    if stats_grpo is not None:
        ax.plot(
            stats_grpo["micros"], stats_grpo["feasible_ratio"],
            linewidth=2.2,
            linestyle="-",
            marker=None,          
            drawstyle="steps-post",
            label="GRPO (candidates)",
        )
    if stats_gfpo is not None:
        ax.plot(
            stats_gfpo["micros"], stats_gfpo["feasible_ratio"],
            linewidth=2.2,
            linestyle=(0, (4, 2)),
            marker=None,           
            drawstyle="steps-post",
            label="GFPO (candidates)",
        )

    ax.set_xlabel("Micro-step")
    ax.set_ylabel(r"Feasible ratio  (#cand with |bg-target|<=tol) / #cand")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle="--", alpha=0.5)
    small_legend(ax, loc="best", title=title)
    add_cms_header(fig, run_label=run_label)
    finalize_diag_fig(fig)
    save_png(fig, str(outpath))
    plt.close(fig)


def plot_feasibility_bar(stats_grpo, stats_gfpo, *, title, outpath, run_label):
    #stats_grpo is gfpo-f
    #stats_gfpo is gfpo-fr
    metrics = ["cand_feas", "kept_feas", "pad_rate", "shield_rate"]
    labels  = ["Feasible ratio", "Kept-feasible ratio", "Pad rate", "Shield rate"]

    def getvals(st, is_gfpo):
        if st is None:
            return [np.nan]*4
        return [
            float(st["feasible_ratio_mean"]),
            float(st["kept_feasible_ratio_mean"]),
            float(st["pad_rate"]),
            float(st["shield_rate"]),
        ]

    vals_grpo = getvals(stats_grpo, is_gfpo=False)
    vals_gfpo = getvals(stats_gfpo, is_gfpo=True)

    x = np.arange(len(labels))
    bw = 0.38

    fig, ax = plt.subplots(figsize=(9, 5.4))
    ax.bar(x - bw/2, vals_grpo, width=bw, label="GFPO-F")
    ax.bar(x + bw/2, vals_gfpo, width=bw, label="GFPO-FR")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Fraction")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    small_legend(ax, loc="best", title=title)
    add_cms_header(fig, run_label=run_label)
    finalize_diag_fig(fig)
    save_png(fig, str(outpath))
    plt.close(fig)


def plot_running_inband_multi(time, inband_by_method, w, title, outpath, run_label):
    """
    inband_by_method: dict(name -> boolean mask per chunk)
    """
    inband_by_method = select_plot_methods(inband_by_method)
    fig, ax = plt.subplots(figsize=(8, 5.2))
    style = {
        "Constant": dict(linestyle="--", linewidth=2.2),
        "PID":      dict(linestyle="-",  linewidth=2.2),
        "DQN":      dict(linestyle=(0, (8, 2, 2, 2)), linewidth=2.6, marker="o", markersize=3, markevery=8),
        "ADT": dict(linestyle=(0, (6, 2)), linewidth=2.6),
        "PPO": dict(linestyle=(0, (3, 2, 1, 2)), linewidth=2.6),
        "GRPO":     dict(linestyle=(0, (10, 2, 2, 2)), linewidth=2.8),
        "GFPO-F":   dict(linestyle=(0, (4, 2)), linewidth=2.6),
        "GFPO-FR":  dict(linestyle=(0, (2, 2)), linewidth=2.6),
    }
    for name, m in inband_by_method.items():
        y = running_mean_bool(m, w=int(w))
        t = np.linspace(0.0, 1.0, len(y))
        ax.plot(t, y, label=f"{name} (w={int(w)})", **style.get(name, {}))


    ax.set_xlabel("Time (Fraction of Run)")
    ax.set_ylabel("Running in-band fraction")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.5)
    small_legend(ax, loc="best", title=title)
    add_cms_header(fig, run_label=run_label)
    finalize_diag_fig(fig)
    save_png(fig, str(outpath))
    plt.close(fig)


def plot_cut_step_hist_multi(
    cut_by_method,
    xlabel,
    title,
    outpath,
    run_label,
    bins=30,  # kept for backward compat; ignored in raw mode
    allow_constant_zeros=True,
    raw=True,                 # <-- NEW: default to raw delta plot
    use_abs=False,            # <-- NEW: if True, plot |Δcut| raw values
    max_points=8000,          # <-- NEW: cap points per method (subsample)
):
    """
    If raw=True: plot per-step raw deltas (no binning) as a scatter over time.
    If raw=False: fall back to the old histogram behavior (binned).
    """
    cut_by_method = select_plot_methods(cut_by_method)

    fig, ax = plt.subplots(figsize=(8, 5.2))
    any_plotted = False

    for name, c in cut_by_method.items():
        c = np.asarray(c, dtype=np.float64)

        dc = np.diff(c) if c.size >= 2 else np.array([], dtype=np.float64)

        if dc.size == 0 and allow_constant_zeros:
            # constant / degenerate history -> show "no motion"
            dc = np.zeros(max(1, c.size - 1), dtype=np.float64)

        if dc.size == 0:
            continue

        y = np.abs(dc) if use_abs else dc

        if raw:
            # subsample for readability / speed
            n = y.size
            stride = max(1, int(np.ceil(n / max(1, int(max_points)))))
            y_s = y[::stride]
            t_s = np.linspace(0.0, 1.0, y_s.size)  # normalized time axis

            ax.plot(
                t_s,
                y_s,
                linestyle="None",
                marker=".",
                markersize=3.0,
                alpha=0.55,
                label=name,
            )
        else:
            # old behavior: binned histogram
            ax.hist(y, bins=int(bins), alpha=0.50, label=name)

        any_plotted = True

    if not any_plotted:
        ax.text(0.5, 0.5, "No cut history to plot", ha="center", va="center",
                transform=ax.transAxes)

    ax.axhline(0.0, linestyle="--", linewidth=1.2, alpha=0.6)
    ax.set_xlabel("Time (Fraction of Run)" if raw else xlabel)
    ax.set_ylabel(r"$\Delta \mathrm{cut}$" if raw and not use_abs else (r"$|\Delta \mathrm{cut}|$" if raw else "Count"))
    ax.grid(True, linestyle="--", alpha=0.4)
    small_legend(ax, loc="best", title=title)

    add_cms_header(fig, run_label=run_label)
    finalize_diag_fig(fig)
    save_png(fig, str(outpath))
    plt.close(fig)



def plot_inband_eff_bars_multi(summary_by_method, title, outpath, run_label):
    """
    summary_by_method: dict(name -> summarize_compact(...) dict)
    """
    summary_by_method = select_plot_methods(summary_by_method)
    if not summary_by_method:
        return
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
    small_legend(ax, loc="best", title=title)
    add_cms_header(fig, run_label=run_label)
    finalize_diag_fig(fig)
    save_png(fig, str(outpath))
    plt.close(fig)

def gfpo_topk_keep_indices(bg_after, tt_after, aa_after, rewards, *,
                           target, tol, feas_mult, mix, k_keep):
    """
    GFPO selection with explicit G_keep < G_sample.

    Returns:
      keep_idx: np.ndarray of length k_keep (best-first ranking)
      feas_count: int number of feasible candidates among G_sample
      used_pad: bool (True if we had to pad from infeasible to reach k_keep)
    Ranking:
      - Feasible first: sort by (mix*tt + (1-mix)*aa), tie-break reward, then closeness
      - If feasible < k_keep: pad remaining from infeasible sorted by closeness, tie-break reward, then score
    """
    bg_after = np.asarray(bg_after, dtype=np.float64)
    tt_after = np.asarray(tt_after, dtype=np.float64)
    aa_after = np.asarray(aa_after, dtype=np.float64)
    rewards  = np.asarray(rewards,  dtype=np.float64)

    G = bg_after.size
    k_keep = int(k_keep)
    if k_keep <= 0:
        raise ValueError("k_keep must be >= 1")
    if k_keep > G:
        raise ValueError("k_keep must be <= number of candidates")

    abs_err = np.abs(bg_after - float(target))
    feas = abs_err <= float(feas_mult) * float(tol)
    feas_idx = np.where(feas)[0]
    infeas_idx = np.where(~feas)[0]

    score_sig = float(mix) * tt_after + (1.0 - float(mix)) * aa_after

    # --- sort feasible: higher score_sig, then higher reward, then smaller abs_err ---
    if feas_idx.size:
        # lexsort sorts by last key primary; we want best-first, so use negatives for descending
        order = np.lexsort((
            abs_err[feas_idx],             # smaller better
            -rewards[feas_idx],            # larger better
            -score_sig[feas_idx],          # larger better
        ))
        feas_sorted = feas_idx[order]
    else:
        feas_sorted = np.array([], dtype=np.int64)

    # --- sort infeasible: smaller abs_err, then higher reward, then higher score_sig ---
    if infeas_idx.size:
        order = np.lexsort((
            -score_sig[infeas_idx],        # larger better (tie)
            -rewards[infeas_idx],          # larger better
            abs_err[infeas_idx],           # smaller better (primary)
        ))
        infeas_sorted = infeas_idx[order]
    else:
        infeas_sorted = np.array([], dtype=np.int64)

    used_pad = False
    if feas_sorted.size >= k_keep:
        keep = feas_sorted[:k_keep]
    else:
        used_pad = True
        need = k_keep - feas_sorted.size
        keep = np.concatenate([feas_sorted, infeas_sorted[:need]]) if need > 0 else feas_sorted

    feas_count = int(feas_sorted.size)
    return keep.astype(np.int64), feas_count, used_pad

# ----------------------------- main -----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--input", default="Data/Trigger_food_MC.h5",
                    choices=["Data/Trigger_food_MC.h5", "Data/Matched_data_2016_dim2.h5", "Data/Trigger_food_MC_ablation_4.h5", "Data/Trigger_food_MC_ablation_6.h5", "Data/Trigger_food_MC_ablation_8.h5", "Data/Trigger_food_MC_ablation_10.h5", "Data/Trigger_food_MC_ablation_12.h5", "Data/Trigger_food_MC_ablation_14.h5", "Data/Trigger_food_MC_ablation_16.h5"])
    ap.add_argument("--outdir", default="outputs/demo_sing_grpo_as_feature")
    ap.add_argument("--control", default="MC", choices=["MC", "RealData"])
    ap.add_argument("--score-dim-hint", type=int, default=2)
    ap.add_argument("--as-dim", type=int, default=2, choices=[1, 2, 4, 6, 8, 10, 12, 14, 16])

    ap.add_argument("--as-deltas", type=str, default="-3,-1.5,0,1.5,3",choices=["-3,-1.5,0,1.5,3","-4,-2,-1,0,1,2,4","-3,-1.5,-1,0,1,1.5,3"])
    ap.add_argument("--as-step", type=float, default=0.5, help = "AS delta step size multiply max of as-deltas above would be maximum delta ad trigger can take.")

    ap.add_argument("--print-keys", action="store_true")
    ap.add_argument("--print-keys-max", type=int, default=None)

    ap.add_argument("--window-events-chunk-size", type=int, default=3)
    ap.add_argument("--seq-len", type=int, default=128)
    # making it a sliding window
    ap.add_argument("--micro-stride", type=int, default=5000,
                help="events per micro update step (small step)")
    ap.add_argument("--micro-window", type=int, default=50000,
                help="events used to evaluate bg rate / reward at each micro step")

    # GRPO kwargs
    ap.add_argument("--train-every", type=int, default=50)
    ap.add_argument("--ht-temperature", type=float, default=1.0) 
    ap.add_argument("--as-temperature", type=float, default=1.0)
    ap.add_argument("--beta-kl", type=float, default=0.02)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--grpo-lr", type=float, default=2e-4) #originally 3e-4
    ap.add_argument("--band-mult-ht", type=float, default=1.0,
                help="HT candidate filter band: |bg-target| <= band-mult * tol (1.0 = exact tolerance)")
    ap.add_argument("--band-mult-as", type=float, default=1.0,
                help="AS candidate filter band: |bg-target| <= band-mult * tol (1.0 = exact tolerance)")
    ap.add_argument("--sig-bonus", type=float, default=1.0,
                help="HT extra bonus weight for signal score inside band (helps avoid bg-only overfit)")
    ap.add_argument("--sig-bonus-as", type=float, default=1.0,
                help="AS extra bonus weight for signal score inside band (helps avoid bg-only overfit)")


    # objective/reward
    ap.add_argument("--target", type=float, default=0.25)   # percent
    ap.add_argument("--tol", type=float, default=0.025,     # percent  (0.025% -> ±10kHz band)
                    help="tolerance in percent units; 0.025 corresponds to [90,110] kHz when target=0.25%")
    ap.add_argument("--alpha", type=float, default=0.3) #alpha ttbar focus
    ap.add_argument("--beta", type=float, default=0.2, help="beta moving penalty weight") 
    ap.add_argument("--violation-penalty", type=float, default=5.0,
                    help="penalty weight for bg rate outside of target±tol band")

    # optional stabilization (AD-specific)
    ap.add_argument("--occ-pen", type=float, default=0.0,
                    help="extra penalty weight for near-cut occupancy * |delta| (0.5-3.0 might be ideal)")
    ap.add_argument("--run-avg-window", type=int, default=3,
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

    # --- Frozen DQN baseline (DQN-F) ---
    ap.add_argument("--dqn-f-train-chunks", type=int, default=1,
                help="Train DQN-F only on the first N chunks, then freeze weights and rollout.")
    ap.add_argument("--dqn-f-eps", type=float, default=0.0,
                help="Epsilon used for DQN-F rollout AFTER freezing (default: greedy).")

    # GFPO (Greedy Feasible Policy Optimization) baseline
    ap.add_argument("--gfpo-filter", type=str, default="abs_err_topk", choices=["abs_err_topk", "feasible_first_sig", "both"]
                    , help="abs_err_topk: pick the top-K candidates with the smallest |bg_after - target|, " \
                        "feasible_first_sig   : feasible-first (|bg-target|<=feas_mult*tol), " \
                        "then rank by mix*tt+(1-mix)*aa; pad with closest if needed" \
                            "both=runs both.")
    ap.add_argument("--group-size-keep", type=int, default=16, choices=[16, 32]) 
    ap.add_argument("--group-size-sample", type=int, default=32)

    ap.add_argument("--gfpo-feas-mult", type=float, default=1.0,
                    help="feasibility band multiplier: |bg-target| <= mult*tol")
    ap.add_argument("--gfpo-mix", type=float, default=0.20, #0.8 originally
                    help="GFPO ranking: mix*tt + (1-mix)*aa within feasible set")



    ap.add_argument(
        "--baselines",
        type=str,
        default="constant,pid,adt,dqn,dqn_f,ppo,grpo,gfpo_f,gfpo_fr",
        help="Comma-separated: constant,pid,adt,dqn,dqn_f,ppo,grpo,gfpo_f,gfpo_fr"
    )


    ap.add_argument("--run-adt", action="store_true", help="Enable ADT baseline (DQN with action-hold + end-of-chunk updates)")
    ap.add_argument("--adt-l", type=int, default=10, help="ADT action-hold: update action every l micro-steps")
    ap.add_argument("--adt-train-steps-per-episode", type=int, default=50, help="ADT: gradient steps ONLY at end of each chunk")

    ap.add_argument("--adt-reward-mode", default="lhc", choices=["lhc", "paper"],
                help="ADT reward: 'lhc' uses rate-tracking reward; 'paper' uses TP/TN/FP/FN style reward")
    ap.add_argument("--adt-alpha", type=float, default=0.7)
    ap.add_argument("--adt-beta",  type=float, default=0.3)



    # PPO knobs
    ap.add_argument("--ppo-lr", type=float, default=3e-4)
    ap.add_argument("--ppo-gamma", type=float, default=0.95)
    ap.add_argument("--ppo-lam", type=float, default=0.95)
    ap.add_argument("--ppo-clip-eps", type=float, default=0.2)
    ap.add_argument("--ppo-epochs", type=int, default=4)
    ap.add_argument("--ppo-minibatch", type=int, default=64)
    ap.add_argument("--ppo-ent-coef", type=float, default=0.01)
    ap.add_argument("--ppo-vf-coef", type=float, default=0.5)
    ap.add_argument("--ppo-max-grad-norm", type=float, default=0.5)
    ap.add_argument("--ppo-temperature", type=float, default=1.0,
                    help="sampling temperature for PPO policy during data collection")


    global chunk_rows
    chunk_rows = []


    args = ap.parse_args()
    target = float(args.target)
    tol    = float(args.tol)


    BASELINES = [x.strip().lower() for x in args.baselines.split(",") if x.strip()]

    

    if args.gfpo_filter == "both":
        GFPO_VARIANTS = [("GFPO-F", "abs_err_topk"), ("GFPO-FR", "feasible_first_sig")]
    else:
        # single baseline run
        name = "GFPO-F" if args.gfpo_filter == "abs_err_topk" else "GFPO-FR"
        GFPO_VARIANTS = [(name, args.gfpo_filter)]

    # --- append gfpo filter to outdir (so runs don't overwrite for GFPO) ---
    suffix="all"
    outdir_str = str(args.outdir)
    if not outdir_str.endswith(f"_{suffix}"):
        args.outdir = f"{outdir_str}_{suffix}_{args.control}"
    # ------------------------------------------------------------- 

    if args.group_size_sample < args.group_size_keep: #sample >= keep
        raise SystemExit("--gfpo-keep-size must be <= --gfpo-sample-size")

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
    if args.as_dim == 1:
        Bas, Tas, Aas = d["Bas1"], d["Tas1"], d["Aas1"]
    elif args.as_dim == 2:
        Bas, Tas, Aas = d["Bas2"], d["Tas2"], d["Aas2"]
    elif args.as_dim == 4:
        Bas, Tas, Aas = d["Bas4"], d["Tas4"], d["Aas4"]
    elif args.as_dim == 6:
        Bas, Tas, Aas = d["Bas6"], d["Tas6"], d["Aas6"]
    elif args.as_dim == 8:
        Bas, Tas, Aas = d["Bas8"], d["Tas8"], d["Aas8"]
    elif args.as_dim == 10:
        Bas, Tas, Aas = d["Bas10"], d["Tas10"], d["Aas10"]
    elif args.as_dim == 12:
        Bas, Tas, Aas = d["Bas12"], d["Tas12"], d["Aas12"]
    elif args.as_dim == 14:
        Bas, Tas, Aas = d["Bas14"], d["Tas14"], d["Aas14"]
    elif args.as_dim == 16:
        Bas, Tas, Aas = d["Bas16"], d["Tas16"], d["Aas16"]
    else:
        raise SystemExit("Unsupported --as-dim")

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
    win_hi = min(start_event + (100000 if args.control == "MC" else 40000), N) #real data 200000 - 240000
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

    # HT PD init (optional baseline; let's keep it for completeness)
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


    # ------------------ infer feature dims before building any agents ------------------
    def infer_feat_dim_as():
        probe_idx = np.arange(win_lo, min(win_lo + max(K, 256), N))
        probe_bas = Bas[probe_idx]
        probe_bnpv = Bnpv[probe_idx]
        probe_bg = Sing_Trigger(probe_bas, fixed_AS_cut)
        obs = make_event_seq_as(
        bas=probe_bas, bnpv=probe_bnpv,
        bg_rate=probe_bg, prev_bg_rate=probe_bg,
        cut=fixed_AS_cut,
        as_mid=as_mid, as_span=as_span,
        target=target, K=K,
        last_delta=0.0, max_delta=MAX_DELTA_AS,
        near_widths=near_widths_as,
        step=AS_STEP,
        tol=tol,
        err_i=0.0,
        d_bg_d_cut=0.0,
        )
        return int(np.asarray(obs).shape[-1])


    feat_dim_as = infer_feat_dim_as()
    print(f"[DEBUG] inferred feat_dim_as={feat_dim_as}")

    # GRPO agent AS
    cfg = GRPOConfig(
        lr=args.grpo_lr,
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
        mode="lex",        # "lex" default; "lag" if adaptive lambda
        mix=args.alpha, #increase for tt
        alpha_sig=1.0,
        beta_move=args.beta,
        gamma_stab=0.25,
        k_violate=args.violation_penalty,
        w_occ=float(args.occ_pen)
    ))
    gfpo_cfg_as = GRPOConfig(
        lr=args.grpo_lr,
        beta_kl=args.beta_kl,
        ent_coef=args.ent_coef,
        device="cpu",
        batch_size=64,
        train_epochs=2,
        ref_update_interval=200,
    )
    gfpo_as = GRPOAgent(seq_len=K, feat_dim=feat_dim_as, n_actions=len(AS_DELTAS), cfg=gfpo_cfg_as, seed=SEED,
        reward_cfg=GRPORewardCfg(
        target=target,
        tol=tol,
        mode="lex",        # "lex" default; "lag" if adaptive lambda
        mix=args.alpha, #increase for tt
        alpha_sig=1.0,
        beta_move=args.beta,
        gamma_stab=0.25,
        k_violate=args.violation_penalty,
        w_occ=float(args.occ_pen)
    ))

    # ---------------- GFPO variants (AS) ----------------
    gfpo_as_agents = {}
    gfpo_as_state = {}   # per-variant running state
    gfpo_as_logs = {}    # per-variant chunk-level logs
    gfpo_as_losses = {}  # per-variant update losses
     
    # logs (background in percent units first)

    run_label = "MC" if args.control == "MC" else "283408"


    
    # ---------------- DQN agent (AS only for now temporarilly, parallel baseline) ----------------
    dqn_cfg = DQNConfig(
        lr=float(args.dqn_lr),
        gamma=float(args.dqn_gamma),
        batch_size=int(args.dqn_batch_size),
        target_update=int(args.dqn_target_update),
    )
    dqn_as = SeqDQNAgent(seq_len=K, feat_dim=feat_dim_as, n_actions=len(AS_DELTAS),
                        cfg=dqn_cfg, seed=SEED)
    

    controllers_as = []

    if "constant" in BASELINES:
        controllers_as.append(ConstantCtrl("Constant", fixed_AS_cut))

    if "pid" in BASELINES:
        controllers_as.append(PIDCtrl("PID", fixed_AS_cut, as_lo, as_hi))

    if "dqn" in BASELINES:
        controllers_as.append(DQNCtrl(
        "DQN", fixed_AS_cut, as_lo, as_hi,
        agent=dqn_as, deltas=AS_DELTAS, step=AS_STEP, max_delta=MAX_DELTA_AS,
        as_mid=as_mid, as_span=as_span, near_widths=near_widths_as, K=K,
        target=target, tol=tol,
        eps_min=args.dqn_eps_min, eps_decay=args.dqn_eps_decay,
        train_steps_per_micro=args.dqn_train_steps_per_micro,
        alpha=args.alpha, beta=args.beta
        ))
    
    if "dqn_f" in BASELINES:
        agent_dqnf_ad = SeqDQNAgent(seq_len=K, feat_dim=feat_dim_as, n_actions=len(AS_DELTAS), cfg=dqn_cfg, seed=SEED+7)

        controllers_as.append(DQNFrozenCtrl(
        "DQN-F", fixed_AS_cut, as_lo, as_hi,
        agent=agent_dqnf_ad, deltas=AS_DELTAS, step=AS_STEP, max_delta=MAX_DELTA_AS,
        as_mid=as_mid, as_span=as_span, near_widths=near_widths_as, K=K,
        target=target, tol=tol,
        eps_min=args.dqn_f_eps, eps_decay=1.0,  # no decay
        train_steps_per_micro=0,               # no training during rollout
        alpha=args.alpha, beta=args.beta, train_chunks=args.dqn_f_train_chunks, eps_after_freeze=args.dqn_f_eps
        ))

    
    # --- ADT baseline (AS): DQN + action-hold + end-of-chunk updates ---
    if args.run_adt and ("adt" in BASELINES):
        adt_cfg_as = DQNConfig(
        lr=float(args.dqn_lr),
        gamma=float(args.dqn_gamma),
        batch_size=int(args.dqn_batch_size),
        target_update=int(args.dqn_target_update),
        )
        agent_adt_as = SeqDQNAgent(
        seq_len=K, feat_dim=feat_dim_as, n_actions=len(AS_DELTAS),
        cfg=adt_cfg_as, seed=SEED + 101
        )

        controllers_as.append(ADTCtrl(
        name="ADT",
        init_cut=fixed_AS_cut, lo=as_lo, hi=as_hi,
        agent=agent_adt_as,
        deltas=AS_DELTAS, step=AS_STEP, max_delta=MAX_DELTA_AS,
        as_mid=as_mid, as_span=as_span,
        near_widths=near_widths_as, K=K,
        target=target, tol=tol,
        eps_min=args.dqn_eps_min, eps_decay=args.dqn_eps_decay,
        train_steps_per_micro=0,               # ignored by ADT (no per-micro train)
        alpha=args.alpha, beta=args.beta,      # used for reward_mode="lhc"
        adt_l=args.adt_l,
        train_steps_per_episode=args.adt_train_steps_per_episode,
        reward_mode=args.adt_reward_mode,
        adt_alpha=args.adt_alpha, adt_beta=args.adt_beta,
        ))

    if "grpo" in BASELINES:
        controllers_as.append(GRPOCtrl(
        "GRPO", fixed_AS_cut, as_lo, as_hi,
        agent=agent, deltas=AS_DELTAS, step=AS_STEP, max_delta=MAX_DELTA_AS,
        as_mid=as_mid, as_span=as_span, near_widths=near_widths_as, K=K,
        target=target, tol=tol,
        train_every=args.train_every, temperature=args.as_temperature,
        group_size_keep=args.group_size_keep
        ))


    if "gfpo_f" in BASELINES:
        controllers_as.append(GFPOCtrl(
        "GFPO-F", fixed_AS_cut, as_lo, as_hi,
        agent=GRPOAgent(seq_len=K, feat_dim=feat_dim_as, n_actions=len(AS_DELTAS), cfg=cfg, seed=SEED,
                        reward_cfg=agent.reward_cfg),
        deltas=AS_DELTAS, step=AS_STEP, max_delta=MAX_DELTA_AS,
        as_mid=as_mid, as_span=as_span, near_widths=near_widths_as, K=K,
        target=target, tol=tol,
        train_every=args.train_every, temperature=args.as_temperature,
        gfpo_filter="abs_err_topk",
        group_size_sample=args.group_size_sample, group_size_keep=args.group_size_keep,
        feas_mult=args.gfpo_feas_mult, mix=args.gfpo_mix,
        band_mult=args.band_mult_as, sig_bonus=args.sig_bonus_as,
        ))

    if "gfpo_fr" in BASELINES:
        controllers_as.append(GFPOCtrl(
        "GFPO-FR", fixed_AS_cut, as_lo, as_hi,
        agent=GRPOAgent(seq_len=K, feat_dim=feat_dim_as, n_actions=len(AS_DELTAS), cfg=cfg, seed=SEED,
                        reward_cfg=agent.reward_cfg),
        deltas=AS_DELTAS, step=AS_STEP, max_delta=MAX_DELTA_AS,
        as_mid=as_mid, as_span=as_span, near_widths=near_widths_as, K=K,
        target=target, tol=tol,
        train_every=args.train_every, temperature=args.as_temperature,
        gfpo_filter="feasible_first_sig",
        group_size_sample=args.group_size_sample, group_size_keep=args.group_size_keep,
        feas_mult=args.gfpo_feas_mult, mix=args.gfpo_mix,
        band_mult=args.band_mult_as, sig_bonus=args.sig_bonus_as
        ))

    if "ppo" in BASELINES:
        print("feat_dim_as for PPO:", feat_dim_as)
        ppo_cfg_as = SeqPPOConfig(
            feat_dim=feat_dim_as,
            n_actions=len(AS_DELTAS),
        )
        ppo_agent_ad = SeqPPOAgent(
            cfg=ppo_cfg_as    
        )
        controllers_as.append(
        PPOCtrl(
                "PPO",
                init_cut=fixed_AS_cut,
                lo=as_lo,
                hi=as_hi,
                agent=ppo_agent_ad,
                deltas=AS_DELTAS,
                step=args.as_step,
                max_delta=MAX_DELTA_AS,
                as_mid=as_mid,
                as_span=as_span,
                near_widths=near_widths_as,
                K=K,
                target=target,
                tol=tol,
                alpha=args.alpha,
                beta=args.beta,
                ppo_temperature=args.as_temperature,
        )
        )

    



    if args.run_ht:

        err_i_ht_dqn  = 0.0
        err_i_ht_grpo = 0.0
        err_i_ht_gfpo = 0.0
        HT_DELTAS = np.array([float(x) for x in args.ht_deltas.split(",")], dtype=np.float32)
        HT_STEP = float(args.ht_step)
        MAX_DELTA_HT = float(np.max(np.abs(HT_DELTAS))) * HT_STEP

        near_widths_ht = (5.0, 10.0, 20.0)
        # feat_dim_ht = 10 + len(near_widths_ht)
        def infer_feat_dim_ht():
            probe_idx = np.arange(win_lo, min(win_lo + max(K, 256), N))
            probe_bht = Bht[probe_idx]
            probe_bnpv = Bnpv[probe_idx]
            probe_bg = Sing_Trigger(probe_bht, fixed_Ht_cut)
            obs = make_event_seq_ht(
            bht=probe_bht, bnpv=probe_bnpv,
            bg_rate=probe_bg, prev_bg_rate=probe_bg,
            cut=fixed_Ht_cut,
            ht_mid=ht_mid, ht_span=ht_span,
            target=target, K=K,
            last_delta=0.0, max_delta=MAX_DELTA_HT,
            near_widths=near_widths_ht,
            step=HT_STEP,
            tol=tol,
            err_i=0.0,
            d_bg_d_cut=0.0,
            )
            return int(np.asarray(obs).shape[-1])
        
        feat_dim_ht = infer_feat_dim_ht()
        print(f"[DEBUG] inferred feat_dim_ht={feat_dim_ht}")

        controllers_ht = []

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
        
        if "constant" in BASELINES:
            controllers_ht.append(ConstantCtrl("Constant", fixed_Ht_cut))

        if "pid" in BASELINES:
            controllers_ht.append(PIDCtrlHT("PID", fixed_Ht_cut, ht_lo, ht_hi))

        if "dqn" in BASELINES:
            controllers_ht.append(DQNCtrlHT(
            "DQN", fixed_Ht_cut, ht_lo, ht_hi,
            agent=dqn_ht, deltas=HT_DELTAS, step=HT_STEP, max_delta=MAX_DELTA_HT,
            ht_mid=ht_mid, ht_span=ht_span, near_widths=near_widths_ht, K=K,
            target=target, tol=tol,
            eps_min=args.dqn_eps_min, eps_decay=args.dqn_eps_decay,
            train_steps_per_micro=args.dqn_train_steps_per_micro,
            alpha=args.alpha, beta=args.beta
            ))
        if "dqn_f" in BASELINES:
            agent_dqnf_ht = SeqDQNAgent(seq_len=K, feat_dim=feat_dim_ht, n_actions=len(HT_DELTAS), cfg=dqn_ht_cfg, seed=SEED+11)

            controllers_ht.append(DQNFrozenCtrlHT(
            "DQN-F", fixed_Ht_cut, ht_lo, ht_hi,
            agent=agent_dqnf_ht, deltas=HT_DELTAS, step=HT_STEP, max_delta=MAX_DELTA_HT,
            ht_mid=ht_mid, ht_span=ht_span, near_widths=near_widths_ht, K=K,
            target=target, tol=tol,
            eps_min=args.dqn_f_eps, eps_decay=1.0,  # no decay
            train_steps_per_micro=0,               # no training during rollout
            alpha=args.alpha, beta=args.beta, train_chunks=args.dqn_f_train_chunks, eps_after_freeze=args.dqn_f_eps
            ))
        
        # --- ADT baseline (HT) ---
        if args.run_adt and ("adt" in BASELINES):
            adt_cfg_ht = DQNConfig(
                lr=float(args.dqn_lr),
                gamma=float(args.dqn_gamma),
                batch_size=int(args.dqn_batch_size),
                target_update=int(args.dqn_target_update),
            )
            agent_adt_ht = SeqDQNAgent(
                seq_len=K, feat_dim=feat_dim_ht, n_actions=len(HT_DELTAS),
                cfg=adt_cfg_ht, seed=SEED + 202
            )

            controllers_ht.append(ADTCtrlHT(
            name="ADT",
            init_cut=fixed_Ht_cut, lo=ht_lo, hi=ht_hi,
            agent=agent_adt_ht,
            deltas=HT_DELTAS, step=HT_STEP, max_delta=MAX_DELTA_HT,
            ht_mid=ht_mid, ht_span=ht_span,
            near_widths=near_widths_ht, K=K,
            target=target, tol=tol,
            eps_min=args.dqn_eps_min, eps_decay=args.dqn_eps_decay,
            train_steps_per_micro=0,              # ignored by ADT
            alpha=args.alpha, beta=args.beta,
            adt_l=args.adt_l,
            train_steps_per_episode=args.adt_train_steps_per_episode,
            reward_mode=args.adt_reward_mode,
            adt_alpha=args.adt_alpha, adt_beta=args.adt_beta,
            ))
        
        if "grpo" in BASELINES:
            cfg_ht = GRPOConfig(
                lr=args.grpo_lr, beta_kl=args.beta_kl, ent_coef=args.ent_coef,
                device="cpu", batch_size=256, train_epochs=2, ref_update_interval=200,
            )
            agent_ht = GRPOAgent(
                seq_len=K, feat_dim=feat_dim_ht, n_actions=len(HT_DELTAS),
                cfg=cfg_ht, seed=SEED,
                reward_cfg=GRPORewardCfg(
                target=target,
                tol=tol,
                mode="lex",        # "lex" recommended
                mix=args.alpha, #increase for tt
                alpha_sig=1.0,
                beta_move=args.beta,
                gamma_stab=0.25,
                k_violate=args.violation_penalty,
                w_occ=float(args.occ_pen)
                )
            )
            controllers_ht.append(GRPOCtrlHT(
            "GRPO", fixed_Ht_cut, ht_lo, ht_hi,
            agent=agent_ht, deltas=HT_DELTAS, step=HT_STEP, max_delta=MAX_DELTA_HT,
            ht_mid=ht_mid, ht_span=ht_span, near_widths=near_widths_ht, K=K,
            target=target, tol=tol,
            train_every=args.train_every, temperature=args.ht_temperature,
            group_size_keep=args.group_size_keep
            ))


        if "gfpo_f" in BASELINES:
            controllers_ht.append(GFPOCtrlHT(
            "GFPO-F", fixed_Ht_cut, ht_lo, ht_hi,
            agent=GRPOAgent(seq_len=K, feat_dim=feat_dim_ht, n_actions=len(HT_DELTAS), cfg=cfg, seed=SEED,
                        reward_cfg=agent.reward_cfg),
                deltas=HT_DELTAS, step=HT_STEP, max_delta=MAX_DELTA_HT,
                ht_mid=ht_mid, ht_span=ht_span, near_widths=near_widths_ht, K=K,
                target=target, tol=tol,
                train_every=args.train_every, temperature=args.ht_temperature,
                gfpo_filter="abs_err_topk",
                group_size_sample=args.group_size_sample, group_size_keep=args.group_size_keep,
                feas_mult=args.gfpo_feas_mult, mix=args.gfpo_mix,
                band_mult=args.band_mult_ht, sig_bonus=args.sig_bonus
                ))

        if "gfpo_fr" in BASELINES:
            controllers_ht.append(GFPOCtrlHT(
            "GFPO-FR", fixed_Ht_cut, ht_lo, ht_hi,
            agent=GRPOAgent(seq_len=K, feat_dim=feat_dim_ht, n_actions=len(HT_DELTAS), cfg=cfg, seed=SEED,
                        reward_cfg=agent.reward_cfg),
                deltas=HT_DELTAS, step=HT_STEP, max_delta=MAX_DELTA_HT,
                ht_mid=ht_mid, ht_span=ht_span, near_widths=near_widths_ht, K=K,
                target=target, tol=tol,
                train_every=args.train_every, temperature=args.ht_temperature,
                gfpo_filter="feasible_first_sig",
                group_size_sample=args.group_size_sample, group_size_keep=args.group_size_keep,
                feas_mult=args.gfpo_feas_mult, mix=args.gfpo_mix,
                band_mult=args.band_mult_ht, sig_bonus=args.sig_bonus
            ))
    
        
        if "ppo" in BASELINES:
            print("feat_dim_ht for PPO:", feat_dim_ht)
            ppo_cfg_ht = SeqPPOConfig(
                feat_dim = feat_dim_ht,
                n_actions = len(HT_DELTAS),
            )
            ppo_agent_ht = SeqPPOAgent(
                cfg=ppo_cfg_ht
            )
            controllers_ht.append(PPOCtrlHT(
                name="PPO",
                init_cut=fixed_Ht_cut, lo=ht_lo, hi=ht_hi,
                agent=ppo_agent_ht,
                deltas=HT_DELTAS, step=HT_STEP, max_delta=max(abs(HT_DELTAS)) * float(args.ht_step),
                ht_mid=ht_mid, ht_span=ht_span,
                near_widths=near_widths_ht, K=K,
                target=target, tol=tol,
                alpha=args.alpha, beta=args.beta, ppo_temperature=args.ht_temperature
            ))

        


    AS_cut_dqn = fixed_AS_cut
    prev_bg_dqn = None
    last_das_dqn = 0.0
    dqn_losses = []
    dqn_rewards = []

    grpo_losses = []


    err_i_as_dqn  = 0.0
    err_i_as_grpo = 0.0
    err_i_as_gfpo = 0.0


    if args.run_ht:
        gfpo_cfg_ht = GRPOConfig(
            lr=args.grpo_lr, beta_kl=args.beta_kl, ent_coef=args.ent_coef,
            device="cpu", batch_size=256, train_epochs=2, ref_update_interval=200,
        )

        gfpo_ht = GRPOAgent(
                seq_len=K, feat_dim=feat_dim_ht, n_actions=len(HT_DELTAS),
                cfg=gfpo_cfg_ht, seed=SEED,
                reward_cfg=GRPORewardCfg(
                target=target,
                tol=tol,
                mode="lex",        # "lex" recommended
                mix=args.alpha, #increase for tt
                alpha_sig=1.0,
                beta_move=args.beta,
                gamma_stab=0.25,
                k_violate=args.violation_penalty,
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

        # GFPO init (HT)

        # GFPO logs (HT)
        R_ht_gfpo_pct = []
        Cut_ht_gfpo   = []
        TT_ht_gfpo    = []
        AA_ht_gfpo    = []

        grpo_ht_losses = []

        gfpo_ht_losses = []



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





    # ---- near-cut occupancy logs (chunk-level) ----
    Occ_const_ad, Occ_pd_ad, Occ_dqn_ad, Occ_grpo_ad = [], [], [], []
    Occ_const_ht, Occ_pd_ht, Occ_dqn_ht, Occ_grpo_ht = [], [], [], []  # if --run-ht      

    batch_starts = list(range(start_event, N, chunk_size))
    micro_counter = 0
    grpo_samples = []   # one table, add column "trigger" = {"AS","HT"}


    # GFPO
    micro_counter_gfpo = 0
    micro_global = 0    # optional: single timeline across AS+HT micro-steps

    # ---- score distribution tracking (chunk-level) ----
    as_edges = _make_edges(Bas[start_event:], lo_q=0.5, hi_q=99.5, nbins=90)
    as_hists = []
    as_stats = []

    if args.run_ht:
        ht_edges = _make_edges(Bht[start_event:], lo_q=0.5, hi_q=99.5, nbins=90)
        ht_hists = []
        ht_stats = []
    

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

        # micro_rewards = []
        # micro_rewards_ht = []   # HT-GRPO executed rewards per micro-step (this chunk)
        micro_stride = max(1, int(args.micro_stride))
        micro_window = max(micro_stride, int(args.micro_window))

        n_micro = max(1, int(np.ceil((end - I) / micro_stride)))

        micro_rewards = [] #AS grpo
        micro_rewards_ht = []   # HT-GRPO executed rewards per micro-step (this chunk)


        micro_rewards_gfpo = [] #AS gfpo
        micro_rewards_gfpo_ht = []


        for j in range(n_micro):
            # new events arriving this micro-step
            j_new_lo = I + j * micro_stride
            j_new_hi = min(j_new_lo + micro_stride, end)
            if j_new_hi <= j_new_lo:
                continue

            idx_new = np.arange(j_new_lo, j_new_hi)
            bas_new = Bas[idx_new]
            bnpv_new = Bnpv[idx_new]

            # update rolling features with ONLY the new arrivals
            roll.append(bas_new, bnpv_new)
            bas_w, bnpv_w = roll.get()

            # evaluation window for bg/reward (overlapping, slides by micro_stride)
            j_eval_lo = max(I, j_new_hi - micro_window)
            idx_eval = np.arange(j_eval_lo, j_new_hi)
            bas_eval = Bas[idx_eval]
            bnpv_eval = Bnpv[idx_eval]
            # ---- micro evaluation slice (THIS replaces old bas_j/bnpv_j/idxj logic) ----
            bas_j = bas_eval
            bnpv_j = bnpv_eval

            for ctrl in controllers_as:
                # only micro-step for methods that actually update on micro
                if ctrl.name in ("DQN", "DQN-F", "ADT", "GRPO", "GFPO-F", "GFPO-FR", "PPO"):
                    out = ctrl.step_micro(
                        chunk=t,
                        bas_w=bas_w, bnpv_w=bnpv_w,
                        bas_j=bas_j,
                        sas_tt=sas_tt, sas_aa=sas_aa,
                        micro_global=micro_global,
                        grpo_samples=grpo_samples,   # keep unified log
                    )
                    micro_global = out.micro_global
 

            if args.run_ht:
                bht_new = Bht[idx_new]
                roll_ht.append(bht_new, bnpv_new)
                bht_w, bnpv_w_ht = roll_ht.get()
 
                # evaluation slice for measuring rate/reward this micro-step
                bht_j = Bht[idx_eval]

                for ctrl in controllers_ht:
                    if ctrl.name in ("DQN", "DQN-F", "ADT", "GRPO", "GFPO-F", "GFPO-FR", "PPO"):
                        out = ctrl.step_micro(
                            chunk=t,
                            bht_w=bht_w, bnpv_w=bnpv_w_ht,
                            bht_j=bht_j,
                            sht_tt=sht_tt, sht_aa=sht_aa,
                            micro_global=micro_global,
                            grpo_samples=grpo_samples,   # if HT GRPO ctrl supports it
                        )
                        micro_global = out.micro_global
                
        # ---- PRINT every 5 chunks for all baselines ----
        Kprint = 5
        if Kprint > 0 and ((t + 1) % Kprint == 0):
            print("\n" + "=" * 140)
            print(f"[WINDOW SUMMARY] chunks {max(0, t - Kprint + 1)}..{t}  (every {Kprint} chunks)")
            print_every_k_chunk_stats(chunk_rows, trigger="AD", c_hi=t, k=Kprint, target_pct=target, tol_pct=tol)
            if args.run_ht:
                print_every_k_chunk_stats(chunk_rows, trigger="HT", c_hi=t, k=Kprint, target_pct=target, tol_pct=tol)
            print("=" * 140 + "\n")

    
        # --- end-of-chunk hooks (ADT trains here) ---
        for ctrl in controllers_as:
            ctrl.end_chunk(chunk=t)

        if args.run_ht:
            for ctrl in controllers_ht:
                ctrl.end_chunk(chunk=t)
        # ============================
        # CHUNK-LEVEL logging (ONCE per chunk)
        # ============================
        logs_by_method = {}
        # PID chunk update happens once per chunk
        # Now log chunk metrics for ALL methods
        for ctrl in controllers_as:
            cut = ctrl.cut_value()
            bg = float(Sing_Trigger(bas, cut))
            tt = float(Sing_Trigger(sas_tt, cut))
            aa = float(Sing_Trigger(sas_aa, cut))
            occ = float(near_occupancy(bas, cut, near_widths_as)[1])

            cm = confusion_counts_at_cut_split(bas_j, sas_tt, sas_aa, cut)
            log_chunk_stats(
                chunk=t, trigger="AD", method=ctrl.name,
                cut=cut, bg_pct=bg, tt=tt, aa=aa,
                occ_mid=occ, target=target, tol=tol,
                tp=cm["tp"], fp=cm["fp"], tn=cm["tn"], fn=cm["fn"],
                tpr=cm["tpr"], fpr=cm["fpr"], precision=cm["precision"], f1=cm["f1"],
                tp_tt=cm["tp_tt"], fn_tt=cm["fn_tt"],
                tp_h4b=cm["tp_h4b"], fn_h4b=cm["fn_h4b"],
                tpr_tt=cm["tpr_tt"], precision_tt=cm["precision_tt"], f1_tt=cm["f1_tt"],
                tpr_h4b=cm["tpr_h4b"], precision_h4b=cm["precision_h4b"], f1_h4b=cm["f1_h4b"],
            )
            ctrl.end_chunk(chunk=t, bas_j=bas) 

        # AD score distribution for this chunk
        h_as, _ = np.histogram(bas, bins=as_edges, density=True)
        as_hists.append(h_as)
        as_stats.append(_score_chunk_stats(bas))

  
        # --- HT chunk-level logs (ONLY if enabled --run-ht) ---
        if args.run_ht:
            for ctrl in controllers_ht:
                cut = ctrl.cut_value()
                bg = float(Sing_Trigger(bht, cut))
                tt = float(Sing_Trigger(sht_tt, cut))
                aa = float(Sing_Trigger(sht_aa, cut))
                occ = float(near_occupancy(bht, cut, near_widths_ht)[1])
                cm = confusion_counts_at_cut_split(bht_j, sht_tt, sht_aa, cut)

                log_chunk_stats(
                    chunk=t, trigger="HT", method=ctrl.name,
                    cut=cut, bg_pct=bg, tt=tt, aa=aa,
                    occ_mid=occ, target=target, tol=tol,
                    tp=cm["tp"], fp=cm["fp"], tn=cm["tn"], fn=cm["fn"],
                    tpr=cm["tpr"], fpr=cm["fpr"], precision=cm["precision"], f1=cm["f1"],
                    tp_tt=cm["tp_tt"], fn_tt=cm["fn_tt"], precision_tt=cm["precision_tt"],
                    f1_tt=cm["f1_tt"],
                    tp_h4b=cm["tp_h4b"], fn_h4b=cm["fn_h4b"], precision_h4b=cm["precision_h4b"],
                    f1_h4b=cm["f1_h4b"],
                )
                ctrl.end_chunk(chunk=t, bht_j=bht)   # HT PID



            tt_ht_const = Sing_Trigger(sht_tt, fixed_Ht_cut)
            aa_ht_const = Sing_Trigger(sht_aa, fixed_Ht_cut)

            tt_ht_pd  = Sing_Trigger(sht_tt, Ht_cut_pd)
            aa_ht_pd  = Sing_Trigger(sht_aa, Ht_cut_pd)

            tt_ht_grpo = Sing_Trigger(sht_tt, Ht_cut_grpo)
            aa_ht_grpo = Sing_Trigger(sht_aa, Ht_cut_grpo)

            tt_ht_dqn = Sing_Trigger(sht_tt, Ht_cut_dqn)
            aa_ht_dqn = Sing_Trigger(sht_aa, Ht_cut_dqn)



    # --- always dump the chunk table ---
    write_chunk_stats_csv(tables_dir / "chunk_stats.csv")

    tag = _run_tag(args, target, tol)

    # ======================
    # (AD trigger) plots
    # ======================
    as_series = build_series_from_chunk_rows(chunk_rows, trigger="AD")

    # ======================
    # HT trigger plots
    # ======================
    if args.run_ht:
        ht_series = build_series_from_chunk_rows(chunk_rows, trigger="HT")
        


    target_khz = target * RATE_SCALE_KHZ
    tol_khz = tol * RATE_SCALE_KHZ
    upper_tol_khz = target_khz + tol_khz
    lower_tol_khz = target_khz - tol_khz


    # ----------------------------- Advantage distribution plots -----------------------------
    # Inspired by: https://arxiv.org/pdf/2504.08837 VL rethinker
    # Note: paper-style normalized advantages use (r - mean)/std. :contentReference[oaicite:1]{index=1}


    write_chunk_stats_csv(tables_dir / "chunk_stats.csv")

    make_original_plots_for_trigger(
        as_series,
        trigger_name="AD",
        fixed_cut=fixed_AS_cut,
        target=target, tol=tol,
        plots_dir=plots_dir,
        run_label=run_label,
        w=args.run_avg_window,
    )

    if args.run_ht:
        ht_series = build_series_from_chunk_rows(chunk_rows, trigger="HT")
        make_original_plots_for_trigger(
        ht_series,
        trigger_name="HT",
        fixed_cut=fixed_Ht_cut,
        target=target, tol=tol,
        plots_dir=plots_dir,
        run_label=run_label,
        w=args.run_avg_window,
        )


    paper_rows = build_paper_rows_from_chunk_rows(chunk_rows, target_pct=target, tol_pct=tol)
    write_paper_table(
        paper_rows,
        out_csv=tables_dir / "paper_table.csv",
        out_tex=tables_dir / "paper_table.tex",
        target_pct=target,
        tol_pct=tol,
    )
    write_confusion_split_tables_tex(
        chunk_rows,
        out_tt_tex=tables_dir / "confusion_tt.tex",
        out_h4b_tex=tables_dir / "confusion_h4b.tex",
    )


    tag = _run_tag(args, target, tol)

    make_gfpo_f_vs_fr_diagnostics(
        grpo_samples,
        trigger="AD",
        target=target, tol=tol,
        mix=float(args.gfpo_mix),
        group_size_keep=int(args.group_size_keep),
        plots_dir=plots_dir,
        run_label=run_label,
        tag=tag,
    )
    #show why filtering is needed (early GRPO collapse) ----
    if grpo_samples is not None and len(grpo_samples) > 0:
        micro_max = 50  # 

        for trig in (["AD"] + (["HT"] if args.run_ht else [])):
            # Entropy collapse comparison
            st_grpo = compute_micro_action_entropy(
                grpo_samples, trigger=trig, method="GRPO",
                target=target, tol=tol, kept_only=False
            )
            # Compare against kept-only distribution from one filter variant (pick one)
            st_gfpo = compute_micro_action_entropy(
                grpo_samples, trigger=trig, method="GFPO-F",
                target=target, tol=tol, kept_only=True
            )

            plot_entropy_timeseries(
                {"GRPO (candidates)": st_grpo, "GFPO-F (kept)": st_gfpo},
                title=f"{trig}: early exploration (entropy) shows GRPO collapse",
                outpath=plots_dir / f"grpo_entropy_collapse_{tag}_{trig.lower()}",
                run_label=run_label
            )

            # Early-window abs error histogram (this screams “filter outputs!”)
            ae_grpo = collect_candidate_abs_err_window(
                grpo_samples, trigger=trig, method="GRPO",
                target=target, micro_max=micro_max, kept_only=False
            )
            ae_gfpo = collect_candidate_abs_err_window(
                grpo_samples, trigger=trig, method="GFPO-F",
                target=target, micro_max=micro_max, kept_only=True
            )
            plot_early_abs_err_hist(
                ae_grpo, ae_gfpo,
                title=f"{trig}: first {micro_max} micro-steps candidate closeness",
                outpath=plots_dir / f"early_abs_err_{tag}_{trig.lower()}",
                run_label=run_label
            )

    # HT (optional)
    if args.run_ht:
        # build per-trigger in-band eff dicts: method -> {"tt":..., "h_to_4b":...}
        def _inband_eff(series):
            series = select_plot_methods(series)
            out = {}
            for m, s in series.items():
                inb = np.asarray(s["inband"], dtype=bool)
                out[m] = {
                    "tt": float(np.mean(np.asarray(s["tt"], dtype=np.float64)[inb])) if np.any(inb) else np.nan,
                    "h_to_4b": float(np.mean(np.asarray(s["aa"], dtype=np.float64)[inb])) if np.any(inb) else np.nan,
                }
            return out

        eff_ad = inband_eff_by_method(chunk_rows, "AD")
        eff_ht = inband_eff_by_method(chunk_rows, "HT")

        # Print a quick comparison table (h→4b)
        print("\nMean in-band h→4b efficiency (AD vs HT)")
        print("Method     AD(h4b)   HT(h4b)")
        for m in PLOT_METHODS:
            ad = eff_ad.get(m, {}).get("h_to_4b", np.nan)
            ht = eff_ht.get(m, {}).get("h_to_4b", np.nan)
            print(f"{m:<9}  {ad:8.4f}  {ht:8.4f}")
        # Print a quick comparison table (ttbar)
        print("\nMean in-band ttbar efficiency (AD vs HT)")
        print("Method     AD(ttbar)   HT(ttbar)")
        for m in PLOT_METHODS:
            ad = eff_ad.get(m, {}).get("tt", np.nan)
            ht = eff_ht.get(m, {}).get("tt", np.nan)
            print(f"{m:<9}  {ad:8.4f}  {ht:8.4f}")

        # ttbar plot (grouped by trigger)
        plot_inband_eff_grouped_by_trigger(
            eff_ad, eff_ht,
            signal_key="tt",
            signal_label=r"$t\bar{t}$",
            outpath=plots_dir / "inband_eff_ttbar_ad_vs_ht",
            run_label=run_label,
        )

        # h->4b plot (grouped by trigger)
        plot_inband_eff_grouped_by_trigger(
            eff_ad, eff_ht,
            signal_key="h_to_4b",
            signal_label=r"$h\rightarrow 4b$",
            outpath=plots_dir / "inband_eff_h4b_ad_vs_ht",
            run_label=run_label,
        )

        make_gfpo_f_vs_fr_diagnostics(
            grpo_samples,
            trigger="HT",
            target=target, tol=tol,
            mix=float(args.gfpo_mix),
            group_size_keep=int(args.group_size_keep),
            plots_dir=plots_dir,
            run_label=run_label,
            tag=tag,
        )


    write_paper_table(
        paper_rows,
        out_csv=tables_dir / "summary_table.csv",
        out_tex=tables_dir / "summary_table.tex",
        target_pct=target,
        tol_pct=tol,
    )

    def write_grpo_samples_csv(path: Path, rows):
        if not rows:
            return
        # union of keys
        cols = sorted({k for r in rows for k in r.keys()})
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
    # ---------------- Diagnostics: feasibility + advantage (GRPO vs GFPO) ----------------
    def _pick_first_existing(names):
        for n in names:
            if any(r.get("method") == n for r in grpo_samples):
                return n
        return None

    if grpo_samples:
        # --- AD diagnostics ---
        gfpo_name_as = _pick_first_existing(["GFPO-FR", "GFPO-F"])
        if any(r.get("method") == "GRPO" and r.get("trigger") == "AD" for r in grpo_samples) and gfpo_name_as:
            st_grpo = compute_feasibility_micro_stats(grpo_samples, trigger="AD", method="GRPO", target=target, tol=tol, group_size_keep=args.group_size_keep)
            st_gfpo = compute_feasibility_micro_stats(grpo_samples, trigger="AD", method=gfpo_name_as, target=target, tol=tol, group_size_keep=args.group_size_keep)
            plot_feasible_ratio_timeseries(
            st_grpo, st_gfpo,
            title=f"AD Trigger  (GRPO vs {gfpo_name_as})",
            outpath=plots_dir / "feasible_ratio_as",
            run_label=run_label,
            )
            plot_feasibility_bar(
            st_grpo, st_gfpo,
            title=f"AD Trigger  (GRPO vs {gfpo_name_as})",
            outpath=plots_dir / "feasibility_bar_as",
            run_label=run_label,
            )

            # advantage ECDF: GRPO candidates vs GFPO kept candidates
            _, adv_grpo_norm, _, _, _ = _group_advantages_from_samples(
            grpo_samples, trigger="AD", method="GRPO",
            baseline="mean", reward_key="reward_raw", kept_only=False
            )
            _, adv_gfpo_norm, _, _, _ = _group_advantages_from_samples(
            grpo_samples, trigger="AD", method=gfpo_name_as,
            baseline="mean", reward_key="reward_train", kept_only=True
            )
            _plot_adv_compare_ecdf(
            adv_grpo_norm, adv_gfpo_norm,
            title=f"AD Trigger  (GRPO vs {gfpo_name_as})",
            outpath=plots_dir / "adv_ecdf_as_grpo_vs_gfpo",
            run_label=run_label,
            )

        # --- HT diagnostics ---
        if args.run_ht:
            gfpo_name_ht = _pick_first_existing(["GFPO-FR", "GFPO-F"])
            if any(r.get("method") == "GRPO" and r.get("trigger") == "HT" for r in grpo_samples) and gfpo_name_ht:
                st_grpo = compute_feasibility_micro_stats(grpo_samples, trigger="HT", method="GRPO", target=target, tol=tol, group_size_keep=args.group_size_keep)
                st_gfpo = compute_feasibility_micro_stats(grpo_samples, trigger="HT", method=gfpo_name_ht, target=target, tol=tol, group_size_keep=args.group_size_keep)
                plot_feasible_ratio_timeseries(
                st_grpo, st_gfpo,
                title=f"HT Trigger  (GRPO vs {gfpo_name_ht})",
                outpath=plots_dir / "feasible_ratio_ht",
                run_label=run_label,
                )
    write_grpo_samples_csv(tables_dir / "grpo_samples.csv", grpo_samples)



if __name__ == "__main__":
    main()
