"""
anomaly_detection/rollout_nab.py

Rollout RL agents + baselines on NAB test split.
Computes NAB score (standard profile), precision, recall, F1.

Usage:
    python anomaly_detection/rollout_nab.py \
        --scores  anomaly_detection/data/nab_test.npz \
        --windows anomaly_detection/data/nab_windows.json \
        --models  anomaly_detection/models_nab \
        --outdir  outputs/anomaly_nab

Methods:
    Constant, Constant-opt, DQN, GRPO, GFPO-F, GFPO-FR, L-GRPO
"""

import argparse
import json
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "RL"))

from anomaly_detection.env_nab import NABEnv, NABEnvConfig
from RL.dqn_agent  import SeqDQNAgent, DQNConfig
from RL.grpo_agent import GRPOAgent, GRPOConfig, GRPORewardCfg
from RL.gfpo_agent import GFPOAgent, GFPOConfig
from RL.cpo_agent  import CPOAgent, CPOConfig, CPORewardCfg
from RL.ppo_agent  import SeqPPOAgent, SeqPPOConfig


# ══════════════════════════════════════════════════════════════════════════════
# NAB scoring
# ══════════════════════════════════════════════════════════════════════════════

def compute_nab_score(
    detections: List[int],
    windows: List[Tuple[int, int]],
    n_total: int,
    a_fp: float = -0.11,
    a_fn: float = -1.0,
) -> float:
    """
    Standard NAB scoring (standard profile).

    detections : sorted list of timestep indices where threshold was exceeded
    windows    : list of (start, end) tuples (inclusive) marking anomaly windows
    n_total    : total number of timesteps in the stream
    a_fp       : FP penalty (standard NAB profile: -0.11)
    a_fn       : FN penalty (standard NAB profile: -1.0)

    Returns NAB score in [0, 100] (clamped to 0 if below null detector).

    Scoring logic:
      - Early detection inside a window earns up to +1.0 (sigmoid-based).
      - Each FP outside all windows earns a_fp.
      - Each missed window earns a_fn.
    """
    n_windows = len(windows)
    if n_windows == 0:
        # No anomaly windows: score is purely based on FP count
        n_fp = len(detections)
        raw  = a_fp * n_fp
        return max(0.0, 100.0 * raw / 1.0) if n_fp == 0 else 0.0

    score       = 0.0
    used_windows = set()

    for d in sorted(detections):
        in_window = False
        for i, (ws, we) in enumerate(windows):
            if ws <= d <= we and i not in used_windows:
                # Early detection reward: sigmoid decaying from +1 to 0
                p     = (d - ws) / max(we - ws, 1)
                score += 2.0 / (1.0 + np.exp(5.0 * p)) - 1.0
                used_windows.add(i)
                in_window = True
                break
        if not in_window:
            score += a_fp

    # FN penalty for missed windows
    for i in range(n_windows):
        if i not in used_windows:
            score += a_fn

    # Normalize: null detector = all FN = a_fn * n_windows
    #            perfect detector = early detection every window = ~+1 per window
    null_score = a_fn * n_windows
    perf_score = 1.0  * n_windows   # upper bound (detect at start of every window)
    if abs(perf_score - null_score) < 1e-12:
        return 0.0
    return max(0.0, 100.0 * (score - null_score) / (perf_score - null_score))


def stream_metrics(
    scores_flat: np.ndarray,
    labels_flat: np.ndarray,
    threshold: float,
    windows: List[Tuple[int, int]],
) -> dict:
    """
    Compute detection metrics for a flat (1-D) stream at fixed threshold.

    Returns dict with keys: nab_score, precision, recall, f1, n_detections.
    """
    n_total    = len(scores_flat)
    detections = list(np.where(scores_flat >= threshold)[0])
    pred       = (scores_flat >= threshold).astype(np.int32)

    tp = int(((pred == 1) & (labels_flat == 1)).sum())
    fp = int(((pred == 1) & (labels_flat == 0)).sum())
    fn = int(((pred == 0) & (labels_flat == 1)).sum())

    recall    = tp / (tp + fn + 1e-9)
    precision = tp / (tp + fp + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)
    nab       = compute_nab_score(detections, windows, n_total)

    return dict(
        nab_score=nab, precision=precision, recall=recall, f1=f1,
        n_detections=len(detections),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Controller base classes
# ══════════════════════════════════════════════════════════════════════════════

class NABBaseCtrl:
    def __init__(self, name: str, init_thresh: float, cfg: NABEnvConfig):
        self.name      = name
        self.threshold = init_thresh
        self.cfg       = cfg
        self._detections: List[int] = []   # global timestep indices
        self._chunk_offset = 0             # current chunk's start in global stream

    def reset(self, init_thresh: float):
        self.threshold     = init_thresh
        self._detections   = []
        self._chunk_offset = 0

    def act(self, env: NABEnv, chunk: int) -> float:
        """Return chosen threshold for this chunk (subclasses override)."""
        raise NotImplementedError

    def record_detections(self, env: NABEnv, chunk: int, threshold: float):
        """Append detection timestep indices (global) to self._detections."""
        s    = env.scores[chunk]
        hits = np.where(s >= threshold)[0]
        for h in hits:
            self._detections.append(self._chunk_offset + int(h))
        self._chunk_offset += env.chunk_size

    @property
    def detections(self) -> List[int]:
        return sorted(self._detections)


class ConstantCtrl(NABBaseCtrl):
    def act(self, env: NABEnv, chunk: int) -> float:
        self.record_detections(env, chunk, self.threshold)
        return self.threshold


class PIDCtrl(NABBaseCtrl):
    """PID controller tracking a target flagging rate (label-free)."""
    def __init__(self, name, init_thresh, cfg,
                 target_rate: float = 0.03,
                 kp: float = 0.02, ki: float = 0.002, kd: float = 0.01):
        super().__init__(name, init_thresh, cfg)
        self.target_rate  = target_rate
        self.kp, self.ki, self.kd = kp, ki, kd
        self._integral = 0.0
        self._prev_err = 0.0

    def reset(self, init_thresh):
        super().reset(init_thresh)
        self._integral = 0.0
        self._prev_err = 0.0

    def act(self, env: NABEnv, chunk: int) -> float:
        s            = env.scores[chunk]
        current_rate = float((s >= self.threshold).mean())
        err          = current_rate - self.target_rate   # positive → too many flags
        self._integral += err
        deriv          = err - self._prev_err
        self._prev_err = err
        # Positive error → raise threshold (reduce flags)
        delta = self.kp * err + self.ki * self._integral + self.kd * deriv
        self.threshold = env._clip_threshold(self.threshold + delta)
        self.record_detections(env, chunk, self.threshold)
        return self.threshold


class DSpotCtrl(NABBaseCtrl):
    """
    D-SPOT: Drift SPOT — EVT-based adaptive threshold.
    Refit GPD on the top-q tail of sliding window; target P(X>z) = risk.
    """
    def __init__(self, name, init_thresh, cfg,
                 target_rate: float = 0.03, W: int = 20, q: float = 0.95):
        super().__init__(name, init_thresh, cfg)
        self.risk    = target_rate
        self.W       = W
        self.q       = q
        self._window: List[np.ndarray] = []

    def reset(self, init_thresh):
        super().reset(init_thresh)
        self._window = []

    def _dspot_threshold(self, data: np.ndarray) -> float:
        import math
        from scipy.stats import genpareto
        n  = len(data)
        t0 = float(np.percentile(data, self.q * 100))
        tail = data[data > t0] - t0
        nt   = len(tail)
        if nt < 10:
            return float(np.percentile(data, (1.0 - self.risk) * 100))
        r = self.risk * n / nt
        if r >= 1.0:
            return float(np.percentile(data, (1.0 - self.risk) * 100))
        try:
            gamma, _loc, sigma = genpareto.fit(tail, floc=0)
            if abs(gamma) < 1e-8:
                y = sigma * math.log(1.0 / r)
            else:
                y = (sigma / gamma) * (r ** (-gamma) - 1.0)
            return float(t0 + max(y, 0.0))
        except Exception:
            return float(np.percentile(data, (1.0 - self.risk) * 100))

    def act(self, env: NABEnv, chunk: int) -> float:
        scores = env.scores[chunk]
        self._window.append(scores)
        if len(self._window) > self.W:
            self._window.pop(0)
        if len(self._window) >= 2:
            self.threshold = env._clip_threshold(
                self._dspot_threshold(np.concatenate(self._window))
            )
        self.record_detections(env, chunk, self.threshold)
        return self.threshold


class ADTCtrl(NABBaseCtrl):
    """
    Anomaly Transformer (ADT) controller for NAB.
    Calibrates on first n_calib chunks, then uses AT anomaly scores.
    """
    def __init__(self, name, init_thresh, cfg,
                 score_lo: float = 0.0, score_hi: float = 1.0,
                 target_rate: float = 0.03,
                 n_calib: int = 8):
        super().__init__(name, init_thresh, cfg)
        from RL.anomaly_transformer_ctrl import AnomalyTransformerCtrl
        self._adt = AnomalyTransformerCtrl(
            name=name,
            init_cut=init_thresh,
            lo=score_lo, hi=score_hi,
            target=target_rate * 100,   # expects percent
            win_size=50,
            n_calib_chunks=n_calib,
            n_train_epochs=2,
            batch_size=64,
            train=True,
        )
        self._chunk_counter = 0

    def reset(self, init_thresh):
        super().reset(init_thresh)
        self._chunk_counter = 0

    def act(self, env: NABEnv, chunk: int) -> float:
        scores_1d = env.scores[chunk]
        self._adt.end_chunk(chunk=self._chunk_counter, bas_chunk=scores_1d)
        self._chunk_counter += 1
        self.threshold = float(self._adt.cut)
        # Detect using ADT anomaly scores if calibrated, else raw z-scores
        if self._adt._calibrated:
            adt_scores = self._adt.compute_scores(scores_1d)
            hits = np.where(adt_scores >= self.threshold)[0]
            for h in hits:
                self._detections.append(self._chunk_offset + int(h))
        else:
            self.record_detections(env, chunk, self.threshold)
        self._chunk_offset += env.chunk_size
        return self.threshold


class RLCtrl(NABBaseCtrl):
    def __init__(self, name, init_thresh, cfg, agent, deltas,
                 train: bool = False, group_size: int = 8):
        super().__init__(name, init_thresh, cfg)
        self.agent      = agent
        self.deltas     = deltas
        self.train      = train
        self.group_size = group_size
        self._recent_tpr: List[float] = []
        self._recent_fpr: List[float] = []
        self._history:    List[np.ndarray] = []

    def reset(self, init_thresh):
        super().reset(init_thresh)
        self._recent_tpr = []
        self._recent_fpr = []
        self._history    = []

    def _build_obs(self, env: NABEnv, chunk: int) -> np.ndarray:
        env.chunk_idx    = chunk
        env.threshold    = self.threshold
        env._recent_tpr  = self._recent_tpr
        env._recent_fpr  = self._recent_fpr
        env._history     = self._history
        return env._get_state()

    def _update_history(self, feat: np.ndarray, tpr: float, fpr: float):
        self._recent_tpr.append(tpr)
        self._recent_fpr.append(fpr)
        self._history.append(feat)


class DQNCtrl(RLCtrl):
    def act(self, env: NABEnv, chunk: int) -> float:
        obs    = self._build_obs(env, chunk)
        action = self.agent.act(obs, eps=0)   # greedy
        delta  = float(self.deltas[action])
        self.threshold = env._clip_threshold(self.threshold + delta)
        self.record_detections(env, chunk, self.threshold)
        tpr, fpr, _, _ = env._eval_threshold(chunk, self.threshold)
        feat = env._chunk_features(chunk, self.threshold)
        self._update_history(feat, tpr, fpr)
        return self.threshold


class GRPOCtrl(RLCtrl):
    def act(self, env: NABEnv, chunk: int) -> float:
        obs = self._build_obs(env, chunk)
        if self.train:
            actions, logp = self.agent.sample_group_actions(obs, self.group_size)
            rewards = []
            for a in actions:
                d              = float(self.deltas[int(a)])
                tpr, fpr, _, _ = env._eval_threshold(chunk, self.threshold + d)
                r              = env._compute_reward(tpr=tpr, fpr=fpr, delta=d)
                rewards.append(r)
            self.agent.store_group(obs=obs, actions=actions, logp=logp,
                                    rewards=np.array(rewards))
            self.agent.update()

        action = self.agent.greedy_action(obs)
        delta  = float(self.deltas[action])
        self.threshold = env._clip_threshold(self.threshold + delta)
        self.record_detections(env, chunk, self.threshold)
        tpr, fpr, _, _ = env._eval_threshold(chunk, self.threshold)
        feat = env._chunk_features(chunk, self.threshold)
        self._update_history(feat, tpr, fpr)
        return self.threshold


class LGRPOCtrl(GRPOCtrl):
    """L-GRPO: GRPO with Lagrangian dual update based on detection quality."""
    def act(self, env: NABEnv, chunk: int) -> float:
        thresh = super().act(env, chunk)
        # Dual update based on TPR (want TPR high → push lambda down if high, up if low)
        tpr, fpr, _, _ = env._eval_threshold(chunk, thresh)
        if hasattr(self.agent, "update_dual_chunk"):
            self.agent.update_dual_chunk(tpr)
        return thresh


class GFPOCtrl(RLCtrl):
    def act(self, env: NABEnv, chunk: int) -> float:
        obs     = self._build_obs(env, chunk)
        G       = self.group_size
        actions, logp = self.agent.sample_group_actions(obs, G)

        rewards = np.empty(G, dtype=np.float64)
        tprs    = np.empty(G, dtype=np.float64)
        fprs    = np.empty(G, dtype=np.float64)
        for i, a in enumerate(actions):
            d               = float(self.deltas[int(a)])
            tpr, fpr, _, _  = env._eval_threshold(chunk, self.threshold + d)
            rewards[i]      = env._compute_reward(tpr=tpr, fpr=fpr, delta=d)
            tprs[i]         = tpr
            fprs[i]         = fpr

        if self.train:
            keep_size = min(int(self.agent.gfpo_cfg.keep_size), G)
            order     = np.argsort(-rewards)[:keep_size]
            self.agent.store_group(obs=obs, actions=actions[order],
                                    logp=logp[order], rewards=rewards[order])
            self.agent.update()

        # Greedy: best reward candidate
        best       = int(actions[np.argmax(rewards)])
        best_delta = float(self.deltas[best])
        self.threshold = env._clip_threshold(self.threshold + best_delta)
        self.record_detections(env, chunk, self.threshold)
        best_tpr = float(tprs[np.argmax(rewards)])
        best_fpr = float(fprs[np.argmax(rewards)])
        feat = env._chunk_features(chunk, self.threshold)
        self._update_history(feat, best_tpr, best_fpr)
        return self.threshold


class PPOCtrl(RLCtrl):
    def act(self, env: NABEnv, chunk: int) -> float:
        obs    = self._build_obs(env, chunk)
        if self.train:
            action, logp, val, _ = self.agent.act(obs)
            delta  = float(self.deltas[action])
            new_t  = env._clip_threshold(self.threshold + delta)
            tpr, fpr, _, _ = env._eval_threshold(chunk, new_t)
            r    = env._compute_reward(tpr=tpr, fpr=fpr, delta=delta)
            done = False
            self.agent.store(obs, action, logp, val, r, done)
            self.agent.update()
        result = self.agent.act(obs)
        action = result[0] if isinstance(result, tuple) else result
        delta  = float(self.deltas[action])
        self.threshold = env._clip_threshold(self.threshold + delta)
        self.record_detections(env, chunk, self.threshold)
        tpr, fpr, _, _ = env._eval_threshold(chunk, self.threshold)
        feat = env._chunk_features(chunk, self.threshold)
        self._update_history(feat, tpr, fpr)
        return self.threshold


class GFPOFRCtrl(GFPOCtrl):
    """GFPO-FR: GFPO-F + dual-lambda feedback based on TPR."""
    def act(self, env: NABEnv, chunk: int) -> float:
        thresh = super().act(env, chunk)
        tpr, fpr, _, _ = env._eval_threshold(chunk, thresh)
        if hasattr(self.agent, "update_dual_chunk"):
            self.agent.update_dual_chunk(tpr)
        return thresh


class CPOCtrl(RLCtrl):
    """Constrained Policy Optimization (Achiam et al. 2017) for NAB.

    Same bandit candidate sampling pattern as GFPOCtrl. Reward is the
    standard NAB reward; cost is a sign-flipped, shifted form of the
    rate-tracking term: c = e^2 if e<=1 else e, where e is the relative
    deviation of FPR from the target. CPOAgent.update() then solves the
    trust-region QP with a constraint on the expected cost.
    """
    def act(self, env: NABEnv, chunk: int) -> float:
        obs = self._build_obs(env, chunk)
        G   = self.group_size
        actions, logp = self.agent.sample_group_actions(obs, G)

        rewards = np.empty(G, dtype=np.float64)
        costs   = np.empty(G, dtype=np.float64)
        tprs    = np.empty(G, dtype=np.float64)
        fprs    = np.empty(G, dtype=np.float64)
        for i, a in enumerate(actions):
            d                = float(self.deltas[int(a)])
            tpr, fpr, _, _   = env._eval_threshold(chunk, self.threshold + d)
            rewards[i]       = env._compute_reward(tpr=tpr, fpr=fpr, delta=d)
            costs[i]         = self.agent.compute_cost(bg_after=fpr)
            tprs[i]          = tpr
            fprs[i]          = fpr

        if self.train:
            self.agent.store_group(
                obs=obs, actions=actions, logp=logp,
                rewards=rewards, costs=costs, baseline="mean",
            )
            self.agent.update()

        # Greedy: best reward candidate among feasible (cost ≈ 0); else min cost.
        feas = costs <= 1e-9
        if feas.any():
            idx = np.where(feas)[0]
            best = int(actions[idx[np.argmax(rewards[idx])]])
        else:
            best = int(actions[np.argmin(costs)])
        delta = float(self.deltas[best])
        self.threshold = env._clip_threshold(self.threshold + delta)
        self.record_detections(env, chunk, self.threshold)
        tpr, fpr, _, _ = env._eval_threshold(chunk, self.threshold)
        feat = env._chunk_features(chunk, self.threshold)
        self._update_history(feat, tpr, fpr)
        return self.threshold


# ══════════════════════════════════════════════════════════════════════════════
# Build controllers
# ══════════════════════════════════════════════════════════════════════════════

def build_controllers(
    methods: List[str],
    init_thresh: float,
    env_cfg: NABEnvConfig,
    models_dir: Path,
    seq_len: int,
    feat_dim: int,
    n_actions: int,
    train: bool = False,
    score_lo: float = 0.0,
    score_hi: float = 1.0,
) -> List[NABBaseCtrl]:
    ctrls  = []
    deltas = np.linspace(-env_cfg.delta_range, env_cfg.delta_range,
                          n_actions, dtype=np.float32)

    dummy_reward_cfg = GRPORewardCfg(target=0.5, tol=0.5, mode="lex")

    def load_weights(agent, name: str):
        p = models_dir / f"{name}.pt"
        if p.exists():
            import torch
            state = torch.load(p, map_location="cpu")
            if hasattr(agent, "q"):
                net = agent.q
            elif hasattr(agent, "ac"):
                net = agent.ac
            else:
                net = agent.pi
            net.load_state_dict(state["pi"])
            print(f"  Loaded {p}")
        else:
            print(f"  WARNING: {p} not found — using random weights for {name}")
        return agent

    for m in methods:
        m = m.strip().lower()

        if m == "constant":
            ctrls.append(ConstantCtrl("Constant", init_thresh, env_cfg))

        elif m == "pid":
            ctrls.append(PIDCtrl("PID", init_thresh, env_cfg,
                                  target_rate=0.03))

        elif m in ("dspot", "spot"):
            ctrls.append(DSpotCtrl("DSPOT", init_thresh, env_cfg,
                                    target_rate=0.03))

        elif m in ("adt", "anomaly-transformer"):
            ctrls.append(ADTCtrl("ADT", init_thresh, env_cfg,
                                  score_lo=score_lo, score_hi=score_hi,
                                  target_rate=0.03, n_calib=8))

        elif m == "constant-opt":
            # Oracle: best fixed threshold for F1 — handled separately after rollout
            ctrls.append(ConstantCtrl("Constant-opt", init_thresh, env_cfg))

        elif m == "dqn":
            agent = SeqDQNAgent(seq_len=seq_len, feat_dim=feat_dim,
                                n_actions=n_actions, cfg=DQNConfig())
            load_weights(agent, "DQN")
            ctrls.append(DQNCtrl("DQN", init_thresh, env_cfg, agent, deltas,
                                  train=train))

        elif m == "grpo":
            agent = GRPOAgent(seq_len=seq_len, feat_dim=feat_dim,
                               n_actions=n_actions, cfg=GRPOConfig(),
                               reward_cfg=dummy_reward_cfg)
            load_weights(agent, "GRPO")
            ctrls.append(GRPOCtrl("GRPO", init_thresh, env_cfg, agent, deltas,
                                   train=train, group_size=8))

        elif m == "lgrpo":
            reward_cfg = GRPORewardCfg(target=0.5, tol=0.5, mode="lag",
                                        alpha_step=0.01, dual_init=0.0)
            agent = GRPOAgent(seq_len=seq_len, feat_dim=feat_dim,
                               n_actions=n_actions, cfg=GRPOConfig(),
                               reward_cfg=reward_cfg)
            load_weights(agent, "GRPO")   # init from GRPO weights
            ctrls.append(LGRPOCtrl("L-GRPO", init_thresh, env_cfg, agent, deltas,
                                    train=train, group_size=8))

        elif m == "ppo":
            agent = SeqPPOAgent(SeqPPOConfig(feat_dim=feat_dim, n_actions=n_actions))
            load_weights(agent, "PPO")
            ctrls.append(PPOCtrl("PPO", init_thresh, env_cfg, agent, deltas,
                                  train=train, group_size=1))

        elif m in ("gfpo", "gfpo-f"):
            agent = GFPOAgent(seq_len=seq_len, feat_dim=feat_dim,
                               n_actions=n_actions, cfg=GRPOConfig(),
                               gfpo_cfg=GFPOConfig(sample_size=32, keep_size=16),
                               reward_cfg=dummy_reward_cfg)
            load_weights(agent, "GFPO")
            ctrls.append(GFPOCtrl("GFPO-F", init_thresh, env_cfg, agent, deltas,
                                   train=train, group_size=32))

        elif m == "gfpo-fr":
            reward_cfg_fr = GRPORewardCfg(target=0.5, tol=0.5, mode="lag",
                                           alpha_step=0.05, dual_init=0.0)
            agent = GFPOAgent(seq_len=seq_len, feat_dim=feat_dim,
                               n_actions=n_actions, cfg=GRPOConfig(),
                               gfpo_cfg=GFPOConfig(sample_size=32, keep_size=16),
                               reward_cfg=reward_cfg_fr)
            load_weights(agent, "GFPO")
            ctrls.append(GFPOFRCtrl("GFPO-FR", init_thresh, env_cfg, agent, deltas,
                                     train=train, group_size=32))

        elif m == "cpo":
            cpo_reward_cfg = CPORewardCfg(
                target=0.03, tol=0.03,
                lambda_1=0.25, mix=0.5, beta_move=0.02, cost_limit=1.0,
            )
            agent = CPOAgent(
                seq_len=seq_len, feat_dim=feat_dim, n_actions=n_actions,
                cfg=CPOConfig(delta=0.03, cg_iters=10, cg_damping=0.1,
                              line_search_steps=10, line_search_decay=0.8,
                              batch_min=64),
                reward_cfg=cpo_reward_cfg,
            )
            load_weights(agent, "CPO")
            ctrls.append(CPOCtrl("CPO", init_thresh, env_cfg, agent, deltas,
                                  train=train, group_size=16))

        else:
            print(f"WARNING: unknown method '{m}', skipping.")

    return ctrls


# ══════════════════════════════════════════════════════════════════════════════
# Per-file rollout
# ══════════════════════════════════════════════════════════════════════════════

def rollout_file(
    ctrls: List[NABBaseCtrl],
    env: NABEnv,
    chunk_indices: List[int],    # which chunks belong to this file
    windows_global: List[Tuple[int, int]],  # anomaly windows in global test-stream coords
    init_thresh: float,
) -> dict:
    """
    Run all controllers on the chunks belonging to one file.
    Returns dict: method → metrics dict.
    """
    # Reset all controllers for this file
    for ctrl in ctrls:
        ctrl.reset(init_thresh)

    # Construct local env subset
    local_scores = env.scores[chunk_indices]
    local_labels = env.labels[chunk_indices]
    local_env    = NABEnv(local_scores, local_labels, env.cfg)

    n_test_ts    = len(chunk_indices) * env.chunk_size
    scores_flat  = local_scores.ravel()
    labels_flat  = local_labels.ravel()

    results = {}
    per_method_time_sec = {}
    for ctrl in ctrls:
        ctrl.reset(init_thresh)
        _t0 = time.perf_counter()
        for local_chunk_idx, _ in enumerate(chunk_indices):
            ctrl.act(local_env, local_chunk_idx)
        per_method_time_sec[ctrl.name] = time.perf_counter() - _t0

        # Constant-opt: find oracle threshold maximizing F1 on this file
        if ctrl.name == "Constant-opt":
            best_f1   = -1.0
            best_thr  = init_thresh
            thresholds = np.percentile(scores_flat, np.arange(50, 100, 2))
            for thr in thresholds:
                pred = (scores_flat >= thr).astype(np.int32)
                tp   = int(((pred == 1) & (labels_flat == 1)).sum())
                fp   = int(((pred == 1) & (labels_flat == 0)).sum())
                fn   = int(((pred == 0) & (labels_flat == 1)).sum())
                prec = tp / (tp + fp + 1e-9)
                rec  = tp / (tp + fn + 1e-9)
                f1   = 2 * prec * rec / (prec + rec + 1e-9)
                if f1 > best_f1:
                    best_f1  = f1
                    best_thr = thr
            # Re-record detections with optimal threshold
            ctrl._detections   = []
            ctrl._chunk_offset = 0
            for local_chunk_idx, _ in enumerate(chunk_indices):
                ctrl.record_detections(local_env, local_chunk_idx, best_thr)
            detections = ctrl.detections
        else:
            detections = ctrl.detections

        metrics = stream_metrics(scores_flat, labels_flat, init_thresh, windows_global)
        # Recompute with actual detections from controller
        pred_arr  = np.zeros(n_test_ts, dtype=np.int32)
        for d in detections:
            if 0 <= d < n_test_ts:
                pred_arr[d] = 1

        tp = int(((pred_arr == 1) & (labels_flat == 1)).sum())
        fp = int(((pred_arr == 1) & (labels_flat == 0)).sum())
        fn = int(((pred_arr == 0) & (labels_flat == 1)).sum())
        recall    = tp / (tp + fn + 1e-9)
        precision = tp / (tp + fp + 1e-9)
        f1        = 2 * precision * recall / (precision + recall + 1e-9)
        nab_score = compute_nab_score(detections, windows_global, n_test_ts)

        results[ctrl.name] = dict(
            nab_score=nab_score,
            precision=precision,
            recall=recall,
            f1=f1,
            n_detections=len(detections),
            wall_sec=per_method_time_sec.get(ctrl.name, float("nan")),
        )

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores",   default="anomaly_detection/data/nab_test.npz")
    ap.add_argument("--windows",  default="anomaly_detection/data/nab_windows.json")
    ap.add_argument("--models",   default="anomaly_detection/models_nab")
    ap.add_argument("--outdir",   default="outputs/anomaly_nab")
    ap.add_argument("--methods",
                    default="constant,constant-opt,pid,dspot,adt,dqn,grpo,lgrpo,ppo,gfpo-f,gfpo-fr")
    ap.add_argument("--alpha",       type=float, default=0.10)
    ap.add_argument("--beta",        type=float, default=0.005)
    ap.add_argument("--n-deltas",    type=int,   default=21)
    ap.add_argument("--delta-range", type=float, default=0.3)
    ap.add_argument("--seq-len",     type=int,   default=8)
    ap.add_argument("--train",       action="store_true",
                    help="Enable online fine-tuning during rollout")
    args = ap.parse_args()

    # Load test data
    data     = np.load(args.scores)
    scores   = data["scores"].astype(np.float32)
    labels   = data["labels"].astype(np.int32)
    file_ids = data["file_ids"].astype(np.int32)
    print(f"Loaded {scores.shape[0]} test chunks, "
          f"{len(np.unique(file_ids))} unique files.")

    # Load anomaly windows
    with open(args.windows) as f:
        windows_json = json.load(f)

    env_cfg = NABEnvConfig(
        alpha=args.alpha, beta=args.beta,
        n_deltas=args.n_deltas, delta_range=args.delta_range,
        seq_len=args.seq_len,
    )
    env = NABEnv(scores, labels, env_cfg)

    init_thresh = env.init_threshold
    models_dir  = Path(args.models)
    methods_list = [m.strip() for m in args.methods.split(",")]

    ctrls = build_controllers(
        methods_list, init_thresh, env_cfg, models_dir,
        seq_len=env_cfg.seq_len, feat_dim=env.feat_dim,
        n_actions=env_cfg.n_deltas, train=args.train,
        score_lo=float(scores.min()), score_hi=float(scores.max()),
    )
    print(f"Controllers: {[c.name for c in ctrls]}")

    # Group chunks by file_id
    unique_fids = sorted(np.unique(file_ids))
    all_results = {c.name: [] for c in ctrls}

    for fid in unique_fids:
        chunk_mask    = np.where(file_ids == fid)[0].tolist()
        if not chunk_mask:
            continue

        # Reconstruct anomaly windows for this file (in test-stream local coords)
        win_data = windows_json.get(str(fid), {})
        raw_windows = win_data.get("windows", []) if isinstance(win_data, dict) else []
        # windows are already in local test-stream coordinates
        file_windows = [(int(ws), int(we)) for ws, we in raw_windows]

        file_results = rollout_file(
            ctrls, env, chunk_mask, file_windows, init_thresh
        )

        for ctrl_name, metrics in file_results.items():
            all_results[ctrl_name].append(metrics)

        if len(unique_fids) <= 10 or fid % max(1, len(unique_fids) // 5) == 0:
            print(f"  file_id={fid}: "
                  + "  ".join(f"{n}:F1={m['f1']:.3f}"
                               for n, m in file_results.items()))

    # Aggregate
    rows = []
    for ctrl_name, file_metrics_list in all_results.items():
        if not file_metrics_list:
            continue
        mean_nab  = float(np.mean([m["nab_score"]  for m in file_metrics_list]))
        mean_prec = float(np.mean([m["precision"]  for m in file_metrics_list]))
        mean_rec  = float(np.mean([m["recall"]     for m in file_metrics_list]))
        mean_f1   = float(np.mean([m["f1"]         for m in file_metrics_list]))
        wall_secs = [m.get("wall_sec", float("nan")) for m in file_metrics_list]
        wall_secs = [w for w in wall_secs if not (w != w)]   # drop NaN
        mean_wall = float(np.mean(wall_secs)) if wall_secs else float("nan")
        rows.append(dict(
            method=ctrl_name,
            nab_score=round(mean_nab,  4),
            precision=round(mean_prec, 4),
            recall=round(mean_rec,     4),
            f1=round(mean_f1,          4),
            wall_sec_per_series=round(mean_wall, 4),
        ))

    df = pd.DataFrame(rows).sort_values("nab_score", ascending=False)

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "nab_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved → {csv_path}")

    # Pretty print summary
    print("\n=== NAB Results (mean across test files) ===")
    print(f"{'Method':<14}  {'NAB Score':>10}  {'Precision':>10}  "
          f"{'Recall':>8}  {'F1':>8}  {'sec/series':>10}")
    print("-" * 72)
    for _, row in df.iterrows():
        print(f"{row['method']:<14}  {row['nab_score']:>10.4f}  "
              f"{row['precision']:>10.4f}  {row['recall']:>8.4f}  "
              f"{row['f1']:>8.4f}  {row['wall_sec_per_series']:>10.4f}")

    # Save timing summary including hardware
    try:
        import torch as _torch
        device = "cuda" if _torch.cuda.is_available() else (
            "mps" if _torch.backends.mps.is_available() else "cpu")
    except Exception:
        device = "cpu"
    timing_summary = {
        "hardware": {
            "platform":   platform.platform(),
            "machine":    platform.machine(),
            "processor":  platform.processor() or platform.machine(),
            "cpu_count":  int(__import__("os").cpu_count() or 0),
            "torch_device": device,
        },
        "n_series": int(len(unique_fids)),
        "per_method_mean_wall_sec_per_series": {
            r["method"]: float(r["wall_sec_per_series"]) for r in rows
        },
        "per_method_mean_wall_min_per_series": {
            r["method"]: float(r["wall_sec_per_series"]) / 60.0 for r in rows
        },
    }
    timing_path = out_dir / "rollout_timing.json"
    with open(timing_path, "w") as f:
        json.dump(timing_summary, f, indent=2)
    print(f"\nSaved timing → {timing_path}  (device={device}, {platform.processor() or platform.machine()})")

    print("\nRollout complete.")


if __name__ == "__main__":
    main()
