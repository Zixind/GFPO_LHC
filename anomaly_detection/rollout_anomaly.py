"""
anomaly_detection/rollout_anomaly.py

Online anomaly detection rollout on UNSW-NB15.
Runs all methods (Constant, PID, DSPOT, DQN, GRPO, L-GRPO, GFPO-F, GFPO-FR, PPO)
and saves per-chunk metrics to outputs/anomaly/<run_name>/tables/chunk_stats.csv.

Usage:
    python anomaly_detection/rollout_anomaly.py \
        --scores  anomaly_detection/data/unsw_scores.npz \
        --models  anomaly_detection/models/ \
        --outdir  outputs/anomaly_unsw \
        --baselines "constant,pid,dspot,dqn,grpo,lgrpo,gfpo,ppo"

Prerequisites:
    1.  python anomaly_detection/preprocess_unsw.py
    2.  python anomaly_detection/base_detector.py
    3.  python anomaly_detection/train_anomaly.py   (to produce models/)
"""

import argparse
import sys
import os
import math
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd

# ── make project root importable ─────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "RL"))

from env_anomaly import AnomalyEnv, AnomalyEnvConfig
from RL.dqn_agent   import SeqDQNAgent, DQNConfig
from RL.grpo_agent  import GRPOAgent, GRPOConfig, GRPORewardCfg
from RL.gfpo_agent  import GFPOAgent, GFPOConfig
from RL.ppo_agent   import SeqPPOAgent, SeqPPOConfig
from RL.cpo_agent   import CPOAgent, CPOConfig, CPORewardCfg


# ══════════════════════════════════════════════════════════════════════════════
# Baseline controllers
# ══════════════════════════════════════════════════════════════════════════════

class BaseCtrl:
    def __init__(self, name: str, init_thresh: float, cfg: AnomalyEnvConfig):
        self.name       = name
        self.threshold  = init_thresh
        self.cfg        = cfg
        self._rows: List[dict] = []

    def act(self, env: AnomalyEnv, chunk: int) -> dict:
        raise NotImplementedError

    def _tpr_by_cat(self, env: AnomalyEnv, chunk: int, threshold: float):
        """Compute TPR_easy (cat==1) and TPR_hard (cat==2) at given threshold."""
        cat = getattr(env, "cat", None)
        if cat is None:
            return float("nan"), float("nan")
        s   = env.scores[chunk]
        c   = cat[chunk]
        pred = (s >= threshold).astype(np.int32)
        def _tpr_for(mask):
            if mask.sum() == 0:
                return float("nan")
            return float(pred[mask].mean())
        return _tpr_for(c == 1), _tpr_for(c == 2)

    def log(self, chunk: int, far: float, tpr: float, threshold: float,
            env: "AnomalyEnv" = None):
        inband = int(abs(far - self.cfg.far_target) <= self.cfg.far_tol)
        if env is not None:
            tpr_easy, tpr_hard = self._tpr_by_cat(env, chunk, threshold)
        else:
            tpr_easy = tpr_hard = float("nan")
        self._rows.append(dict(
            chunk=chunk, method=self.name,
            threshold=threshold, far=far, tpr=tpr, inband=inband,
            tpr_easy=tpr_easy, tpr_hard=tpr_hard,
        ))

    @property
    def rows(self):
        return self._rows


class ConstantCtrl(BaseCtrl):
    def act(self, env: AnomalyEnv, chunk: int) -> dict:
        far, tpr, _, _ = env._eval_threshold(chunk, self.threshold)
        self.log(chunk, far, tpr, self.threshold, env=env)
        return dict(far=far, tpr=tpr)


class PIDCtrl(BaseCtrl):
    def __init__(self, name, init_thresh, cfg,
                 kp=0.05, ki=0.005, kd=0.02):
        super().__init__(name, init_thresh, cfg)
        self.kp, self.ki, self.kd = kp, ki, kd
        self._integral = 0.0
        self._prev_err = 0.0

    def act(self, env: AnomalyEnv, chunk: int) -> dict:
        far, tpr, _, _ = env._eval_threshold(chunk, self.threshold)
        err             = far - self.cfg.far_target
        self._integral += err
        deriv           = err - self.prev_err
        self._prev_err  = err
        delta = self.kp * err + self.ki * self._integral + self.kd * deriv
        self.threshold += delta
        far2, tpr2, _, _ = env._eval_threshold(chunk, self.threshold)
        self.log(chunk, far2, tpr2, self.threshold, env=env)
        return dict(far=far2, tpr=tpr2)

    @property
    def prev_err(self):
        return self._prev_err


class DSpotCtrl(BaseCtrl):
    """
    D-SPOT: Drift Streaming Peaks Over Threshold.

    Reference: Siffer, Fouque, Termier & Largouet, "Anomaly Detection in
    Streams with Extreme Value Theory", KDD 2017.

    At each chunk the window slides forward by one chunk.  The POT threshold
    t0 is recomputed as the q-th quantile of the current window, the GPD is
    refit on exceedances above t0, and the anomaly threshold z_α is derived
    analytically so that P(score > z_α) ≈ far_target over the window.

    Parameters
    ----------
    W : int   – sliding window width (chunks).  Larger W → slower adaptation.
    q : float – quantile level for the POT pre-threshold t0 within the window
                (scores above t0 are treated as extreme; 0.98 ≈ top 2%).

    Note on UNSW-NB15
    -----------------
    D-SPOT is label-free: it fits the GPD to the top-q tail of ALL scores in
    the window (benign + attack).  In UNSW-NB15 the test stream is ~55%
    attacks, whose scores are ~5x higher than benign scores.  Consequently the
    top-2% tail is entirely in attack-score territory, the derived threshold
    exceeds virtually all benign scores, and FAR ≈ 0%.  This is the expected
    behaviour of D-SPOT when anomaly prevalence is high — the algorithm assumes
    a mostly-normal stream with rare anomalies, which UNSW-NB15 violates.
    """

    def __init__(self, name: str, init_thresh: float, cfg: AnomalyEnvConfig,
                 W: int = 30, q: float = 0.98):
        super().__init__(name, init_thresh, cfg)
        self.W    = W
        self.q    = q
        self.risk = cfg.far_target          # target FAR = risk of exceedance
        self._window: List[np.ndarray] = [] # sliding buffer, one array per chunk

    # ── public ────────────────────────────────────────────────────────────────

    def act(self, env: AnomalyEnv, chunk: int) -> dict:
        scores = env.scores[chunk]

        # slide window
        self._window.append(scores)
        if len(self._window) > self.W:
            self._window.pop(0)

        # refit on every chunk once we have at least 2 chunks of data
        if len(self._window) >= 2:
            self.threshold = self._dspot_threshold(np.concatenate(self._window))

        far, tpr, _, _ = env._eval_threshold(chunk, self.threshold)
        self.log(chunk, far, tpr, self.threshold, env=env)
        return dict(far=far, tpr=tpr)

    # ── core EVT fitting ──────────────────────────────────────────────────────

    def _dspot_threshold(self, data: np.ndarray) -> float:
        """
        Fit GPD to exceedances and return z_α such that P(X > z_α) ≈ risk.

        POT formula (Pickands–Balkema–de Haan):
          P(X > z_α) = (Nt/n) · (1 + γ·y/σ)^(−1/γ)  where y = z_α − t0
          Setting equal to `risk` and solving for y:
            γ ≠ 0:  y = (σ/γ) · ((risk·n/Nt)^(−γ) − 1)
            γ = 0:  y = σ · ln(Nt / (risk·n))
        """
        from scipy.stats import genpareto

        n  = len(data)
        t0 = float(np.percentile(data, self.q * 100))

        tail = data[data > t0] - t0
        nt   = len(tail)

        # Fallback: not enough exceedances for a reliable GPD fit
        if nt < 10:
            return float(np.percentile(data, (1.0 - self.risk) * 100))

        r = self.risk * n / nt      # conditional exceedance probability target
        if r >= 1.0:
            # risk is so high that the empirical quantile is more stable
            return float(np.percentile(data, (1.0 - self.risk) * 100))

        try:
            gamma, _loc, sigma = genpareto.fit(tail, floc=0)
            if abs(gamma) < 1e-8:                          # exponential case
                y = sigma * math.log(1.0 / r)
            else:
                y = (sigma / gamma) * (r ** (-gamma) - 1.0)
            return float(t0 + max(y, 0.0))
        except Exception:
            return float(np.percentile(data, (1.0 - self.risk) * 100))


class ADTCtrl(BaseCtrl):
    """
    Anomaly Transformer (ADT) controller.
    Calibrates on first n_calib chunks, then adapts threshold using learned scores.
    """
    def __init__(self, name, init_thresh, cfg, n_calib=8):
        super().__init__(name, init_thresh, cfg)
        from RL.anomaly_transformer_ctrl import AnomalyTransformerCtrl
        self._adt = AnomalyTransformerCtrl(
            name=name,
            init_cut=init_thresh,
            lo=0.0, hi=1.0,
            target=cfg.far_target * 100,   # expects percent
            win_size=50,
            n_calib_chunks=n_calib,
            n_train_epochs=2,
            batch_size=64,
            train=True,
        )
        self._chunk_counter = 0

    def act(self, env: AnomalyEnv, chunk: int) -> dict:
        scores_1d = env.scores[chunk]
        self._adt.end_chunk(chunk=self._chunk_counter, bas_chunk=scores_1d)
        self._chunk_counter += 1
        self.threshold = float(self._adt.cut)
        far, tpr, _, _ = env._eval_threshold(chunk, self.threshold)
        self.log(chunk, far, tpr, self.threshold, env=env)
        return dict(far=far, tpr=tpr)


# ── RL controller base ────────────────────────────────────────────────────────

class RLCtrl(BaseCtrl):
    def __init__(self, name, init_thresh, cfg, agent, deltas,
                 train: bool = True, group_size: int = 8):
        super().__init__(name, init_thresh, cfg)
        self.agent      = agent
        self.deltas     = deltas
        self.train      = train
        self.group_size = group_size

    def _build_obs(self, env: AnomalyEnv, chunk: int) -> np.ndarray:
        env.chunk_idx  = chunk
        env.threshold  = self.threshold
        return env._get_state()

    def act(self, env: AnomalyEnv, chunk: int) -> dict:
        obs = self._build_obs(env, chunk)

        if self.train:
            self._train_step(obs, env, chunk)

        # greedy action — different agents expose this differently
        if hasattr(self.agent, "greedy_action"):
            action = self.agent.greedy_action(obs)
        elif hasattr(self.agent, "act"):
            result = self.agent.act(obs)
            action = result[0] if isinstance(result, tuple) else result
        else:
            raise AttributeError(f"{type(self.agent)} has no greedy_action or act method")
        delta  = float(self.deltas[action])
        self.threshold += delta
        far, tpr, _, _ = env._eval_threshold(chunk, self.threshold)
        self.log(chunk, far, tpr, self.threshold, env=env)
        return dict(far=far, tpr=tpr)

    def _train_step(self, obs, env, chunk):
        raise NotImplementedError


class DQNCtrl(RLCtrl):
    def act(self, env: AnomalyEnv, chunk: int) -> dict:
        obs = self._build_obs(env, chunk)
        if self.train:
            self._train_step(obs, env, chunk)
        action = self.agent.act(obs, eps=0)   # greedy
        delta  = float(self.deltas[action])
        self.threshold += delta
        far, tpr, _, _ = env._eval_threshold(chunk, self.threshold)
        self.log(chunk, far, tpr, self.threshold, env=env)
        return dict(far=far, tpr=tpr)

    def _train_step(self, obs, env, chunk):
        action = self.agent.act(obs)
        delta  = float(self.deltas[action])
        new_t  = self.threshold + delta
        far, tpr, _, _ = env._eval_threshold(chunk, new_t)
        reward = env._compute_reward(
            far=far, tpr=tpr,
            delta=delta, prev_threshold=self.threshold,
        )
        next_obs = obs  # simplified: same state used for next
        done     = (chunk == env.n_chunks - 1)
        self.agent.buf.push(obs, action, reward, next_obs, done)
        self.agent.train_step()


class GRPOCtrl(RLCtrl):
    def _train_step(self, obs, env, chunk):
        actions, logp = self.agent.sample_group_actions(obs, self.group_size)
        rewards = []
        for a in actions:
            delta = float(self.deltas[int(a)])
            far, tpr, _, _ = env._eval_threshold(chunk, self.threshold + delta)
            r = env._compute_reward(
                far=far, tpr=tpr,
                delta=delta, prev_threshold=self.threshold,
            )
            rewards.append(r)
        self.agent.store_group(
            obs=obs, actions=actions, logp=logp,
            rewards=np.array(rewards),
        )
        self.agent.update()


class PPOCtrl(RLCtrl):
    """PPO controller — uses SeqPPOAgent.act/update interface."""
    def act(self, env: AnomalyEnv, chunk: int) -> dict:
        obs = self._build_obs(env, chunk)
        if self.train:
            self._train_step(obs, env, chunk)
        result = self.agent.act(obs)
        action = result[0] if isinstance(result, tuple) else result
        delta  = float(self.deltas[action])
        self.threshold = env._clip_threshold(self.threshold + delta)
        far, tpr, _, _ = env._eval_threshold(chunk, self.threshold)
        self.log(chunk, far, tpr, self.threshold, env=env)
        return dict(far=far, tpr=tpr)

    def _train_step(self, obs, env, chunk):
        result = self.agent.act(obs)
        action, logp, val, _logits = result  # act returns (int, logp, value, logits)
        delta = float(self.deltas[action])
        new_t = env._clip_threshold(self.threshold + delta)
        far, tpr, _, _ = env._eval_threshold(chunk, new_t)
        r = env._compute_reward(far=far, tpr=tpr, delta=delta, prev_threshold=self.threshold)
        done = (chunk == env.n_chunks - 1)
        if hasattr(self.agent, "store"):
            self.agent.store(obs, action, logp, val, r, done)
        if hasattr(self.agent, "update"):
            self.agent.update()


class LGRPOCtrl(GRPOCtrl):
    def act(self, env: AnomalyEnv, chunk: int) -> dict:
        result = super().act(env, chunk)
        # dual update at chunk boundary
        far = result["far"]
        if hasattr(self.agent, "update_dual_chunk"):
            self.agent.update_dual_chunk(far)
        return result


class GFPOCtrl(RLCtrl):
    """Feasibility-first: sample G candidates, filter to keep_size via select_keep_indices."""

    def _sample_evaluate(self, obs, env, chunk):
        """Sample G candidates, evaluate FAR/TPR/reward for each. Returns arrays."""
        G = self.group_size
        actions, logp = self.agent.sample_group_actions(obs, G)
        far_arr = np.empty(G, dtype=np.float64)
        tpr_arr = np.empty(G, dtype=np.float64)
        rew_arr = np.empty(G, dtype=np.float64)
        for i, a in enumerate(actions):
            delta = float(self.deltas[int(a)])
            far, tpr, _, _ = env._eval_threshold(chunk, self.threshold + delta)
            rew_arr[i] = env._compute_reward(
                far=far, tpr=tpr, delta=delta, prev_threshold=self.threshold)
            far_arr[i] = far
            tpr_arr[i] = tpr
        return actions, logp, far_arr, tpr_arr, rew_arr

    def _best_action(self, actions, far_arr, tpr_arr, rew_arr, env):
        """Pick the best action: feasible-first (max TPR), then all-attack TPR, then closest to target."""
        feas_mask = np.abs(far_arr - env.cfg.far_target) <= env.cfg.far_tol * 2
        if feas_mask.any():
            # In-band: maximize TPR directly (movement penalty already handled by policy)
            feas_tpr = np.where(feas_mask, tpr_arr, -np.inf)
            return int(actions[np.argmax(feas_tpr)])
        elif np.std(far_arr) < 1e-6:
            # All candidates have identical FAR — no movement (preserve threshold stability)
            return int(actions[np.argmin(np.abs(self.deltas[actions]))])
        else:
            return int(actions[np.argmin(np.abs(far_arr - env.cfg.far_target))])

    def act(self, env: AnomalyEnv, chunk: int) -> dict:
        obs = self._build_obs(env, chunk)
        actions, logp, far_arr, tpr_arr, rew_arr = self._sample_evaluate(obs, env, chunk)

        if self.train:
            keep_idx, _, _ = self.agent.select_keep_indices(
                bg_after=far_arr, tt_after=tpr_arr, aa_after=tpr_arr,
                rewards=rew_arr,
                target=env.cfg.far_target, tol=env.cfg.far_tol,
            )
            self.agent.store_group(obs=obs, actions=actions[keep_idx],
                                   logp=logp[keep_idx], rewards=rew_arr[keep_idx])
            self.agent.update()

        # Greedy: best feasible candidate (or move toward target if none)
        action = self._best_action(actions, far_arr, tpr_arr, rew_arr, env)
        delta  = float(self.deltas[action])
        self.threshold = env._clip_threshold(self.threshold + delta)
        far, tpr, _, _ = env._eval_threshold(chunk, self.threshold)
        self.log(chunk, far, tpr, self.threshold, env=env)
        return dict(far=far, tpr=tpr)

    def _train_step(self, obs, env, chunk):
        pass  # Logic moved to act()


class GFPOFRCtrl(GFPOCtrl):
    """GFPO-F + dual-λ rate feedback (GFPO-FR): online fine-tuning with constraint signal."""
    def act(self, env: AnomalyEnv, chunk: int) -> dict:
        result = super().act(env, chunk)
        far = result["far"]
        if hasattr(self.agent, "update_dual_chunk"):
            self.agent.update_dual_chunk(far)
        return result


class CPOCtrl(RLCtrl):
    """Constrained Policy Optimization (Achiam et al. 2017).

    Same bandit candidate sampling pattern as GFPOCtrl, but:
      - per-candidate reward and cost are both evaluated under the env,
        cost = e^2 if e<=1 else e, where e = |far - far_target|/far_tol;
      - the buffer stores both group-relative advantages;
      - update() solves the constrained QP with a KL trust region and
        runs a backtracking line search.
    Executed action: best feasible-by-cost candidate by reward (matches the
    GFPOCtrl heuristic so the comparison reflects CPO's *update* rather
    than a different action-selection rule).
    """
    def _evaluate_candidates(self, obs, env, chunk):
        G = self.group_size
        actions, logp = self.agent.sample_group_actions(obs, G)
        far_arr = np.empty(G, dtype=np.float64)
        tpr_arr = np.empty(G, dtype=np.float64)
        rew_arr = np.empty(G, dtype=np.float64)
        cost_arr = np.empty(G, dtype=np.float64)
        for i, a in enumerate(actions):
            d = float(self.deltas[int(a)])
            far, tpr, _, _ = env._eval_threshold(chunk, self.threshold + d)
            rew_arr[i] = env._compute_reward(
                far=far, tpr=tpr, delta=d, prev_threshold=self.threshold)
            far_arr[i] = far
            tpr_arr[i] = tpr
            cost_arr[i] = self.agent.compute_cost(bg_after=far)
        return actions, logp, far_arr, tpr_arr, rew_arr, cost_arr

    def _best_action(self, actions, far_arr, tpr_arr, rew_arr, env):
        feas_mask = np.abs(far_arr - env.cfg.far_target) <= env.cfg.far_tol * 2
        if feas_mask.any():
            feas_tpr = np.where(feas_mask, tpr_arr, -np.inf)
            return int(actions[np.argmax(feas_tpr)])
        if np.std(far_arr) < 1e-6:
            return int(actions[np.argmin(np.abs(self.deltas[actions]))])
        return int(actions[np.argmin(np.abs(far_arr - env.cfg.far_target))])

    def act(self, env: AnomalyEnv, chunk: int) -> dict:
        obs = self._build_obs(env, chunk)
        actions, logp, far_arr, tpr_arr, rew_arr, cost_arr = \
            self._evaluate_candidates(obs, env, chunk)

        if self.train:
            self.agent.store_group(
                obs=obs, actions=actions, logp=logp,
                rewards=rew_arr, costs=cost_arr, baseline="mean",
            )
            self.agent.update()

        action = self._best_action(actions, far_arr, tpr_arr, rew_arr, env)
        delta  = float(self.deltas[action])
        self.threshold = env._clip_threshold(self.threshold + delta)
        far, tpr, _, _ = env._eval_threshold(chunk, self.threshold)
        self.log(chunk, far, tpr, self.threshold, env=env)
        return dict(far=far, tpr=tpr)


# ══════════════════════════════════════════════════════════════════════════════
# Build controllers
# ══════════════════════════════════════════════════════════════════════════════

def build_controllers(baselines: List[str], init_thresh: float,
                      env_cfg: AnomalyEnvConfig, models_dir: Path,
                      seq_len: int, feat_dim: int, n_actions: int,
                      train: bool) -> List[BaseCtrl]:
    ctrls = []
    deltas = np.linspace(-env_cfg.delta_range, env_cfg.delta_range, n_actions,
                         dtype=np.float32)

    def load_or_init(agent, name):
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
        return agent

    for b in baselines:
        b = b.strip().lower()
        if b == "constant":
            ctrls.append(ConstantCtrl("Constant", init_thresh, env_cfg))
        elif b == "pid":
            ctrls.append(PIDCtrl("PID", init_thresh, env_cfg))
        elif b in ("spot", "dspot"):
            ctrls.append(DSpotCtrl("DSPOT", init_thresh, env_cfg))
        elif b == "adt":
            ctrls.append(ADTCtrl("ADT", init_thresh, env_cfg))
        elif b == "dqn":
            from RL.dqn_agent import SeqDQNAgent, DQNConfig
            agent = SeqDQNAgent(seq_len=seq_len, feat_dim=feat_dim,
                                n_actions=n_actions, cfg=DQNConfig())
            load_or_init(agent, "DQN")
            ctrls.append(DQNCtrl("DQN", init_thresh, env_cfg, agent, deltas, train=train))
        elif b == "grpo":
            reward_cfg = GRPORewardCfg(
                target=env_cfg.far_target, tol=env_cfg.far_tol, mode="lex")
            agent = GRPOAgent(seq_len=seq_len, feat_dim=feat_dim,
                              n_actions=n_actions, cfg=GRPOConfig(),
                              reward_cfg=reward_cfg)
            load_or_init(agent, "GRPO")
            ctrls.append(GRPOCtrl("GRPO", init_thresh, env_cfg, agent, deltas, train=train))
        elif b == "lgrpo":
            reward_cfg = GRPORewardCfg(
                target=env_cfg.far_target, tol=env_cfg.far_tol, mode="lag",
                alpha_step=0.01, dual_init=0.25)
            agent = GRPOAgent(seq_len=seq_len, feat_dim=feat_dim,
                              n_actions=n_actions, cfg=GRPOConfig(),
                              reward_cfg=reward_cfg)
            load_or_init(agent, "GRPO")  # initialise from GRPO weights
            ctrls.append(LGRPOCtrl("L-GRPO", init_thresh, env_cfg, agent, deltas, train=train))
        elif b in ("gfpo", "gfpo-f", "gfpo-fr"):
            from RL.gfpo_agent import GFPOAgent, GFPOConfig
            reward_cfg = GRPORewardCfg(
                target=env_cfg.far_target, tol=env_cfg.far_tol,
                alpha_step=0.05, dual_init=0.25)
            # G=32, K=16 — matches LHC default (group_size_sample=32, group_size_keep=16)
            gfpo_cfg = GFPOConfig(keep_size=16, feas_mult=2.0)
            agent = GFPOAgent(seq_len=seq_len, feat_dim=feat_dim,
                              n_actions=n_actions, cfg=GRPOConfig(),
                              gfpo_cfg=gfpo_cfg,
                              reward_cfg=reward_cfg)
            load_or_init(agent, "GFPO")
            name   = "GFPO-FR" if b == "gfpo-fr" else "GFPO-F"
            CtrlCls = GFPOFRCtrl if b == "gfpo-fr" else GFPOCtrl
            ctrl = CtrlCls(name, init_thresh, env_cfg, agent, deltas, train=train,
                           group_size=32)
            ctrls.append(ctrl)
        elif b == "ppo":
            from RL.ppo_agent import SeqPPOAgent, SeqPPOConfig
            agent = SeqPPOAgent(SeqPPOConfig(
                feat_dim=feat_dim, n_actions=n_actions))
            load_or_init(agent, "PPO")
            ctrls.append(PPOCtrl("PPO", init_thresh, env_cfg, agent, deltas, train=train))
        elif b == "cpo":
            reward_cfg = CPORewardCfg(
                target=env_cfg.far_target, tol=env_cfg.far_tol,
                lambda_1=0.25, mix=0.5, beta_move=0.02, cost_limit=1.0,
            )
            agent = CPOAgent(
                seq_len=seq_len, feat_dim=feat_dim, n_actions=n_actions,
                cfg=CPOConfig(delta=0.03, cg_iters=10, cg_damping=0.1,
                              line_search_steps=10, line_search_decay=0.8,
                              batch_min=64),
                reward_cfg=reward_cfg,
            )
            load_or_init(agent, "CPO")
            ctrls.append(CPOCtrl(
                "CPO", init_thresh, env_cfg, agent, deltas,
                train=train, group_size=16,
            ))
    return ctrls


# ══════════════════════════════════════════════════════════════════════════════
# Main rollout loop
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores",      default="anomaly_detection/data/unsw_scores.npz")
    ap.add_argument("--models",      default="anomaly_detection/models")
    ap.add_argument("--outdir",      default="outputs/anomaly_unsw")
    ap.add_argument("--baselines",   default="constant,pid,dspot,dqn,grpo,lgrpo,gfpo,ppo")
    ap.add_argument("--start-chunk", type=int, default=0,
                    help="First chunk index for rollout (for train/val split)")
    ap.add_argument("--train",       action="store_true",
                    help="Enable online RL training during rollout")
    ap.add_argument("--recalibrate", action="store_true",
                    help="Reset each controller threshold to local p(1-FAR_target) "
                         "at start of every chunk (label-free; helps non-stationary streams)")
    ap.add_argument("--far-target", type=float, default=0.005)
    ap.add_argument("--far-tol",    type=float, default=0.0005)
    ap.add_argument("--n-deltas",   type=int,   default=21)
    ap.add_argument("--delta-range",type=float, default=0.5)
    ap.add_argument("--seq-len",    type=int,   default=8)
    ap.add_argument("--lambda1",    type=float, default=0.25,
                    help="Rate-tracking weight in reward (0=pure TPR, 1=pure rate)")
    args = ap.parse_args()

    data = np.load(args.scores)
    scores = data["scores"].astype(np.float32)
    labels = data["y"].astype(np.int32)
    cat    = data["cat"].astype(np.int8) if "cat" in data else None
    if args.start_chunk > 0:
        scores = scores[args.start_chunk:]
        labels = labels[args.start_chunk:]
        if cat is not None:
            cat = cat[args.start_chunk:]
    n_chunks, chunk_size = scores.shape
    print(f"Stream: {n_chunks} chunks × {chunk_size} records, "
          f"attack prevalence={labels.mean():.3f}")

    env_cfg = AnomalyEnvConfig(
        far_target=args.far_target, far_tol=args.far_tol,
        n_deltas=args.n_deltas, delta_range=args.delta_range,
        seq_len=args.seq_len, lambda_1=args.lambda1,
    )
    env = AnomalyEnv(scores, labels, env_cfg)
    env.cat = cat  # attach category array (or None) for TPR_easy / TPR_hard logging

    models_dir = Path(args.models)
    models_dir.mkdir(parents=True, exist_ok=True)
    baselines  = [b.strip() for b in args.baselines.split(",")]
    init_thresh = env.init_threshold

    ctrls = build_controllers(
        baselines, init_thresh, env_cfg, models_dir,
        seq_len=env_cfg.seq_len, feat_dim=env.feat_dim,
        n_actions=env_cfg.n_deltas, train=args.train,
    )
    print(f"Running {len(ctrls)} controllers for {n_chunks} chunks …")

    for chunk in range(n_chunks):
        if args.recalibrate:
            # Label-free per-chunk threshold reset: use local p(1-FAR_target)
            # quantile of ALL scores (attack rate ~2% so dominated by normal).
            local_t = float(np.percentile(scores[chunk],
                                          (1.0 - args.far_target) * 100))
            for ctrl in ctrls:
                ctrl.threshold = local_t
        for ctrl in ctrls:
            ctrl.act(env, chunk)
        if (chunk + 1) % 20 == 0:
            print(f"  chunk {chunk+1}/{n_chunks}")

    # ── aggregate and save ────────────────────────────────────────────────────
    all_rows = []
    for ctrl in ctrls:
        all_rows.extend(ctrl.rows)
    df = pd.DataFrame(all_rows)

    out_dir = Path(args.outdir) / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "chunk_stats.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved → {csv_path}")

    # ── 5-metric summary table ────────────────────────────────────────────────
    def p95_abs_err(x):
        return float(np.percentile(np.abs(x - env_cfg.far_target), 95)) * 100

    summary = df.groupby("method").agg(
        InBand   =("inband",   "mean"),
        MAE_pct  =("far",      lambda x: (x - env_cfg.far_target).abs().mean() * 100),
        P95_pct  =("far",      p95_abs_err),
        TPR_easy =("tpr_easy", lambda x: x.dropna().mean() if x.notna().any() else float("nan")),
        TPR_hard =("tpr_hard", lambda x: x.dropna().mean() if x.notna().any() else float("nan")),
    ).round(4)

    # Display order: InBand first
    print("\n=== Summary (5-metric) ===")
    print(f"{'Method':<12}  {'InBand':>8}  {'MAE%':>8}  {'P95|e|%':>8}  "
          f"{'TPR_easy':>9}  {'TPR_hard':>9}")
    print("-" * 65)
    for method, row in summary.iterrows():
        def _fmt(v):
            return f"{v:.4f}" if not (v != v) else "   n/a "  # nan check
        print(f"{method:<12}  {row['InBand']:>8.4f}  {row['MAE_pct']:>8.4f}  "
              f"{row['P95_pct']:>8.4f}  {_fmt(row['TPR_easy']):>9}  {_fmt(row['TPR_hard']):>9}")
    summary.to_csv(out_dir / "summary.csv")
    print(f"\nSummary saved → {out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
