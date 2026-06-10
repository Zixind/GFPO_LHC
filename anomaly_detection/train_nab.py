"""
anomaly_detection/train_nab.py

Train RL agents (DQN, GRPO, GFPO) on NAB training split.
Uses pure detection quality reward: TPR - alpha*FPR  (no rate constraint).

Usage:
    python anomaly_detection/train_nab.py \
        --scores anomaly_detection/data/nab_train.npz \
        --epochs 5 \
        --outdir anomaly_detection/models_nab
"""

import argparse
import json
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "RL"))

from anomaly_detection.env_nab import NABEnv, NABEnvConfig
from RL.dqn_agent  import SeqDQNAgent, DQNConfig
from RL.grpo_agent import GRPOAgent, GRPOConfig, GRPORewardCfg
from RL.gfpo_agent import GFPOAgent, GFPOConfig
from RL.ppo_agent  import SeqPPOAgent, SeqPPOConfig
from RL.cpo_agent  import CPOAgent, CPOConfig, CPORewardCfg

import torch


# ══════════════════════════════════════════════════════════════════════════════
# Per-agent training helpers
# ══════════════════════════════════════════════════════════════════════════════

def _train_dqn(agent: SeqDQNAgent, env: NABEnv, chunk: int):
    """One DQN step on the given chunk."""
    env.chunk_idx  = chunk
    obs            = env._get_state()
    action         = agent.act(obs)
    delta          = float(env.deltas[action])
    new_t          = env.threshold + delta
    tpr, fpr, _, _ = env._eval_threshold(chunk, new_t)
    reward         = env._compute_reward(tpr=tpr, fpr=fpr, delta=delta)
    agent.buf.push(obs, action, reward, obs, False)
    agent.train_step()
    env.threshold  = env._clip_threshold(new_t)
    return tpr, fpr


def _train_grpo(agent: GRPOAgent, env: NABEnv, chunk: int, group_size: int = 8):
    """One GRPO step on the given chunk — raw reward from NABEnv, no constraint shaping."""
    env.chunk_idx = chunk
    obs           = env._get_state()
    actions, logp = agent.sample_group_actions(obs, group_size)
    rewards       = np.empty(group_size, dtype=np.float64)
    tprs          = np.empty(group_size, dtype=np.float64)
    fprs          = np.empty(group_size, dtype=np.float64)
    for i, a in enumerate(actions):
        delta          = float(env.deltas[int(a)])
        tpr, fpr, _, _ = env._eval_threshold(chunk, env.threshold + delta)
        rewards[i]     = env._compute_reward(tpr=tpr, fpr=fpr, delta=delta)
        tprs[i]        = tpr
        fprs[i]        = fpr

    # store_group uses raw rewards directly (relative advantage computed inside)
    agent.store_group(obs=obs, actions=actions, logp=logp, rewards=rewards)
    agent.update()

    best       = int(actions[np.argmax(rewards)])
    best_delta = float(env.deltas[best])
    env.threshold = env._clip_threshold(env.threshold + best_delta)
    best_tpr = float(tprs[np.argmax(rewards)])
    best_fpr = float(fprs[np.argmax(rewards)])
    return best_tpr, best_fpr


def _train_gfpo(agent: GFPOAgent, env: NABEnv, chunk: int, group_size: int = 32):
    """
    One GFPO step on the given chunk.

    In the NAB context there is no hard feasibility constraint, so we skip
    select_keep_indices and just keep all candidates (pure quality ranking).
    We still call store_group with all G candidates, which is equivalent to
    GFPO-F with no feasibility filter (i.e., keep_size = group_size).
    """
    env.chunk_idx = chunk
    obs           = env._get_state()
    actions, logp = agent.sample_group_actions(obs, group_size)
    rewards       = np.empty(group_size, dtype=np.float64)
    tprs          = np.empty(group_size, dtype=np.float64)
    fprs          = np.empty(group_size, dtype=np.float64)
    for i, a in enumerate(actions):
        delta          = float(env.deltas[int(a)])
        tpr, fpr, _, _ = env._eval_threshold(chunk, env.threshold + delta)
        rewards[i]     = env._compute_reward(tpr=tpr, fpr=fpr, delta=delta)
        tprs[i]        = tpr
        fprs[i]        = fpr

    # Sort by reward descending; keep top keep_size — quality-ranked, no FAR filter
    keep_size = min(int(agent.gfpo_cfg.keep_size), group_size)
    order     = np.argsort(-rewards)[:keep_size]
    agent.store_group(obs=obs, actions=actions[order],
                      logp=logp[order], rewards=rewards[order])
    agent.update()

    best       = int(actions[np.argmax(rewards)])
    best_delta = float(env.deltas[best])
    env.threshold = env._clip_threshold(env.threshold + best_delta)
    best_tpr = float(tprs[np.argmax(rewards)])
    best_fpr = float(fprs[np.argmax(rewards)])
    return best_tpr, best_fpr


def _train_cpo(agent: CPOAgent, env: NABEnv, chunk: int, group_size: int = 16):
    """One CPO step on the given chunk — bandit sampling + trust-region QP."""
    env.chunk_idx = chunk
    obs           = env._get_state()
    actions, logp = agent.sample_group_actions(obs, group_size)
    rewards = np.empty(group_size, dtype=np.float64)
    costs   = np.empty(group_size, dtype=np.float64)
    tprs    = np.empty(group_size, dtype=np.float64)
    fprs    = np.empty(group_size, dtype=np.float64)
    for i, a in enumerate(actions):
        delta          = float(env.deltas[int(a)])
        tpr, fpr, _, _ = env._eval_threshold(chunk, env.threshold + delta)
        rewards[i]     = env._compute_reward(tpr=tpr, fpr=fpr, delta=delta)
        costs[i]       = agent.compute_cost(bg_after=fpr)
        tprs[i]        = tpr
        fprs[i]        = fpr

    agent.store_group(obs=obs, actions=actions, logp=logp,
                      rewards=rewards, costs=costs, baseline="mean")
    agent.update()

    feas = costs <= 1e-9
    if feas.any():
        idx = np.where(feas)[0]
        best = int(actions[idx[np.argmax(rewards[idx])]])
    else:
        best = int(actions[np.argmin(costs)])
    best_delta = float(env.deltas[best])
    env.threshold = env._clip_threshold(env.threshold + best_delta)
    return tprs[np.argmax(rewards)], fprs[np.argmax(rewards)]


def _train_ppo(agent: SeqPPOAgent, env: NABEnv, chunk: int):
    """One PPO step on the given chunk."""
    env.chunk_idx = chunk
    obs = env._get_state()
    action, logp, val, _ = agent.act(obs)
    delta = float(env.deltas[action])
    new_t = env._clip_threshold(env.threshold + delta)
    tpr, fpr, _, _ = env._eval_threshold(chunk, new_t)
    reward = env._compute_reward(tpr=tpr, fpr=fpr, delta=delta)
    done = (chunk == env.n_chunks - 1)
    agent.store(obs, action, logp, val, reward, done)
    agent.update()
    env.threshold = new_t
    return tpr, fpr


def eval_f1(env: NABEnv, n_chunks: int) -> float:
    """Compute mean F1 across chunks at current threshold."""
    f1s = []
    for c in range(min(n_chunks, env.n_chunks)):
        tpr, fpr, prec, f1 = env._eval_threshold(c, env.threshold)
        f1s.append(f1)
    return float(np.mean(f1s)) if f1s else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores",      default="anomaly_detection/data/nab_train.npz")
    ap.add_argument("--epochs",      type=int,   default=5)
    ap.add_argument("--alpha",       type=float, default=0.10,
                    help="FPR penalty weight in NABEnv reward (controls precision/recall tradeoff)")
    ap.add_argument("--beta",        type=float, default=0.005,
                    help="Threshold movement penalty")
    ap.add_argument("--n-deltas",    type=int,   default=21)
    ap.add_argument("--delta-range", type=float, default=0.3)
    ap.add_argument("--seq-len",     type=int,   default=8)
    ap.add_argument("--group-size",  type=int,   default=8,
                    help="Group size G for GRPO; GFPO uses 4x this")
    ap.add_argument("--outdir",      default="anomaly_detection/models_nab")
    args = ap.parse_args()

    data   = np.load(args.scores)
    scores = data["scores"].astype(np.float32)
    labels = data["labels"].astype(np.int32)
    print(f"Loaded {scores.shape[0]} training chunks "
          f"({scores.shape[1]} timesteps each). "
          f"Anomaly prevalence: {labels.mean():.4f}")

    env_cfg = NABEnvConfig(
        alpha=args.alpha, beta=args.beta,
        n_deltas=args.n_deltas, delta_range=args.delta_range,
        seq_len=args.seq_len,
    )
    env = NABEnv(scores, labels, env_cfg)

    seq_len  = env_cfg.seq_len
    feat_dim = env.feat_dim
    n_act    = env_cfg.n_deltas
    n_chunks = env.n_chunks
    gfpo_group_size = args.group_size * 4   # G=32 by default

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ── Instantiate agents ────────────────────────────────────────────────────
    # GRPORewardCfg is required by GRPOAgent constructor, but we bypass its
    # reward shaping entirely: rewards are computed by NABEnv._compute_reward
    # and passed directly to store_group as raw values.
    dummy_reward_cfg = GRPORewardCfg(target=0.5, tol=0.5, mode="lex")

    dqn_agent  = SeqDQNAgent(seq_len=seq_len, feat_dim=feat_dim,
                              n_actions=n_act, cfg=DQNConfig())
    grpo_agent = GRPOAgent(seq_len=seq_len, feat_dim=feat_dim,
                            n_actions=n_act, cfg=GRPOConfig(),
                            reward_cfg=dummy_reward_cfg)
    gfpo_agent = GFPOAgent(seq_len=seq_len, feat_dim=feat_dim,
                            n_actions=n_act, cfg=GRPOConfig(),
                            gfpo_cfg=GFPOConfig(
                                sample_size=gfpo_group_size,
                                keep_size=max(1, gfpo_group_size // 2),
                            ),
                            reward_cfg=dummy_reward_cfg)

    ppo_agent = SeqPPOAgent(SeqPPOConfig(feat_dim=feat_dim, n_actions=n_act))

    cpo_agent = CPOAgent(
        seq_len=seq_len, feat_dim=feat_dim, n_actions=n_act,
        cfg=CPOConfig(delta=0.03, cg_iters=10, cg_damping=0.1,
                      line_search_steps=10, line_search_decay=0.8,
                      batch_min=64),
        reward_cfg=CPORewardCfg(
            target=0.03, tol=0.03,
            lambda_1=0.25, mix=0.5, beta_move=args.beta, cost_limit=1.0,
        ),
    )

    agents = {
        "DQN":  (dqn_agent,  "dqn",  args.group_size),
        "GRPO": (grpo_agent, "grpo", args.group_size),
        "GFPO": (gfpo_agent, "gfpo", gfpo_group_size),
        "PPO":  (ppo_agent,  "ppo",  1),
        "CPO":  (cpo_agent,  "cpo",  16),
    }

    best_f1   = {name: -1.0 for name in agents}
    best_ckpt = {name: None  for name in agents}

    # Per-method cumulative training wall time (across all epochs and chunks)
    method_time_sec = defaultdict(float)
    overall_t0 = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        for name, (agent, atype, gs) in agents.items():
            # Reset environment state for each agent each epoch
            env.threshold    = env.init_threshold
            env._recent_tpr  = []
            env._recent_fpr  = []
            env._history     = []

            t0 = time.perf_counter()
            tpr_list, fpr_list = [], []
            for chunk in range(n_chunks):
                if atype == "dqn":
                    tpr, fpr = _train_dqn(dqn_agent, env, chunk)
                elif atype == "grpo":
                    tpr, fpr = _train_grpo(grpo_agent, env, chunk, gs)
                elif atype == "ppo":
                    tpr, fpr = _train_ppo(ppo_agent, env, chunk)
                elif atype == "cpo":
                    tpr, fpr = _train_cpo(cpo_agent, env, chunk, gs)
                else:
                    tpr, fpr = _train_gfpo(gfpo_agent, env, chunk, gs)
                tpr_list.append(tpr)
                fpr_list.append(fpr)
            method_time_sec[name] += time.perf_counter() - t0

            mean_tpr = float(np.mean(tpr_list))
            mean_fpr = float(np.mean(fpr_list))
            mean_prec = mean_tpr / (mean_tpr + mean_fpr + 1e-9)
            mean_f1  = (2 * mean_prec * mean_tpr
                        / (mean_prec + mean_tpr + 1e-9))
            print(f"  {name}: mean_TPR={mean_tpr:.4f}  mean_FPR={mean_fpr:.4f}  "
                  f"approx_F1={mean_f1:.4f}")

            net = agent.q if atype == "dqn" else (agent.ac if atype == "ppo" else agent.pi)
            if mean_f1 > best_f1[name]:
                best_f1[name]   = mean_f1
                best_ckpt[name] = {k: v.clone() for k, v in net.state_dict().items()}
                print(f"    → new best checkpoint for {name} (F1={mean_f1:.4f})")

    # ── Save checkpoints ──────────────────────────────────────────────────────
    for name, (agent, atype, _) in agents.items():
        p    = outdir / f"{name}.pt"
        ckpt = best_ckpt[name]
        net  = agent.q if atype == "dqn" else (agent.ac if atype == "ppo" else agent.pi)
        if ckpt is None:
            ckpt = net.state_dict()
        torch.save({"pi": ckpt}, p)
        print(f"Saved {name} → {p}  (best approx_F1={best_f1[name]:.4f})")

    # ── Save timing summary ───────────────────────────────────────────────────
    total_sec = time.perf_counter() - overall_t0
    timing = {
        "hardware": {
            "platform": platform.platform(),
            "machine":  platform.machine(),
            "processor": platform.processor() or platform.machine(),
            "cpu_count": int(__import__("os").cpu_count() or 0),
            "torch_device": "cuda" if torch.cuda.is_available()
                            else ("mps" if torch.backends.mps.is_available() else "cpu"),
        },
        "config": {
            "epochs":   int(args.epochs),
            "n_chunks": int(n_chunks),
            "n_train_files": int(len(set(env._file_ids.tolist()))) if hasattr(env, "_file_ids") else None,
        },
        "per_method_train_time_sec": {k: float(v) for k, v in method_time_sec.items()},
        "per_method_train_time_min": {k: float(v) / 60.0 for k, v in method_time_sec.items()},
        "total_wall_time_sec": float(total_sec),
        "total_wall_time_min": float(total_sec) / 60.0,
    }
    timing_path = outdir / "train_timing.json"
    with open(timing_path, "w") as f:
        json.dump(timing, f, indent=2)
    print(f"\n=== Timing summary ({timing['hardware']['torch_device'].upper()} on {timing['hardware']['processor']}) ===")
    for k, v in method_time_sec.items():
        print(f"  {k:<6}  train wall = {v:7.2f} s  ({v/60.0:.2f} min)")
    print(f"  total wall = {total_sec:7.2f} s  ({total_sec/60.0:.2f} min)")
    print(f"Saved timing → {timing_path}")

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
