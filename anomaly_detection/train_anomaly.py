"""
anomaly_detection/train_anomaly.py

Train all RL agents on UNSW-NB15 anomaly detection (MC-equivalent: training split).
Saves trained model weights to anomaly_detection/models/.

Usage:
    python anomaly_detection/train_anomaly.py \
        --n-chunks 60 \
        --epochs   3  \
        --outdir   anomaly_detection/models
"""

import argparse, sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "RL"))

from env_anomaly import AnomalyEnv, AnomalyEnvConfig
from RL.dqn_agent  import SeqDQNAgent, DQNConfig
from RL.grpo_agent import GRPOAgent, GRPOConfig, GRPORewardCfg
from RL.gfpo_agent import GFPOAgent, GFPOConfig
from RL.ppo_agent  import SeqPPOAgent, SeqPPOConfig
from RL.cpo_agent  import CPOAgent, CPOConfig, CPORewardCfg
import torch


def train_agent(agent, env: AnomalyEnv, deltas: np.ndarray,
                n_chunks: int, group_size: int = 8, agent_type: str = "grpo"):
    for chunk in range(min(n_chunks, env.n_chunks)):
        obs = env._get_state()
        env.chunk_idx = chunk

        if agent_type == "dqn":
            action = agent.act(obs)
            delta  = float(deltas[action])
            new_t  = env.threshold + delta
            far, tpr, _, _ = env._eval_threshold(chunk, new_t)
            reward = env._compute_reward(far=far, tpr=tpr, delta=delta,
                                         prev_threshold=env.threshold)
            agent.buf.push(obs, action, reward, obs, False)
            agent.train_step()
            env.threshold = env._clip_threshold(new_t)
        elif agent_type == "ppo":
            result = agent.act(obs)
            action, logp, val, _ = result
            delta  = float(deltas[action])
            new_t  = env._clip_threshold(env.threshold + delta)
            far, tpr, _, _ = env._eval_threshold(chunk, new_t)
            reward = env._compute_reward(far=far, tpr=tpr, delta=delta,
                                         prev_threshold=env.threshold)
            done = (chunk == min(n_chunks, env.n_chunks) - 1)
            if hasattr(agent, "store"):
                agent.store(obs, action, logp, val, reward, done)
            if hasattr(agent, "update"):
                agent.update()
            env.threshold = new_t
        elif agent_type == "cpo":
            # CPO: bandit candidate sampling with separate reward + cost,
            # constrained policy update via trust-region QP.
            G = group_size
            actions, logp = agent.sample_group_actions(obs, G)
            far_arr = np.empty(G, dtype=np.float64)
            tpr_arr = np.empty(G, dtype=np.float64)
            rew_arr = np.empty(G, dtype=np.float64)
            cost_arr = np.empty(G, dtype=np.float64)
            for i, a in enumerate(actions):
                d = float(deltas[int(a)])
                far, tpr, _, _ = env._eval_threshold(chunk, env.threshold + d)
                rew_arr[i] = env._compute_reward(far=far, tpr=tpr, delta=d,
                                                 prev_threshold=env.threshold)
                cost_arr[i] = agent.compute_cost(bg_after=far)
                far_arr[i] = far; tpr_arr[i] = tpr
            agent.store_group(
                obs=obs, actions=actions, logp=logp,
                rewards=rew_arr, costs=cost_arr, baseline="mean",
            )
            agent.update()
            # greedy step: same heuristic as GFPO (feas → max TPR; else closest)
            feas_mask = np.abs(far_arr - env.cfg.far_target) <= env.cfg.far_tol * 2
            if feas_mask.any():
                feas_tpr = np.where(feas_mask, tpr_arr, -np.inf)
                best = int(actions[np.argmax(feas_tpr)])
            elif np.std(far_arr) < 1e-6:
                best = int(actions[np.argmin(np.abs(deltas[actions]))])
            else:
                best = int(actions[np.argmin(np.abs(far_arr - env.cfg.far_target))])
            env.threshold = env._clip_threshold(env.threshold + float(deltas[best]))
        elif agent_type == "gfpo":
            # GFPO: sample G candidates, filter to keep_size via select_keep_indices
            G = group_size
            actions, logp = agent.sample_group_actions(obs, G)
            far_arr = np.empty(G, dtype=np.float64)
            tpr_arr = np.empty(G, dtype=np.float64)
            rew_arr = np.empty(G, dtype=np.float64)
            for i, a in enumerate(actions):
                d = float(deltas[int(a)])
                far, tpr, _, _ = env._eval_threshold(chunk, env.threshold + d)
                rew_arr[i] = env._compute_reward(far=far, tpr=tpr, delta=d,
                                                 prev_threshold=env.threshold)
                far_arr[i] = far; tpr_arr[i] = tpr
            keep_idx, n_feas, _ = agent.select_keep_indices(
                bg_after=far_arr, tt_after=tpr_arr, aa_after=tpr_arr,
                rewards=rew_arr,
                target=env.cfg.far_target, tol=env.cfg.far_tol,
            )
            agent.store_group(obs=obs, actions=actions[keep_idx], logp=logp[keep_idx],
                              rewards=rew_arr[keep_idx])
            agent.update()
            # greedy step: best among feasible candidates (or no move if all equal)
            if n_feas > 0:
                feas_mask = np.abs(far_arr - env.cfg.far_target) <= env.cfg.far_tol * 2
                # Maximize TPR among feasible candidates (FAR already satisfied)
                feas_tpr = np.where(feas_mask, tpr_arr, -np.inf)
                best = int(actions[np.argmax(feas_tpr)])
            elif np.std(far_arr) < 1e-6:
                # All candidates have identical FAR → no movement (preserve threshold stability)
                best = int(actions[np.argmin(np.abs(deltas[actions]))])
            else:
                # Move toward target: smallest FAR error
                best = int(actions[np.argmin(np.abs(far_arr - env.cfg.far_target))])
            env.threshold = env._clip_threshold(env.threshold + float(deltas[best]))
        else:
            actions, logp = agent.sample_group_actions(obs, group_size)
            rewards = []
            for a in actions:
                d = float(deltas[int(a)])
                far, tpr, _, _ = env._eval_threshold(chunk, env.threshold + d)
                r = env._compute_reward(far=far, tpr=tpr, delta=d,
                                        prev_threshold=env.threshold)
                rewards.append(r)
            agent.store_group(obs=obs, actions=actions, logp=logp,
                               rewards=np.array(rewards))
            agent.update()
            # take greedy step
            best = int(actions[np.argmax(rewards)])
            env.threshold = env._clip_threshold(env.threshold + float(deltas[best]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores",     default="anomaly_detection/data/unsw_scores_train.npz",
                    help="Period 1 (training stream) scores by default; use unsw_scores.npz for in-stream warmup")
    ap.add_argument("--n-chunks",   type=int,   default=175)
    ap.add_argument("--epochs",     type=int,   default=3)
    ap.add_argument("--far-target", type=float, default=0.005)
    ap.add_argument("--far-tol",    type=float, default=0.0005)
    ap.add_argument("--n-deltas",   type=int,   default=21)
    ap.add_argument("--delta-range",type=float, default=0.5)
    ap.add_argument("--seq-len",    type=int,   default=8)
    ap.add_argument("--lambda1",    type=float, default=0.25,
                    help="Rate-tracking weight in reward (0=pure TPR, 1=pure rate)")
    ap.add_argument("--outdir",     default="anomaly_detection/models")
    ap.add_argument("--seed",       type=int, default=0)
    args = ap.parse_args()

    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data   = np.load(args.scores)
    scores = data["scores"].astype(np.float32)
    labels = data["y"].astype(np.int32)

    env_cfg = AnomalyEnvConfig(
        far_target=args.far_target, far_tol=args.far_tol,
        n_deltas=args.n_deltas, delta_range=args.delta_range,
        seq_len=args.seq_len, lambda_1=args.lambda1,
    )
    env = AnomalyEnv(scores, labels, env_cfg)

    deltas   = env.deltas
    seq_len  = env_cfg.seq_len
    feat_dim = env.feat_dim
    n_act    = env_cfg.n_deltas

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    gfpo_group_size = 64   # NAB-aligned: G=64 candidates, keep_size=16
    grpo_group_size = 16   # NAB-aligned: G=16
    agents = {
        "PPO": (SeqPPOAgent(SeqPPOConfig(feat_dim=feat_dim, n_actions=n_act)),
                "ppo", 1),
        "DQN": (SeqDQNAgent(seq_len=seq_len, feat_dim=feat_dim,
                            n_actions=n_act, cfg=DQNConfig(), seed=args.seed), "dqn", 1),
        "GRPO": (GRPOAgent(seq_len=seq_len, feat_dim=feat_dim,
                           n_actions=n_act, cfg=GRPOConfig(), seed=args.seed,
                           reward_cfg=GRPORewardCfg(
                               target=args.far_target, tol=args.far_tol)), "grpo", grpo_group_size),
        "GFPO": (GFPOAgent(seq_len=seq_len, feat_dim=feat_dim,
                           n_actions=n_act, cfg=GRPOConfig(),
                           gfpo_cfg=GFPOConfig(keep_size=16, feas_mult=2.0),
                           reward_cfg=GRPORewardCfg(
                               target=args.far_target, tol=args.far_tol)), "gfpo", gfpo_group_size),
        "CPO": (CPOAgent(seq_len=seq_len, feat_dim=feat_dim,
                         n_actions=n_act,
                         cfg=CPOConfig(delta=0.03, cg_iters=10, cg_damping=0.1,
                                       line_search_steps=10, line_search_decay=0.8,
                                       batch_min=64),
                         reward_cfg=CPORewardCfg(
                             target=args.far_target, tol=args.far_tol,
                             lambda_1=args.lambda1, mix=0.5, beta_move=0.02,
                             cost_limit=1.0),
                         seed=args.seed), "cpo", 16),
    }

    # best-checkpoint tracking: save the epoch with lowest mean |FAR - target|
    # on training chunks that have benign samples (skip all-attack chunks)
    best_mae  = {name: float("inf") for name in agents}
    best_ckpt = {name: None         for name in agents}

    benign_chunks = [c for c in range(min(args.n_chunks, env.n_chunks))
                     if (labels[c] == 0).any()]

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        for name, (agent, atype, gs) in agents.items():
            env.reset()
            env.threshold = env.init_threshold
            train_agent(agent, env, deltas, args.n_chunks,
                        group_size=gs, agent_type=atype)
            # Evaluate on benign-containing chunks only (all-attack chunks give
            # FAR=0 regardless of threshold and dilute the signal)
            far_list = [env._eval_threshold(c, env.threshold)[0]
                        for c in benign_chunks]
            mae = float(np.mean(np.abs(np.array(far_list) - args.far_target)))
            print(f"  {name}: benign-MAE={mae*100:.3f}%  (n_benign_chunks={len(benign_chunks)})")

            if atype == "dqn":
                net = agent.q
            elif atype == "ppo":
                net = agent.ac
            else:
                net = agent.pi
            if mae < best_mae[name]:
                best_mae[name]  = mae
                best_ckpt[name] = {k: v.clone() for k, v in net.state_dict().items()}
                print(f"    → new best checkpoint for {name}")

    for name, (agent, atype, _) in agents.items():
        p    = outdir / f"{name}.pt"
        ckpt = best_ckpt[name]
        if ckpt is None:
            if atype == "dqn":
                net = agent.q
            elif atype == "ppo":
                net = agent.ac
            else:
                net = agent.pi
            ckpt = net.state_dict()
        torch.save({"pi": ckpt}, p)
        print(f"Saved {name} → {p}  (best benign-MAE={best_mae[name]*100:.3f}%)")

    print("Training complete.")


if __name__ == "__main__":
    main()
