# RL/ppo_agent.py
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SeqPPOConfig:
    feat_dim: int          # F
    n_actions: int

    # optimization
    lr: float = 3e-4
    epochs: int = 4
    minibatch_size: int = 64
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5

    # RL
    gamma: float = 0.95
    lam: float = 0.95

    adv_norm: bool = True
    device: str = "cpu"

    # match SeqDQN default hidden=64 (your SeqQNet uses hidden=64)
    hidden: int = 64


class SeqActorCritic(nn.Module):
    """
    Input:  x (B, K, F)
    Output: logits (B, A), value (B,)
    """
    def __init__(self, feat_dim: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.gru = nn.GRU(input_size=feat_dim, hidden_size=hidden, batch_first=True)
        self.pi = nn.Sequential(
            nn.Linear(hidden, 64), nn.Tanh(),
            nn.Linear(64, n_actions),
        )
        self.v = nn.Sequential(
            nn.Linear(hidden, 64), nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B,K,F)
        _, h = self.gru(x)          # h: (1,B,H)
        h = h[-1]                   # (B,H)
        logits = self.pi(h)         # (B,A)
        value = self.v(h).squeeze(-1)  # (B,)
        return logits, value


class PPOBufferSeq:
    def __init__(self):
        self.obs: List[np.ndarray] = []   # each (K,F)
        self.act: List[int] = []
        self.logp: List[float] = []
        self.val: List[float] = []
        self.rew: List[float] = []
        self.done: List[bool] = []
        self.adv: Optional[np.ndarray] = None
        self.ret: Optional[np.ndarray] = None

    def clear(self):
        self.__init__()

    def store(self, obs_seq, act, logp, val, rew, done):
        obs_seq = np.asarray(obs_seq, dtype=np.float32)
        if obs_seq.ndim != 2:
            raise ValueError(f"Expected obs_seq (K,F), got shape {obs_seq.shape}")
        self.obs.append(obs_seq)
        self.act.append(int(act))
        self.logp.append(float(logp))
        self.val.append(float(val))
        self.rew.append(float(rew))
        self.done.append(bool(done))

    def __len__(self):
        return len(self.obs)


class SeqPPOAgent:
    def __init__(self, cfg: SeqPPOConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        self.ac = SeqActorCritic(cfg.feat_dim, cfg.n_actions, hidden=cfg.hidden).to(self.device)
        self.opt = torch.optim.Adam(self.ac.parameters(), lr=cfg.lr)
        self.buf = PPOBufferSeq()

    def _check_obs(self, obs_seq: np.ndarray) -> np.ndarray:
        obs_seq = np.asarray(obs_seq, dtype=np.float32)
        if obs_seq.ndim != 2:
            raise ValueError(f"PPO expects (K,F), got {obs_seq.shape}")
        if obs_seq.shape[1] != int(self.cfg.feat_dim):
            raise ValueError(f"feat_dim mismatch: expected F={self.cfg.feat_dim}, got {obs_seq.shape}")
        return obs_seq

    @torch.no_grad()
    def act(self, obs_seq: np.ndarray, *, temperature: float = 1.0) -> Tuple[int, float, float, np.ndarray]:
        """
        Returns: a, logp(a), v(s), logits_np
        """
        obs_seq = self._check_obs(obs_seq)
        x = torch.as_tensor(obs_seq, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1,K,F)
        logits, v = self.ac(x)  # logits (1,A), v (1,)
        logits = logits / max(1e-6, float(temperature))
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()                 # (1,)
        logp = dist.log_prob(a)           # (1,)
        return int(a.item()), float(logp.item()), float(v.item()), logits.squeeze(0).detach().cpu().numpy()

    @torch.no_grad()
    def value(self, obs_seq: np.ndarray) -> float:
        obs_seq = self._check_obs(obs_seq)
        x = torch.as_tensor(obs_seq, dtype=torch.float32, device=self.device).unsqueeze(0)
        _, v = self.ac(x)
        return float(v.item())

    def store(self, obs_seq, act, logp, val, rew, done=False):
        self.buf.store(obs_seq, act, logp, val, rew, done)

    def finish_path(self, last_value: float):
        n = len(self.buf)
        if n == 0:
            return

        rews = np.asarray(self.buf.rew, dtype=np.float32)
        vals = np.asarray(self.buf.val, dtype=np.float32)
        dones = np.asarray(self.buf.done, dtype=np.bool_)

        adv = np.zeros(n, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(n)):
            nonterminal = 0.0 if dones[t] else 1.0
            v_next = float(last_value) if (t == n - 1) else float(vals[t + 1])
            delta = rews[t] + self.cfg.gamma * v_next * nonterminal - vals[t]
            last_gae = delta + self.cfg.gamma * self.cfg.lam * nonterminal * last_gae
            adv[t] = last_gae

        ret = adv + vals
        if self.cfg.adv_norm:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        self.buf.adv = adv
        self.buf.ret = ret

    def update(self) -> dict:
        n = len(self.buf)
        if n == 0 or self.buf.adv is None or self.buf.ret is None:
            return {"n": n, "loss_pi": np.nan, "loss_v": np.nan, "ent": np.nan}

        obs = torch.as_tensor(np.stack(self.buf.obs), dtype=torch.float32, device=self.device)  # (N,K,F)
        act = torch.as_tensor(np.asarray(self.buf.act), dtype=torch.int64, device=self.device)  # (N,)
        logp_old = torch.as_tensor(np.asarray(self.buf.logp), dtype=torch.float32, device=self.device)  # (N,)
        adv = torch.as_tensor(self.buf.adv, dtype=torch.float32, device=self.device)  # (N,)
        ret = torch.as_tensor(self.buf.ret, dtype=torch.float32, device=self.device)  # (N,)

        idx = np.arange(n)
        mb = int(self.cfg.minibatch_size)

        last_loss_pi = last_loss_v = last_ent = 0.0

        for _ in range(int(self.cfg.epochs)):
            np.random.shuffle(idx)
            for start in range(0, n, mb):
                j = idx[start:start + mb]
                oj = obs[j]                # (B,K,F)
                aj = act[j]                # (B,)
                logp_oj = logp_old[j]      # (B,)
                advj = adv[j]              # (B,)
                retj = ret[j]              # (B,)

                logits, v = self.ac(oj)    # logits (B,A), v (B,)
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(aj)   # (B,)
                ratio = torch.exp(logp - logp_oj)

                unclipped = ratio * advj
                clipped = torch.clamp(ratio, 1.0 - self.cfg.clip_eps, 1.0 + self.cfg.clip_eps) * advj
                loss_pi = -(torch.min(unclipped, clipped)).mean()

                loss_v = F.mse_loss(v, retj)
                ent = dist.entropy().mean()

                loss = loss_pi + self.cfg.vf_coef * loss_v - self.cfg.ent_coef * ent

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), self.cfg.max_grad_norm)
                self.opt.step()

                last_loss_pi = float(loss_pi.item())
                last_loss_v = float(loss_v.item())
                last_ent = float(ent.item())

        info = {"n": n, "loss_pi": last_loss_pi, "loss_v": last_loss_v, "ent": last_ent}
        self.buf.clear()
        return info
    

    @torch.no_grad()
    def eval_logp_v(self, obs_seq: np.ndarray, act: int) -> Tuple[float, float]:
        """
        Evaluate log π(act|obs_seq) and V(obs_seq) for a *given* action.
        obs_seq: (K,F)
        act: int in [0, n_actions)
        Returns: (logp, value) as floats
        """
        obs_seq = self._check_obs(obs_seq)

        x = torch.as_tensor(obs_seq, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1,K,F)
        logits, v = self.ac(x)  # logits (1,A), v (1,)

        a = torch.as_tensor([int(act)], dtype=torch.int64, device=self.device)  # (1,)
        dist = torch.distributions.Categorical(logits=logits)
        logp = dist.log_prob(a)  # (1,)

        return float(logp.item()), float(v.item())

    @torch.no_grad()
    def evaluate(self, obs_seq: np.ndarray, act: int) -> Tuple[float, float]:
        """
        Alias for compatibility with callers expecting `evaluate(obs, act)`.
        """
        return self.eval_logp_v(obs_seq, act)
