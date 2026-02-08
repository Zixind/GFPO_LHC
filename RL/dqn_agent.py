# rl/dqn_agent.py
"""
Minimal DQN module (PyTorch) for threshold-control tasks.

Exports:
  - DQNAgent
  - make_obs(...)
  - shield_delta(...)
  - compute_reward(...)

No domain-specific code here.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple
import random
import numpy as np
import math
from triggers import Sing_Trigger
# --- torch import guarded so main script can error nicely if missing ---
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "PyTorch is required.\nInstall: pip install torch\n\n"
        f"Import error: {e}"
    )
def _tail_shape_features(scores, cut, step, eps=1e-8):
    """
    Tail shape above cut using background scores only.
    Returns 4 scalars: p1, p2, ratio1, ratio2.
    All in "percent units" consistent with Sing_Trigger outputs.
    """
    if step is None or step <= 0:
        # fallback: no tail info
        return 0.0, 0.0, 0.0, 0.0

    p0 = float(Sing_Trigger(scores, cut))
    p1 = float(Sing_Trigger(scores, cut + 1.0 * step))
    p2 = float(Sing_Trigger(scores, cut + 2.0 * step))

    r1 = p1 / (p0 + eps)
    r2 = p2 / (p1 + eps)
    return p1, p2, r1, r2

def _near_cut_fractions(x: np.ndarray, cut: float, widths: Sequence[float]) -> np.ndarray:
    """
    Generic 'how much mass is near threshold' features.
    Returns frac(|x-cut| < w) for each w in widths.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return np.zeros(len(widths), dtype=np.float32)
    m = np.abs(x - float(cut))
    out = [(m < float(w)).mean() for w in widths]
    return np.asarray(out, dtype=np.float32)

def _robust_stats(x: np.ndarray) -> Tuple[float, float, float]:
    """(median, p10, p90) robust window stats."""
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return 0.0, 0.0, 0.0
    return (float(np.median(x)), float(np.percentile(x, 10)), float(np.percentile(x, 90)))

def _downsample_last_K(x: np.ndarray, K: int) -> np.ndarray:
    """Return exactly K samples from x, biased toward most-recent."""
    x = np.asarray(x)
    n = len(x)
    if n == 0:
        return np.zeros(K, dtype=np.float32)
    if n >= K:
        # choose K indices from the last n points
        idx = np.linspace(n - K, n - 1, K).astype(int)
        return x[idx].astype(np.float32)
    # pad on the left with the first element
    pad = np.full(K - n, x[0], dtype=np.float32)
    return np.concatenate([pad, x.astype(np.float32)], axis=0)
# ------------------------ replay buffer ------------------------
class ReplayBuffer:
    def __init__(self, capacity: int = 50000):
        self.capacity = int(capacity)
        self.data = []
        self.i = 0

    def push(self, s, a, r, sp, done: bool):
        item = (
            np.asarray(s, np.float32),
            int(a),
            float(r),
            np.asarray(sp, np.float32),
            float(done),
        )
        if len(self.data) < self.capacity:
            self.data.append(item)
        else:
            self.data[self.i] = item
        self.i = (self.i + 1) % self.capacity

    def sample(self, batch_size: int = 128):
        batch = random.sample(self.data, batch_size)
        s, a, r, sp, done = zip(*batch)
        return (
            np.stack(s),
            np.asarray(a, np.int64),
            np.asarray(r, np.float32),
            np.stack(sp),
            np.asarray(done, np.float32),
        )

    def __len__(self):
        return len(self.data)

# ------------------------ networks ------------------------
class QNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x):
        return self.net(x)

# ------------------------ agent ------------------------
@dataclass
class DQNConfig:
    lr: float = 5e-4
    gamma: float = 0.95
    batch_size: int = 128
    target_update: int = 200
    buffer_capacity: int = 50_000
    grad_clip: float = 5.0

class DQNAgent:
    """
    Vanilla Double-DQN with:
      - SmoothL1 (Huber)
      - target network
      - replay buffer
      - epsilon-greedy action selection
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        seed: int = 0,
        device: Optional[str] = None,
        cfg: Optional[DQNConfig] = None,
    ):
        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self.cfg = cfg or DQNConfig()

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self.q = QNet(self.obs_dim, self.n_actions).to(self.device)
        self.tgt = QNet(self.obs_dim, self.n_actions).to(self.device)
        self.tgt.load_state_dict(self.q.state_dict())

        self.opt = optim.Adam(self.q.parameters(), lr=self.cfg.lr)
        self.buf = ReplayBuffer(capacity=self.cfg.buffer_capacity)

        self.train_steps = 0

    def act(self, obs: np.ndarray, eps: float = 0.1) -> int:
        """Epsilon-greedy."""
        if random.random() < eps:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            x = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            qvals = self.q(x)[0]
            return int(torch.argmax(qvals).item())

    def train_step(self) -> Optional[float]:
        """One gradient step. Returns loss or None if not enough data."""
        bs = self.cfg.batch_size
        if len(self.buf) < bs:
            return None

        s, a, r, sp, done = self.buf.sample(bs)

        s = torch.tensor(s, dtype=torch.float32, device=self.device)
        a = torch.tensor(a, dtype=torch.int64, device=self.device).unsqueeze(1)
        r = torch.tensor(r, dtype=torch.float32, device=self.device).unsqueeze(1)
        sp = torch.tensor(sp, dtype=torch.float32, device=self.device)
        done = torch.tensor(done, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_sa = self.q(s).gather(1, a)

        # Double DQN target
        with torch.no_grad():
            a_star = torch.argmax(self.q(sp), dim=1, keepdim=True)
            q_sp = self.tgt(sp).gather(1, a_star)
            y = r + (1.0 - done) * self.cfg.gamma * q_sp

        loss = nn.SmoothL1Loss()(q_sa, y)

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), self.cfg.grad_clip)
        self.opt.step()

        self.train_steps += 1
        if self.train_steps % self.cfg.target_update == 0:
            self.tgt.load_state_dict(self.q.state_dict())

        return float(loss.item())

# ------------------------ observation + reward helpers ------------------------
def make_obs(
    bg_rate: float,
    prev_bg_rate: float,
    cut: float,
    cut_mid: float,
    cut_span: float,
    target: float,
    last_delta: float = 0.0,
    max_delta: float = 1.0,
    frac_near: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Default 3D observation used for RL agent.:
      [ normalized_error, normalized_delta_error, normalized_cut ]
    """
    cut_span = max(1e-12, float(cut_span))
    target = max(1e-12, float(target))
    max_delta = max(1e-12, float(max_delta))

    x_rate = (float(bg_rate) - target) / target
    x_drate = (float(bg_rate) - float(prev_bg_rate)) / target
    x_cut = (float(cut) - float(cut_mid)) / cut_span
    x_last = float(last_delta) / max_delta
    
    base = [x_rate, x_drate, x_cut, x_last]
    
    if frac_near is not None:
        base.extend([float(v) for v in frac_near])

    return np.asarray(base, dtype=np.float32)

def _downsample_or_pad(x: np.ndarray, K: int) -> np.ndarray:
    x = np.asarray(x)
    n = len(x)
    if n == 0:
        return np.zeros(K, dtype=np.float32)
    if n >= K:
        idx = np.linspace(0, n - 1, K).astype(int)
        return x[idx].astype(np.float32)
    # pad by repeating last value
    out = np.empty(K, dtype=np.float32)
    out[:n] = x.astype(np.float32)
    out[n:] = float(x[-1])
    return out

def make_event_seq_ht(
    *,
    bht, bnpv,
    bg_rate, prev_bg_rate,
    cut,
    ht_mid, ht_span,
    target,
    K,
    last_delta,
    max_delta,
    near_widths=(5.0, 10.0, 20.0),
    step = None,
    # (optional)
    tol=None,            # to compute inband
    err_i=None,          # leaky integral of err (pass from outside)
    d_bg_d_cut=None,     # sensitivity probe (pass from outside)
):
    # 1) downsample/pad raw event streams to length K
    htK  = _downsample_last_K(bht,  K)
    npvK = _downsample_last_K(bnpv, K)

    # 2) normalize per-event quantities
    ht_norm  = (htK - ht_mid) / max(ht_span, 1e-6)
    # simple npv normalization (center/scale by window stats)
    npv_mu, npv_sd = float(np.mean(npvK)), float(np.std(npvK) + 1e-6)
    npv_norm = (npvK - npv_mu) / npv_sd

    cut_norm = (cut - ht_mid) / max(ht_span, 1e-6)
    dist_norm = (htK - cut) / max(ht_span, 1e-6)
    pass_flag = (htK >= cut).astype(np.float32)

    # 3) chunk-level scalars broadcast to each timestep
    err = (bg_rate - target) / max(target, 1e-6)             # rate error (fractional)
    dbr = (bg_rate - prev_bg_rate) / max(target, 1e-6)       # rate drift
    abs_err = abs(bg_rate - target) / max(target, 1e-6)
    inband = 0.0 if tol is None else float(abs(bg_rate - target) <= float(tol))

    last_d = last_delta / max(max_delta, 1e-6)               # last action
    tpos = np.linspace(0.0, 1.0, K).astype(np.float32)       # time position inside seq

    # 4) “near cut” indicators: |ht - cut| <= width
    near_feats = np.stack(
        [(np.abs(htK - cut) <= float(w)).astype(np.float32) for w in near_widths],
        axis=1
    )  # (K, W)

    p1, p2, tr1, tr2 = _tail_shape_features(bht, cut, step)

    err_i = 0.0 if err_i is None else float(err_i)
    d_bg_d_cut = 0.0 if d_bg_d_cut is None else float(d_bg_d_cut)
    # 5) base 10 features (K,10)
    base = np.stack([
        ht_norm,          # 0
        npv_norm,         # 1
        pass_flag,        # 2
        dist_norm,        # 3
        np.full(K, err,  dtype=np.float32),     # 4
        np.full(K, dbr,  dtype=np.float32),     # 5
        np.full(K, cut_norm, dtype=np.float32), # 6
        np.full(K, last_d,   dtype=np.float32), # 7
        tpos,             # 8
        # np.full(K, target / 100.0, dtype=np.float32), # 8 (optional constant)

        #  “how bad” + feasibility
        np.full(K, abs_err, dtype=np.float32),   # 9
        np.full(K, inband, dtype=np.float32),    # 10

        #  NPV regime scalars (scale lightly so magnitudes stay sane)
        np.full(K, npv_mu / 50.0, dtype=np.float32),  # 11 (choose divisor appropriate for your NPV scale)
        np.full(K, npv_sd / 20.0, dtype=np.float32),  # 12

        #  tail/shape (already available)
        np.full(K, float(p1), dtype=np.float32),   # 13
        np.full(K, float(p2), dtype=np.float32),   # 14
        np.full(K, float(tr1), dtype=np.float32),  # 15
        np.full(K, float(tr2), dtype=np.float32),  # 16

        # optional integrator + sensitivity probe
        np.full(K, err_i, dtype=np.float32),       # 17
        np.full(K, d_bg_d_cut, dtype=np.float32),  # 18
        
    ], axis=1)

    obs = np.concatenate([base, near_feats], axis=1)  # (K, 10+W)
    return obs.astype(np.float32)


def make_event_seq_as(
    *,
    bas, bnpv,
    bg_rate, prev_bg_rate,
    cut,
    as_mid, as_span,
    target,
    K,
    last_delta,
    max_delta,
    near_widths=(0.01, 0.02, 0.05),
    step=None,
    # (optional)
    tol=None,
    err_i=None,
    d_bg_d_cut=None,
):
    asK  = _downsample_last_K(bas,  K)
    npvK = _downsample_last_K(bnpv, K)

    as_norm = (asK - as_mid) / max(as_span, 1e-6)

    npv_mu, npv_sd = float(np.mean(npvK)), float(np.std(npvK) + 1e-6)
    npv_norm = (npvK - npv_mu) / npv_sd

    cut_norm  = (cut - as_mid) / max(as_span, 1e-6)
    dist_norm = (asK - cut) / max(as_span, 1e-6)
    pass_flag = (asK >= cut).astype(np.float32)

    err  = (bg_rate - target) / max(target, 1e-6)
    dbr  = (bg_rate - prev_bg_rate) / max(target, 1e-6)
    abs_err = abs(bg_rate - target) / max(target, 1e-6)
    inband = 0.0 if tol is None else float(abs(bg_rate - target) <= float(tol))

    last_d = last_delta / max(max_delta, 1e-6)
    tpos = np.linspace(0.0, 1.0, K).astype(np.float32)

    near_feats = np.stack(
        [(np.abs(asK - cut) <= float(w)).astype(np.float32) for w in near_widths],
        axis=1
    )

    p1, p2, tr1, tr2 = _tail_shape_features(bas, cut, step)

    err_i = 0.0 if err_i is None else float(err_i)
    d_bg_d_cut = 0.0 if d_bg_d_cut is None else float(d_bg_d_cut)

    base = np.stack([
        as_norm,
        pass_flag,
        dist_norm,
        npv_norm,
        np.full(K, err, dtype=np.float32),
        np.full(K, dbr, dtype=np.float32),
        np.full(K, cut_norm, dtype=np.float32),
        np.full(K, last_d, dtype=np.float32),
        tpos,

        np.full(K, abs_err, dtype=np.float32),
        np.full(K, inband, dtype=np.float32),

        np.full(K, npv_mu / 50.0, dtype=np.float32),
        np.full(K, npv_sd / 20.0, dtype=np.float32),

        np.full(K, float(p1), dtype=np.float32),
        np.full(K, float(p2), dtype=np.float32),
        np.full(K, float(tr1), dtype=np.float32),
        np.full(K, float(tr2), dtype=np.float32),

        np.full(K, err_i, dtype=np.float32),
        np.full(K, d_bg_d_cut, dtype=np.float32),
    ], axis=1)

    obs = np.concatenate([base, near_feats], axis=1)
    return obs.astype(np.float32)

def shield_delta(
    bg_rate: float,
    target: float,
    tol: float,
    max_delta: float,
) -> Optional[float]:
    """
    If agent is too far from target, force a strong move in the correct direction.
      - bg too high => increase cut (positive delta)
      - bg too low  => decrease cut (negative delta)
    """
    if bg_rate > target + tol:
        return +float(max_delta)
    if bg_rate < target - tol:
        return -float(max_delta)
    return None

def compute_reward(
    bg_rate: float,
    target: float,
    tol: float,
    sig_rate_1: float,
    sig_rate_2: float,
    delta_applied: float,
    max_delta: float,
    alpha: float = 0.2,
    beta: float = 0.02,
    clip: Tuple[float, float] = (-10.0, 10.0),
    prev_bg_rate: Optional[float] = None,
    gamma_stab: float = 0.25, # weight for stability penalty 0.25 default
) -> float:
    """
    sig_rate_1: first signal rate (e.g. TTbar) focuses more on TTbar
    sig_rate_2: second signal rate (e.g. HToAATo4B)

    Generic reward:
      + in-band tracking bonus (encourages holding)
      - out-of-band penalty grows smoothly
    #   - background penalty: |bg-target|/tol
      + signal bonus: alpha * mean(sig)/100
      - movement penalty: beta * |delta|/max_delta
    """
    tol = max(1e-12, float(tol))
    max_delta = max(1e-12, float(max_delta))

    # normalized error
    e = (float(bg_rate) - float(target)) / tol
    ae = abs(e)

    # Rate Tracking: reward being within tolerance, penalize being outside
    if ae <= 1.0:
        # max +1 at center; smoothly decreases to 0 at band edge
        track = 1.0 - ae**2
    else:
        # linear penalty outside band, continuous at ae=1
        track = - (ae - 1.0)


    # bg_pen = abs(float(bg_rate) - float(target)) / tol
    # sig_term = 0.5 * (2 * float(sig_rate_1) + float(sig_rate_2)) / 100.0

    # signal mix in ~[0,1]
    tt = float(sig_rate_1) / 100.0
    aa = float(sig_rate_2) / 100.0
    sig_term = float(alpha) * tt + (1.0 - alpha) * aa #alpha ttbar focus

    move_pen = abs(float(delta_applied)) / max_delta

    if prev_bg_rate is None:
        stab_pen = 0.0
    else:
        db = abs(float(bg_rate) - float(prev_bg_rate)) / tol
        stab_pen = db * db
    # r = -bg_pen + alpha * sig_term - beta * move_pen
    r = track + sig_term - beta * move_pen - gamma_stab * stab_pen
    lo, hi = clip 
    return float(np.clip(r, lo, hi))




### Encode event-level sequences with a GRU-based Q-network ###
# ------------------------ sequence network ------------------------
class SeqQNet(nn.Module):
    """
    Q-network for event-level sequences.
    Input:  x of shape (B, K, F)
    Output: Q-values of shape (B, n_actions)
    """
    def __init__(self, feat_dim: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.gru = nn.GRU(input_size=feat_dim, hidden_size=hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x):
        # x: (B, K, F)
        _, h = self.gru(x)      # h: (1, B, hidden)
        h = h[-1]               # (B, hidden)
        return self.head(h)

class SeqQNet_ligher(nn.Module):
    def __init__(self, feat_dim: int, n_actions: int, hidden: int = 32):
        super().__init__()
        self.gru = nn.GRU(input_size=feat_dim, hidden_size=hidden, batch_first=True)
        self.head = nn.Linear(hidden, n_actions)

    def forward(self, x):
        _, h = self.gru(x)          # h: (1, B, hidden)
        return self.head(h[-1])     # (B, n_actions)

class ReplayBufferSeq:
    def __init__(self, capacity: int = 50_000):
        self.capacity = int(capacity)
        self.data = []
        self.i = 0

    def push(self, s_seq, a, r, sp_seq, done: bool):
        item = (
            np.asarray(s_seq, np.float32),   # (K, F)
            int(a),
            float(r),
            np.asarray(sp_seq, np.float32),  # (K, F)
            float(done),
        )
        if len(self.data) < self.capacity:
            self.data.append(item)
        else:
            self.data[self.i] = item
        self.i = (self.i + 1) % self.capacity

    def sample(self, batch_size: int = 128):
        batch = random.sample(self.data, batch_size)
        s, a, r, sp, done = zip(*batch)
        return (
            np.stack(s),  # (B, K, F)
            np.asarray(a, np.int64),
            np.asarray(r, np.float32),
            np.stack(sp), # (B, K, F)
            np.asarray(done, np.float32),
        )

    def __len__(self):
        return len(self.data)


class SeqDQNAgent:
    """
    Double-DQN for event-level sequences.
    Same API style as DQNAgent, but obs is (K, F) instead of (obs_dim,).
    """
    def __init__(
        self,
        seq_len: int,
        feat_dim: int,
        n_actions: int,
        seed: int = 0,
        device: Optional[str] = None,
        cfg: Optional[DQNConfig] = None,
    ):
        self.seq_len = int(seq_len)
        self.feat_dim = int(feat_dim)
        self.n_actions = int(n_actions)
        self.cfg = cfg or DQNConfig()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self.q = SeqQNet(self.feat_dim, self.n_actions).to(self.device)
        self.tgt = SeqQNet(self.feat_dim, self.n_actions).to(self.device)
        self.tgt.load_state_dict(self.q.state_dict())

        self.opt = optim.Adam(self.q.parameters(), lr=self.cfg.lr)
        self.buf = ReplayBufferSeq(capacity=self.cfg.buffer_capacity)
        self.train_steps = 0

    def act(self, obs_seq: np.ndarray, eps: float = 0.1) -> int:
        if random.random() < eps:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            x = torch.tensor(obs_seq, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1,K,F)
            qvals = self.q(x)[0]
            return int(torch.argmax(qvals).item())

    def train_step(self) -> Optional[float]:
        bs = self.cfg.batch_size
        if len(self.buf) < bs:
            return None

        s, a, r, sp, done = self.buf.sample(bs)

        s = torch.tensor(s, dtype=torch.float32, device=self.device)            # (B,K,F)
        sp = torch.tensor(sp, dtype=torch.float32, device=self.device)          # (B,K,F)
        a = torch.tensor(a, dtype=torch.int64, device=self.device).unsqueeze(1)
        r = torch.tensor(r, dtype=torch.float32, device=self.device).unsqueeze(1)
        done = torch.tensor(done, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_sa = self.q(s).gather(1, a)

        with torch.no_grad():
            a_star = torch.argmax(self.q(sp), dim=1, keepdim=True)
            q_sp = self.tgt(sp).gather(1, a_star)
            y = r + (1.0 - done) * self.cfg.gamma * q_sp

        loss = nn.SmoothL1Loss()(q_sa, y)

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), self.cfg.grad_clip)
        self.opt.step()

        self.train_steps += 1
        if self.train_steps % self.cfg.target_update == 0:
            self.tgt.load_state_dict(self.q.state_dict())

        return float(loss.item())
    
    @staticmethod
    def compute_reward(
        *,
        bg_rate: float,
        target: float,
        tol: float,
        sig_rate_1: float,
        sig_rate_2: float,
        delta_applied: float,
        max_delta: float,
        alpha: float = 0.2,
        beta: float = 0.02,
        clip: tuple[float, float] = (-10.0, 10.0),
        prev_bg_rate: Optional[float] = None,
        gamma_stab: float = 0.25,
    ) -> float:
        # call the module-level function defined above
        return compute_reward(
            bg_rate=bg_rate,
            target=target,
            tol=tol,
            sig_rate_1=sig_rate_1,
            sig_rate_2=sig_rate_2,
            delta_applied=delta_applied,
            max_delta=max_delta,
            alpha=alpha,
            beta=beta,
            clip=clip,
            prev_bg_rate=prev_bg_rate,
            gamma_stab=gamma_stab,
        )




# ADT baseline for now
def make_event_seq_ht_v0(
    *,
    bht, bnpv,
    bg_rate, prev_bg_rate,
    cut,
    ht_mid, ht_span,
    target,
    K,
    last_delta,
    max_delta,
    near_widths=(5.0, 10.0, 20.0),
):
    # 1) downsample/pad raw event streams to length K
    htK  = _downsample_last_K(bht,  K)
    npvK = _downsample_last_K(bnpv, K)

    # 2) normalize per-event quantities
    ht_norm  = (htK - ht_mid) / max(ht_span, 1e-6)
    # simple npv normalization (center/scale by window stats)
    npv_mu, npv_sd = float(np.mean(npvK)), float(np.std(npvK) + 1e-6)
    npv_norm = (npvK - npv_mu) / npv_sd

    cut_norm = (cut - ht_mid) / max(ht_span, 1e-6)
    dist_norm = (htK - cut) / max(ht_span, 1e-6)
    pass_flag = (htK >= cut).astype(np.float32)

    # 3) chunk-level scalars broadcast to each timestep
    err = (bg_rate - target) / max(target, 1e-6)             # rate error (fractional)
    dbr = (bg_rate - prev_bg_rate) / max(target, 1e-6)       # rate drift
    last_d = last_delta / max(max_delta, 1e-6)               # last action
    tpos = np.linspace(0.0, 1.0, K).astype(np.float32)       # time position inside seq

    # 4) “near cut” indicators: |ht - cut| <= width
    near_feats = []
    for w in near_widths:
        near_feats.append((np.abs(htK - cut) <= float(w)).astype(np.float32))
    near_feats = np.stack(near_feats, axis=1)  # (K, W)

    # 5) base 10 features (K,10)
    base = np.stack([
        ht_norm,          # 0
        npv_norm,         # 1
        pass_flag,        # 2
        dist_norm,        # 3
        np.full(K, err,  dtype=np.float32),     # 4
        np.full(K, dbr,  dtype=np.float32),     # 5
        np.full(K, cut_norm, dtype=np.float32), # 6
        np.full(K, last_d,   dtype=np.float32), # 7
        np.full(K, target / 100.0, dtype=np.float32), # 8 (optional constant)
        tpos,             # 9
    ], axis=1)

    obs = np.concatenate([base, near_feats], axis=1)  # (K, 10+W)
    return obs.astype(np.float32)
def make_event_seq_as_v0(
    *,
    bas, bnpv,
    bg_rate, prev_bg_rate,
    cut,
    as_mid, as_span,
    target,
    K,
    last_delta,
    max_delta,
    near_widths=(0.01, 0.02, 0.05),
    step = None
):
    asK  = _downsample_last_K(bas,  K)
    npvK = _downsample_last_K(bnpv, K)

    as_norm = (asK - as_mid) / max(as_span, 1e-6)
    npv_mu, npv_sd = float(np.mean(npvK)), float(np.std(npvK) + 1e-6)
    npv_norm = (npvK - npv_mu) / npv_sd

    cut_norm  = (cut - as_mid) / max(as_span, 1e-6)
    dist_norm = (asK - cut) / max(as_span, 1e-6)
    pass_flag = (asK >= cut).astype(np.float32)

    err  = (bg_rate - target) / max(target, 1e-6)
    dbr  = (bg_rate - prev_bg_rate) / max(target, 1e-6)
    last_d = last_delta / max(max_delta, 1e-6)
    tpos = np.linspace(0.0, 1.0, K).astype(np.float32)

    near_feats = []
    for w in near_widths:
        near_feats.append((np.abs(asK - cut) <= float(w)).astype(np.float32))
    near_feats = np.stack(near_feats, axis=1)  # (K, W)

    p1, p2, tr1, tr2 = _tail_shape_features(bas, cut, step)

    base = np.stack([
        as_norm, pass_flag, dist_norm, npv_norm,
        np.full(K, err, dtype=np.float32),
        np.full(K, dbr, dtype=np.float32),
        np.full(K, cut_norm, dtype=np.float32),
        np.full(K, last_d, dtype=np.float32),
        np.full(K, target / 100.0, dtype=np.float32),
        tpos,
    ], axis=1)

    obs = np.concatenate([base, near_feats], axis=1)  # (K, 10+W)
    return obs.astype(np.float32)