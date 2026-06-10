"""
anomaly_detection/env_anomaly.py

Online anomaly-detection environment for UNSW-NB15.

Analogy to the particle physics trigger:
  background rate   →  false alert rate  (FAR = FP / (FP+TN))
  signal efficiency →  attack recall     (TPR = TP / (TP+FN))
  cut threshold     →  anomaly score threshold
  rate target ±τ    →  FAR budget ±tolerance

At each chunk the agent observes a state vector built from the current
chunk's anomaly-score distribution and recent history, then picks a
discrete threshold delta.  The environment evaluates FAR and TPR at
the resulting threshold.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List


@dataclass
class AnomalyEnvConfig:
    # Rate control
    far_target: float = 0.005      # target false alert rate (0.5 %) — same relative stringency as LHC 0.25%±0.025%
    far_tol:    float = 0.0005     # tolerance ±0.05 %

    # Threshold action space
    n_deltas:   int   = 21         # number of discrete threshold steps
    delta_range: float = 0.5       # ±delta_range around current threshold (score units)

    # State history length
    seq_len:    int   = 8          # number of recent chunks in state

    # Reward weights (same structure as particle physics reward)
    lambda_1:   float = 0.25       # rate-tracking weight; 1-lambda_1=0.75 → strong TPR signal in-band
    alpha:      float = 0.0        # recall mix weight (0 = pure recall, 1 = pure precision)
    beta:       float = 0.005      # movement penalty (small — allow fast threshold adjustment)


class AnomalyEnv:
    """
    Online anomaly-detection environment.

    State   : (seq_len, feat_dim) array of per-chunk statistics
    Action  : index into discrete threshold deltas
    Reward  : same lexicographic/fixed-weight structure as particle physics
    """

    def __init__(self, scores: np.ndarray, labels: np.ndarray,
                 cfg: AnomalyEnvConfig = None):
        """
        scores : (n_chunks, chunk_size) float32 — anomaly scores
        labels : (n_chunks, chunk_size) int32   — 0=normal, 1=attack
        """
        self.scores  = scores.astype(np.float32)
        self.labels  = labels.astype(np.int32)
        self.n_chunks, self.chunk_size = scores.shape
        self.cfg     = cfg or AnomalyEnvConfig()

        # discrete threshold deltas (like AS_DELTAS in the trigger code)
        self.deltas = np.linspace(
            -self.cfg.delta_range, self.cfg.delta_range, self.cfg.n_deltas
        ).astype(np.float32)

        # Initialise threshold at the (1 - far_target) quantile of normal-only scores,
        # so that roughly far_target fraction of normals are flagged initially.
        normal_scores = scores[labels == 0].ravel()
        if len(normal_scores) == 0:
            normal_scores = scores.ravel()
        self.init_threshold = float(
            np.percentile(normal_scores, (1.0 - self.cfg.far_target) * 100)
        )
        # min threshold = min score, max = 99.99th pct of ALL scores (avoids trapping at 0)
        self.min_threshold = float(np.percentile(scores.ravel(), 1))
        self.max_threshold = float(np.percentile(scores.ravel(), 99.99))
        self.reset()

    # ── internal helpers ──────────────────────────────────────────────────────

    def _clip_threshold(self, t: float) -> float:
        return float(np.clip(t, self.min_threshold, self.max_threshold))

    def _eval_threshold(self, chunk_idx: int, threshold: float
                        ) -> Tuple[float, float, float, float]:
        """Return (far, tpr, tp, fp) for a given threshold on chunk_idx."""
        threshold = self._clip_threshold(threshold)
        s = self.scores[chunk_idx]
        y = self.labels[chunk_idx]
        pred  = (s >= threshold).astype(np.int32)
        tp    = int(((pred == 1) & (y == 1)).sum())
        fp    = int(((pred == 1) & (y == 0)).sum())
        tn    = int(((pred == 0) & (y == 0)).sum())
        fn    = int(((pred == 0) & (y == 1)).sum())
        far   = fp / (fp + tn + 1e-9)
        tpr   = tp / (tp + fn + 1e-9)
        return far, tpr, float(tp), float(fp)

    def _chunk_features(self, chunk_idx: int, threshold: float) -> np.ndarray:
        """
        Build a feature vector for one chunk — analogous to AS_features.
        Features: score percentiles, current FAR, current TPR, threshold,
                  attack prevalence, err (normalised distance to FAR target).
        """
        s = self.scores[chunk_idx]
        pcts = np.percentile(s, [10, 25, 50, 75, 90, 95, 99]).astype(np.float32)
        far, tpr, _, _ = self._eval_threshold(chunk_idx, threshold)
        err  = (far - self.cfg.far_target) / max(self.cfg.far_tol, 1e-9)
        prev = float(self.labels[chunk_idx].mean())
        feat = np.array([
            *pcts,
            float(threshold),
            float(far),
            float(tpr),
            float(err),
            float(prev),
        ], dtype=np.float32)
        return feat

    @property
    def feat_dim(self) -> int:
        return 7 + 5   # 7 percentiles + 5 scalars

    @property
    def n_actions(self) -> int:
        return len(self.deltas)

    # ── public API ────────────────────────────────────────────────────────────

    def reset(self) -> np.ndarray:
        self.threshold = self.init_threshold
        self.chunk_idx = 0
        self._history: List[np.ndarray] = []
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """Return (seq_len, feat_dim) state array, zero-padded at start."""
        feat = self._chunk_features(self.chunk_idx, self.threshold)
        self._history.append(feat)
        K = self.cfg.seq_len
        if len(self._history) < K:
            pad = [np.zeros_like(feat)] * (K - len(self._history))
            seq = pad + list(self._history)
        else:
            seq = list(self._history[-K:])
        return np.stack(seq, axis=0)   # (K, feat_dim)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Apply threshold delta, evaluate metrics, return (next_state, reward, done, info).
        """
        delta      = float(self.deltas[action])
        new_thresh = self._clip_threshold(self.threshold + delta)
        far, tpr, tp, fp = self._eval_threshold(self.chunk_idx, new_thresh)

        reward = self._compute_reward(
            far=far, tpr=tpr,
            delta=delta, prev_threshold=self.threshold,
        )
        self.threshold = new_thresh
        self.chunk_idx += 1
        done = self.chunk_idx >= self.n_chunks

        info = dict(far=far, tpr=tpr, threshold=new_thresh,
                    inband=int(abs(far - self.cfg.far_target) <= self.cfg.far_tol),
                    chunk=self.chunk_idx - 1)

        state = self._get_state() if not done else np.zeros(
            (self.cfg.seq_len, self.feat_dim), dtype=np.float32)
        return state, reward, done, info

    def _compute_reward(self, *, far: float, tpr: float,
                        delta: float, prev_threshold: float) -> float:
        cfg   = self.cfg
        tol   = max(1e-9, cfg.far_tol)
        err   = abs(far - cfg.far_target) / tol

        if err <= 1.0:
            track = 1.0 - err * err       # quadratic reward in-band
            # TPR bonus only when within the rate budget (matches LHC reward structure)
            sig_mix = (1.0 - cfg.alpha) * tpr + cfg.alpha * (1.0 - far)
        else:
            track = -(err - 1.0)          # linear penalty out-of-band
            # No TPR incentive when out-of-band: prevents agents from chasing TPR
            # at the cost of violating the FAR constraint (key fix for all-attack chunks)
            sig_mix = 0.0

        move_pen = abs(delta) / (self.cfg.delta_range * 2 + 1e-9)

        r = cfg.lambda_1 * track + (1.0 - cfg.lambda_1) * sig_mix \
            - cfg.beta * move_pen
        return float(np.clip(r, -50.0, 10.0))

    # ── group sampling (for GRPO / GFPO) ─────────────────────────────────────

    def sample_candidates(self, n: int = 8) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (far_arr, tpr_arr) for n candidate threshold deltas
        (uniformly sampled from self.deltas), evaluated on current chunk.
        """
        idx   = np.random.choice(len(self.deltas), size=n, replace=True)
        deltas = self.deltas[idx]
        fars, tprs = [], []
        for d in deltas:
            f, t, _, _ = self._eval_threshold(self.chunk_idx, self.threshold + d)
            fars.append(f); tprs.append(t)
        return np.array(idx), np.array(fars), np.array(tprs)
