"""
anomaly_detection/env_nab.py

RL environment for NAB streaming anomaly detection.

Key difference from env_anomaly.py:
  - NO rate constraint (no FAR target, no lambda_1)
  - Reward = TPR - alpha * FPR   (pure detection quality)
  - State features focused on score distribution + detection quality history
  - Same action space: discrete threshold deltas
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class NABEnvConfig:
    alpha:       float = 0.10   # FPR penalty weight (controls precision/recall tradeoff)
    beta:        float = 0.005  # movement penalty
    n_deltas:    int   = 21
    delta_range: float = 0.3    # +-0.3 around current threshold
    seq_len:     int   = 8


class NABEnv:
    """
    Online anomaly-detection environment for NAB.

    State   : (seq_len, feat_dim) array of per-chunk statistics
    Action  : index into discrete threshold deltas
    Reward  : TPR - alpha * FPR - beta * |delta| / delta_range  (pure detection quality)

    Unlike AnomalyEnv, there is NO rate-constraint objective here.
    The agent is free to trade off precision vs recall as it sees fit.
    """

    def __init__(self, scores: np.ndarray, labels: np.ndarray,
                 cfg: Optional[NABEnvConfig] = None):
        """
        scores : (n_chunks, chunk_size) float32 — anomaly scores in [0,1]
        labels : (n_chunks, chunk_size) int32   — 1=anomaly, 0=normal
        """
        self.scores    = scores.astype(np.float32)
        self.labels    = labels.astype(np.int32)
        self.n_chunks, self.chunk_size = scores.shape
        self.cfg       = cfg or NABEnvConfig()

        # Discrete threshold deltas (same pattern as particle physics trigger)
        self.deltas = np.linspace(
            -self.cfg.delta_range, self.cfg.delta_range, self.cfg.n_deltas
        ).astype(np.float32)

        # Clip bounds: 1st and 99.99th percentile of all scores
        self.min_threshold = float(np.percentile(scores.ravel(), 1.0))
        self.max_threshold = float(np.percentile(scores.ravel(), 99.99))

        # History of recent TPR/FPR for state features
        self._recent_tpr: List[float] = []
        self._recent_fpr: List[float] = []
        self._history:    List[np.ndarray] = []

        self.reset()

    # ── properties ────────────────────────────────────────────────────────────

    @property
    def feat_dim(self) -> int:
        # p25, p50, p75, p90, p95, p99, threshold, flagging_rate, recent_tpr, recent_fpr
        return 10

    @property
    def n_actions(self) -> int:
        return len(self.deltas)

    @property
    def init_threshold(self) -> float:
        """Initialize at p97 of all training scores — targets ~3% flagging rate for NAB."""
        return float(np.percentile(self.scores.ravel(), 97.0))

    # ── internal helpers ──────────────────────────────────────────────────────

    def _clip_threshold(self, t: float) -> float:
        return float(np.clip(t, self.min_threshold, self.max_threshold))

    def _eval_threshold(self, chunk_idx: int, threshold: float
                        ) -> Tuple[float, float, float, float]:
        """
        Return (tpr, fpr, precision, f1) for a given threshold on chunk_idx.
        Handles edge cases (all-normal, all-anomaly chunks) gracefully.
        """
        threshold = self._clip_threshold(threshold)
        s    = self.scores[chunk_idx]
        y    = self.labels[chunk_idx]
        pred = (s >= threshold).astype(np.int32)

        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())

        tpr       = tp / (tp + fn + 1e-9)
        fpr       = fp / (fp + tn + 1e-9)
        precision = tp / (tp + fp + 1e-9)
        f1        = 2 * precision * tpr / (precision + tpr + 1e-9)
        return float(tpr), float(fpr), float(precision), float(f1)

    def _chunk_features(self, chunk_idx: int, threshold: float) -> np.ndarray:
        """
        Build a 10-dim feature vector for one chunk.

        Features:
          [0-5] score percentiles: p25, p50, p75, p90, p95, p99
          [6]   current threshold
          [7]   current flagging rate (fraction of chunk flagged)
          [8]   recent TPR (mean of last seq_len evaluations; 0 if no history)
          [9]   recent FPR (mean of last seq_len evaluations; 0 if no history)
        """
        s    = self.scores[chunk_idx]
        pcts = np.percentile(s, [25, 50, 75, 90, 95, 99]).astype(np.float32)

        flagging_rate = float((s >= threshold).mean())

        recent_tpr = float(np.mean(self._recent_tpr[-self.cfg.seq_len:])
                           if self._recent_tpr else 0.0)
        recent_fpr = float(np.mean(self._recent_fpr[-self.cfg.seq_len:])
                           if self._recent_fpr else 0.0)

        feat = np.array([
            pcts[0], pcts[1], pcts[2], pcts[3], pcts[4], pcts[5],
            float(threshold),
            flagging_rate,
            recent_tpr,
            recent_fpr,
        ], dtype=np.float32)
        return feat

    def _compute_reward(self, tpr: float, fpr: float, delta: float) -> float:
        """
        Pure detection quality reward (no rate constraint).
            reward = TPR - alpha * FPR - beta * |delta| / delta_range
        """
        move_pen = abs(delta) / (self.cfg.delta_range + 1e-9)
        r = tpr - self.cfg.alpha * fpr - self.cfg.beta * move_pen
        return float(np.clip(r, -10.0, 10.0))

    # ── public API ────────────────────────────────────────────────────────────

    def reset(self) -> np.ndarray:
        self.threshold  = self.init_threshold
        self.chunk_idx  = 0
        self._history   = []
        self._recent_tpr = []
        self._recent_fpr = []
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """Return (seq_len, feat_dim) state array, zero-padded at start."""
        feat = self._chunk_features(self.chunk_idx, self.threshold)
        self._history.append(feat)
        K = self.cfg.seq_len
        if len(self._history) < K:
            pad = [np.zeros(self.feat_dim, dtype=np.float32)] * (K - len(self._history))
            seq = pad + list(self._history)
        else:
            seq = list(self._history[-K:])
        return np.stack(seq, axis=0)   # (K, feat_dim)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Apply threshold delta, evaluate metrics, return (next_state, reward, done, info).
        """
        delta     = float(self.deltas[action])
        new_thresh = self._clip_threshold(self.threshold + delta)

        tpr, fpr, precision, f1 = self._eval_threshold(self.chunk_idx, new_thresh)
        reward = self._compute_reward(tpr=tpr, fpr=fpr, delta=delta)

        self._recent_tpr.append(tpr)
        self._recent_fpr.append(fpr)
        self.threshold = new_thresh
        self.chunk_idx += 1
        done = self.chunk_idx >= self.n_chunks

        info = dict(
            tpr=tpr, fpr=fpr, precision=precision, f1=f1,
            threshold=new_thresh, chunk=self.chunk_idx - 1,
        )

        if done:
            state = np.zeros((self.cfg.seq_len, self.feat_dim), dtype=np.float32)
        else:
            state = self._get_state()

        return state, reward, done, info

    # ── group sampling (for GRPO / GFPO) ─────────────────────────────────────

    def sample_candidates(self, n: int = 8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return (action_indices, tpr_arr, fpr_arr) for n candidate threshold deltas
        (uniformly sampled from self.deltas), evaluated on current chunk.
        """
        idx   = np.random.choice(len(self.deltas), size=n, replace=True)
        deltas = self.deltas[idx]
        tprs, fprs = [], []
        for d in deltas:
            tpr, fpr, _, _ = self._eval_threshold(self.chunk_idx, self.threshold + d)
            tprs.append(tpr)
            fprs.append(fpr)
        return idx, np.array(tprs), np.array(fprs)
