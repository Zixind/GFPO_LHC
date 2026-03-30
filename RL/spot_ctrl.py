"""
SPOT / DSPOT baseline controller for adaptive trigger threshold control.

Reference: Siffer et al., "Anomaly Detection in Streams with Extreme Value
           Theory", KDD 2017.  https://hal.science/hal-01640325v1/document
Code:      https://github.com/limjcst/ads-evt

Algorithm overview
------------------
SPOT models the tail of the score distribution with a Generalized Pareto
Distribution (GPD).  Given a target false-positive rate q (= target background
rate), it computes the threshold z_q such that P(X >= z_q) = q.

    t        = initial threshold  = quantile(X_calib, 1 - INIT_QF * q)
    peaks    = {x - t : x > t}   (exceedances above t)
    GPD fit  : P(peak > y) ≈ (1 + γ·y/σ)^{-1/γ}
    z_q      = t + (σ/γ) · ((n·q / n_t)^{-γ} - 1)   [γ ≠ 0]
               t - σ · log(n·q / n_t)                  [γ = 0]

DSPOT extends SPOT to concept-drifting streams by re-fitting GPD on a
sliding window of the most recent chunks.

Usage in this codebase
----------------------
  Phase 1 – calibration (MC training chunks, train=True):
      end_chunk() accumulates background scores.  After n_calib_chunks the
      GPD is fitted and the controller switches to DSPOT online updates.

  Phase 2 – deployment (remaining MC chunks or CMS real data):
      end_chunk() calls dspot_updater.update(scores) every chunk, re-fitting
      GPD on the sliding window and updating self.cut.

  Persistence:
      state_dict() / load_state_dict() mirror the PyTorch convention used by
      RL agents so the calibration can be saved and reloaded for the rollout.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional

try:
    from scipy.stats import genpareto
    _SCIPY = True
except ImportError:
    _SCIPY = False

try:
    from triggers import Sing_Trigger  # noqa: F401 (used in tests)
except ImportError:
    pass


# ──────────────────────────────────────────────────────────────────────────────
# GPD helpers
# ──────────────────────────────────────────────────────────────────────────────

def _fit_gpd(peaks: np.ndarray):
    """MLE fit of Generalised Pareto Distribution to peaks.

    Returns (gamma, sigma).  Falls back to method-of-moments when MLE fails
    or scipy is unavailable.
    """
    peaks = np.asarray(peaks, dtype=np.float64)
    n = len(peaks)
    if n < 5:
        mu = float(np.mean(peaks)) if n > 0 else 1.0
        return 0.0, max(mu, 1e-9)

    if _SCIPY:
        try:
            gamma, _, sigma = genpareto.fit(peaks, floc=0)
            return float(gamma), max(float(sigma), 1e-9)
        except Exception:
            pass

    # Method-of-moments fallback
    mu  = float(np.mean(peaks))
    var = float(np.var(peaks)) + 1e-12
    gamma = 0.5 * (mu ** 2 / var - 1.0)
    sigma = 0.5 * mu * (mu ** 2 / var + 1.0)
    return float(gamma), max(float(sigma), 1e-9)


def _gpd_threshold(t: float, gamma: float, sigma: float,
                   n: int, n_t: int, q: float) -> float:
    """SPOT threshold formula.  Returns z such that P(X >= z) ≈ q."""
    if n_t == 0:
        return float(t)
    r = (n * q) / n_t  # expected exceedance fraction at alarm level
    if abs(gamma) < 1e-8:
        return float(t - sigma * np.log(r))
    else:
        return float(t + (sigma / gamma) * (r ** (-gamma) - 1.0))


# ──────────────────────────────────────────────────────────────────────────────
# SPOTCalibrator – one-shot calibration on MC training data
# ──────────────────────────────────────────────────────────────────────────────

class SPOTCalibrator:
    """Accumulates MC background chunks and fits GPD once.

    Parameters
    ----------
    target_rate_pct : float
        Target background rate in percent (e.g. 0.2725 for 0.2725 %).
    init_qf : float
        Initial threshold depth: t = quantile(X, 1 − init_qf · q).
        Larger values → deeper tail, more stable GPD fit.  Default 5.
    """

    def __init__(self, target_rate_pct: float, init_qf: float = 5.0):
        self.q    = target_rate_pct / 100.0
        self.init_qf = float(init_qf)
        self._chunks: list[np.ndarray] = []

        # set after fit()
        self.t:         Optional[float] = None
        self.gamma:     Optional[float] = None
        self.sigma:     Optional[float] = None
        self.n_total:   int = 0
        self.n_peaks:   int = 0
        self.threshold: Optional[float] = None

    def add_chunk(self, scores: np.ndarray) -> None:
        self._chunks.append(np.asarray(scores, dtype=np.float64))

    def fit(self) -> float:
        """Fit GPD over all accumulated chunks; return threshold."""
        if not self._chunks:
            raise RuntimeError("SPOTCalibrator.fit() called before any chunks added.")

        X = np.concatenate(self._chunks)
        self.n_total = int(len(X))

        p = max(0.0, min(1.0, 1.0 - self.init_qf * self.q))
        self.t = float(np.quantile(X, p))

        peaks = (X[X > self.t] - self.t).astype(np.float64)
        self.n_peaks = int(len(peaks))

        if self.n_peaks < 5:
            # Not enough peaks: fall back to empirical quantile
            self.threshold = float(np.quantile(X, max(0.0, 1.0 - self.q)))
            self.gamma = 0.0
            self.sigma = 1.0
        else:
            self.gamma, self.sigma = _fit_gpd(peaks)
            self.threshold = _gpd_threshold(
                self.t, self.gamma, self.sigma,
                self.n_total, self.n_peaks, self.q
            )
        return self.threshold

    # ── persistence ──

    def state_dict(self) -> dict:
        return {
            "q": self.q, "init_qf": self.init_qf,
            "t": self.t, "gamma": self.gamma, "sigma": self.sigma,
            "n_total": self.n_total, "n_peaks": self.n_peaks,
            "threshold": self.threshold,
        }

    def load_state_dict(self, d: dict) -> None:
        self.q        = float(d["q"])
        self.init_qf  = float(d.get("init_qf", 5.0))
        self.t        = float(d["t"])
        self.gamma    = float(d["gamma"])
        self.sigma    = float(d["sigma"])
        self.n_total  = int(d["n_total"])
        self.n_peaks  = int(d["n_peaks"])
        self.threshold = float(d["threshold"])


# ──────────────────────────────────────────────────────────────────────────────
# DSPOTUpdater – online sliding-window refinement
# ──────────────────────────────────────────────────────────────────────────────

class DSPOTUpdater:
    """Re-fits GPD on a sliding window of the most recent chunks.

    Parameters
    ----------
    cal : SPOTCalibrator
        Already-fitted calibrator to initialise from.
    window_chunks : int
        Number of recent chunks to retain in the window.
    """

    def __init__(self, cal: SPOTCalibrator, window_chunks: int = 5):
        self.q        = cal.q
        self.init_qf  = cal.init_qf
        self._wsize   = int(window_chunks)
        self._window: list[np.ndarray] = []

        # warm-start from calibration
        self.t         = cal.t
        self.gamma     = cal.gamma
        self.sigma     = cal.sigma
        self.threshold = cal.threshold

    def update(self, new_scores: np.ndarray) -> float:
        """Add new chunk to window, re-fit GPD, return updated threshold."""
        self._window.append(np.asarray(new_scores, dtype=np.float64))
        if len(self._window) > self._wsize:
            self._window.pop(0)

        X = np.concatenate(self._window)
        p = max(0.0, min(1.0, 1.0 - self.init_qf * self.q))
        t = float(np.quantile(X, p))
        peaks = (X[X > t] - t).astype(np.float64)

        if len(peaks) < 5:
            self.threshold = float(np.quantile(X, max(0.0, 1.0 - self.q)))
            return self.threshold

        self.t, (self.gamma, self.sigma) = t, _fit_gpd(peaks)
        self.threshold = _gpd_threshold(
            self.t, self.gamma, self.sigma, len(X), len(peaks), self.q
        )
        return self.threshold


# ──────────────────────────────────────────────────────────────────────────────
# Controller shim (matches BaseCtrl interface in rollout script)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class _CtrlOut:
    micro_global: int


class SPOTCtrl:
    """DSPOT controller for AS/AD trigger.

    Calibration (train=True, first n_calib_chunks via end_chunk):
        Accumulates background AS scores; fits GPD after n_calib_chunks.

    Deployment (rest of MC run or CMS real data):
        Calls dspot_updater.update() each end_chunk to track distribution drift.

    For rollout-only runs (train=False), call load_state_dict() before the
    main loop to restore calibration from a saved file.
    """

    def __init__(
        self,
        name:             str,
        init_cut:         float,
        lo:               float,
        hi:               float,
        *,
        target:           float,           # background rate target in percent
        n_calib_chunks:   int   = 50,
        window_chunks:    int   = 5,
        init_qf:          float = 5.0,
        train:            bool  = True,
    ):
        self.name  = name
        self.cut   = float(init_cut)
        self.lo    = float(lo)
        self.hi    = float(hi)
        self.train = bool(train)

        self._n_calib   = int(n_calib_chunks)
        self._win       = int(window_chunks)
        self.cal        = SPOTCalibrator(target, init_qf)
        self.updater: Optional[DSPOTUpdater] = None
        self._cnt       = 0          # calibration chunk counter
        self._calibrated = False

    # ── BaseCtrl interface ──────────────────────────────────────────────────

    def cut_value(self) -> float:
        return self.cut

    def step_micro(self, *, micro_global: int, **kwargs) -> _CtrlOut:
        return _CtrlOut(micro_global=micro_global)

    def end_chunk(
        self,
        *,
        chunk: int,
        bas_j:    Optional[np.ndarray] = None,
        bas_chunk: Optional[np.ndarray] = None,
        **kwargs,
    ) -> None:
        scores = bas_j if bas_j is not None else bas_chunk
        if scores is None or len(scores) == 0:
            return
        scores = np.asarray(scores, dtype=np.float64)

        if not self._calibrated:
            # ── Phase 1: accumulate calibration data ──
            self.cal.add_chunk(scores)
            self._cnt += 1
            if self._cnt >= self._n_calib:
                self.cal.fit()
                self.updater = DSPOTUpdater(self.cal, self._win)
                self._calibrated = True
                self.cut = float(np.clip(self.cal.threshold, self.lo, self.hi))
        else:
            # ── Phase 2: DSPOT online update ──
            new_z = self.updater.update(scores)
            self.cut = float(np.clip(new_z, self.lo, self.hi))

    # ── Persistence ────────────────────────────────────────────────────────

    def state_dict(self) -> dict:
        return {
            "calibrated":  self._calibrated,
            "cut":         self.cut,
            "cal":         self.cal.state_dict() if self._calibrated else {},
        }

    def load_state_dict(self, d: dict) -> None:
        if d.get("calibrated"):
            self.cal.load_state_dict(d["cal"])
            self.updater    = DSPOTUpdater(self.cal, self._win)
            self._calibrated = True
            self.cut = float(np.clip(self.cal.threshold, self.lo, self.hi))
        if "cut" in d:
            self.cut = float(np.clip(float(d["cut"]), self.lo, self.hi))


class SPOTCtrlHT(SPOTCtrl):
    """DSPOT controller for HT trigger — same logic, different score key."""

    def end_chunk(
        self,
        *,
        chunk: int,
        bht_j:    Optional[np.ndarray] = None,
        bht_chunk: Optional[np.ndarray] = None,
        **kwargs,
    ) -> None:
        scores = bht_j if bht_j is not None else bht_chunk
        super().end_chunk(chunk=chunk, bas_j=scores)
