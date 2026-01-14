# RL/gfpo_agent.py
from dataclasses import dataclass
import numpy as np
from RL.grpo_agent import GRPOAgent

@dataclass
class GFPOConfig:
    sample_size: int = 32   # G_sample
    keep_size: int = 16     # G_keep
    feas_mult: float = 1.0
    mix: float = 0.80       # ranking: mix*tt + (1-mix)*aa
    baseline: str = "mean"  # "mean" or "median"

class GFPOAgent(GRPOAgent):
    """
    GFPO = GRPO-style policy, but each micro-step:
      1) sample G_sample actions from policy
      2) evaluate candidates
      3) keep top G_keep after feasibility/ranking
      4) store only kept + update policy
    """
    def __init__(self, *args, gfpo_cfg: GFPOConfig, **kwargs):
        super().__init__(*args, **kwargs)
        self.gfpo_cfg = gfpo_cfg

    def select_keep_indices(self, bg_after, tt_after, aa_after, rewards, *, target, tol):
        bg_after = np.asarray(bg_after, dtype=np.float64)
        tt_after = np.asarray(tt_after, dtype=np.float64)
        aa_after = np.asarray(aa_after, dtype=np.float64)
        rewards  = np.asarray(rewards,  dtype=np.float64)

        G = bg_after.size
        Gk = int(self.gfpo_cfg.keep_size)
        if Gk > G:
            raise ValueError("keep_size must be <= number of candidates")

        abs_err = np.abs(bg_after - float(target))
        feas = abs_err <= float(self.gfpo_cfg.feas_mult) * float(tol)

        score_sig = float(self.gfpo_cfg.mix) * tt_after + (1.0 - float(self.gfpo_cfg.mix)) * aa_after

        feas_idx = np.where(feas)[0]
        infeas_idx = np.where(~feas)[0]

        # Feasible first: high score_sig, then high reward, then small abs_err
        if feas_idx.size:
            order = np.lexsort((
                abs_err[feas_idx],
                -rewards[feas_idx],
                -score_sig[feas_idx],
            ))
            feas_sorted = feas_idx[order]
        else:
            feas_sorted = np.array([], dtype=np.int64)

        # If need padding: pick closest-to-target among infeasible
        if infeas_idx.size:
            order = np.lexsort((
                -score_sig[infeas_idx],
                -rewards[infeas_idx],
                abs_err[infeas_idx],
            ))
            infeas_sorted = infeas_idx[order]
        else:
            infeas_sorted = np.array([], dtype=np.int64)

        if feas_sorted.size >= Gk:
            keep = feas_sorted[:Gk]
            used_pad = False
        else:
            need = Gk - feas_sorted.size
            keep = np.concatenate([feas_sorted, infeas_sorted[:need]]) if need > 0 else feas_sorted
            used_pad = True

        return keep.astype(np.int64), int(feas_sorted.size), bool(used_pad)

    def store_kept_group(self, obs, actions, logp, rewards, keep_idx):
        # Train only on kept candidates
        keep_idx = np.asarray(keep_idx, dtype=np.int64)
        acts_k = np.asarray(actions, dtype=np.int64)[keep_idx]
        logp_k = np.asarray(logp, dtype=np.float32)[keep_idx]
        rew_k  = np.asarray(rewards, dtype=np.float32)[keep_idx]

        # reuse GRPO buffer format
        self.store_group(
            obs=obs,
            actions=acts_k,
            logp=logp_k,
            rewards=rew_k,
            baseline=self.gfpo_cfg.baseline,
        )
