#!/usr/bin/env python3
import argparse, csv
from pathlib import Path
import numpy as np
import h5py
from RL.utils import read_any_h5
from triggers import Sing_Trigger

"""python -m RL.compute_chunk_quantile_cuts \
  --h5 Data/Trigger_food_MC.h5 \
  --mode as --score-dim 2 \
  --chunk-size 50000 \
  --bg-reject-frac 0.9975 \
  --out chunk_cuts_as2_9975_mc.csv

  python -m RL.compute_chunk_quantile_cuts \
  --h5 Data/Trigger_food_MC.h5 \
  --mode as --score-dim 2 \
  --chunk-size 50000 \
  --bg-reject-frac 0.99725 \
  --out chunk_cuts_as2_99725_mc.csv

    python -m RL.compute_chunk_quantile_cuts \
  --h5 Data/Trigger_food_MC.h5 \
  --mode ht --score-dim 2 \
  --chunk-size 50000 \
  --bg-reject-frac 0.9975 \
  --out chunk_cuts_ht2_9975_mc.csv

    python -m RL.compute_chunk_quantile_cuts \
  --h5 Data/Trigger_food_MC.h5 \
  --mode ht --score-dim 2 \
  --chunk-size 50000 \
  --bg-reject-frac 0.99725 \
  --out chunk_cuts_ht2_99725_mc.csv

  """

# ---- compute_chunk_quantile_cuts.py ----
def choose_cut_for_bg_reject(bg_scores: np.ndarray, *, reject_frac: float, bump_if_ties: bool = False) -> float:
    """Set cut so that exactly round(n * (1 - reject_frac)) background events pass."""
    bg = np.asarray(bg_scores, dtype=np.float64)
    bg = bg[np.isfinite(bg)]
    if bg.size == 0:
        return np.nan
    target_accept = 1.0 - float(reject_frac)
    k = max(1, int(np.round(bg.size * target_accept)))
    sorted_bg = np.sort(bg)
    return float(sorted_bg[-k])

def eff_at_cut(scores: np.ndarray, cut: float) -> float:
    x = np.asarray(scores, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0 or not np.isfinite(cut):
        return  np.nan #suppose all gets accepted?
    return float(np.mean(x >= float(cut)))

def min_finite(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    return float(np.min(x))

# ---- helpers referenced by your read_any_h5() ----
def _first_present(keys, candidates):
    for k in candidates:
        if k in keys:
            return k
    return None

def _read_score(h5, prefix: str, hint: int):
    # expects keys like f"{prefix}_score02"
    k = f"{prefix}_score{hint:02d}"
    if k in h5:
        return h5[k][:]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--mode", choices=["as", "ht"], required=True, help="Use anomaly score (as) or HT (ht) arrays.")
    ap.add_argument("--score-dim", type=int, default=2, help="Which score dimension to use for AS mode (default: 2).")

    ap.add_argument("--chunk-size", type=int, default=10000)
    ap.add_argument("--stride", type=int, default=0, help="Default: chunk-size")
    ap.add_argument("--drop-last", action="store_true")

    ap.add_argument("--bg-reject-frac", type=float, default=0.99725)
    ap.add_argument("--bump-if-ties", action="store_true")

    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d = read_any_h5(args.h5, score_dim_hint=args.score_dim)
    stride = args.stride if args.stride > 0 else args.chunk_size

    if args.mode == "as":
        k = args.score_dim
        bg = np.asarray(d[f"Bas{k}"], dtype=np.float64)
        tt = np.asarray(d[f"Tas{k}"], dtype=np.float64)
        aa = np.asarray(d[f"Aas{k}"], dtype=np.float64)
    else:
        bg = np.asarray(d["Bht"], dtype=np.float64)
        tt = np.asarray(d["Tht"], dtype=np.float64)
        aa = np.asarray(d["Aht"], dtype=np.float64)

    n = len(d["Bas2"])
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print('len(bg), len(tt), len(aa)', len(bg), len(tt), len(aa))
    cols = [
        "chunk","start","end","n_bg",
        "cut","bg_accept","bg_reject",
        "tt_eff","aa_eff","sig_eff_combined",
        "tt_min","aa_min",
        "tt_eff_avg","aa_eff_avg","n_tt_eff","n_aa_eff",
    ]
    tt_eff_vals = []
    aa_eff_vals = []

    with open(out_path, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=cols)
        w.writeheader()

        i = 0
        start = 0
        while start < n:
            end = start + args.chunk_size
            if end > n and args.drop_last:
                break
            end = min(end, n)

            bg_chunk = bg[start:end]
            cut = choose_cut_for_bg_reject(bg_chunk, reject_frac=args.bg_reject_frac, bump_if_ties=args.bump_if_ties)
            bg_accept = float(np.mean(bg_chunk[np.isfinite(bg_chunk)] >= cut))
            bg_reject = 1.0 - bg_accept

            tt_chunk = tt[start:end]
            aa_chunk = aa[start:end]

            # tt_eff = eff_at_cut(tt_chunk, cut)
            # aa_eff = eff_at_cut(aa_chunk, cut)
            tt_eff = Sing_Trigger(tt_chunk, cut)
            aa_eff = Sing_Trigger(aa_chunk, cut)

            if np.isfinite(tt_eff):
                tt_eff_vals.append(tt_eff)
            if np.isfinite(aa_eff):
                aa_eff_vals.append(aa_eff)

            # print(np.mean(tt_chunk), np.mean(aa_eff))
            tt_min = min_finite(tt_chunk)
            aa_min = min_finite(aa_chunk)

            sig_eff_comb = float(np.mean([x for x in [tt_eff, aa_eff] if np.isfinite(x)]))

            # w.writerow(dict(
            #     chunk=i, start=start, end=end, n_bg=int(np.isfinite(bg_chunk).sum()),
            #     cut=float(cut), bg_accept=bg_accept, bg_reject=bg_reject,
            #     tt_eff=tt_eff, aa_eff=aa_eff, sig_eff_combined=sig_eff_comb
            # ))

            w.writerow(dict(
                chunk=i, start=start, end=end, n_bg=int(np.isfinite(bg_chunk).sum()),
                cut=float(cut), bg_accept=bg_accept, bg_reject=bg_reject,
                tt_eff=tt_eff, aa_eff=aa_eff, sig_eff_combined=sig_eff_comb,
                tt_min=tt_min, aa_min=aa_min,
            ))

            i += 1
            start += stride

    tt_eff_avg = float(np.mean(tt_eff_vals)) if tt_eff_vals else np.nan
    aa_eff_avg = float(np.mean(aa_eff_vals)) if aa_eff_vals else np.nan

    # append a summary row
    with open(out_path, "a", newline="") as fp: 
        w = csv.DictWriter(fp, fieldnames=cols)
        w.writerow(dict(
        chunk="AVG_FINITE",
        start="",
        end="",
        n_bg="",
        cut=np.nan,
        bg_accept=np.nan,
        bg_reject=np.nan,
        tt_eff=np.nan,
        aa_eff=np.nan,
        sig_eff_combined=np.nan,
        tt_min=np.nan,
        aa_min=np.nan,
        tt_eff_avg=tt_eff_avg,
        aa_eff_avg=aa_eff_avg,
        n_tt_eff=len(tt_eff_vals),
        n_aa_eff=len(aa_eff_vals),
        ))

    
    print(f"[avg over finite chunks] tt_eff_avg={tt_eff_avg:.6g}  aa_eff_avg={aa_eff_avg:.6g}  "
      f"(n_tt={len(tt_eff_vals)}, n_aa={len(aa_eff_vals)})")
    
    print(f"[done] wrote {out_path}")

if __name__ == "__main__":
    main()
