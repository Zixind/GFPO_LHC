#!/usr/bin/env python3
"""
plot_as_distributions.py

Make anomaly score (AS) distribution plots for:
  - background
  - ttbar
  - HToAATo4B

Also draw the Constant Menu threshold defined as:
  fixed_AS_cut = percentile(Bas[first_chunk_window], 99.75)

Works with:
  - Trigger_food_MC.h5 (keys: mc_bkg_*, mc_tt_*, mc_aa_*)
  - Trigger_food_Data.h5 (keys: data_bkg_*, data_tt_*, data_aa_*)
  - Matched_data_2016_dim2.h5 (paired realdata; may have data_Npv)

Outputs (in --outdir):
  - as_dist_bkg.png
  - as_dist_tt.png
  - as_dist_aa.png
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import h5py
import hdf5plugin  # noqa: F401

from RL.utils import add_cms_header, save_png


def _first_present(keys, candidates):
    for k in candidates:
        if k in keys:
            return k
    return None


def _read_score(h5, prefix: str, dim: int):
    """
    Builder writes: f"{prefix}_score{dim:02d}"
      e.g. mc_bkg_score02, data_tt_score02, ...
    """
    d2 = f"{int(dim):02d}"
    for k in (f"{prefix}_score{d2}", f"{prefix}_scores{d2}"):
        if k in h5:
            return h5[k][:]
    return None


def read_any_h5_scores_only(path: str, score_dim_hint: int = 2):
    """
    Returns dict with Bas, Tas, Aas arrays (float32) + meta.
    """
    with h5py.File(path, "r") as h5:
        keys = set(h5.keys())
        hint = int(score_dim_hint)

        # MC layout
        if ("mc_bkg_ht" in keys) and ("mc_bkg_Npv" in keys):
            Bas = _read_score(h5, "mc_bkg", hint)
            Tas = _read_score(h5, "mc_tt", hint)
            Aas = _read_score(h5, "mc_aa", hint)

            if Bas is None or Tas is None or Aas is None:
                raise SystemExit(
                    f"[read_any_h5_scores_only] Missing score{hint:02d} in MC file.\n"
                    f"Expected keys like mc_bkg_score{hint:02d}, mc_tt_score{hint:02d}, mc_aa_score{hint:02d}.\n"
                    f"Top-level keys: {sorted(list(keys))}"
                )

            return dict(
                Bas=np.asarray(Bas, np.float32),
                Tas=np.asarray(Tas, np.float32),
                Aas=np.asarray(Aas, np.float32),
                meta=dict(control="MC", matched_by_index=False),
            )

        # Data layout (Trigger_food_Data or Matched_data_*)
        has_bkg = ("data_bkg_ht" in keys)
        has_tt = ("data_tt_ht" in keys)
        has_aa = ("data_aa_ht" in keys)
        has_npvs_any = ("data_Npv" in keys) or ("data_bkg_Npv" in keys)

        if has_bkg and has_tt and has_aa and has_npvs_any:
            Bas = _read_score(h5, "data_bkg", hint)
            Tas = _read_score(h5, "data_tt", hint)
            Aas = _read_score(h5, "data_aa", hint)

            if Bas is None or Tas is None or Aas is None:
                raise SystemExit(
                    f"[read_any_h5_scores_only] Missing score{hint:02d} in Data file.\n"
                    f"Expected keys like data_bkg_score{hint:02d}, data_tt_score{hint:02d}, data_aa_score{hint:02d}.\n"
                    f"Top-level keys: {sorted(list(keys))}"
                )

            matched_by_index = ("data_Npv" in keys)
            return dict(
                Bas=np.asarray(Bas, np.float32),
                Tas=np.asarray(Tas, np.float32),
                Aas=np.asarray(Aas, np.float32),
                meta=dict(control="RealData", matched_by_index=matched_by_index),
            )

        raise SystemExit(
            "[read_any_h5_scores_only] Unrecognized H5 layout.\n"
            f"Top-level keys: {sorted(list(keys))}"
        )


def _plot_one(
    scores: np.ndarray,
    fixed_cut: float,
    outpath: Path,
    title: str,
    run_label: str,
    bins: int,
    logy: bool = True,
):
    scores = np.asarray(scores, np.float32)
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        raise SystemExit(f"[plot] No finite scores for {title}")

    # Robust x-range so tails don't dominate the view
    xlo = float(np.percentile(scores, 0.1))
    xhi = float(np.percentile(scores, 99.9))
    if not np.isfinite(xlo) or not np.isfinite(xhi) or xhi <= xlo:
        xlo = float(np.min(scores))
        xhi = float(np.max(scores))

    fig, ax = plt.subplots(figsize=(10, 6))

    # Histogram (density)
    ax.hist(scores, bins=bins, range=(xlo, xhi), density=True, alpha=0.75)
    if logy:
        ax.set_yscale("log")

    ax.axvline(fixed_cut, linewidth=2.0, linestyle="--", label=f"fixed_AS_cut (p99.75) = {fixed_cut:.6g}")

    ax.set_xlabel("Anomaly score (AE MSE)", loc="center")
    ax.set_ylabel("Density", loc="center")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_title(title)

    # CDF on twin axis
    ax2 = ax.twinx()
    xs = np.sort(scores)
    ys = (np.arange(xs.size) + 1) / xs.size
    ax2.plot(xs, ys, linewidth=1.5)
    ax2.set_ylabel("CDF", loc="center")
    ax2.set_ylim(0.0, 1.0)

    ax.legend(loc="best", frameon=True)

    # Header + save (use your utils if available)
    try:
        add_cms_header(fig, run_label=run_label)
        save_png(fig, str(outpath.with_suffix("")))
    except Exception:
        # fallback if RL.utils isn't importable
        outpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="Data/Trigger_food_MC.h5",
                    help="Trigger_food_*.h5 or Matched_data_*.h5")
    ap.add_argument("--outdir", default="RL_outputs/as_distributions",
                    help="Output directory for plots")
    ap.add_argument("--score-dim-hint", type=int, default=2,
                    help="score dimension: 2 means score02")
    ap.add_argument("--control", default="MC", choices=["MC", "RealData"],
                    help="Control type: MC or RealData")
    ap.add_argument("--percentile", type=float, default=99.75,
                    help="Constant-menu percentile threshold")
    ap.add_argument("--bins", type=int, default=120,
                    help="Histogram bins")

    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    d = read_any_h5_scores_only(args.input, score_dim_hint=args.score_dim_hint)
    Bas, Tas, Aas = d["Bas"], d["Tas"], d["Aas"]

    # Define "first chunk" window and then compute fixed cut from its first ref_window
    N = len(Bas)
    if args.control == "MC":
        chunk_size = 50000
    else:
        #real data
        chunk_size = 20000
    chunk_size = int(min(chunk_size, N))
    ref_window = 10000 if args.control == "RealData" else 100000 #MC: NOTE: separate by Data_SingleTrigger and Run_SingleTrigger.py
    if ref_window < 10:
        raise SystemExit(f"ref-window too small after clipping: {ref_window}")

    ref = Bas[:ref_window]
    fixed_AS_cut = float(np.percentile(ref, float(args.percentile)))

    print(f"[INFO] input={args.input}")
    print(f"[INFO] N={N} chunk_size={chunk_size} ref_window={ref_window}")
    print(f"[INFO] fixed_AS_cut = percentile(Bas[:{ref_window}], {args.percentile}) = {fixed_AS_cut:.8g}")

    if args.control == "MC":
        run_label = "MC"
    else:
        run_label = "283408"
    _plot_one(
        Bas, fixed_AS_cut,
        outdir / f"as_dist_bkg_{args.control}.png",
        title="Anomaly score distribution: Background",
        run_label=run_label,
        bins=int(args.bins),
    )
    _plot_one(
        Tas, fixed_AS_cut,
        outdir / f"as_dist_tt_{args.control}.png",
        title="Anomaly score distribution: ttbar",
        run_label=run_label,
        bins=int(args.bins),
    )
    _plot_one(
        Aas, fixed_AS_cut,
        outdir / f"as_dist_aa_{args.control}.png",
        title="Anomaly score distribution: HToAATo4B",
        run_label=run_label,
        bins=int(args.bins),
    )

    print(f"[OK] Wrote plots to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
