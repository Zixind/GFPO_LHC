# derived_info/build_trigger_food_v2.py
# h5 returns (jest, ht, npv)
# X = [flatten(jets(8×3))=24, npv=1]

from __future__ import annotations 
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("CMS")
import numpy as np
import h5py

# use the updated readers from ae/data.py
# (adjust import if your package layout differs)
from ..ae.data import process_h5_file_MC, process_h5_file_Data #process_h5_file_newMC, process_h5_file0_newData
#only read in load_autoencoder for dim=2 model
from .scoring import load_autoencoder, calculate_H_met, count_njets
from .data_io import write_trigger_food


# -------------------------
# Helpers
# -------------------------

def ensure_parent_dir(path_like: str) -> None:
    p = Path(path_like)
    if p.parent and str(p.parent) != "":
        p.parent.mkdir(parents=True, exist_ok=True)


def jets_npv_to_X(jets: np.ndarray, npv: np.ndarray) -> np.ndarray:
    """
    jets: (N, 8, 3) with [eta, phi, pt]
    npv:  (N,) or (N,1)

    returns X: (N, 25) = flatten(jets)->24 + npv->1
    """
    jets = np.asarray(jets, dtype=np.float32)
    npv = np.asarray(npv, dtype=np.float32)
    jets_flat = jets.reshape(jets.shape[0], -1)  # (N, 24)
    if npv.ndim == 1:
        npv = npv.reshape(-1, 1)
    return np.concatenate([jets_flat, npv], axis=1).astype(np.float32)


def ae_mse_scores(ae, X: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    """
    Score AE with per-event MSE on X (N,25).
    Works for your ReLU AE trained as reconstruct(X)->X.
    """
    X = np.asarray(X, dtype=np.float32)
    Xhat = ae.predict(X, batch_size=batch_size, verbose=0)
    return np.mean((X - Xhat) ** 2, axis=1).astype(np.float32)


def load_sample(control: str, path: str, is_background: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns jets, ht, npv.
    Background uses RealData reader only if control=RealData.
    AA/TT are always MC (your setup).
    """
    if is_background and control == "RealData":
        jets, ht, npv = process_h5_file_Data(path)#process_h5_file0_newData(path)
    else:
        jets, ht, npv = process_h5_file_MC(path)#process_h5_file_newMC(path)
    return jets, ht, npv

def plot_anomaly_score_distribution(
    h5_path: str,
    control: str,
    ae_dim: int,
    out_dir: str | None = None,
    cut_quantile: float = 99.75,
    max_points: int = 300_000,
    bins: int = 90,
    show: bool = True,
) -> None:
    """
    Plot anomaly score (AE MSE) distributions for bkg vs tt vs aa
    from the Trigger_food_*.h5 written by this script.

    - Saves: anomaly_score_dist_raw_*.pdf/png and anomaly_score_dist_log10_*.pdf/png
    - Also shows interactively if show=True (may no-op on headless nodes).
    """
    score_key = f"score{ae_dim:02d}"  # e.g. score02

    # dataset names depend on control
    if control == "MC":
        k_bkg = f"mc_bkg_{score_key}"
        k_tt  = f"mc_tt_{score_key}"
        k_aa  = f"mc_aa_{score_key}"
    else:
        k_bkg = f"data_bkg_{score_key}"
        k_tt  = f"data_tt_{score_key}"
        k_aa  = f"data_aa_{score_key}"

    def _read_downsample(dset, max_points: int):
        n = dset.shape[0]
        if max_points is None or max_points <= 0 or n <= max_points:
            return np.asarray(dset[:], dtype=np.float32)
        stride = int(np.ceil(n / max_points))
        return np.asarray(dset[::stride], dtype=np.float32)

    h5_path = str(h5_path)
    out_dir = out_dir or str(Path(h5_path).parent)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, "r") as h5:
        for k in (k_bkg, k_tt, k_aa):
            if k not in h5:
                raise KeyError(f"Missing dataset '{k}' in {h5_path}. Keys: {list(h5.keys())}")

        bkg = _read_downsample(h5[k_bkg], max_points)
        tt  = _read_downsample(h5[k_tt],  max_points)
        aa  = _read_downsample(h5[k_aa],  max_points)

    tag = f"{Path(h5_path).stem}_{control}_dim{ae_dim:02d}"

    # background cut
    cut = float(np.percentile(bkg, cut_quantile))
    pass_b = 100.0 * float(np.mean(bkg > cut))
    eff_tt = 100.0 * float(np.mean(tt  > cut))
    eff_aa = 100.0 * float(np.mean(aa  > cut))

    def _cms_header(fig, left_x=0.13, right_x=0.90, y=0.94):
        fig.text(left_x, y, "CMS Open Data", ha="left", va="top",
                 fontweight="bold", fontsize=22)
        fig.text(right_x, y, tag, ha="right", va="top", fontsize=18)

    # ------------------------
    # RAW histogram
    # ------------------------
    fig = plt.figure(figsize=(10, 6))
    _cms_header(fig)

    allv = np.concatenate([bkg, tt, aa])
    lo = max(0.0, float(np.min(allv)))
    hi = float(np.percentile(allv, 99.9))
    bins_raw = np.linspace(lo, hi, bins)

    plt.hist(bkg, bins=bins_raw, density=True, histtype="step", linewidth=2.2, label=f"Background (n={len(bkg):,})")
    plt.hist(tt,  bins=bins_raw, density=True, histtype="step", linewidth=2.2, label=f"ttbar (n={len(tt):,})")
    plt.hist(aa,  bins=bins_raw, density=True, histtype="step", linewidth=2.2, label=f"H→AA→4b (n={len(aa):,})")

    plt.axvline(
        cut, linestyle="--", linewidth=1.8,
        label=(f"{cut_quantile:.2f}th % bkg cut\n"
               f"bkg pass={pass_b:.3f}% | tt={eff_tt:.2f}% | AA={eff_aa:.2f}%")
    )

    plt.xlabel("Anomaly score (AE MSE)", loc="center")
    plt.ylabel("Density", loc="center")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(frameon=True, fontsize=12)
    plt.tight_layout()

    raw_pdf = Path(out_dir) / f"anomaly_score_dist_raw_{tag}.pdf"
    raw_png = Path(out_dir) / f"anomaly_score_dist_raw_{tag}.png"
    fig.savefig(raw_pdf, bbox_inches="tight")
    fig.savefig(raw_png, bbox_inches="tight", dpi=300)
    if show:
        plt.show()
    plt.close(fig)

    # ------------------------
    # log10(score) histogram
    # ------------------------
    eps = 1e-12
    bkgL = np.log10(bkg + eps)
    ttL  = np.log10(tt  + eps)
    aaL  = np.log10(aa  + eps)
    cutL = np.log10(cut + eps)

    fig = plt.figure(figsize=(10, 6))
    _cms_header(fig)

    allL = np.concatenate([bkgL, ttL, aaL])
    loL = float(np.percentile(allL, 0.1))
    hiL = float(np.percentile(allL, 99.9))
    bins_log = np.linspace(loL, hiL, bins)

    plt.hist(bkgL, bins=bins_log, density=True, histtype="step", linewidth=2.2, label="Background")
    plt.hist(ttL,  bins=bins_log, density=True, histtype="step", linewidth=2.2, label="ttbar")
    plt.hist(aaL,  bins=bins_log, density=True, histtype="step", linewidth=2.2, label="H→AA→4b")
    plt.axvline(cutL, linestyle="--", linewidth=1.8, label=f"log10 cut = {cutL:.3f}")

    plt.xlabel("log10(Anomaly score + 1e-12)", loc="center")
    plt.ylabel("Density", loc="center")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(frameon=True, fontsize=12)
    plt.tight_layout()

    log_pdf = Path(out_dir) / f"anomaly_score_dist_log10_{tag}.pdf"
    log_png = Path(out_dir) / f"anomaly_score_dist_log10_{tag}.png"
    fig.savefig(log_pdf, bbox_inches="tight")
    fig.savefig(log_png, bbox_inches="tight", dpi=300)
    if show:
        plt.show()
    plt.close(fig)

    print(f"[PLOT] Saved anomaly score plots:\n  {raw_pdf}\n  {log_pdf}")
    print(f"[PLOT] Cut@{cut_quantile:.2f}th bkg percentile = {cut:.6g}")
    print(f"[PLOT] Pass: bkg={pass_b:.4f}% | tt={eff_tt:.3f}% | AA={eff_aa:.3f}%")


# -------------------------
# Main pipeline
# -------------------------
def run_pipeline(
    control: str,
    bkg_path: str,
    htoaa_path: str,
    tt_path: str,
    ae_dim: int,
    ae_path: str,
    out_path: str,
) -> None:
    # Load model (load_autoencoders returns tuple; we pass 1 path for dim=2
    ae = load_autoencoder(ae_path)

    # Load and preprocess datasets (jets, ht, npv)
    bkg_jets, bkg_ht, bkg_npv = load_sample(control, bkg_path, is_background=True)
    aa_jets,  aa_ht,  aa_npv  = load_sample(control, htoaa_path, is_background=False)
    tt_jets,  tt_ht,  tt_npv  = load_sample(control, tt_path, is_background=False)

    # # Derived features (should still work if calculate_H_met/count_njets assume pt is index 2)
    # bkg_Hmets = calculate_H_met(bkg_jets, bkg_ht)
    # aa_Hmets  = calculate_H_met(aa_jets,  aa_ht)
    # tt_Hmets  = calculate_H_met(tt_jets,  tt_ht)

    bkg_njets = count_njets(bkg_jets)
    aa_njets  = count_njets(aa_jets)
    tt_njets  = count_njets(tt_jets)

    # Keys for bookkeeping
    aa_key = np.ones_like(aa_npv, dtype=np.int32) * 1
    tt_key = np.ones_like(tt_npv, dtype=np.int32) * 2

    # AE scores on X = [jets_flat, npv]
    X_bkg = jets_npv_to_X(bkg_jets, bkg_npv)
    X_aa  = jets_npv_to_X(aa_jets,  aa_npv)
    X_tt  = jets_npv_to_X(tt_jets,  tt_npv)

    bkg_score = ae_mse_scores(ae, X_bkg)
    aa_score  = ae_mse_scores(ae, X_aa)
    tt_score  = ae_mse_scores(ae, X_tt)

    # Pack & write
    # NOTE: keep "mc_bkg_*" naming for compatibility with downstream pairing code,
    # even when control=RealData (your pairing reader treats mc_bkg_* as "data_*").
    score_key = f"score{ae_dim:02d}"  # e.g. score02

    if control == "MC":
        # different naming for MC control
        arrays: Dict[str, Any] = {
            "mc_bkg_ht":        np.asarray(bkg_ht, dtype=np.float32),
            f"mc_bkg_{score_key}": bkg_score,
            "mc_bkg_Npv":       np.asarray(bkg_npv, dtype=np.float32),

            "mc_aa_ht":         np.asarray(aa_ht, dtype=np.float32),
            f"mc_aa_{score_key}": aa_score,
            "aa_Npv":           np.asarray(aa_npv, dtype=np.float32),

            "mc_tt_ht":         np.asarray(tt_ht, dtype=np.float32),
            f"mc_tt_{score_key}": tt_score,
            "tt_Npv":           np.asarray(tt_npv, dtype=np.float32),
        }
    else:
        # different naming for RealData control
        arrays: Dict[str, Any] = {
            "data_bkg_ht":       np.asarray(bkg_ht, dtype=np.float32),
            f"data_bkg_{score_key}":  bkg_score,
            "data_bkg_Npv":      np.asarray(bkg_npv, dtype=np.float32),
            "data_bkg_njets":    np.asarray(bkg_njets, dtype=np.float32),

            "data_aa_ht":        np.asarray(aa_ht, dtype=np.float32),
            f"data_aa_{score_key}":  aa_score,
            "data_aa_Npv":       np.asarray(aa_npv, dtype=np.float32),
            "data_aa_njets":     np.asarray(aa_njets, dtype=np.float32),

            "data_tt_ht":        np.asarray(tt_ht, dtype=np.float32),
            f"data_tt_{score_key}":  tt_score,
            "data_tt_Npv":       np.asarray(tt_npv, dtype=np.float32),
            "data_tt_njets":     np.asarray(tt_njets, dtype=np.float32),
        }

    ensure_parent_dir(out_path)
    write_trigger_food(out_path, arrays)
    print(f"[OK] Wrote Trigger_food to {out_path} (AE dim={ae_dim})")
    # --- plot anomaly score distributions for the above file ---
    plot_anomaly_score_distribution(
        h5_path=out_path,
        control=control,
        ae_dim=ae_dim,
        out_dir=str(Path(out_path).parent),  # save next to the H5
        cut_quantile=99.75,
        max_points=300_000,   # downsample if huge
        bins=90,
        show=True,            # set False on headless machines
    )


# -------------------------
# Pairing (update to new score key)
# -------------------------
def run_pairing_npv(input_file: str, output_file: str, ae_dim: int) -> None:
    score_key = f"score{ae_dim:02d}"  # e.g. score02
    # Only applies to real data control for pairing npv
    with h5py.File(input_file, "r") as h5:
        data_ht     = h5["data_bkg_ht"][:]
        data_score  = h5[f"data_bkg_{score_key}"][:]
        data_npvs   = h5["data_bkg_Npv"][:]
        data_njets  = h5["data_bkg_njets"][:]

        tt_ht       = h5["data_tt_ht"][:]
        tt_score    = h5[f"data_tt_{score_key}"][:]
        tt_npvs     = h5["data_tt_Npv"][:]
        tt_njets = h5["data_tt_njets"][:]

        aa_ht       = h5["data_aa_ht"][:]
        aa_score    = h5[f"data_aa_{score_key}"][:]
        aa_npvs     = h5["data_aa_Npv"][:]
        aa_njets = h5["data_aa_njets"][:]

    def _match_to_data(data_npvs, sig_ht, sig_score, sig_npvs, sig_njets):
        order = np.argsort(sig_npvs)
        sig_ht = sig_ht[order]
        sig_score = sig_score[order]
        sig_npvs = sig_npvs[order]
        sig_njets = sig_njets[order]

        m_ht, m_sc, m_npv, m_nj = [], [], [], []
        for npv in data_npvs:
            L = np.searchsorted(sig_npvs, npv, side="left")
            R = np.searchsorted(sig_npvs, npv, side="right")
            if L >= len(sig_npvs):
                idx = len(sig_npvs) - 1
            elif L == R:
                idx = L
            else:
                idx = np.random.randint(L, R)
            m_ht.append(sig_ht[idx]); m_sc.append(sig_score[idx])
            m_npv.append(sig_npvs[idx]); m_nj.append(sig_njets[idx])
        return np.array(m_ht), np.array(m_sc), np.array(m_npv), np.array(m_nj)

    matched_tt_ht, matched_tt_score, matched_tt_npv, matched_tt_njets = _match_to_data(
        data_npvs, tt_ht, tt_score, tt_npvs, tt_njets
    )
    matched_aa_ht, matched_aa_score, matched_aa_npv, matched_aa_njets = _match_to_data(
        data_npvs, aa_ht, aa_score, aa_npvs, aa_njets
    )

    ensure_parent_dir(output_file)
    with h5py.File(output_file, "w") as h5_out:
        h5_out.create_dataset("data_bkg_ht", data=data_ht)
        h5_out.create_dataset(f"data_bkg_{score_key}", data=data_score)
        h5_out.create_dataset("data_Npv", data=data_npvs)
        # h5_out.create_dataset("data_njets", data=data_njets)

        h5_out.create_dataset("data_aa_ht", data=matched_aa_ht)
        h5_out.create_dataset(f"data_aa_{score_key}", data=matched_aa_score)
        h5_out.create_dataset("data_aa_Npv", data=matched_aa_npv)
        # h5_out.create_dataset("matched_aa_njets", data=matched_aa_njets)

        h5_out.create_dataset("data_tt_ht", data=matched_tt_ht)
        h5_out.create_dataset(f"data_tt_{score_key}", data=matched_tt_score)
        h5_out.create_dataset("data_tt_Npv", data=matched_tt_npv)
        # h5_out.create_dataset("matched_tt_njets", data=matched_tt_njets)

    print(f"[RealData] Pairing saved to: {output_file} (AE dim={ae_dim})")


# -------------------------
# CLI
# -------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build Trigger_food HDF5 using new readers + ReLU AE on X=(jets_flat,npv)")

    p.add_argument("--control", default="MC", choices=["MC", "RealData"])

    # Inputs
    p.add_argument("--minbias", default="Data/MinBias_2.h5", help="MC background (if control=MC)")
    p.add_argument("--data",    default="Data/data_Run_2016_283876.h5", help="Real background (if control=RealData)")
    p.add_argument("--htoaa",   default="Data/HToAATo4B.h5")
    p.add_argument("--tt",      default="Data/TT_1.h5")

    # AE (default dim=2)
    p.add_argument("--ae-dim", type=int, default=2)

    # Outputs
    p.add_argument("--out", default="Data/Trigger_food_MC.h5")
    p.add_argument("--out-paired", default="Data/Matched_data_2016.h5")

    return p


def main():
    args = build_argparser().parse_args()

    # Choose background file
    bkg_path = args.data if args.control == "RealData" else args.minbias

    # Default output names by control
    if args.out == "Data/Trigger_food_MC.h5" and args.control == "RealData":
        out = "Data/Trigger_food_Data.h5"
        ae_path_string = f"SampleProcessing/models/autoencoder_model_realdata_{args.ae_dim}.keras"
    else:
        out = args.out
        ae_path_string = f"SampleProcessing/models/autoencoder_model_mc_{args.ae_dim}.keras"

    run_pipeline(
        control=args.control,
        bkg_path=bkg_path,
        htoaa_path=args.htoaa,
        tt_path=args.tt,
        ae_dim=args.ae_dim,
        ae_path=ae_path_string,
        out_path=out,
    )

    if args.control == "RealData":
        run_pairing_npv(input_file=out, output_file=args.out_paired, ae_dim=args.ae_dim)


if __name__ == "__main__":
    main()
