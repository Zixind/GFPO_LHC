#experiment_testae_v2.py: using autoencoder with RELU activation
#save latent dimension as 2 for both MC and real data
import numpy as np
from tensorflow import keras
from keras.layers import Dense, Flatten, Reshape, InputLayer
from keras.models import Sequential
import argparse
import numpy as np
import random
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import os
# from .data import process_h5_file_Data, process_h5_file_MC #process_h5_file0_newData, process_h5_file_newMC, load_bkg_aa_tt #new readin H5 file functions V2
from .models import build_autoencoder_data # the new AE model with RELU V2
from .losses import masked_mse_loss  # mse loss
from ..derived_info.scoring import (
    batch_mse_scores,
    percentile_threshold,
    pass_rate_above,
)
from .plots import plot_signal_pass_vs_dim, plot_hist_pair, plot_signal_pass_vs_dim_data, plot_hist_for_dim
from pathlib import Path
try:
    import atlas_mpl_style as aplt
    aplt.use_atlas_style()
except Exception:
    pass
from pathlib import Path
def ensure_parent_dir(path_like):
    """Create parent directory for a file path (e.g., outputs/xx.pdf)."""
    p = Path(path_like)
    if p.parent and str(p.parent) != "":
        p.parent.mkdir(parents=True, exist_ok=True)
THIS_DIR = Path(__file__).resolve().parent        # .../SampleProcessing/ae
PKG_ROOT = THIS_DIR.parent                        # .../SampleProcessing
DEFAULT_MODEL_DIR = PKG_ROOT / "models"           # .../SampleProcessing/models
# ----------------------------
# Helpers
# ----------------------------
def set_all_seeds(seed: int):
    """Helpers to set all random seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)



def jets_npv_to_X(jets: np.ndarray, npv: np.ndarray) -> np.ndarray:
    """
    jets: (N, 8, 3)
    npv:  (N,) or (N,1)

    returns X: (N, 25) = flatten(jets)->24 + npv->1
    """
    jets_flat = jets.reshape(jets.shape[0], -1)  # (N, 24)
    if npv.ndim == 1:
        npv = npv.reshape(-1, 1)
    return np.concatenate([jets_flat, npv], axis=1).astype(np.float32)



def train_one_dim(X_train, X_val, img_shape, code_dim, loss_name="mse", lr = 2e-3):
    """
    ReLU AE (build_autoencoder_data) + optional masked loss.
    """
    enc, dec = build_autoencoder_data(img_shape, code_dim)

    inp = keras.Input(img_shape)
    z = enc(inp)
    out = dec(z)
    ae = keras.Model(inp, out)

    loss = loss_name if loss_name != "masked" else masked_mse_loss
    opt = keras.optimizers.Adam(learning_rate=lr)
    ae.compile(optimizer=opt, loss=loss)
    # ae.compile(optimizer="adamax", loss=loss)

    es = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3, restore_best_weights=True
    )
    ae.fit(
        X_train, X_train,
        validation_data=(X_val, X_val),
        epochs=100,
        callbacks=[es],
        verbose=0,
        shuffle=True,
    )
    return ae

def add_mode_suffix(path_like: str, mode: str) -> str:
    """
    Return a new path with _<mode> inserted before extension.
    Example: outputs/x.pdf -> outputs/x_mc.pdf
    If it already ends with _mc/_realdata (before extension), keep it.
    """
    p = Path(path_like)
    stem = p.stem
    mode = mode.lower()
    # normalize RealData -> realdata
    if mode == "realdata":
        mode_tag = "realdata"
    else:
        mode_tag = "mc"

    if stem.endswith(f"_{mode_tag}"):
        return str(p)

    return str(p.with_name(f"{stem}_{mode_tag}{p.suffix}"))


# ----------------------------
# Main pipeline
# ----------------------------
def run_experiment(args):
    # ---- Load ----
    (bkg_jets, bkg_ht, bkg_npv), (aa_jets, aa_ht, aa_npv), (tt_jets, tt_ht, tt_npv) = load_bkg_aa_tt(args)

    # ---- Apply stride to background only (like your older scripts) ----
    stride = args.stride
    bkg_jets = bkg_jets[::stride]
    bkg_ht   = bkg_ht[::stride]
    bkg_npv  = bkg_npv[::stride]

    # ---- Build X vectors (24 jets + 1 npv = 25) ----
    X_bkg = jets_npv_to_X(bkg_jets, bkg_npv)
    X_AA  = jets_npv_to_X(aa_jets,  aa_npv)
    X_TT  = jets_npv_to_X(tt_jets,  tt_npv)

    HT_bkg = np.asarray(bkg_ht)
    HT_AA  = np.asarray(aa_ht)
    HT_TT  = np.asarray(tt_ht)

    print("[SHAPES] X_bkg:", X_bkg.shape, "X_AA:", X_AA.shape, "X_TT:", X_TT.shape)

    # ---- Train/test split for background ----
    # Keep semantics:
    #  - RealData: HT baseline is percentile of HT_test 
    #  - MC:       HT baseline is percentile of HT_bkg full 
    X_tr, X_te, HT_tr, HT_te = train_test_split(
        X_bkg, HT_bkg, test_size=0.2, random_state=40
    )

    img_shape = X_tr.shape[1:]  # (25,)

    # ---- HT baseline ----
    if args.control == "RealData":
        thr_ht = percentile_threshold(HT_te, 99.75)
    else:
        thr_ht = percentile_threshold(HT_bkg, 99.75)

    AA_ht_pass = 100.0 * np.mean(HT_AA > thr_ht)
    TT_ht_pass = 100.0 * np.mean(HT_TT > thr_ht)
    print("AA_ht_passed:", AA_ht_pass)
    print("TT_ht_passed:", TT_ht_pass)

    # ---- AE sweep over dims (default [2]) ----
    results_dim = {}
    aa_rates, tt_rates = [], []

    for d in args.dims:
        print(f"\n=== Training ReLU autoencoder dim={d} (control={args.control}) ===")
        ae = train_one_dim(X_tr, X_te, img_shape, code_dim=d, loss_name=args.loss)

        # save selected dims (default [2]) for BOTH MC and RealData
        if d in args.save_model_dims:
            os.makedirs(args.model_out_dir, exist_ok=True)
            save_path = os.path.join(args.model_out_dir, f"{args.model_prefix}{d}.keras")
            print(f"Saving autoencoder dim={d} to {save_path}")
            ae.save(save_path)

        # scores
        bkg_scores = batch_mse_scores(ae, X_te)
        aa_scores  = batch_mse_scores(ae, X_AA)
        tt_scores  = batch_mse_scores(ae, X_TT)

        thr = percentile_threshold(bkg_scores, 99.75)
        aa_pass = pass_rate_above(aa_scores, thr)
        tt_pass = pass_rate_above(tt_scores, thr)

        results_dim[d] = dict(bkg_scores=bkg_scores, AA_scores=aa_scores, TT_scores=tt_scores)
        aa_rates.append(aa_pass)
        tt_rates.append(tt_pass)

        print(f"dim={d:2d}  thr={thr:.4g}  AA={aa_pass:6.2f}%  TT={tt_pass:6.2f}%")

    # ---- Efficiency vs dim plot ----
    if args.control == "MC":
        plot_signal_pass_vs_dim(
            args.dims,
            aa_rates,
            tt_rates,
            AA_ht_pass,
            TT_ht_pass,
            args.out_pass_vs_dim,
        )
    else:
        # meeting-style plot for RealData mode (same as your old wrapper)
        base, _ = os.path.splitext(args.out_pass_vs_dim)
        plot_signal_pass_vs_dim_data(
            args.dims,
            aa_rates,
            tt_rates,
            aa_ht_eff=21.33,
            tt_ht_eff=97.26,
            out_prefix=base,
        )

    # ---- Hist pair ----
    # If only one dim (e.g. [2]), plot it twice so plot_hist_pair won't break.
    if len(args.dims) == 1:
        d1 = d2 = args.dims[0]
    else:
        d1 = args.dims[0]
        d2 = 4 if 4 in results_dim else args.dims[min(1, len(args.dims) - 1)]

    plot_hist_pair(
        d1,
        d2,
        results_dim[d1],
        results_dim[d2],
        out_pair=args.out_hist_pair,
        out_a=args.out_hist_a,
        out_b=args.out_hist_b,
    )


def main():
    ap = argparse.ArgumentParser(description="Autoencoder test experiment (ReLU AE + new H5 readers)")

    ap.add_argument("--control", default="MC", choices=["MC", "RealData"],
                    help="MC: Monte Carlo simulated samples MinBias bkg; RealData: real data background run. AA/TT are always MC.")

    ap.add_argument("--minbias", default="Data/MinBias_1.h5",
                    help="MC MinBias background file (used if --control MC).")
    ap.add_argument("--data", default="Data/data_Run_2016_283876.h5",
                    help="Real data background file (used if --control RealData).")

    ap.add_argument("--aa", default="Data/HToAATo4B.h5")
    ap.add_argument("--tt", default="Data/TT_1.h5")

    ap.add_argument("--stride", type=int, default=100,
                    help="Subsample stride for background only (bkg = bkg[::stride]).")

    # default latent dim sweep
    ap.add_argument("--dims", type=int, nargs="+", default=[1, 2, 4, 6, 8, 16])


    ap.add_argument("--loss", choices=["mse", "masked"], default="mse")

    ap.add_argument("--out_pass_vs_dim", default="outputs/autoencoders/signal_pass_vs_dimension.pdf")
    ap.add_argument("--out_hist_pair",   default="outputs/autoencoders/AS_hist_comparison2016.pdf")
    ap.add_argument("--out_hist_a",      default="outputs/autoencoders/AS_hist_comparison2016-a.pdf")
    ap.add_argument("--out_hist_b",      default="outputs/autoencoders/AS_hist_comparison2016-b.pdf")

    # default save dim=2 for BOTH modes
    ap.add_argument("--save-model-dims", type=int, nargs="+", default=[2])
    ap.add_argument("--model-out-dir", default=str(DEFAULT_MODEL_DIR))
    ap.add_argument("--model-prefix", default="autoencoder_model_")

    ap.add_argument("--seed", type=int, default=20251208)

    args = ap.parse_args()
    mode_tag = "realdata" if args.control == "RealData" else "mc"
    args.out_pass_vs_dim = add_mode_suffix(args.out_pass_vs_dim, mode_tag)
    args.out_hist_pair   = add_mode_suffix(args.out_hist_pair, mode_tag)
    args.out_hist_a      = add_mode_suffix(args.out_hist_a, mode_tag)
    args.out_hist_b      = add_mode_suffix(args.out_hist_b, mode_tag)

    # Make sure output dirs exist
    ensure_parent_dir(args.out_pass_vs_dim)
    ensure_parent_dir(args.out_hist_pair)
    ensure_parent_dir(args.out_hist_a)
    ensure_parent_dir(args.out_hist_b)

    if not args.model_prefix.endswith(f"_{mode_tag}_"):
        args.model_prefix = args.model_prefix.rstrip("_") + f"_{mode_tag}_"

    # Model directory too
    Path(args.model_out_dir).mkdir(parents=True, exist_ok=True)

    os.makedirs(args.model_out_dir, exist_ok=True)
    set_all_seeds(args.seed)

    print(f"[INFO] control={args.control} dims={args.dims} save_dims={args.save_model_dims} stride={args.stride}")
    run_experiment(args)


if __name__ == "__main__":
    main()