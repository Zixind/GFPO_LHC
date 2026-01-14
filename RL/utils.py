import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

def cummean(x):
    x = np.asarray(x, dtype=np.float64)
    return np.cumsum(x) / (np.arange(len(x)) + 1.0)

def rel_to_t0(x):
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return x
    return x / max(1e-12, float(x[0]))




# ------------------------- plotting helpers -------------------------
def add_cms_header(fig, left_x=0.13, right_x=0.90, y=0.94, run_label="Run 283408"):
    fig.text(left_x, y, "CMS Open Data",
             ha="left", va="top", fontweight="bold", fontsize=20)
    fig.text(right_x, y, run_label,
             ha="right", va="top", fontsize=20)

def save_pdf_png(fig, basepath, dpi_png=300):
    Path(basepath).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{basepath}.pdf", bbox_inches="tight")
    fig.savefig(f"{basepath}.png", bbox_inches="tight", dpi=dpi_png)

def plot_rate_with_tolerance(
    time,
    y_const,
    y_pd,
    y_dqn,
    outbase,
    *,
    run_label="Run 283408",
    legend_title="HT Trigger",
    ylim=(0, 200),
    ylabel="Background Rate [kHz]",
    xlabel="Time (Fraction of Run)",
    tol_upper=110.0,
    tol_lower=90.0,
    grid_alpha=0.6,
    label_fs = 22,
    tick_fs = 18,
    legend_fs = 16,
    legend_title_fs = 18,
    # styles (match your paper family)
    const_style=dict(color="tab:blue", linestyle="dashed", linewidth=3.0),
    pd_style=dict(color="mediumblue", linestyle="solid", linewidth=2.5),
    dqn_style=dict(color="tab:purple", linestyle="solid", linewidth=2.5),
    const_label="Constant Menu",
    pd_label="PD Controller",
    dqn_label="DQN",
    add_cms_header=None,
    save_pdf_png=None,
    dpi_png=300,
):
    """
    Plot background rate vs time with:
      - main legend: Constant/PD/DQN under legend_title
      - reference legend: Upper/Lower tolerance lines
    outbase: path WITHOUT extension (str or Path)
    add_cms_header(fig, run_label=...) and save_pdf_png(fig, basepath, dpi_png=...)
      are passed in (so utils doesn't need to import them from your script).
    """
    time = np.asarray(time)
    y_const = np.asarray(y_const)
    y_pd = np.asarray(y_pd)
    y_dqn = np.asarray(y_dqn)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(time, y_const, **const_style)
    ax.plot(time, y_pd,    **pd_style)
    ax.plot(time, y_dqn,   **dqn_style)

    ax.axhline(y=tol_upper, color="gray", linestyle="--", linewidth=1.5)
    ax.axhline(y=tol_lower, color="gray", linestyle="--", linewidth=1.5)

    ax.set_xlabel(xlabel, loc="center", fontsize = label_fs)
    ax.set_ylabel(ylabel, loc="center", fontsize = label_fs)
    # Bigger tick labels (x and y)
    ax.tick_params(axis="both", which="major", labelsize=tick_fs)
    ax.set_ylim(*ylim)
    ax.grid(True, linestyle="--", alpha=grid_alpha)

    # ---- Main legend (Constant/PD/DQN) ----
    h_const = mlines.Line2D([], [], **const_style)
    h_pd    = mlines.Line2D([], [], **pd_style)
    h_dqn   = mlines.Line2D([], [], **dqn_style)

    leg_main = ax.legend(
        [h_const, h_pd, h_dqn],
        [const_label, pd_label, dqn_label],
        title=legend_title,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        frameon=True,
        fontsize=legend_fs,
    )
    leg_main.get_title().set_fontsize(legend_title_fs)
    ax.add_artist(leg_main)

    # ---- Reference legend (tolerances) ----
    upper = mlines.Line2D([], [], color="gray", linestyle="--", linewidth=1.5)
    lower = mlines.Line2D([], [], color="gray", linestyle="--", linewidth=1.5)
    leg_ref = ax.legend(
        [upper, lower],
        [f"Upper Tolerance ({int(tol_upper)})", f"Lower Tolerance ({int(tol_lower)})"],
        title="Reference",
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        frameon=True,
        fontsize=legend_fs,
    )
    leg_ref.get_title().set_fontsize(legend_title_fs)
    if add_cms_header is not None:
        # your add_cms_header(fig, run_label=...)
        add_cms_header(fig, run_label=run_label)

    if save_pdf_png is not None:
        save_pdf_png(fig, str(outbase), dpi_png=dpi_png)
    
    plt.close(fig)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

def plot_rate_with_tolerance_4(
    time,
    y_const,
    y_pd,
    y_dqn,
    y_dqnf,
    outbase,
    *,
    run_label="Run 283408",
    legend_title="HT Trigger",
    ylim=(0, 200),
    ylabel="Background Rate [kHz]",
    xlabel="Time (Fraction of Run)",
    tol_upper=110.0,
    tol_lower=90.0,
    grid_alpha=0.6,
    label_fs=22,
    tick_fs=18,
    legend_fs=16,
    legend_title_fs=18,
    # styles (match your paper family)
    const_style=dict(color="tab:blue", linestyle="dashed", linewidth=3.0),
    pd_style=dict(color="mediumblue", linestyle="solid", linewidth=2.5),
    dqn_style=dict(color="tab:purple", linestyle="solid", linewidth=2.5),
    dqnf_style=dict(color="tab:red", linestyle=":", linewidth=3.0),
    const_label="Constant Menu",
    pd_label="PD Controller",
    dqn_label="DQN",
    dqnf_label="DQN-F",
    add_cms_header=None,
    save_pdf_png=None,
    dpi_png=300,
):
    """
    Plot background rate vs time with:
      - main legend: Constant/PD/DQN/DQN-F under legend_title
      - reference legend: Upper/Lower tolerance lines
    outbase: path WITHOUT extension (str or Path)
    add_cms_header(fig, run_label=...) and save_pdf_png(fig, basepath, dpi_png=...)
      are passed in.
    """
    time = np.asarray(time)
    y_const = np.asarray(y_const)
    y_pd = np.asarray(y_pd)
    y_dqn = np.asarray(y_dqn)
    y_dqnf = np.asarray(y_dqnf)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(time, y_const, **const_style)
    ax.plot(time, y_pd,    **pd_style)
    ax.plot(time, y_dqn,   **dqn_style)
    ax.plot(time, y_dqnf,  **dqnf_style)

    ax.axhline(y=tol_upper, color="gray", linestyle="--", linewidth=1.5)
    ax.axhline(y=tol_lower, color="gray", linestyle="--", linewidth=1.5)

    ax.set_xlabel(xlabel, loc="center", fontsize=label_fs)
    ax.set_ylabel(ylabel, loc="center", fontsize=label_fs)
    ax.tick_params(axis="both", which="major", labelsize=tick_fs)
    ax.set_ylim(*ylim)
    ax.grid(True, linestyle="--", alpha=grid_alpha)

    # ---- Main legend (Constant/PD/DQN/DQN-F) ----
    h_const = mlines.Line2D([], [], **const_style)
    h_pd    = mlines.Line2D([], [], **pd_style)
    h_dqn   = mlines.Line2D([], [], **dqn_style)
    h_dqnf  = mlines.Line2D([], [], **dqnf_style)

    leg_main = ax.legend(
        [h_const, h_pd, h_dqn, h_dqnf],
        [const_label, pd_label, dqn_label, dqnf_label],
        title=legend_title,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        frameon=True,
        fontsize=legend_fs,
    )
    leg_main.get_title().set_fontsize(legend_title_fs)
    ax.add_artist(leg_main)

    # ---- Reference legend (tolerances) ----
    upper = mlines.Line2D([], [], color="gray", linestyle="--", linewidth=1.5)
    lower = mlines.Line2D([], [], color="gray", linestyle="--", linewidth=1.5)
    leg_ref = ax.legend(
        [upper, lower],
        [f"Upper Tolerance ({int(tol_upper)})", f"Lower Tolerance ({int(tol_lower)})"],
        title="Reference",
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        frameon=True,
        fontsize=legend_fs,
    )
    leg_ref.get_title().set_fontsize(legend_title_fs)

    if add_cms_header is not None:
        add_cms_header(fig, run_label=run_label)

    if save_pdf_png is not None:
        save_pdf_png(fig, str(outbase), dpi_png=dpi_png)

    plt.close(fig)


def save_png(fig, basepath, dpi_png=300):
    """Save figure as PNG only. basepath: path without extension."""
    p = Path(basepath)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(p.with_suffix(".png")), bbox_inches="tight", dpi=dpi_png)






#### H5 reading utility ####

# ------------------------- H5 reading (the unified reader) -------------------------
import h5py
import hdf5plugin  # noqa: F401
def _collect_datasets(h5):
    """Return dict: dataset_path -> h5py.Dataset (supports nested groups)."""
    dsets = {}
    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            dsets[name] = obj
    h5.visititems(visitor)
    return dsets

def _basename(x: str) -> str:
    return x.split("/")[-1]

def _find_key(keys, candidates):
    """
    Find a dataset key by trying:
      1) exact match
      2) basename match (for nested datasets)
    """
    for c in candidates:
        if c in keys:
            return c
    # basename match
    for c in candidates:
        for k in keys:
            if _basename(k) == c:
                return k
    return None
def _first_present(h5_keys, candidates):
    for k in candidates:
        if k in h5_keys:
            return k
    return None

def _read_score(h5, prefix: str, dim: int):
    """
    Trigger_food_MC.h5 or Trigger_food_Data.h5 or Matched_data_2016_dim2.h5 : f"{prefix}_score{dim:02d}" like mc_bkg_score02 or data_bkg_score02.
    """
    d2 = f"{int(dim):02d}"
    for k in (f"{prefix}_score0{d2}", f"{prefix}_scores0{d2}"):
        if k in h5:
            return h5[k][:]
    return None

def print_h5_tree(path: str, max_items: int | None = None) -> None:
    """
    Print ALL keys in an HDF5 file, including nested groups/datasets.
    Shows dataset shape + dtype. Use max_items to truncate.
    """
    print(f"\n[H5] Inspect: {path}")
    n = 0

    def visitor(name, obj):
        nonlocal n
        if max_items is not None and n >= max_items:
            return
        if isinstance(obj, h5py.Group):
            print(f"  [G] {name}/")
        elif isinstance(obj, h5py.Dataset):
            print(f"  [D] {name}  shape={obj.shape}  dtype={obj.dtype}")
        else:
            print(f"  [?] {name}  type={type(obj)}")
        n += 1

    with h5py.File(path, "r") as h5:
        # root keys (top-level)
        print("  Top-level:", list(h5.keys()))
        h5.visititems(visitor)

    if max_items is not None:
        print(f"  ... printed up to max_items={max_items}")
    print("")
def _read_score(h5, prefix: str, dim: int):
    """
    Supports either:
      - top-level datasets: f"{prefix}_score02"
      - groups: h5[prefix][f"score02"]
      - minor naming variants: score2, score_02, etc.
    Returns None if not found.
    """
    d = int(dim)

    # 1) Common top-level dataset names
    candidates = [
        f"{prefix}_score{d:02d}",   # data_bkg_score02  (your file)
        f"{prefix}_score{d}",       # data_bkg_score2
        f"{prefix}_score_{d:02d}",  # data_bkg_score_02
        f"{prefix}_score_{d}",      # data_bkg_score_2
    ]
    for k in candidates:
        if k in h5:
            return h5[k][:]

    # 2) Group-style layout: h5["data_bkg"]["score02"]
    if prefix in h5 and hasattr(h5[prefix], "keys"):
        g = h5[prefix]
        gcands = [f"score{d:02d}", f"score{d}", f"score_{d:02d}", f"score_{d}"]
        for kk in gcands:
            if kk in g:
                return g[kk][:]

    # 3) Path-style layout: "data_bkg/score02"
    path_candidates = [
        f"{prefix}/score{d:02d}",
        f"{prefix}/score{d}",
        f"{prefix}/score_{d:02d}",
        f"{prefix}/score_{d}",
    ]
    for k in path_candidates:
        if k in h5:
            return h5[k][:]

    return None


def read_any_h5(path: str, score_dim_hint: int = 2):
    """
    Unified outputs (same keys as your DQN code expects):
      Bht, Bnpv, Bas2, #background Ht, Npv, anomaly score with dim 2
      Tht, Tnpv, Tas2, #ttbar Ht, Npv, anomaly score with dim 2
      Aht, Anpv, Aas2, #aa Ht, Npv, anomaly score with dim 2
      meta['matched_by_index']

    Supported input files:
      - Trigger_food_MC.h5          (MC control) Background Data/MinBias_2.h5 + aa Data/HToAATo4B.h5 + ttbar Data/TT_1.h5
      - Trigger_food_Data.h5        (RealData control, unpaired) Background Data/data_Run_2016_283408_longest.h5
      - Matched_data_2016.h5        (RealData paired) Matched_data_2016_dim2.h5
    """
    with h5py.File(path, "r") as h5:
        keys = set(h5.keys())
        hint = int(score_dim_hint)

        # ------------------------------------------------------------
        # Case A) MC Trigger_food (control="MC")
        # ------------------------------------------------------------
        if ("mc_bkg_ht" in keys) and ("mc_bkg_Npv" in keys):
            Bht  = h5["mc_bkg_ht"][:]
            Bnpv = h5["mc_bkg_Npv"][:]

            Tht  = h5["mc_tt_ht"][:]
            Aht  = h5["mc_aa_ht"][:]

            # tt_Npv / aa_Npv (not mc_tt_Npv / mc_aa_Npv)
            Tnpv = h5["tt_Npv"][:] if "tt_Npv" in keys else np.zeros_like(Tht, dtype=np.float32)
            Anpv = h5["aa_Npv"][:] if "aa_Npv" in keys else np.zeros_like(Aht, dtype=np.float32)
            # suppose dim = 2
            Bas = _read_score(h5, "mc_bkg", hint)

            Tas = _read_score(h5, "mc_tt",  hint)

            Aas = _read_score(h5, "mc_aa",  hint)

            if Bas is None or Tas is None or Aas is None:
                raise SystemExit(
                    f"[read_any_h5] MC file missing score{hint:02d}. "
                    f"Expected keys like mc_bkg_score{hint:02d}, mc_tt_score{hint:02d}, mc_aa_score{hint:02d}. "
                    f"Top-level keys: {sorted(list(keys))}"
                )



            out = dict(
                Bht=Bht, Bnpv=Bnpv,
                Tht=Tht, Tnpv=Tnpv,
                Aht=Aht, Anpv=Anpv,
                meta=dict(matched_by_index=False, score_dim=hint),
            )
            out[f"Bas{hint}"] = Bas
            out[f"Tas{hint}"] = Tas
            out[f"Aas{hint}"] = Aas
            return out


        # ------------------------------------------------------------
        # Case B) RealData Trigger_food_Data (unpaired) OR paired Matched_data_2016_dim2.h5
        #   - Paired Matched_data has data_Npv not data_bkg_Npv
        #   - Unpaired Trigger_food_Data has data_bkg_Npv and also data_tt_Npv / data_aa_Npv
        # ------------------------------------------------------------
        has_bkg = ("data_bkg_ht" in keys)
        has_npvs_any = ("data_Npv" in keys) or ("data_bkg_Npv" in keys)
        has_tt = ("data_tt_ht" in keys)
        has_aa = ("data_aa_ht" in keys)

        if has_bkg and has_npvs_any and has_tt and has_aa:
            # Background arrays
            Bht = h5["data_bkg_ht"][:]
            # paired file uses data_Npv; unpaired uses data_bkg_Npv
            npv_key = _first_present(keys, ["data_Npv", "data_bkg_Npv"])
            Bnpv = h5[npv_key][:]

            # Signal arrays (already aligned to the background npv distribution if paired)
            Tht = h5["data_tt_ht"][:]
            Aht = h5["data_aa_ht"][:]

            # keep these for the "mask by npv range" branch
            # (in paired file they exist as data_tt_Npv / data_aa_Npv written by run_pairing_npv)
            Tnpv_k = _first_present(keys, ["data_tt_Npv", "data_tt_npv"])
            Anpv_k = _first_present(keys, ["data_aa_Npv", "data_aa_npv"])
            Tnpv = h5[Tnpv_k][:] if Tnpv_k else np.zeros_like(Tht, dtype=np.float32)
            Anpv = h5[Anpv_k][:] if Anpv_k else np.zeros_like(Aht, dtype=np.float32)

            # Bas2 = _read_score(h5, "data_bkg", hint)

            # Tas2 = _read_score(h5, "data_tt",  hint)

            # Aas2 = _read_score(h5, "data_aa",  hint)
            Bas2 = _read_score(h5, "data_bkg", hint)
            Tas2 = _read_score(h5, "data_tt",  hint)
            Aas2 = _read_score(h5, "data_aa",  hint)

            if Bas2 is None or Tas2 is None or Aas2 is None:
                raise SystemExit(
                    f"[read_any_h5] Data file missing score{hint:02d}. "
                    f"Expected keys like data_bkg_score{hint:02d}, data_tt_score{hint:02d}, data_aa_score{hint:02d}. "
                    f"Top-level keys: {sorted(list(keys))}"
                )


            # IMPORTANT:
            # - If file has data_Npv, tt/aa were already matched: should start with Matched_data_2016_dim2.h5 -> treat as matched_by_index=True
            # - If file has data_bkg_Npv, it’s unpaired Trigger_food_Data.h5 -> matched_by_index=False
            matched_by_index = ("data_Npv" in keys)

            out = dict(
                Bht=Bht, Bnpv=Bnpv,
                Tht=Tht, Tnpv=Tnpv,
                Aht=Aht, Anpv=Anpv,
                meta=dict(matched_by_index=matched_by_index, score_dim=hint),
            )
            print("keys: {}".format(keys))
            out[f"Bas{hint}"] = Bas2
            out[f"Tas{hint}"] = Tas2
            out[f"Aas{hint}"] = Aas2
            return out

        # ------------------------------------------------------------
        # Fall back: unknown layout
        # ------------------------------------------------------------
        raise SystemExit(
            "[read_any_h5] Unrecognized H5 layout. "
            "Run with --print-keys to inspect keys.\n"
            f"Top-level keys: {sorted(list(keys))}"
        )



# AUROC plotting per chunk
# ------------------------- ROC/AUC helpers -------------------------
def _downsample_pair(scores, labels, max_n=200_000, seed=20251213):
    """
    Keep ROC/AUC computation fast by downsampling to max_n points.
    NOTE: Avoid printing here; it will spam during loops.
    """
    scores = np.asarray(scores, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int32)

    n = scores.size
    if n <= max_n:
        return scores, labels

    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_n, replace=False)
    return scores[idx], labels[idx]



def roc_curve_np(scores, labels):
    """
    Compute ROC curve points (FPR, TPR) given scores and binary labels.
    Higher score => more likely positive.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)

    m = np.isfinite(scores)
    scores = scores[m]
    labels = labels[m]
    if scores.size == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])

    order = np.argsort(scores)[::-1]
    y = labels[order]

    P = float(np.sum(y == 1))
    N = float(np.sum(y == 0))
    if P == 0 or N == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])

    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)

    tpr = tp / P
    fpr = fp / N

    # endpoints
    tpr = np.concatenate([[0.0], tpr, [1.0]])
    fpr = np.concatenate([[0.0], fpr, [1.0]])
    return fpr, tpr

def cut_at_event(cut_hist, event_idx, start_event, update_chunk_size):
    """
    event_idx -> index into cut_hist (piecewise-constant policy).
    """
    if len(cut_hist) == 0:
        return np.nan
    j = int((event_idx - start_event) // int(update_chunk_size))
    j = max(0, min(j, len(cut_hist) - 1))
    return float(cut_hist[j])

def auc_trapz(fpr, tpr):
    fpr = np.asarray(fpr, dtype=np.float64)
    tpr = np.asarray(tpr, dtype=np.float64)
    o = np.argsort(fpr)
    return float(np.trapz(tpr[o], fpr[o]))

# ------------------------- AUC + operating point -------------------------
def chunk_auc_binary_from_margin(x_bkg, x_sig, cut, max_n=200000, seed=20251213):
    """
    AUROC for ONE signal class vs background.

    score = margin = x - cut
    labels: bkg=0, sig=1

    NOTE: AUROC is invariant to subtracting a constant 'cut' (ranking unchanged).
    So for the same events, AUROC(x-cut) == AUROC(x).
    """
    cut = float(cut)
    b = np.asarray(x_bkg, dtype=np.float32)
    s = np.asarray(x_sig, dtype=np.float32) if x_sig is not None else np.empty(0, np.float32)

    if b.size == 0 or s.size == 0:
        return np.nan

    scores = np.concatenate([b - cut, s - cut]).astype(np.float32, copy=False)
    labels = np.concatenate([
        np.zeros(b.size, dtype=np.int32),
        np.ones(s.size, dtype=np.int32),
    ])

    scores, labels = _downsample_pair(scores, labels, max_n=max_n, seed=seed)
    fpr, tpr = roc_curve_np(scores, labels)
    return auc_trapz(fpr, tpr)


def chunk_operating_point_at_zero(x_bkg, x_sig, cut):
    """
    Operating point at threshold 0 on margin:
      margin = x - cut
      accept if margin > 0  (anomaly decision)

    Returns:
      fpr0: background accept fraction
      tpr0: signal accept fraction
    """
    cut = float(cut)
    b = np.asarray(x_bkg, dtype=np.float32)
    s = np.asarray(x_sig, dtype=np.float32) if x_sig is not None else np.empty(0, np.float32)

    if b.size == 0 or s.size == 0:
        return np.nan, np.nan

    fpr0 = float(np.mean((b - cut) > 0.0))
    tpr0 = float(np.mean((s - cut) > 0.0))
    return fpr0, tpr0



def chunk_auc_from_margin(x_bkg, x_tt, x_aa, cut, max_n=200_000, seed=20251213):
    """
    AUROC for the policy defined by a cut:
      margin = x - cut       
    Trigger rule:
      accept if margin > 0

    Then AUROC is computed by sweeping a threshold over this margin score
      - threshold very high => no event accepted => (FPR, TPR) ~ (0, 0)
      - threshold very low  => every event accepted => (FPR, TPR) ~ (1, 1)
    
    Labels:
      - background (bkg) -> label 0
      - signal = ttbar + aa -> label 1 (pooled together)

    Define Accepted if score > 0
      label: bkg=0, (tt+aa)=1
    """
    cut = float(cut)

    b = np.asarray(x_bkg, dtype=np.float32)
    s_parts = []
    if x_tt is not None and len(x_tt) > 0:
        s_parts.append(np.asarray(x_tt, dtype=np.float32))
    if x_aa is not None and len(x_aa) > 0:
        s_parts.append(np.asarray(x_aa, dtype=np.float32))
    if len(s_parts) == 0 or b.size == 0:
        return np.nan

    s = np.concatenate(s_parts)

    scores = np.concatenate([b - cut, s - cut]).astype(np.float32, copy=False)
    labels = np.concatenate([
        np.zeros(b.size, dtype=np.int32),
        np.ones(s.size, dtype=np.int32),
    ])

    scores, labels = _downsample_pair(scores, labels, max_n=max_n, seed=seed)
    fpr, tpr = roc_curve_np(scores, labels)
    return auc_trapz(fpr, tpr)


def compute_auroc_windows_separate(
    *,
    start_event,
    window_events,
    update_chunk_size,
    matched_by_index,
    Bnpv, Tnpv, Anpv,
    Bx, Tx, Ax,              # HT or AS arrays for bkg/tt/aa
    cut_hist_pd,
    cut_hist_dqn,
    max_n=200_000,
    seed=20251213,
):
    """
    Returns:
      t_mid: time fraction for each window midpoint
      auc_tt_pd,  auc_tt_dqn: AUROC(bkg vs tt) per window
      auc_aa_pd,  auc_aa_dqn: AUROC(bkg vs aa) per window

    Notes:
      - AUROC is computed on score = (x - cut) but AUROC is invariant to the cut.
      - If the signal selection per window is identical, PD and DQN AUROC will overlap.
    """
    N = len(Bx)
    w = int(window_events)
    if w <= 0:
        raise ValueError("window_events must be > 0")

    window_starts = list(range(int(start_event), N, w))

    t_mid = []
    auc_tt_pd = []
    auc_tt_dqn = []
    auc_aa_pd = []
    auc_aa_dqn = []

    denom = max(1, (N - int(start_event)))

    for k, ws in enumerate(window_starts):
        we = min(ws + w, N)
        if we <= ws:
            continue

        # background in this window
        b = Bx[ws:we]
        bnpv = Bnpv[ws:we] if Bnpv is not None else None

        # signal in this window (tt, aa)
        if matched_by_index:
            we_sig = min(we, len(Tx), len(Ax))
            if ws >= we_sig:
                tt = np.empty(0, dtype=np.float32)
                aa = np.empty(0, dtype=np.float32)
            else:
                tt = Tx[ws:we_sig]
                aa = Ax[ws:we_sig]
        else:
            if bnpv is None or len(bnpv) == 0:
                tt = np.empty(0, dtype=np.float32)
                aa = np.empty(0, dtype=np.float32)
            else:
                npv_min = float(np.min(bnpv))
                npv_max = float(np.max(bnpv))
                mtt = (Tnpv >= npv_min) & (Tnpv <= npv_max)
                maa = (Anpv >= npv_min) & (Anpv <= npv_max)
                tt = Tx[mtt]
                aa = Ax[maa]

        # cuts used at this time
        c_pd  = cut_at_event(cut_hist_pd,  ws, start_event, update_chunk_size)
        c_dqn = cut_at_event(cut_hist_dqn, ws, start_event, update_chunk_size)

        # AUROC per class (bkg vs tt) and (bkg vs aa)
        auc_tt_pd.append(chunk_auc_binary_from_margin(b, tt, c_pd,  max_n=max_n, seed=seed + 10*k + 1))
        auc_tt_dqn.append(chunk_auc_binary_from_margin(b, tt, c_dqn, max_n=max_n, seed=seed + 10*k + 2))

        auc_aa_pd.append(chunk_auc_binary_from_margin(b, aa, c_pd,  max_n=max_n, seed=seed + 10*k + 3))
        auc_aa_dqn.append(chunk_auc_binary_from_margin(b, aa, c_dqn, max_n=max_n, seed=seed + 10*k + 4))

        # time coordinate
        mid = 0.5 * (ws + we)
        t_mid.append((mid - int(start_event)) / denom)

    return (
        np.asarray(t_mid),
        np.asarray(auc_tt_pd), np.asarray(auc_tt_dqn),
        np.asarray(auc_aa_pd), np.asarray(auc_aa_dqn),
    )


def compute_operating_point_windows_separate(
    *,
    start_event,
    window_events,
    update_chunk_size,
    matched_by_index,
    Bnpv, Tnpv, Anpv,
    Bx, Tx, Ax,
    cut_hist_pd,
    cut_hist_dqn,
):
    """
    Returns per window:
      - fpr0_* : background accept fraction at margin>0
      - tpr0_tt_* : tt accept fraction at margin>0
      - tpr0_aa_* : aa accept fraction at margin>0
    """
    N = len(Bx)
    w = int(window_events)
    if w <= 0:
        raise ValueError("window_events must be > 0")

    window_starts = list(range(int(start_event), N, w))

    denom = max(1, (N - int(start_event)))
    t_mid = []

    fpr0_pd = []; fpr0_dqn = []
    tpr0_tt_pd = []; tpr0_tt_dqn = []
    tpr0_aa_pd = []; tpr0_aa_dqn = []

    for ws in window_starts:
        we = min(ws + w, N)
        if we <= ws:
            continue

        b = Bx[ws:we]
        bnpv = Bnpv[ws:we] if Bnpv is not None else None

        if matched_by_index:
            we_sig = min(we, len(Tx), len(Ax))
            if ws >= we_sig:
                tt = np.empty(0, np.float32)
                aa = np.empty(0, np.float32)
            else:
                tt = Tx[ws:we_sig]
                aa = Ax[ws:we_sig]
        else:
            if bnpv is None or len(bnpv) == 0:
                tt = np.empty(0, np.float32)
                aa = np.empty(0, np.float32)
            else:
                npv_min = float(np.min(bnpv))
                npv_max = float(np.max(bnpv))
                tt = Tx[(Tnpv >= npv_min) & (Tnpv <= npv_max)]
                aa = Ax[(Anpv >= npv_min) & (Anpv <= npv_max)]

        c_pd  = cut_at_event(cut_hist_pd,  ws, start_event, update_chunk_size)
        c_dqn = cut_at_event(cut_hist_dqn, ws, start_event, update_chunk_size)

        # PD operating point
        fpr_pd, tpr_tt_pd = chunk_operating_point_at_zero(b, tt, c_pd)
        _fpr_pd2, tpr_aa_pd2 = chunk_operating_point_at_zero(b, aa, c_pd)

        # DQN operating point
        fpr_dq, tpr_tt_dq = chunk_operating_point_at_zero(b, tt, c_dqn)
        _fpr_dq2, tpr_aa_dq2 = chunk_operating_point_at_zero(b, aa, c_dqn)

        # background FPR should match regardless of which signal you pair with;
        # we still compute it once and store it.
        fpr0_pd.append(fpr_pd)
        fpr0_dqn.append(fpr_dq)

        tpr0_tt_pd.append(tpr_tt_pd)
        tpr0_tt_dqn.append(tpr_tt_dq)

        tpr0_aa_pd.append(tpr_aa_pd2)
        tpr0_aa_dqn.append(tpr_aa_dq2)

        mid = 0.5 * (ws + we)
        t_mid.append((mid - int(start_event)) / denom)

    return (
        np.asarray(t_mid),
        np.asarray(fpr0_pd), np.asarray(fpr0_dqn),
        np.asarray(tpr0_tt_pd), np.asarray(tpr0_tt_dqn),
        np.asarray(tpr0_aa_pd), np.asarray(tpr0_aa_dqn),
    )







def cummean(x):
    x = np.asarray(x, dtype=np.float64)
    return np.cumsum(x) / np.arange(1, len(x) + 1)

def rel_to_t0(x):
    x = np.asarray(x, dtype=np.float64)
    return x / (x[0] + 1e-12)
def near_occupancy(x, cut, widths):
    x = np.asarray(x, dtype=np.float32)
    out = []
    for w in widths:
        out.append(float(np.mean(np.abs(x - cut) <= float(w))))
    return np.array(out, dtype=np.float32)




# plotting utils
# --- consistent paper fonts ---
AX_LABEL_FS = 22
TICK_FS     = 18
LEGEND_FS   = 14
LEGEND_TITLE_FS = 16

DIAG_AX_LABEL_FS = AX_LABEL_FS
DIAG_TICK_FS     = TICK_FS
DIAG_LEGEND_FS   = LEGEND_FS
DIAG_LEGEND_TITLE_FS = LEGEND_TITLE_FS

def style_diag_axes(ax, xlabel, ylabel, ylim=None):
    ax.set_xlabel(xlabel, fontsize=DIAG_AX_LABEL_FS)
    ax.set_ylabel(ylabel, fontsize=DIAG_AX_LABEL_FS)
    ax.tick_params(axis="both", which="major", labelsize=DIAG_TICK_FS)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True, linestyle="--", alpha=0.5)

def style_diag_legend(ax, title=None, loc="best"):
    leg = ax.legend(loc=loc, frameon=True, fontsize=DIAG_LEGEND_FS, title=title)
    if title is not None and leg is not None:
        leg.get_title().set_fontsize(DIAG_LEGEND_TITLE_FS)
    return leg

def finalize_diag_fig(fig, top=0.86):
    # Reserve space for CMS header so it doesn’t collide with ticks/title
    fig.tight_layout()
    fig.subplots_adjust(top=top)

import matplotlib as mpl

def apply_paper_style():
    """Call once, early (after hep.style.use('CMS') if you use mplhep)."""
    mpl.rcParams.update({
        # label/tick sizes
        "axes.labelsize": AX_LABEL_FS,
        "xtick.labelsize": TICK_FS,
        "ytick.labelsize": TICK_FS,

        # legend sizes
        "legend.fontsize": LEGEND_FS,
        "legend.title_fontsize": LEGEND_TITLE_FS,

        # consistent look
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.5,
        "lines.linewidth": 2.4,

        # saving defaults
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
    })




import matplotlib as mpl
import matplotlib.pyplot as plt

AX_LABEL_FS = 22
TICK_FS     = 18
LEGEND_FS   = 14
LEGEND_TITLE_FS = 16

DIAG_AX_LABEL_FS = AX_LABEL_FS
DIAG_TICK_FS     = TICK_FS
DIAG_LEGEND_FS   = LEGEND_FS
DIAG_LEGEND_TITLE_FS = LEGEND_TITLE_FS

def set_paper_style():
    """Call once at program start (after any plt.style.use / hep.style.use)."""
    mpl.rcParams.update({
        # text
        "axes.labelsize": AX_LABEL_FS,
        "xtick.labelsize": TICK_FS,
        "ytick.labelsize": TICK_FS,
        "legend.fontsize": LEGEND_FS,
        "legend.title_fontsize": LEGEND_TITLE_FS,

        # lines
        "lines.linewidth": 2.4,
        "lines.markersize": 4,

        # layout/save
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })

def style_diag_axes(ax, xlabel, ylabel, ylim=None):
    ax.set_xlabel(xlabel, fontsize=DIAG_AX_LABEL_FS)
    ax.set_ylabel(ylabel, fontsize=DIAG_AX_LABEL_FS)
    ax.tick_params(axis="both", which="major", labelsize=DIAG_TICK_FS)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True, linestyle="--", alpha=0.5)

def style_diag_legend(ax, title=None, loc="best"):
    leg = ax.legend(loc=loc, frameon=True, fontsize=DIAG_LEGEND_FS, title=title)
    if title is not None and leg is not None:
        leg.get_title().set_fontsize(DIAG_LEGEND_TITLE_FS)
    return leg

def finalize_diag_fig(fig, top=0.86):
    fig.tight_layout()
    fig.subplots_adjust(top=top)






def plot_inband_eff_single_signal_ad_vs_ht(
    summ_ad, summ_ht, *, signal_key, signal_label, outpath, run_label,
    ymin=None, ymax_pad=2.0, GFPO=True
):
    """
    Create one plot per signal:
      x-axis: triggers {AD, HT}
      bars: methods {Constant, PID, DQN, GRPO}
      y: mean in-band efficiency for `signal_key` in summarize_paper_table outputs
    """
    triggers = ["AD", "HT"]
    trigger_titles = {"AD": "AD Trigger", "HT": "HT Trigger"}
    if not GFPO:
        methods = ["Constant", "PID", "DQN", "GRPO"]
    else:
        methods = ["Constant", "PID", "DQN", "GRPO", "GFPO"]


    # Build values: shape (T, M)
    vals = np.zeros((2, len(methods)), dtype=np.float64)
    for ti, tr in enumerate(triggers):
        summ = summ_ad if tr == "AD" else summ_ht
        for mi, m in enumerate(methods):
            vals[ti, mi] = float(summ[m][signal_key])

    x = np.arange(len(triggers))
    bw = 0.80 / max(1, len(methods))

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    for mi, m in enumerate(methods):
        ax.bar(x - 0.40 + (mi + 0.5) * bw, vals[:, mi], width=bw, label=m)

    ax.set_xticks(x)
    ax.set_xticklabels([trigger_titles[t] for t in triggers])
    ax.set_ylabel(f"In-band efficiency ({signal_label}) [%]")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    # y-limits: force lower start as requested
    finite = vals[np.isfinite(vals)]
    if finite.size:
        y_top = float(np.max(finite) + float(ymax_pad))
    else:
        y_top = None

    if ymin is not None and y_top is not None:
        ax.set_ylim(float(ymin), y_top)
    elif ymin is not None:
        ax.set_ylim(float(ymin), ax.get_ylim()[1])

    ax.legend(loc="best", frameon=True, title="Method")
    add_cms_header(fig, run_label=run_label)
    finalize_diag_fig(fig)
    save_png(fig, str(outpath))
    plt.close(fig)
