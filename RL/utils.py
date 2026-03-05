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

from collections import deque, defaultdict

def inband_eff_by_method(chunk_rows, trigger):
    """
    Returns dict: method -> {"tt": mean_tt_inband, "h_to_4b": mean_aa_inband}
    Robust to trigger labels: AD may appear as "AD" or "AS".
    """
    if trigger == "AD":
        trig_ok = {"AD", "AS"}
    else:
        trig_ok = {trigger}

    acc = defaultdict(lambda: {"tt": [], "h_to_4b": []})

    for r in chunk_rows:
        tr = str(r.get("trigger", ""))
        if tr not in trig_ok:
            continue
        if int(r.get("inband", 0)) != 1:
            continue

        m = str(r.get("method", "UNK"))
        acc[m]["tt"].append(float(r.get("tt", np.nan)))
        acc[m]["h_to_4b"].append(float(r.get("aa", np.nan)))  # aa == h→4b

    out = {}
    for m, d in acc.items():
        tt = np.asarray(d["tt"], dtype=np.float64)
        aa = np.asarray(d["h_to_4b"], dtype=np.float64)

        tt = tt[np.isfinite(tt)]
        aa = aa[np.isfinite(aa)]

        out[m] = {
            "tt": float(np.mean(tt)) if tt.size else np.nan,
            "h_to_4b": float(np.mean(aa)) if aa.size else np.nan,
        }
    return out

from triggers import Sing_Trigger
def d_bg_d_cut_norm(scores, cut, step, target):
    # normalized derivative: (d bg_rate / d cut) / target
    step = float(step)
    if step <= 0:
        return 0.0
    p_plus  = float(Sing_Trigger(scores, float(cut) + step))
    p_minus = float(Sing_Trigger(scores, float(cut) - step))
    dp_dcut = (p_plus - p_minus) / (2.0 * step)  # typically negative
    return float(dp_dcut) / max(float(target), 1e-6)

def _make_edges(x, lo_q=0.5, hi_q=99.5, nbins=80):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    lo = float(np.percentile(x, lo_q))
    hi = float(np.percentile(x, hi_q))
    if not (hi > lo):
        hi = lo + 1.0
    return np.linspace(lo, hi, int(nbins) + 1)

def _score_chunk_stats(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return dict(mean=np.nan, p05=np.nan, p50=np.nan, p95=np.nan)
    return dict(
        mean=float(np.mean(x)),
        p05=float(np.percentile(x, 5)),
        p50=float(np.percentile(x, 50)),
        p95=float(np.percentile(x, 95)),
    )

LEGEND_FONTSIZE = 13
LEGEND_TITLE_FONTSIZE = 13

def small_legend(ax, *, title=None, loc="best", ncol=1, **kwargs):
    """
    Consistent compact legend across all plots.
    """
    if "fontsize" not in kwargs:
        kwargs["fontsize"] = LEGEND_FONTSIZE
    if title and ("title_fontsize" not in kwargs):
        kwargs["title_fontsize"] = LEGEND_TITLE_FONTSIZE
    return ax.legend(
        loc=loc,
        frameon=True,
        title=title,
        ncol=ncol,
        handlelength=1.6,
        handletextpad=0.4,
        labelspacing=0.25,
        borderpad=0.30,
        columnspacing=0.8,
        markerscale=0.9,
        **kwargs,
    )

def plot_inband_eff_grouped_by_trigger(eff_ad, eff_ht, *, signal_key, signal_label,
                                       outpath, run_label,
                                       trigger_order=("HT", "AD"), control = "MC", PLOT_METHODS=["Constant", "PID", "DQN", "DQN-F", "PPO", "ADT", "GRPO", "GFPO-F", "GFPO-FR"]
):
    """
    Grouped bars like the CMS figure:
      x-axis: triggers (AD Trigger, HT Trigger)
      bars within each group: methods (Constant, PID, ADT, DQN, GRPO, GFPO-F, GFPO-FR)

    eff_ad/eff_ht: dict method -> {"tt": val, "h_to_4b": val}
    signal_key: "tt" or "h_to_4b"
    """
    # which methods exist in either trigger

    methods = [m for m in PLOT_METHODS if (m in eff_ad) or (m in eff_ht)]
    if not methods:
        return

    # trigger groups
    trig_map = {"AD": eff_ad, "HT": eff_ht}
    triggers = [t for t in trigger_order if t in trig_map]
    if not triggers:
        return

    # values: shape (T, M)
    vals = np.zeros((len(triggers), len(methods)), dtype=np.float64)
    for ti, tr in enumerate(triggers):
        eff = trig_map[tr]
        for mi, m in enumerate(methods):
            vals[ti, mi] = float(eff.get(m, {}).get(signal_key, np.nan))

    x = np.arange(len(triggers), dtype=np.float64)
    bw = 0.80 / max(1, len(methods))  # fill 80% of group width

    fig, ax = plt.subplots(figsize=(10, 5.6))

    # bars (one legend entry per method)
    for mi, m in enumerate(methods):
        ax.bar(
            x - 0.40 + (mi + 0.5) * bw,
            vals[:, mi],
            width=bw,
            label=m,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{tr} Trigger" for tr in triggers])
    ax.set_ylabel(f"In-band efficiency ({signal_label})")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    # start y-axis at 80 for ttbar
    if signal_key == "tt":
        if control == "MC":
            ax.set_ylim(bottom=85)          # keep top auto
        # or: ax.set_ylim(80, 100)       # if want fixed top
        else:
            ax.set_ylim(bottom=70)          # keep top auto
    else:
        if control == "MC":
            ax.set_ylim(bottom=15)           # keep top auto
        else:
            ax.set_ylim(bottom=25)           # keep top auto
    # legend is methods 
    small_legend(ax, loc="best", ncol=1)

    add_cms_header(fig, run_label=run_label)
    finalize_diag_fig(fig)
    save_png(fig, str(outpath))
    plt.close(fig)


def plot_entropy_timeseries(stats_by_label, *, title, outpath, run_label):
    fig, ax = plt.subplots(figsize=(9, 5.4))
    for label, st in stats_by_label.items():
        if st is None:
            continue
        ax.plot(st["micros"], st["entropy"], linewidth=2.2, drawstyle="steps-post", label=label)
    ax.set_xlabel("Micro-step")
    ax.set_ylabel("Normalized action entropy (empirical)")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle="--", alpha=0.5)
    small_legend(ax, loc="best", title=title)
    add_cms_header(fig, run_label=run_label)
    finalize_diag_fig(fig)
    save_png(fig, str(outpath))
    plt.close(fig)


def plot_early_abs_err_hist(grpo_ae, gfpo_ae, *, title, outpath, run_label):
    grpo_ae = np.asarray(grpo_ae, dtype=np.float64); grpo_ae = grpo_ae[np.isfinite(grpo_ae)]
    gfpo_ae = np.asarray(gfpo_ae, dtype=np.float64); gfpo_ae = gfpo_ae[np.isfinite(gfpo_ae)]
    if grpo_ae.size == 0 and gfpo_ae.size == 0:
        return

    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    if grpo_ae.size:
        ax.hist(grpo_ae, bins=60, density=True, alpha=0.55, label="GRPO candidates")
    if gfpo_ae.size:
        ax.hist(gfpo_ae, bins=60, density=True, alpha=0.55, label="GFPO kept candidates")

    ax.set_xlabel(r"$|bg-target|$  (percent units)")
    ax.set_ylabel("Density")
    ax.grid(True, linestyle="--", alpha=0.4)
    small_legend(ax, loc="best", title=title)
    add_cms_header(fig, run_label=run_label)
    finalize_diag_fig(fig)
    save_png(fig, str(outpath))
    plt.close(fig)


def ecdf(x):
    """Creating error cdf"""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.array([]), np.array([])
    x = np.sort(x)
    y = (np.arange(1, x.size + 1) / x.size)
    return x, y


def select_plot_methods(d, PLOT_METHODS = ["Constant", "PID", "DQN", "DQN-F", "PPO", "ADT", "GRPO", "GFPO-F", "GFPO-FR"]):
    """
    Filter + order a dict keyed by method name.
    Keeps ONLY methods in PLOT_METHODS, in that order.
    """
    if not d:
        return {}
    return {m: d[m] for m in PLOT_METHODS if m in d}

def plot_cdf_abs_err_multi(rate_khz_by_method, target_khz, tol_khz, title, outpath, run_label):
    """
    rate_khz_by_method: dict(name -> 1D array of rates in kHz)
    """
    rate_khz_by_method = select_plot_methods(rate_khz_by_method)
    fig, ax = plt.subplots(figsize=(8, 5.2))
    for name, r_khz in rate_khz_by_method.items():
        e = np.abs(np.asarray(r_khz, dtype=np.float64) - float(target_khz))
        x, y = ecdf(e)
        if x.size:
            ax.plot(x, y, linewidth=2.2, label=name)

    ax.axvline(float(tol_khz), linestyle="--", linewidth=1.6, label=f"Tolerance = {tol_khz:.1f} kHz")
    ax.set_xlabel(r"$|r-r^*|$ [kHz]")
    ax.set_ylabel("CDF")
    ax.grid(True, linestyle="--", alpha=0.5)
    small_legend(ax, loc="best", title=title)
    add_cms_header(fig, run_label=run_label)
    finalize_diag_fig(fig)
    save_png(fig, str(outpath))
    plt.close(fig)


def running_mean_bool(mask, w=3):
    m = np.asarray(mask, dtype=np.float64)
    k = np.ones(int(w), dtype=np.float64)
    return np.convolve(m, k, mode="same") / np.convolve(np.ones_like(m), k, mode="same")


def plot_running_inband_multi(time, inband_by_method, w, title, outpath, run_label):
    """
    inband_by_method: dict(name -> boolean mask per chunk)
    """
    inband_by_method = select_plot_methods(inband_by_method)
    fig, ax = plt.subplots(figsize=(8, 5.2))
    style = {
        "Constant": dict(linestyle="--", linewidth=2.2),
        "PID":      dict(linestyle="-",  linewidth=2.2),
        "DQN":      dict(linestyle=(0, (8, 2, 2, 2)), linewidth=2.6, marker="o", markersize=3, markevery=8),
        "ADT": dict(linestyle=(0, (6, 2)), linewidth=2.6),
        "PPO": dict(linestyle=(0, (3, 2, 1, 2)), linewidth=2.6),
        "GRPO":     dict(linestyle=(0, (10, 2, 2, 2)), linewidth=2.8),
        "GFPO-F":   dict(linestyle=(0, (4, 2)), linewidth=2.6),
        "GFPO-FR":  dict(linestyle=(0, (2, 2)), linewidth=2.6),
    }
    for name, m in inband_by_method.items():
        y = running_mean_bool(m, w=int(w))
        t = np.linspace(0.0, 1.0, len(y))
        ax.plot(t, y, label=f"{name} (w={int(w)})", **style.get(name, {}))


    ax.set_xlabel("Time (Fraction of Run)")
    ax.set_ylabel("Running in-band fraction")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.5)
    small_legend(ax, loc="best", title=title)
    add_cms_header(fig, run_label=run_label)
    finalize_diag_fig(fig)
    save_png(fig, str(outpath))
    plt.close(fig)


def plot_cut_step_hist_multi(
    cut_by_method,
    xlabel,
    title,
    outpath,
    run_label,
    bins=30,  # kept for backward compat; ignored in raw mode
    allow_constant_zeros=True,
    raw=True,                 # <-- NEW: default to raw delta plot
    use_abs=False,            # <-- NEW: if True, plot |Δcut| raw values
    max_points=8000,          # <-- NEW: cap points per method (subsample)
):
    """
    If raw=True: plot per-step raw deltas (no binning) as a scatter over time.
    If raw=False: fall back to the old histogram behavior (binned).
    """
    cut_by_method = select_plot_methods(cut_by_method)

    fig, ax = plt.subplots(figsize=(8, 5.2))
    any_plotted = False

    for name, c in cut_by_method.items():
        c = np.asarray(c, dtype=np.float64)

        dc = np.diff(c) if c.size >= 2 else np.array([], dtype=np.float64)

        if dc.size == 0 and allow_constant_zeros:
            # constant / degenerate history -> show "no motion"
            dc = np.zeros(max(1, c.size - 1), dtype=np.float64)

        if dc.size == 0:
            continue

        y = np.abs(dc) if use_abs else dc

        if raw:
            # subsample for readability / speed
            n = y.size
            stride = max(1, int(np.ceil(n / max(1, int(max_points)))))
            y_s = y[::stride]
            t_s = np.linspace(0.0, 1.0, y_s.size)  # normalized time axis

            ax.plot(
                t_s,
                y_s,
                linestyle="None",
                marker=".",
                markersize=3.0,
                alpha=0.55,
                label=name,
            )
        else:
            # old behavior: binned histogram
            ax.hist(y, bins=int(bins), alpha=0.50, label=name)

        any_plotted = True

    if not any_plotted:
        ax.text(0.5, 0.5, "No cut history to plot", ha="center", va="center",
                transform=ax.transAxes)

    ax.axhline(0.0, linestyle="--", linewidth=1.2, alpha=0.6)
    ax.set_xlabel("Time (Fraction of Run)" if raw else xlabel)
    ax.set_ylabel(r"$\Delta \mathrm{cut}$" if raw and not use_abs else (r"$|\Delta \mathrm{cut}|$" if raw else "Count"))
    ax.grid(True, linestyle="--", alpha=0.4)
    small_legend(ax, loc="best", title=title)

    add_cms_header(fig, run_label=run_label)
    finalize_diag_fig(fig)
    save_png(fig, str(outpath))
    plt.close(fig)

def plot_rate_from_series(series_by_method, *, target, tol, title, outpath, run_label, RATE_SCALE_KHZ = 400.0):
    if not series_by_method:
        return

    # choose a reference length for x-axis
    max_len = max(len(v["bg_khz"]) for v in series_by_method.values())
    time_ref = np.linspace(0.0, 1.0, max_len)

    target_khz = float(target) * RATE_SCALE_KHZ
    tol_khz    = float(tol)    * RATE_SCALE_KHZ

    fig, ax = plt.subplots(figsize=(10, 6))
    for method, s in select_plot_methods(series_by_method).items():
        y = s["bg_khz"]
        t = np.linspace(0.0, 1.0, len(y))
        ax.plot(t, y, linewidth=2.4, label=method)

    ax.axhline(target_khz + tol_khz, linestyle="--", linewidth=1.2)
    ax.axhline(target_khz - tol_khz, linestyle="--", linewidth=1.2)
    ax.fill_between(time_ref, target_khz - tol_khz, target_khz + tol_khz, alpha=0.12, label="Tolerance band")

    ax.set_xlabel("Time (Fraction of Run)")
    ax.set_ylabel("Background rate [kHz]")
    ax.set_ylim(0, 200)
    ax.grid(True, linestyle="--", alpha=0.5)
    small_legend(ax, loc="best", title=title)
    add_cms_header(fig, run_label=run_label)
    finalize_diag_fig(fig)
    save_png(fig, str(outpath))
    plt.close(fig)


def plot_cut_from_series(series_by_method, *, fixed_cut, ylabel, title, outpath, run_label):
    if not series_by_method:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    for method, s in select_plot_methods(series_by_method).items():
        y = s["cut"]
        t = np.linspace(0.0, 1.0, len(y))
        ax.plot(t, y, linewidth=2.4, label=method)

    ax.axhline(float(fixed_cut), color="gray", linestyle="--", linewidth=1.5, label="fixed")
    ax.set_xlabel("Time (Fraction of Run)")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.5)
    small_legend(ax, loc="best", title=title)
    add_cms_header(fig, run_label=run_label)
    finalize_diag_fig(fig)
    save_png(fig, str(outpath))
    plt.close(fig)


def make_original_plots_for_trigger(series, *, trigger_name, fixed_cut, target, tol, plots_dir, run_label, w=3, RATE_SCALE_KHZ = 400.0, PLOT_METHODS = ["Constant", "PID", "DQN", "DQN-F", "PPO", "ADT", "GRPO", "GFPO-F", "GFPO-FR"]):
    if not series:
        return
    # Keep ONLY: PID, DQN, GRPO, GFPO-F, GFPO-FR (in this order)
    series = select_plot_methods(series, PLOT_METHODS=PLOT_METHODS)
    if not series:
        return
    # 1) CDF of |rate error| (kHz)
    target_khz = float(target) * RATE_SCALE_KHZ
    tol_khz    = float(tol)    * RATE_SCALE_KHZ
    rate_khz_by_method = {m: s["bg_khz"] for m, s in series.items()}
    plot_cdf_abs_err_multi(
        rate_khz_by_method=rate_khz_by_method,
        target_khz=target_khz, tol_khz=tol_khz,
        title=f"{trigger_name} Trigger",
        outpath=plots_dir / f"cdf_abs_err_{trigger_name.lower()}",
        run_label=run_label,
    )

    # 2) Running in-band fraction vs time
    inband_by_method = {m: s["inband"] for m, s in series.items()}
    max_len = max(len(s["inband"]) for s in series.values())
    time_ref = np.linspace(0.0, 1.0, max_len)
    plot_running_inband_multi(
        time=time_ref,
        inband_by_method=inband_by_method,
        w=int(w),
        title=f"{trigger_name} Trigger",
        outpath=plots_dir / f"running_inband_{trigger_name.lower()}",
        run_label=run_label,
    )

    # 3) Cut-step magnitude histogram |Δcut|
    cut_by_method = {m: s["cut"] for m, s in series.items()}
    plot_cut_step_hist_multi(
        cut_by_method=cut_by_method,
        xlabel=r"$|\Delta \mathrm{cut}|$",
        title=f"{trigger_name} Trigger",
        outpath=plots_dir / f"cut_step_hist_{trigger_name.lower()}",
        run_label=run_label,
        raw=True,
        use_abs=False,
        max_points=8000,
    )

    # 5) Rate + cut time-series (“core plots”)
    plot_rate_from_series(
        series,
        target=target, tol=tol,
        title=f"{trigger_name} Trigger",
        outpath=plots_dir / f"rate_{trigger_name.lower()}",
        run_label=run_label,
        RATE_SCALE_KHZ=RATE_SCALE_KHZ
    )
    plot_cut_from_series(
        series,
        fixed_cut=fixed_cut,
        ylabel=f"{trigger_name}_cut",
        title=f"{trigger_name} Cut",
        outpath=plots_dir / f"cut_{trigger_name.lower()}",
        run_label=run_label,
    )


def _sig_score(tt, aa, mix):
    return float(mix) * float(tt) + (1.0 - float(mix)) * float(aa)

def collect_kept_candidate_arrays(samples, *, trigger, method, target, tol, mix):
    """
    Collect arrays over *kept candidates* for a given (trigger, method).
    Returns dict with abs_err, sig_score, feasible (bool).
    """
    abs_err = []
    sig = []
    feas = []
    for r in samples:
        if r.get("trigger") != trigger: 
            continue
        if r.get("method") != method:
            continue
        if r.get("phase") != "candidate":
            continue
        if int(r.get("kept", 0)) != 1:
            continue

        bg = float(r.get("bg_after", np.nan))
        tt = float(r.get("tt_after", np.nan))
        aa = float(r.get("aa_after", np.nan))
        if not np.isfinite(bg) or not np.isfinite(tt) or not np.isfinite(aa):
            continue

        ae = abs(bg - float(target))
        abs_err.append(ae)
        sig.append(_sig_score(tt, aa, mix))
        feas.append(ae <= float(tol))

    return {
        "abs_err": np.asarray(abs_err, dtype=np.float64),
        "sig": np.asarray(sig, dtype=np.float64),
        "feas": np.asarray(feas, dtype=bool),
    }


def _plot_score_density_heatmap(time, hists, edges, *, title, outpath, run_label):
    """
    hists: shape (T, nbins) where nbins = len(edges)-1, density per chunk
    edges: bin edges (len = nbins+1)
    """
    H = np.asarray(hists, dtype=np.float64)
    if H.size == 0:
        return

    # transpose so y-axis is score
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    im = ax.imshow(
        H.T,
        origin="lower",
        aspect="auto",
        extent=[float(time[0]), float(time[-1]), float(edges[0]), float(edges[-1])],
        interpolation="nearest",
    )
    ax.set_xlabel("Time (Fraction of Run)")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="Density")
    add_cms_header(fig, run_label=run_label)
    finalize_diag_fig(fig)
    save_png(fig, str(outpath))
    plt.close(fig)


def _plot_score_summary(time, stats_list, *, title, outpath, run_label):
    """
    stats_list: list of dicts with keys mean, p05, p50, p95 (one per chunk)
    """
    if not stats_list:
        return
    mean = np.array([s["mean"] for s in stats_list], dtype=np.float64)
    p05  = np.array([s["p05"]  for s in stats_list], dtype=np.float64)
    p50  = np.array([s["p50"]  for s in stats_list], dtype=np.float64)
    p95  = np.array([s["p95"]  for s in stats_list], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.plot(time, mean, linewidth=2.2, label="Mean")
    ax.plot(time, p50,  linewidth=2.2, linestyle="--", label="Median (p50)")
    ax.fill_between(time, p05, p95, alpha=0.15, label="p05–p95 band")
    ax.set_xlabel("Time (Fraction of Run)")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.5)
    small_legend(ax, loc="best")
    add_cms_header(fig, run_label=run_label)
    finalize_diag_fig(fig)
    save_png(fig, str(outpath))
    plt.close(fig)


def _plot_adv_compare_ecdf(x_grpo, x_gfpo, *, title, outpath, run_label):
    x_grpo = np.asarray(x_grpo, dtype=np.float64); x_grpo = x_grpo[np.isfinite(x_grpo)]
    x_gfpo = np.asarray(x_gfpo, dtype=np.float64); x_gfpo = x_gfpo[np.isfinite(x_gfpo)]
    if x_grpo.size == 0 and x_gfpo.size == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 5.2))
    if x_grpo.size:
        xs, ys = ecdf(x_grpo)
        ax.plot(xs, ys, linewidth=2.2, label="GRPO (candidates)")
    if x_gfpo.size:
        xs, ys = ecdf(x_gfpo)
        ax.plot(xs, ys, linewidth=2.2, linestyle=(0, (4, 2)), label="GFPO (kept candidates)")

    ax.set_xlabel(r"Normalized advantage  $\hat A$")
    ax.set_ylabel("CDF")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_title(title)
    small_legend(ax, loc="best")
    add_cms_header(fig, run_label=run_label)
    finalize_diag_fig(fig)
    save_png(fig, str(outpath))
    plt.close(fig)


def _plot_adv_hist_and_ecdf(values, *, title, xlabel, outpath_prefix, run_label):
    """
    Saves:
      - {outpath_prefix}_hist.png
      - {outpath_prefix}_ecdf.png
    """
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return

    # Histogram
    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.hist(x, bins=60, density=True, alpha=0.75)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_title(title)
    add_cms_header(fig, run_label=run_label)
    finalize_diag_fig(fig)
    save_png(fig, str(outpath_prefix) + "_hist")
    plt.close(fig)

    # ECDF
    xs, ys = ecdf(x)
    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.plot(xs, ys, linewidth=2.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("CDF")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_title(title)
    add_cms_header(fig, run_label=run_label)
    finalize_diag_fig(fig)
    save_png(fig, str(outpath_prefix) + "_ecdf")
    plt.close(fig)




def _plot_two_hists(x1, x2, *, label1, label2, title, xlabel, outpath, run_label):
    x1 = np.asarray(x1, dtype=np.float64); x1 = x1[np.isfinite(x1)]
    x2 = np.asarray(x2, dtype=np.float64); x2 = x2[np.isfinite(x2)]
    if x1.size == 0 and x2.size == 0:
        return
    fig, ax = plt.subplots(figsize=(8, 5.2))
    if x1.size:
        ax.hist(x1, bins=60, density=True, alpha=0.55, label=label1)
    if x2.size:
        ax.hist(x2, bins=60, density=True, alpha=0.55, label=label2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.grid(True, linestyle="--", alpha=0.4)
    # ax.set_title(title)
    small_legend(ax, loc="best")
    add_cms_header(fig, run_label=run_label)
    finalize_diag_fig(fig)
    save_png(fig, str(outpath))
    plt.close(fig)


def plot_feasibility_bar(stats_grpo, stats_gfpo, *, title, outpath, run_label):
    #stats_grpo is gfpo-f
    #stats_gfpo is gfpo-fr
    metrics = ["cand_feas", "kept_feas", "pad_rate", "shield_rate"]
    labels  = ["Feasible ratio", "Kept-feasible ratio", "Pad rate", "Shield rate"]

    def getvals(st, is_gfpo):
        if st is None:
            return [np.nan]*4
        return [
            float(st["feasible_ratio_mean"]),
            float(st["kept_feasible_ratio_mean"]),
            float(st["pad_rate"]),
            float(st["shield_rate"]),
        ]

    vals_grpo = getvals(stats_grpo, is_gfpo=False)
    vals_gfpo = getvals(stats_gfpo, is_gfpo=True)

    x = np.arange(len(labels))
    bw = 0.38

    fig, ax = plt.subplots(figsize=(9, 5.4))
    ax.bar(x - bw/2, vals_grpo, width=bw, label="GFPO-F")
    ax.bar(x + bw/2, vals_gfpo, width=bw, label="GFPO-FR")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Fraction")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    small_legend(ax, loc="best", title=title)
    add_cms_header(fig, run_label=run_label)
    finalize_diag_fig(fig)
    save_png(fig, str(outpath))
    plt.close(fig)


def plot_feasible_ratio_timeseries(stats_grpo, stats_gfpo, *, title, outpath, run_label):
    fig, ax = plt.subplots(figsize=(9, 5.4))

    if stats_grpo is not None:
        ax.plot(
            stats_grpo["micros"], stats_grpo["feasible_ratio"],
            linewidth=2.2,
            linestyle="-",
            marker=None,          
            drawstyle="steps-post",
            label="GRPO (candidates)",
        )
    if stats_gfpo is not None:
        ax.plot(
            stats_gfpo["micros"], stats_gfpo["feasible_ratio"],
            linewidth=2.2,
            linestyle=(0, (4, 2)),
            marker=None,           
            drawstyle="steps-post",
            label="GFPO (candidates)",
        )

    ax.set_xlabel("Micro-step")
    ax.set_ylabel(r"Feasible ratio  (#cand with |bg-target|<=tol) / #cand")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle="--", alpha=0.5)
    small_legend(ax, loc="best", title=title)
    add_cms_header(fig, run_label=run_label)
    finalize_diag_fig(fig)
    save_png(fig, str(outpath))
    plt.close(fig)



def plot_inband_eff_bars_multi(summary_by_method, title, outpath, run_label):
    """
    summary_by_method: dict(name -> summarize_compact(...) dict)
    """
    summary_by_method = select_plot_methods(summary_by_method)
    if not summary_by_method:
        return
    labels = [r"$t\bar{t}$", r"$h\rightarrow 4b$"]
    keys   = ["tt", "h_to_4b"]
    methods = list(summary_by_method.keys())

    vals = np.array([[summary_by_method[m][k] for k in keys] for m in methods], dtype=np.float64)  # (M,3)

    x = np.arange(len(labels))
    bw = 0.80 / max(1, len(methods))  # fill 80% of tick width

    fig, ax = plt.subplots(figsize=(8, 5.2))
    for i, m in enumerate(methods):
        ax.bar(x - 0.40 + (i + 0.5) * bw, vals[i], width=bw, label=m)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean signal efficiency (in-band)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    small_legend(ax, loc="best", title=title)
    add_cms_header(fig, run_label=run_label)
    finalize_diag_fig(fig)
    save_png(fig, str(outpath))
    plt.close(fig)


def _plot_exec_tradeoff(exec_f, exec_fr, *, title, outpath, run_label):
    """
    Scatter: abs_err (x) vs sig_score (y), executed-only. Feasible points are filled, infeasible hollow.
    """
    def scatter_one(ax, d, label):
        x = d["abs_err"]; y = d["sig"]; feas = d["feas"]
        if x.size == 0:
            return
        ax.scatter(x[~feas], y[~feas], s=18, alpha=0.55, facecolors="none", label=f"{label} (infeas)")
        ax.scatter(x[feas],  y[feas],  s=18, alpha=0.55, label=f"{label} (feas)")

    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    scatter_one(ax, exec_f,  "GFPO-F")
    scatter_one(ax, exec_fr, "GFPO-FR")
    ax.set_xlabel(r"$|bg-target|$  (percent units)")
    ax.set_ylabel(r"Signal score  $mix\cdot t\bar t + (1-mix)\cdot h\to4b$")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_title(title)
    small_legend(ax, loc="best", ncol=1)
    add_cms_header(fig, run_label=run_label)
    finalize_diag_fig(fig)
    save_png(fig, str(outpath))
    plt.close(fig)
