"""
anomaly_detection/preprocess_nab.py

Download NAB (Numenta Anomaly Benchmark) data, compute sliding-window z-score
anomaly scores, and package into train/test npz arrays for the RL pipeline.

Usage:
    python anomaly_detection/preprocess_nab.py [--nab-dir anomaly_detection/data/nab_raw]

Output:
    anomaly_detection/data/nab_train.npz  — training chunks (first 70% of each file)
    anomaly_detection/data/nab_test.npz   — test chunks (last 30% of each file)
    anomaly_detection/data/nab_windows.json — anomaly windows (timestep indices per test chunk)
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Configuration ─────────────────────────────────────────────────────────────

SUBSETS = ["realKnownCause", "realAWSCloudwatch"]
CHUNK_SIZE = 100
WINDOW_SIZE = 100
TRAIN_FRAC = 0.70


def clone_nab(nab_dir: Path):
    """Clone NAB repo if not already present."""
    if nab_dir.exists():
        print(f"NAB directory already exists at {nab_dir}, skipping clone.")
        return
    nab_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"Cloning NAB into {nab_dir} …")
    ret = subprocess.run(
        ["git", "clone", "--depth", "1",
         "https://github.com/numenta/NAB", str(nab_dir)],
        check=False,
    )
    if ret.returncode != 0:
        raise RuntimeError("git clone failed — check network or install git.")
    print("Clone complete.")


def robust_zscore(values: np.ndarray, window_size: int = 100) -> np.ndarray:
    """
    Sliding-window robust z-score:
        score_t = |x_t - median(window_t)| / (MAD(window_t) + 1e-6)
    """
    n = len(values)
    scores = np.zeros(n, dtype=np.float32)
    for i in range(n):
        lo = max(0, i - window_size + 1)
        window = values[lo : i + 1]
        med = np.median(window)
        mad = np.median(np.abs(window - med))
        scores[i] = abs(values[i] - med) / (mad + 1e-6)
    return scores


def normalize_scores_to_unit(scores_train: np.ndarray, scores_test: np.ndarray):
    """
    Fit CDF on training scores; map both train and test to [0,1] via percentile rank.
    """
    sorted_train = np.sort(scores_train.ravel())
    n = len(sorted_train)

    def _cdf_map(x):
        idx = np.searchsorted(sorted_train, x, side="right")
        return (idx / n).astype(np.float32)

    return _cdf_map(scores_train), _cdf_map(scores_test)


def parse_windows_for_file(
    windows_json: dict, file_key: str, timestamps: pd.Series
) -> list:
    """
    Convert ISO timestamp window strings to integer index ranges.
    Returns list of (start_idx, end_idx) inclusive.
    """
    raw = windows_json.get(file_key, [])
    if not raw:
        return []
    ts_arr = pd.to_datetime(timestamps.values)
    result = []
    for w_start_str, w_end_str in raw:
        w_start = pd.Timestamp(w_start_str)
        w_end = pd.Timestamp(w_end_str)
        # find first ts >= w_start and last ts <= w_end
        start_idx = int(np.searchsorted(ts_arr, w_start, side="left"))
        end_idx   = int(np.searchsorted(ts_arr, w_end,   side="right")) - 1
        start_idx = max(0, min(start_idx, len(ts_arr) - 1))
        end_idx   = max(0, min(end_idx,   len(ts_arr) - 1))
        if start_idx <= end_idx:
            result.append((start_idx, end_idx))
    return result


def chunk_array(arr: np.ndarray, chunk_size: int) -> np.ndarray:
    """Truncate arr to a multiple of chunk_size, reshape to (N, chunk_size)."""
    n = (len(arr) // chunk_size) * chunk_size
    return arr[:n].reshape(-1, chunk_size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nab-dir", default="anomaly_detection/data/nab_raw")
    args = ap.parse_args()

    nab_dir = Path(args.nab_dir).resolve()
    out_dir = nab_dir.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    clone_nab(nab_dir)

    # Load NAB anomaly windows
    windows_path = nab_dir / "labels" / "combined_windows.json"
    if not windows_path.exists():
        raise FileNotFoundError(f"Cannot find {windows_path}. NAB clone may be incomplete.")
    with open(windows_path) as f:
        windows_json = json.load(f)

    train_scores_all, train_labels_all, train_file_ids = [], [], []
    test_scores_all,  test_labels_all,  test_file_ids  = [], [], []
    test_offsets = []
    nab_windows_out = {}   # file_id (int) → list of (start, end) in global TEST timesteps

    file_id = 0
    subset_stats = {}

    for subset in SUBSETS:
        data_dir = nab_dir / "data" / subset
        if not data_dir.exists():
            print(f"WARNING: subset directory {data_dir} not found, skipping.")
            continue

        csv_files = sorted(data_dir.glob("*.csv"))
        if not csv_files:
            print(f"WARNING: no CSV files found in {data_dir}, skipping.")
            continue

        n_files_subset = 0
        n_train_chunks_subset = 0
        n_test_chunks_subset  = 0

        for csv_path in csv_files:
            rel_key = f"{subset}/{csv_path.name}"
            df = pd.read_csv(csv_path)
            if "value" not in df.columns:
                # some files use different column names
                val_cols = [c for c in df.columns if c.lower() not in ("timestamp",)]
                if not val_cols:
                    print(f"  Skipping {rel_key}: no value column found.")
                    continue
                df = df.rename(columns={val_cols[0]: "value"})
            if "timestamp" not in df.columns:
                ts_cols = [c for c in df.columns if "time" in c.lower()]
                if ts_cols:
                    df = df.rename(columns={ts_cols[0]: "timestamp"})

            values = df["value"].values.astype(np.float64)
            n_ts   = len(values)
            if n_ts < CHUNK_SIZE * 2:
                print(f"  Skipping {rel_key}: too short ({n_ts} timesteps).")
                continue

            # Compute anomaly scores
            scores_raw = robust_zscore(values, WINDOW_SIZE)

            # Parse anomaly windows → binary labels
            labels_full = np.zeros(n_ts, dtype=np.int32)
            win_list = parse_windows_for_file(windows_json, rel_key, df["timestamp"])
            for ws, we in win_list:
                labels_full[ws : we + 1] = 1

            # Train/test split on timesteps
            n_train_ts = int(n_ts * TRAIN_FRAC)
            scores_train_raw = scores_raw[:n_train_ts]
            scores_test_raw  = scores_raw[n_train_ts:]
            labels_train     = labels_full[:n_train_ts]
            labels_test      = labels_full[n_train_ts:]

            # CDF normalization fitted on training portion
            scores_train_norm, scores_test_norm = normalize_scores_to_unit(
                scores_train_raw, scores_test_raw
            )

            # Chunk training portion
            tr_scores = chunk_array(scores_train_norm, CHUNK_SIZE)
            tr_labels = chunk_array(labels_train,       CHUNK_SIZE)
            n_tr = len(tr_scores)

            # Chunk test portion
            te_scores = chunk_array(scores_test_norm, CHUNK_SIZE)
            te_labels = chunk_array(labels_test,       CHUNK_SIZE)
            n_te = len(te_scores)

            if n_tr == 0 and n_te == 0:
                print(f"  Skipping {rel_key}: no full chunks.")
                continue

            # Store training chunks
            for i in range(n_tr):
                train_scores_all.append(tr_scores[i])
                train_labels_all.append(tr_labels[i])
                train_file_ids.append(file_id)

            # Store test chunks + windows relative to test offset
            # anomaly window in test portion: offset by n_train_ts
            test_windows_for_file = []
            for ws, we in win_list:
                # Restrict to test portion
                ws2 = ws - n_train_ts
                we2 = we - n_train_ts
                if we2 < 0 or ws2 >= len(labels_test):
                    continue
                ws2 = max(0, ws2)
                we2 = min(len(labels_test) - 1, we2)
                test_windows_for_file.append((int(ws2), int(we2)))

            for i in range(n_te):
                global_offset = i * CHUNK_SIZE
                test_scores_all.append(te_scores[i])
                test_labels_all.append(te_labels[i])
                test_file_ids.append(file_id)
                test_offsets.append(global_offset)

            if test_windows_for_file:
                nab_windows_out[str(file_id)] = {
                    "file": rel_key,
                    "windows": test_windows_for_file,
                }

            n_files_subset  += 1
            n_train_chunks_subset += n_tr
            n_test_chunks_subset  += n_te
            anomaly_rate = labels_full.mean()
            print(f"  {rel_key}: {n_ts} ts, {n_tr} train chunks, "
                  f"{n_te} test chunks, anomaly_rate={anomaly_rate:.3f}, "
                  f"file_id={file_id}")
            file_id += 1

        subset_stats[subset] = dict(
            n_files=n_files_subset,
            n_train_chunks=n_train_chunks_subset,
            n_test_chunks=n_test_chunks_subset,
        )

    if not train_scores_all and not test_scores_all:
        raise RuntimeError("No data was processed. Check NAB directory structure.")

    # Save train npz
    if train_scores_all:
        np.savez_compressed(
            out_dir / "nab_train.npz",
            scores=np.stack(train_scores_all).astype(np.float32),
            labels=np.stack(train_labels_all).astype(np.int32),
            file_ids=np.array(train_file_ids, dtype=np.int32),
        )
        print(f"\nSaved nab_train.npz: {len(train_scores_all)} chunks")

    # Save test npz
    if test_scores_all:
        np.savez_compressed(
            out_dir / "nab_test.npz",
            scores=np.stack(test_scores_all).astype(np.float32),
            labels=np.stack(test_labels_all).astype(np.int32),
            file_ids=np.array(test_file_ids, dtype=np.int32),
            chunk_global_offset=np.array(test_offsets, dtype=np.int64),
        )
        print(f"Saved nab_test.npz:  {len(test_scores_all)} chunks")

    # Save windows JSON
    windows_out_path = out_dir / "nab_windows.json"
    with open(windows_out_path, "w") as f:
        json.dump(nab_windows_out, f, indent=2)
    print(f"Saved nab_windows.json: {len(nab_windows_out)} files with anomaly windows")

    # Summary
    print("\n=== Summary per subset ===")
    for subset, s in subset_stats.items():
        print(f"  {subset}: {s['n_files']} files, "
              f"{s['n_train_chunks']} train chunks, "
              f"{s['n_test_chunks']} test chunks")
    print(f"\nTotal file_ids: {file_id}")
    print("Preprocessing complete.")


if __name__ == "__main__":
    main()
