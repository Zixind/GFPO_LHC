"""
anomaly_detection/preprocess_unsw.py

Download instructions and preprocessing for UNSW-NB15.

Download the two pre-split CSV files from:
  https://research.unsw.edu.au/projects/unsw-nb15-dataset
  → UNSW_NB15_training-set.csv   (175,341 records)
  → UNSW_NB15_testing-set.csv    (82,332  records)

Place them in:   anomaly_detection/data/

Then run:
  python anomaly_detection/preprocess_unsw.py

Outputs:
  anomaly_detection/data/unsw_train.npz   — normal-only (for detector training)
  anomaly_detection/data/unsw_stream.npz  — full test stream (chunks for RL)
"""

import os, sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import joblib

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# 49 feature columns (drop 'id', 'attack_cat', keep 'label')
CATEGORICAL = ["proto", "service", "state"]
DROP_COLS   = ["id", "attack_cat"]
LABEL_COL   = "label"

CHUNK_SIZE  = 1000   # records per streaming chunk


def load_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip().str.lower()
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    for col in CATEGORICAL:
        if col in df.columns:
            df[col] = pd.Categorical(df[col]).codes.astype(np.float32)
    return df


# Attack-category encoding for TPR_easy / TPR_hard split
# Mirrors the LHC  tt̄ (easy) / h→4b (hard)  signal split.
#   0 = normal
#   1 = easy attacks  (Generic — highest AE score separation from normal)
#   2 = hard attacks  (Backdoor — score distribution overlaps with normal)
#   3 = other attacks
EASY_CATS = {"generic"}                 # well-separated from normal (analogue: tt̄)
HARD_CATS = {"backdoor", "exploits"}    # overlaps with normal  (analogue: h→4b)

def encode_cat(cat_series: pd.Series) -> np.ndarray:
    out = np.full(len(cat_series), 3, dtype=np.int8)  # default: other attack
    cats_lower = cat_series.str.lower().str.strip()
    out[cats_lower == "normal"] = 0
    for c in EASY_CATS:
        out[cats_lower == c] = 1
    for c in HARD_CATS:
        out[cats_lower == c] = 2
    return out


def preprocess(train_csv: Path, test_csv: Path, chunk_size: int = CHUNK_SIZE):
    print(f"Loading {train_csv} ...")
    train_df = load_raw(train_csv)
    print(f"Loading {test_csv} ...")
    test_df  = load_raw(test_csv)

    # Encode attack categories BEFORE dropping the column
    cat_train = encode_cat(train_df.get("attack_cat", pd.Series(["normal"]*len(train_df))))
    cat_test  = encode_cat(test_df.get("attack_cat",  pd.Series(["normal"]*len(test_df))))

    for df in [train_df, test_df]:
        for c in DROP_COLS:
            if c in df.columns:
                df.drop(columns=c, inplace=True)
        encode_categoricals(df)

    # feature columns = everything except label
    feat_cols = [c for c in train_df.columns if c != LABEL_COL]

    X_train = train_df[feat_cols].values.astype(np.float32)
    y_train = train_df[LABEL_COL].values.astype(np.int32)

    X_test  = test_df[feat_cols].values.astype(np.float32)
    y_test  = test_df[LABEL_COL].values.astype(np.int32)

    # fill NaN with 0
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_test  = np.nan_to_num(X_test,  nan=0.0)

    # fit scaler on normal training traffic only
    normal_mask = (y_train == 0)
    scaler = StandardScaler()
    scaler.fit(X_train[normal_mask])
    X_train = scaler.transform(X_train)
    X_test  = scaler.transform(X_test)

    scaler_path = DATA_DIR / "scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved → {scaler_path}")

    # save training split normal-only (for base detector training)
    np.savez_compressed(
        DATA_DIR / "unsw_train.npz",
        X=X_train[normal_mask],
        y=y_train[normal_mask],
        feat_names=np.array(feat_cols),
    )
    print(f"Train (normal only): {normal_mask.sum()} records → unsw_train.npz")

    # chunk the full training set (for Exp1 RL training on "MC equivalent")
    n_tr = len(X_train)
    n_chunks_tr = n_tr // chunk_size
    X_tr_chunks   = X_train[: n_chunks_tr * chunk_size].reshape(n_chunks_tr, chunk_size, -1)
    y_tr_chunks   = y_train[: n_chunks_tr * chunk_size].reshape(n_chunks_tr, chunk_size)
    cat_tr_chunks = cat_train[: n_chunks_tr * chunk_size].reshape(n_chunks_tr, chunk_size)
    np.savez_compressed(
        DATA_DIR / "unsw_train_stream.npz",
        X=X_tr_chunks,
        y=y_tr_chunks,
        cat=cat_tr_chunks,
        feat_names=np.array(feat_cols),
    )
    print(f"Train stream: {n_chunks_tr} chunks × {chunk_size} records → unsw_train_stream.npz")

    # chunk the test stream preserving original order (for Exp2/3 rollout on "CMS equivalent")
    n = len(X_test)
    n_chunks = n // chunk_size
    X_chunks   = X_test[: n_chunks * chunk_size].reshape(n_chunks, chunk_size, -1)
    y_chunks   = y_test[: n_chunks * chunk_size].reshape(n_chunks, chunk_size)
    cat_chunks = cat_test[: n_chunks * chunk_size].reshape(n_chunks, chunk_size)

    np.savez_compressed(
        DATA_DIR / "unsw_stream.npz",
        X=X_chunks,
        y=y_chunks,
        cat=cat_chunks,
        feat_names=np.array(feat_cols),
    )
    print(f"Test stream: {n_chunks} chunks × {chunk_size} records → unsw_stream.npz")


if __name__ == "__main__":
    train_csv = DATA_DIR / "UNSW_NB15_training-set.csv"
    test_csv  = DATA_DIR / "UNSW_NB15_testing-set.csv"

    if not train_csv.exists() or not test_csv.exists():
        print("ERROR: Download the UNSW-NB15 train/test CSVs and place them in:")
        print(f"  {DATA_DIR}/UNSW_NB15_training-set.csv")
        print(f"  {DATA_DIR}/UNSW_NB15_testing-set.csv")
        print()
        print("Download from: https://research.unsw.edu.au/projects/unsw-nb15-dataset")
        sys.exit(1)

    preprocess(train_csv, test_csv)
    print("Done.")
