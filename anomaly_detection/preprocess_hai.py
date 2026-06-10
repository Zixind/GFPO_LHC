"""
anomaly_detection/preprocess_hai.py

Preprocess HAI 21.03 dataset and generate anomaly scores.

Train split  : hai-21.03/train1-3.csv  (normal operation)
Deploy split : hai-21.03/test1-5.csv   (contains labeled attacks)

Attack categories:
  cat=0  normal
  cat=1  Process attacks  (attack_P1=1, easy signal — large AE loss)
  cat=2  Measurement attacks (attack_P2=1, hard signal — subtle, overlaps normal)
  (P1 & P2 overlap → cat=2, hard takes precedence)

Context variable: P4_ST_PT01 (steam turbine pressure, index 77 in feature vector)

Outputs:
  data/hai_train.npz          — normal-only records for AE training
  data/hai_scores.npz         — scored deploy stream  (n_chunks, chunk_size)
  data/hai_scores_train.npz   — scored train stream   (n_chunks, chunk_size)
  data/hai_scaler.pkl         — fitted StandardScaler
  data/hai_detector.pkl       — trained Autoencoder

Usage:
    python anomaly_detection/preprocess_hai.py
"""

import glob, pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn

ROOT    = Path(__file__).resolve().parent
DATA    = ROOT / "data"
HAI_DIR = DATA / "hai_raw" / "hai-21.03"
CHUNK_SIZE = 5000

LABEL_COLS = ["attack", "attack_P1", "attack_P2", "attack_P3"]


# ── helpers ───────────────────────────────────────────────────────────────────
def load_csvs(pattern):
    files = sorted(glob.glob(str(HAI_DIR / pattern)))
    dfs = [pd.read_csv(f) for f in files]
    df  = pd.concat(dfs, ignore_index=True)
    print(f"  loaded {len(files)} files → {len(df):,} rows")
    return df


def make_cat(df):
    """Return int8 category array: 0=normal, 1=process(easy), 2=measurement(hard)."""
    cat = np.zeros(len(df), dtype=np.int8)
    cat[df["attack_P1"].values == 1] = 1
    cat[df["attack_P2"].values == 1] = 2   # hard takes precedence over easy
    return cat


def chunk_array(arr, chunk_size):
    n = (len(arr) // chunk_size) * chunk_size
    return arr[:n].reshape(-1, chunk_size, *arr.shape[1:])


# ── Autoencoder ───────────────────────────────────────────────────────────────
class Autoencoder(nn.Module):
    def __init__(self, in_dim, hidden=64, bottleneck=32):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, bottleneck), nn.ReLU(),
        )
        self.dec = nn.Sequential(
            nn.Linear(bottleneck, hidden), nn.ReLU(),
            nn.Linear(hidden, in_dim),
        )

    def forward(self, x):
        return self.dec(self.enc(x))

    def score(self, x):
        with torch.no_grad():
            recon = self.forward(x)
            return ((recon - x) ** 2).mean(dim=1)


def train_autoencoder(X_normal, epochs=10, batch=512, lr=1e-3):
    d     = X_normal.shape[1]
    model = Autoencoder(d)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    t     = torch.tensor(X_normal, dtype=torch.float32)
    ds    = torch.utils.data.TensorDataset(t)
    dl    = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True)
    for ep in range(1, epochs + 1):
        total = 0.0
        for (xb,) in dl:
            loss = nn.functional.mse_loss(model(xb), xb)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * len(xb)
        print(f"  AE epoch {ep}/{epochs}  loss={total/len(t):.5f}")
    return model


def score_stream(model, X, batch=2048):
    model.eval()
    scores = []
    t = torch.tensor(X, dtype=torch.float32)
    for i in range(0, len(t), batch):
        scores.append(model.score(t[i:i+batch]).numpy())
    raw = np.concatenate(scores)
    return np.log1p(raw).astype(np.float32)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    DATA.mkdir(parents=True, exist_ok=True)

    # ── Load train (normal only) ──────────────────────────────────────────────
    print("Loading HAI 21.03 train files ...")
    df_train = load_csvs("train*.csv")
    feat_cols = [c for c in df_train.columns if c not in LABEL_COLS and c != "time"]
    print(f"  features: {len(feat_cols)}")

    X_train_all = df_train[feat_cols].values.astype(np.float32)
    y_train_all = df_train["attack"].values.astype(np.int32)
    X_normal    = X_train_all[y_train_all == 0]
    print(f"  normal training records: {len(X_normal):,}")

    # ── Fit scaler on normal training data ───────────────────────────────────
    print("Fitting scaler ...")
    scaler = StandardScaler()
    scaler.fit(X_normal)
    X_normal_sc   = scaler.transform(X_normal).astype(np.float32)
    X_train_sc    = scaler.transform(X_train_all).astype(np.float32)

    # Save normal-only for reference
    np.savez_compressed(DATA / "hai_train.npz",
                        X=X_normal_sc,
                        y=np.zeros(len(X_normal_sc), dtype=np.int32),
                        feat_names=np.array(feat_cols))
    with open(DATA / "hai_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print("  Saved hai_train.npz and hai_scaler.pkl")

    # ── Train autoencoder ─────────────────────────────────────────────────────
    print("Training autoencoder on normal data ...")
    model = train_autoencoder(X_normal_sc, epochs=10)
    with open(DATA / "hai_detector.pkl", "wb") as f:
        pickle.dump(model, f)
    print("  Saved hai_detector.pkl")

    # ── Score train stream ────────────────────────────────────────────────────
    print("Scoring train stream ...")
    scores_train = score_stream(model, X_train_sc)
    y_train      = y_train_all
    cat_train    = make_cat(df_train)

    n_trn = (len(scores_train) // CHUNK_SIZE) * CHUNK_SIZE
    np.savez_compressed(
        DATA / "hai_scores_train.npz",
        scores = scores_train[:n_trn].reshape(-1, CHUNK_SIZE),
        y      = y_train[:n_trn].reshape(-1, CHUNK_SIZE),
        cat    = cat_train[:n_trn].reshape(-1, CHUNK_SIZE),
    )
    print(f"  Saved hai_scores_train.npz  ({n_trn//CHUNK_SIZE} chunks)")

    # ── Load test (deploy) stream ─────────────────────────────────────────────
    print("Loading HAI 21.03 test files ...")
    df_test = load_csvs("test*.csv")
    X_test  = scaler.transform(df_test[feat_cols].values.astype(np.float32))
    y_test  = df_test["attack"].values.astype(np.int32)
    cat_test = make_cat(df_test)

    print(f"  attack rate: {y_test.mean()*100:.2f}%")
    print(f"  process (easy) attacks: {(cat_test==1).sum():,}")
    print(f"  measurement (hard) attacks: {(cat_test==2).sum():,}")

    # ── Score test stream ─────────────────────────────────────────────────────
    print("Scoring test stream ...")
    scores_test = score_stream(model, X_test.astype(np.float32))

    n_tst = (len(scores_test) // CHUNK_SIZE) * CHUNK_SIZE
    np.savez_compressed(
        DATA / "hai_scores.npz",
        scores = scores_test[:n_tst].reshape(-1, CHUNK_SIZE),
        y      = y_test[:n_tst].reshape(-1, CHUNK_SIZE),
        cat    = cat_test[:n_tst].reshape(-1, CHUNK_SIZE),
    )
    print(f"  Saved hai_scores.npz  ({n_tst//CHUNK_SIZE} chunks)")

    # ── Chunk stats ───────────────────────────────────────────────────────────
    scores_r = scores_test[:n_tst].reshape(-1, CHUNK_SIZE)
    y_r      = y_test[:n_tst].reshape(-1, CHUNK_SIZE)
    cat_r    = cat_test[:n_tst].reshape(-1, CHUNK_SIZE)
    all_attack = (y_r == 1).all(axis=1).sum()
    all_normal = (y_r == 0).all(axis=1).sum()
    n_chunks   = n_tst // CHUNK_SIZE
    print(f"\nDeploy stream stats:")
    print(f"  {n_chunks} chunks of {CHUNK_SIZE}")
    print(f"  all-attack chunks : {all_attack}  (structural ceiling impact)")
    print(f"  all-normal chunks : {all_normal}")
    print(f"  max achievable inband: {(n_chunks-all_attack)/n_chunks*100:.1f}%")
    print(f"  chunks with process attacks : {(cat_r==1).any(axis=1).sum()}")
    print(f"  chunks with measurement attacks: {(cat_r==2).any(axis=1).sum()}")
    print("\nDone.")


if __name__ == "__main__":
    main()
