"""
anomaly_detection/base_detector.py

Train a base anomaly detector on normal UNSW-NB15 traffic.
Produces a per-record anomaly score ∈ ℝ (higher = more anomalous).

Two detector options:
  --detector iforest   IsolationForest  (fast, no GPU needed)
  --detector ae        Autoencoder      (PyTorch, better scores)

The trained detector is saved to anomaly_detection/data/detector.pkl
and anomaly scores for the full stream are saved to
anomaly_detection/data/unsw_scores.npz.

Usage:
  python anomaly_detection/base_detector.py --detector iforest
  python anomaly_detection/base_detector.py --detector ae
"""

import argparse
import numpy as np
import joblib
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"


# ── IsolationForest detector ──────────────────────────────────────────────────
def train_iforest(X_normal: np.ndarray):
    from sklearn.ensemble import IsolationForest
    clf = IsolationForest(n_estimators=200, contamination=0.05,
                          random_state=42, n_jobs=-1)
    clf.fit(X_normal)
    return clf


def score_iforest(clf, X: np.ndarray) -> np.ndarray:
    # decision_function returns higher for normal; negate so higher = anomalous
    return -clf.decision_function(X).astype(np.float32)


# ── Autoencoder detector ──────────────────────────────────────────────────────
def train_ae(X_normal: np.ndarray, epochs: int = 20, batch: int = 512,
             hidden: int = 64, device: str = "cpu"):
    import torch
    import torch.nn as nn
    import torch.optim as optim

    d = X_normal.shape[1]
    model = nn.Sequential(
        nn.Linear(d, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden // 2), nn.ReLU(),
        nn.Linear(hidden // 2, hidden), nn.ReLU(),
        nn.Linear(hidden, d),
    ).to(device)

    opt = optim.Adam(model.parameters(), lr=1e-3)
    X_t = torch.tensor(X_normal, dtype=torch.float32, device=device)
    ds  = torch.utils.data.TensorDataset(X_t)
    dl  = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True)

    for ep in range(1, epochs + 1):
        losses = []
        for (xb,) in dl:
            xr = model(xb)
            loss = ((xb - xr) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        if ep % 5 == 0:
            print(f"  AE epoch {ep}/{epochs}  loss={np.mean(losses):.5f}")

    return model.cpu()


def score_ae(model, X: np.ndarray, batch: int = 4096, device: str = "cpu") -> np.ndarray:
    import torch
    model.eval()
    scores = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.tensor(X[i:i+batch], dtype=torch.float32, device=device)
            xr = model(xb)
            rec = ((xb - xr) ** 2).mean(dim=1).cpu().numpy()
            scores.append(rec)
    return np.concatenate(scores).astype(np.float32)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--detector", default="iforest", choices=["iforest", "ae"])
    p.add_argument("--ae-epochs", type=int, default=20)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    train_npz        = np.load(DATA_DIR / "unsw_train.npz")
    stream_npz       = np.load(DATA_DIR / "unsw_stream.npz")
    train_stream_npz = np.load(DATA_DIR / "unsw_train_stream.npz")
    X_normal         = train_npz["X"]
    X_stream         = stream_npz["X"]               # (n_chunks, chunk_size, n_feat)
    y_stream         = stream_npz["y"]
    cat_stream       = stream_npz.get("cat", None)   # (n_chunks, chunk_size) int8 category
    X_tr_stream      = train_stream_npz["X"]         # (n_chunks_tr, chunk_size, n_feat)
    y_tr_stream      = train_stream_npz["y"]
    cat_tr_stream    = train_stream_npz.get("cat", None)
    n_chunks, chunk_size, n_feat = X_stream.shape

    print(f"Training {args.detector} on {len(X_normal)} normal records …")

    if args.detector == "iforest":
        clf = train_iforest(X_normal)
        joblib.dump(clf, DATA_DIR / "detector.pkl")
        score_fn = lambda X: score_iforest(clf, X)
    else:
        model = train_ae(X_normal, epochs=args.ae_epochs, device=args.device)
        joblib.dump(model, DATA_DIR / "detector.pkl")
        score_fn = lambda X: score_ae(model, X, device=args.device)

    # Score test stream (Exp2/3 deployment set)
    s_flat = score_fn(X_stream.reshape(-1, n_feat))
    # log1p-normalise so the threshold range is compact and delta_range makes sense
    s_flat = np.log1p(s_flat)
    scores = s_flat.reshape(n_chunks, chunk_size)
    save_kwargs = dict(scores=scores, y=y_stream)
    if cat_stream is not None:
        save_kwargs["cat"] = cat_stream
    np.savez_compressed(DATA_DIR / "unsw_scores.npz", **save_kwargs)
    print(f"Test scores saved → {DATA_DIR / 'unsw_scores.npz'}")
    print(f"  range: [{scores.min():.4f}, {scores.max():.4f}], attack prevalence: {y_stream.mean():.3f}")

    # Score training stream (Exp1 training set)
    n_chunks_tr = X_tr_stream.shape[0]
    s_tr_flat = score_fn(X_tr_stream.reshape(-1, n_feat))
    s_tr_flat = np.log1p(s_tr_flat)
    scores_tr = s_tr_flat.reshape(n_chunks_tr, chunk_size)
    save_kwargs_tr = dict(scores=scores_tr, y=y_tr_stream)
    if cat_tr_stream is not None:
        save_kwargs_tr["cat"] = cat_tr_stream
    np.savez_compressed(DATA_DIR / "unsw_scores_train.npz", **save_kwargs_tr)
    print(f"Train scores saved → {DATA_DIR / 'unsw_scores_train.npz'}")
    print(f"  range: [{scores_tr.min():.4f}, {scores_tr.max():.4f}], attack prevalence: {y_tr_stream.mean():.3f}")


if __name__ == "__main__":
    main()
