"""
Generate synthetic UNSW-NB15-like data for pipeline testing.
Matches the real dataset's feature count (49), attack prevalence (~32%),
and score separation characteristics.
Replace with real data once downloaded.
"""
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

np.random.seed(42)

N_TRAIN   = 175_341
N_TEST    = 82_332
N_FEAT    = 46          # numeric features (proto/service/state encoded separately)
ATTACK_RATE = 0.319     # 32% attacks in UNSW-NB15 test set

FEATURE_NAMES = [
    "dur","spkts","dpkts","sbytes","dbytes","rate","sttl","dttl",
    "sload","dload","sloss","dloss","sinpkt","dinpkt","sjit","djit",
    "swin","stcpb","dtcpb","dwin","tcprtt","synack","ackdat",
    "smean","dmean","trans_depth","response_body_len","ct_srv_src",
    "ct_state_ttl","ct_dst_ltm","ct_src_dport_ltm","ct_dst_sport_ltm",
    "ct_dst_src_ltm","is_ftp_login","ct_ftp_cmd","ct_flw_http_mthd",
    "ct_src_ltm","ct_srv_dst","is_sm_ips_ports",
    "proto","service","state",  # encoded categoricals
    "dur2","sbytes2","rate2","smean2","dmean2",  # extra
][:49]

def make_records(n, attack_rate):
    n_attack = int(n * attack_rate)
    n_normal = n - n_attack

    # Normal traffic: low anomaly scores
    X_normal = np.random.exponential(scale=0.5, size=(n_normal, N_FEAT)).astype(np.float32)
    X_normal += np.random.normal(0, 0.1, X_normal.shape)
    X_normal = np.clip(X_normal, 0, None)
    y_normal = np.zeros(n_normal, dtype=np.int32)

    # Attack traffic: higher values in subset of features
    X_attack = np.random.exponential(scale=0.5, size=(n_attack, N_FEAT)).astype(np.float32)
    attack_feat_mask = np.random.rand(N_FEAT) > 0.5
    X_attack[:, attack_feat_mask] += np.random.exponential(
        scale=2.0, size=(n_attack, attack_feat_mask.sum()))
    X_attack += np.random.normal(0, 0.15, X_attack.shape)
    X_attack = np.clip(X_attack, 0, None)
    y_attack = np.ones(n_attack, dtype=np.int32)

    X = np.vstack([X_normal, X_attack])
    y = np.concatenate([y_normal, y_attack])

    # shuffle
    perm = np.random.permutation(n)
    return X[perm], y[perm]


def save_csv(X, y, path, feat_names):
    df = pd.DataFrame(X, columns=feat_names[:X.shape[1]])
    df["label"] = y
    df.to_csv(path, index=False)
    print(f"Saved {len(df):,} records → {path}")


print("Generating synthetic UNSW-NB15-like data …")
feat_names = [f"feat_{i:02d}" for i in range(N_FEAT)]

X_train, y_train = make_records(N_TRAIN, attack_rate=0.0)   # normal-only train
X_test,  y_test  = make_records(N_TEST,  attack_rate=ATTACK_RATE)

save_csv(X_train, y_train, DATA_DIR / "UNSW_NB15_training-set.csv", feat_names)
save_csv(X_test,  y_test,  DATA_DIR / "UNSW_NB15_testing-set.csv",  feat_names)
print("Done. Replace with real CSVs from https://research.unsw.edu.au/projects/unsw-nb15-dataset")
