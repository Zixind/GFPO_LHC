# Online Anomaly Detection — UNSW-NB15 Benchmark

Adapts all RL trigger-control methods (DQN, GRPO, L-GRPO, GFPO-F, PPO)
to online network intrusion detection on the UNSW-NB15 dataset.

## Conceptual mapping

| Particle Physics Trigger | UNSW-NB15 Anomaly Detection |
|---|---|
| Background rate | False alert rate (FAR) |
| Signal efficiency (TT / h→4b) | Attack recall (TPR) |
| Cut threshold | Anomaly score threshold |
| Rate target ± τ | FAR budget ± tolerance |
| Chunk (time window) | Streaming traffic window |

## Step-by-step

### 1. Download the data

Download the pre-split train/test CSVs from:
  https://research.unsw.edu.au/projects/unsw-nb15-dataset

Place them in `anomaly_detection/data/`:
```
anomaly_detection/data/UNSW_NB15_training-set.csv
anomaly_detection/data/UNSW_NB15_testing-set.csv
```

### 2. Preprocess

```bash
conda run -n adaptive python anomaly_detection/preprocess_unsw.py
```

Outputs:
- `anomaly_detection/data/unsw_train.npz`  — normal-only training split
- `anomaly_detection/data/unsw_stream.npz` — chunked test stream

### 3. Train base anomaly detector

```bash
# Fast option (IsolationForest)
conda run -n adaptive python anomaly_detection/base_detector.py --detector iforest

# Better scores (Autoencoder, ~2 min)
conda run -n adaptive python anomaly_detection/base_detector.py --detector ae
```

Outputs:
- `anomaly_detection/data/detector.pkl`
- `anomaly_detection/data/unsw_scores.npz`  — per-record anomaly scores

### 4. Train RL agents

```bash
conda run -n adaptive python anomaly_detection/train_anomaly.py \
    --n-chunks 82 --epochs 5 \
    --far-target 0.005 --far-tol 0.0005 \
    --outdir anomaly_detection/models
```

### 5. Run rollout (all methods)

```bash
conda run -n adaptive python anomaly_detection/rollout_anomaly.py \
    --baselines "constant,pid,spot,dqn,grpo,lgrpo,gfpo" \
    --outdir outputs/anomaly_unsw
```

Add `--train` to enable online RL adaptation during deployment.

### 6. Plot results

```bash
conda run -n adaptive python anomaly_detection/plot_anomaly.py \
    --csv outputs/anomaly_unsw/tables/chunk_stats.csv \
    --out outputs/anomaly_unsw/
```

## Output metrics (per chunk, per method)

| Column | Description |
|---|---|
| `far` | False alert rate at current threshold |
| `tpr` | Attack recall (true positive rate) |
| `inband` | 1 if FAR within ± tolerance of target |
| `threshold` | Current anomaly score threshold |
