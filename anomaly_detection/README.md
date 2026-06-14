# Online Anomaly Detection — NAB & UNSW-NB15 Benchmarks

Adapts all RL trigger-control methods (DQN, GRPO, L-GRPO, GFPO-F, GFPO-FR, PPO)
to two online anomaly-detection benchmarks: the **Numenta Anomaly Benchmark
(NAB)** and **UNSW-NB15** network-intrusion detection.

## Conceptual mapping

| Particle Physics Trigger | Anomaly Detection |
|---|---|
| Background rate | False alert rate (FAR) |
| Signal efficiency (TT / h→4b) | Attack recall / anomaly recall (TPR) |
| Cut threshold | Anomaly score threshold |
| Rate target ± τ | FAR budget ± tolerance |
| Chunk (time window) | Streaming traffic / time window |

## Datasets — download & placement

Both datasets live under **`anomaly_detection/data/`**, which is **gitignored**
(not shipped with the repo). Download them once into that directory as below.

### UNSW-NB15 (manual download)

Download the two pre-split CSVs from the official UNSW page:

- **Source:** https://research.unsw.edu.au/projects/unsw-nb15-dataset
  (CSV Files → `UNSW_NB15_training-set.csv` 175,341 records, `UNSW_NB15_testing-set.csv` 82,332 records)

Place them exactly here:
```
anomaly_detection/data/UNSW_NB15_training-set.csv
anomaly_detection/data/UNSW_NB15_testing-set.csv
```

### NAB (auto-downloaded)

The Numenta Anomaly Benchmark (NAB; Lavin & Ahmad, 2015) is a standard benchmark
for streaming anomaly detection: real-world univariate metrics, sampled at
regular intervals and hand-labeled with *anomaly windows*, processed strictly
online (one timestep at a time, no lookahead) with detections rewarded for being
early and within a labeled window. The streams cover everyday monitoring domains
— cloud-server metrics (Amazon CloudWatch CPU/network/disk), machine and ambient
temperatures, server CPU during known faults, taxi ridership, Twitter volume,
freeway traffic, and online-ad cost-per-click. We use the `realKnownCause` and
`realAWSCloudwatch` categories (24 streams of server and machine telemetry), as
they best match the trigger setting: a single live metric on which the policy
must raise alerts on the fly under a controlled alert budget.

NAB is cloned automatically by the preprocessing script — **no manual download
needed** (requires `git` + network). It is fetched from:

- **Source:** https://github.com/numenta/NAB

into:
```
anomaly_detection/data/nab_raw/        # git clone --depth 1 https://github.com/numenta/NAB
```

To pre-clone manually instead:
```bash
git clone --depth 1 https://github.com/numenta/NAB anomaly_detection/data/nab_raw
```

---

## UNSW-NB15 pipeline

```bash
# 1. Preprocess  (reads the two CSVs above)
conda run -n adaptive python anomaly_detection/preprocess_unsw.py
#   → anomaly_detection/data/unsw_train.npz   (normal-only training split)
#   → anomaly_detection/data/unsw_stream.npz  (chunked test stream)

# 2. Train the base anomaly detector
conda run -n adaptive python anomaly_detection/base_detector.py --detector iforest   # fast (IsolationForest)
# conda run -n adaptive python anomaly_detection/base_detector.py --detector ae       # better scores (Autoencoder, ~2 min)
#   → anomaly_detection/data/detector.pkl
#   → anomaly_detection/data/unsw_scores.npz

# 3. Train RL agents
conda run -n adaptive python anomaly_detection/train_anomaly.py \
    --n-chunks 82 --epochs 5 --far-target 0.005 --far-tol 0.0005 \
    --outdir anomaly_detection/models

# 4. Roll out all methods  (add --train for online RL adaptation)
conda run -n adaptive python anomaly_detection/rollout_anomaly.py \
    --baselines "constant,pid,spot,dqn,grpo,lgrpo,gfpo" \
    --outdir outputs/anomaly_unsw

# 5. Plot
conda run -n adaptive python anomaly_detection/plot_anomaly.py \
    --csv outputs/anomaly_unsw/tables/chunk_stats.csv --out outputs/anomaly_unsw/
```

## NAB pipeline

```bash
# 1. Preprocess  (auto-clones NAB into data/nab_raw on first run)
conda run -n adaptive python anomaly_detection/preprocess_nab.py
#   → anomaly_detection/data/nab_train.npz     (first 70% of each file)
#   → anomaly_detection/data/nab_test.npz      (last 30% of each file)
#   → anomaly_detection/data/nab_windows.json  (anomaly windows per test chunk)

# 2. Train RL agents
conda run -n adaptive python anomaly_detection/train_nab.py \
    --scores anomaly_detection/data/nab_train.npz \
    --epochs 5 --outdir anomaly_detection/models_nab

# 3. Roll out all methods
conda run -n adaptive python anomaly_detection/rollout_nab.py \
    --scores  anomaly_detection/data/nab_test.npz \
    --windows anomaly_detection/data/nab_windows.json \
    --models  anomaly_detection/models_nab \
    --methods "constant,constant-opt,pid,dspot,adt,dqn,grpo,lgrpo,ppo,gfpo-f,gfpo-fr" \
    --outdir  outputs/anomaly_nab

# 4. Plot
conda run -n adaptive python anomaly_detection/plot_anomaly.py \
    --csv outputs/anomaly_nab/tables/chunk_stats.csv --out outputs/anomaly_nab/
```

## Output metrics (per chunk, per method)

| Column | Description |
|---|---|
| `far` | False alert rate at current threshold |
| `tpr` | Attack/anomaly recall (true positive rate) |
| `inband` | 1 if FAR within ± tolerance of target |
| `threshold` | Current anomaly score threshold |
