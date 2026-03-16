# Towards a Self-Driving Trigger at the LHC: Adaptive Response in Real Time v2

For Control-based methods, please refer: https://github.com/Shaghayegh-E/Adaptive-ParticlePhysics-Triggers

This repo is a follow-up on RL methods for LHC triggers.

## Datasets

These datasets are derived from the **CMS 2016 Open Data** for Level-1 (L1) hadronic objects (jets).  
Each file contains reconstructed jet features and the **number of primary vertices ($N_{PV}$)** per event, which serve as the core observables for our anomaly detection and control algorithm studies. The gradual decrease of $N_{PV}$ over time reflects the drop in luminosity and pileup as the fill progresses.

Two main dataset categories are included:

- **Base physics samples:** used for training and evaluating Anomaly Detection (AD) models and for benchmarking control algorithms.  
- **Trigger_food datasets:** precomputed control-variable files containing **Anomaly Scores** and **Hadronic Transverse Momentum (HT)** for each event, generated using our trained AD model to accelerate trigger-control experiments.

| File | Description | Usage |
|------|--------------|-------|
| **`MinBias_1.h5`** | Minimum-bias Monte Carlo (MC) Simulated sample with reconstructed jets and NPV. | Only Used for Anomaly Detection **training** (MC Simulated background) for Autoencoder. |
| **`MinBias_2.h5`** | Alternate minimum-bias MC background sample. | Used for **control-algorithm studies** (MC-only). |
| **`TT_1.h5`** | MC signal sample for **Standard Model** hadronic decay of the $\(t\bar{t}\)$ process. | Simulated **Standard Model** signal sample |
| **`HToAATo4B.h5`** | MC signal sample for **Beyond Standard Model** process $\(H \rightarrow AA \rightarrow 4b\)$. | Simulated **Beyond Standard Model** signal sample |
| **`data_Run_2016_283876.h5`** | Real CMS 2016 run with reconstructed jets and NPV. | Used for **training** the AD model with real-data background. |
| **`data_Run_2016_283408_longest.h5`** | Longest CMS 2016 real-data run. | Used for **control-algorithm testing** with real-data background. |
| **`Trigger_food_MC.h5`** | Precomputed **control-variables dataset** (MC): includes anomaly scores, HT, and NPV for each event across multiple MC processes. | Used for fast control-algorithm studies with **MC** background. |
| **`Trigger_food_Data.h5`** | Precomputed **control-variables dataset** (real data): includes anomaly scores, HT, NPV, and matched MC signal + real background (matched by NPV). | Used for fast control-algorithm studies with **real-data** background. |

> **Notes:**  
> • Some datasets are reserved exclusively for control-algorithm benchmarks to avoid overlap with AD model training.  
> • “Trigger_food” files store pre-evaluated anomaly scores and kinematic variables, reducing runtime for repeated experiments.  
> • All datasets originate from **CMS 2016 Open Data**.

> • Trigger_food_MC.h5 combines event information from MinBias_2.h5, TT_1.h5, HToAATo4B.h5.
> • Trigger_food_Data.h5 combines event information from data_Run_2016_283408_longest.h5, TT_1.h5, HToAATo4B.h5.
---

### Dataset Link

All datasets (base samples and precomputed control-variable files) are publicly hosted on **Zenodo**:

➡️ **Zenodo Record:** [https://zenodo.org/records/17399948?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjgwZmU5ZDg3LTYxMTYtNGE5OC05M2ZlLTQ5ZjdmYjE2NDRkMyIsImRhdGEiOnt9LCJyYW5kb20iOiIwNTQzMjkyYWVlMTQ2ZDE0NmI5MGIyZGFkYzFlN2VkZSJ9.rl-hT8qA2Og1SAncUUlR-98JWpI5FreQ9YOcwsZ5_utfP2Y8mHLYDXxDC5ErF-cxb2AS-6xQjBJx6ynofYVkeQ](https://zenodo.org/records/17399948?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjgwZmU5ZDg3LTYxMTYtNGE5OC05M2ZlLTQ5ZjdmYjE2NDRkMyIsImRhdGEiOnt9LCJyYW5kb20iOiIwNTQzMjkyYWVlMTQ2ZDE0NmI5MGIyZGFkYzFlN2VkZSJ9.rl-hT8qA2Og1SAncUUlR-98JWpI5FreQ9YOcwsZ5_utfP2Y8mHLYDXxDC5ErF-cxb2AS-6xQjBJx6ynofYVkeQ)

> The record includes `.h5` files for all datasets listed above.  

---

### Example: Load dataset in Python
```python
import h5py
import pandas as pd

with h5py.File("Trigger_food_MC.h5", "r") as f:
    print("Available keys:", list(f.keys()))
    df = pd.DataFrame({
        "HT": f["HT"][:],
        "NPV": f["NPV"][:],
        "score": f["anomaly_score"][:],
        "process": [p.decode() for p in f["process"][:]],
    })
    print(df.head())
```
## Required Packages

| Package | Purpose |
|---------|---------|
| `numpy` | Numerical computing |
| `pandas` | Data manipulation and analysis |
| `matplotlib` | Data visualization |
| `seaborn` | Statistical data visualization |
| `h5py` | HDF5 file I/O for datasets |
| `hdf5plugin` | HDF5 compression filters |
| `mplhep` | HEP-style matplotlib plots |
| `atlas_mpl_style` | ATLAS experiment plot styling |
| `scikit-learn` | Machine learning utilities (preprocessing, metrics) |
| `torch` (PyTorch) | Deep learning framework for RL agents |
| `tensorflow` / `keras` | Deep learning framework for Autoencoder training |
| `wandb` | Weights & Biases experiment tracking and hyperparameter sweeps |
| `pytest` | Testing framework |

## Setup
```bash
# clone
git clone https://github.com/Shaghayegh-E/Adaptive-ParticlePhysics-Triggers.git
cd Adaptive-ParticlePhysics-Triggers

# create env (recommended)
conda create -n AutoTrig python=3.9 -y
conda activate AutoTrig

# install required packages
pip install -r requirements.txt
```

## File Structure
Download data from Zenodo: All `.h5` datasets can be downloaded from the public Zenodo record:
After downloading, place them under the following structure.

```text
Adaptive-ParticlePhysics-Triggers/
├── Data/                     # Place downloaded .h5 datasets here (from Zenodo and should be ignored by Git)
│   ├── MinBias_1.h5 #used for training
│   ├── MinBias_2.h5
│   ├── TT_1.h5
│   ├── HToAATo4B.h5
│   ├── data_Run_2016_283876.h5 #used for training
│   ├── data_Run_2016_283408_longest.h5
│   ├── Trigger_food_MC.h5
│   └── Trigger_food_Data.h5
│
├── SampleProcessing/         
│   ├── ae/                   # Autoencoder models & training scripts & building autoencoders for Anomaly Detection Algorithm. Data Samples: Data/MinBias_1 Data/HToAATo4B.h5 Data/TT_1.h5
│   │   ├── data.py
│   │   ├── experiment_testae.py 
│   │   ├── losses.py
│   │   ├── models.py
│   │   └── plots.py
│   │
│   ├── derived_info/         # Build Data/MinBias_2.h5, Data/HToAATo4B.h5, Data/TT_1.h5, models/autoencoder_model_2_mc.keras -> Data/trigger_food_MC (monte carlo samples)
│   │   ├── build_trigger_food.py
│   │   ├── data_io.py
│   │   ├── preprocess.py
│   │   └── scoring.py
│   ├── models/               #saving trained autoencoders with dimension = 2
│   │   ├── autoencoder_model_mc_2.keras #autoencoder with dimension = 2
│   
├── Control/   #Running single trigger / Local Multi Trigger and Multi Path trigger
│   ├── agents.py
│   ├── mc_localmulti.py
│   ├── mc_multipath.py
│   ├── mc_singletrigger_io.py
│   ├── mc_singletrigger_plots.py
│   ├── mc_singletrigger.py
│   ├── summary.py
│   └── metrics.py
│── RL/ # Running RL algorithms
│
├── firmware/                 # FPGA firmware export (StateEncoder → HLS C++ via hls4ml)
│   ├── README.md
│   ├── config.yaml
│   ├── extract_weights.py
│   ├── unroll_gru.py
│   ├── unroll_rnn.py
│   ├── convert_hls.py
│   └── validate.py
│
├── outputs/                  # Generated plots & results (create your own outputs folder to store the plots)
├── controllers.py                  
├── triggers.py                  
└── README.md
```
## Step 1 Training Autoencoder
### Training Autoencoder with dimension = 2 

# use simulated events as background
```
python3 -m SampleProcessing.ae.experiment_testae --dims=2
```
# use real experiment events as background
```
python3 -m SampleProcessing.ae.experiment_testae --dims=2 --bkgType=RealData
```

## Step 2 Building Trigger_food
### Building Trigger_food_MC.h5 or Trigger_food_Data.h5 under Data Folder
#### use simulated events as background
```
python3 -m SampleProcessing.derived_info.build_trigger_food
```
#### use real experiment events as background
```
python3 -m SampleProcessing.derived_info.build_trigger_food --bkgType=RealData
```

## Step 3 Choose different agents for Trigger Control (Control-only framework)

### Single-path demo (PD controller on HT & AD)
#### use --bkgType=MC or RealData (default=MC)
```
python3 -m Control.singletrigger --bkgType=RealData
python3 -m Control.singletrigger_plots --bkgType=RealData
```
### Multi Trigger Control Framework Case 1/2/3
#### Running CompCost_Eval reports reference cost parameters for Case 3 
### (default: MC)
```
python3 -m Control.idealMultiTrigger --agent v1 --bkgType=MC --path "Data/Trigger_food_MC.h5" \
--outdir outputs/demo_IdealMultiTrigger_mc

python3 -m Control.idealMultiTrigger --agent v2

python3 -m Control.compCost_eval --bkgType=MC  --path Data/Trigger_food_MC.h5\
--outdir outputs/demo_IdealMultiTrigger_mc
python3 -m Control.idealMultiTrigger --agent v3 --costRef 5.6 2.7 --forceCostRef
```


### A Real Controller Case 1/2/3 (default: MC)
```
python3 -m Control.realMultiTrigger --agent v1 \
    --bkgType RealData \
    --path Data/Trigger_food_Data.h5 \
    --outdir outputs/demo_RealMultiTrigger_realdata

python3 -m Control.realMultiTrigger --agent v2 \
    --bkgType RealData \
    --path Data/Trigger_food_Data.h5 \
    --outdir outputs/demo_RealMultiTrigger_realdata

python3 -m Control.realMultiTrigger --agent v3 \
    --bkgType RealData \
    --path Data/Trigger_food_Data.h5 \
    --outdir outputs/demo_RealMultiTrigger_realdata
```

## Step 4 RL-based Trigger Control (GFPO framework)

The RL pipeline trains adaptive trigger policies on Monte Carlo (MC) simulation, validates on held-out MC data, and deploys on CMS real collision data. The MC dataset (`Trigger_food_MC.h5`) contains 185 chunks after the calibration window; we use an 80/20 temporal split (148 train / 37 validation) for hyperparameter tuning.

### 4a. Hyperparameter sweep on 80% MC (optional)

Grid search over reward weights $\lambda_1$ and $\lambda_2$. Each run trains on the first 148 chunks with `--max-chunks 148`.

```bash
# Create the sweep (5x5 grid: lambda_1, lambda_2 in {0.0, 0.25, 0.5, 0.75, 1.0})
wandb sweep sweep_lambda_train80.yaml

# Launch the sweep agent (runs all 25 configurations sequentially)
wandb agent <SWEEP_ID>
```

After the sweep, inspect the wandb dashboard to select the best $(\lambda_1, \lambda_2)$ based on validation signal efficiency and InBand rate.

**Recommended defaults:** Based on Pareto frontier analysis across all methods (DQN, PPO, ADT, GRPO, GFPO-F, GFPO-FR) and both triggers (AD, HT), we select $\lambda_1 = 0.25$, $\lambda_2 = 1.0$. This combination achieves mean InBand = 0.952 with strong signal efficiency across all methods while keeping both penalty terms active. GFPO-F is the most robust method, achieving InBand $\geq$ 0.993 regardless of $(\lambda_1, \lambda_2)$.

### 4b. Train on 80% MC

Train RL agents (GFPO-F, GFPO-FR, GRPO, DQN, PPO, ADT) on the first 148 chunks and save model checkpoints.

```bash
python RL/demo_single_trigger_grpo_as_feature_all_training.py \
  --run-ht --run-adt --save-models \
  --max-chunks 148 \
  --lambda_1 0.25 --lambda_2 1.0 \
  --models-dir outputs/best_mc/models_mc \
  --outdir outputs/best_mc
```

### 4c. Validate on held-out 20% MC

Evaluate the trained models on chunks 149-185 (the held-out validation set) to confirm generalization without data leakage.

```bash
python RL/demo_single_trigger_grpo_as_feature_all_rollout.py \
  --input Data/Trigger_food_MC.h5 --control MC \
  --models-dir outputs/best_mc/models_mc \
  --run-ht --run-adt \
  --skip-chunks 148 \
  --lambda_1 0.25 --lambda_2 1.0 \
  --outdir outputs/val_20pct_mc
```

### 4d. Deploy on CMS real data

Roll out the MC-trained models on CMS Run 2016 collision data to evaluate sim-to-real transfer.

```bash
python RL/demo_single_trigger_grpo_as_feature_all_rollout.py \
  --input Data/Matched_data_2016_dim2.h5 --control RealData \
  --models-dir outputs/best_mc/models_mc \
  --run-ht --run-adt \
  --lambda_1 0.25 --lambda_2 1.0 \
  --outdir outputs/rollout_real
```

### 4e. (Optional) Evaluate on full MC

Roll out trained models on the entire MC dataset (all 185 chunks) for completeness.

```bash
python RL/demo_single_trigger_grpo_as_feature_all_rollout.py \
  --input Data/Trigger_food_MC.h5 --control MC \
  --models-dir outputs/best_mc/models_mc \
  --run-ht --run-adt \
  --lambda_1 0.25 --lambda_2 1.0 \
  --outdir outputs/rollout_full_mc
```

### 4f. (Optional) Train and deploy directly on CMS real data

For comparison, train directly on real data and evaluate on it (no sim-to-real gap).

```bash
# Train on real data
python RL/demo_single_trigger_grpo_as_feature_all_training.py \
  --input Data/Matched_data_2016_dim2.h5 --control RealData \
  --run-ht --run-adt --save-models \
  --lambda_1 0.25 --lambda_2 1.0 \
  --models-dir outputs/best_real/models_real \
  --outdir outputs/best_real

# Deploy real-trained models on real data
python RL/demo_single_trigger_grpo_as_feature_all_rollout.py \
  --input Data/Matched_data_2016_dim2.h5 --control RealData \
  --models-dir outputs/best_real/models_real \
  --run-ht --run-adt \
  --lambda_1 0.25 --lambda_2 1.0 \
  --outdir outputs/rollout_real_on_real
```

### Key RL arguments

| Argument | Description |
|----------|-------------|
| `--max-chunks N` | Use only first N chunks for training (148 = 80% train split) |
| `--skip-chunks N` | Skip first N chunks for validation (148 = start at 20% held-out) |
| `--save-models` | Save trained model checkpoints to disk |
| `--models-dir PATH` | Directory to load/save model `.pt` files |
| `--run-ht` | Enable HT trigger alongside AD trigger |
| `--run-adt` | Enable ADT baseline (DQN with action-hold) |
| `--alpha` | $t\bar{t}$ focus weight (default: 0.7) |
| `--lambda_1` | Background rate tracking reward weight (default: 0.25) |
| `--lambda_2` | Threshold movement penalty weight (default: 1.0) |
| `--input PATH` | Input dataset path |
| `--control MC\|RealData` | Data source type for calibration logic |
| `--outdir PATH` | Output directory for plots and logs |

## Step 5 FPGA Firmware Export (StateEncoder → HLS C++)

Export trained RL policy networks to synthesizable HLS C++ for Xilinx FPGAs via [hls4ml](https://fastmachinelearning.org/hls4ml/). The `firmware/` directory is fully self-contained and does **not** modify any files in `RL/`.

**Architecture:** The `StateEncoder` (GRU + Linear head) is unrolled over K timesteps into explicit Dense-layer gate operations, producing an equivalent feedforward Keras model that hls4ml synthesizes natively.

```
PyTorch StateEncoder (.pt)  →  Unrolled Keras (Dense layers)  →  HLS C++ (Vivado)  →  FPGA bitstream
```

### 5a. Install hls4ml

```bash
conda run -n adaptive pip install hls4ml[profiling]
```

### 5b. Extract weights from trained checkpoint

```bash
python firmware/extract_weights.py \
  --checkpoint outputs/best_mc/models_mc/model.pt \
  --rnn-type gru \
  --output-dir firmware/weights/
```

### 5c. Convert to HLS C++

```bash
python firmware/convert_hls.py \
  --weights-dir firmware/weights/ \
  --seq-len 10 \
  --output-dir firmware/hls_output/ \
  --precision "ap_fixed<16,6>" \
  --reuse-factor 1 \
  --clock-period 5 \
  --fpga-part xcu250-figd2104-2L-e
```

### 5d. Validate numerical equivalence (PyTorch vs Keras)

```bash
python firmware/validate.py \
  --checkpoint outputs/best_mc/models_mc/model.pt \
  --weights-dir firmware/weights/ \
  --seq-len 10 \
  --rnn-type gru
```

### 5e. (Optional) Run Vivado C-synthesis for resource estimates

Requires Xilinx Vivado HLS installed.

```bash
python firmware/convert_hls.py \
  --weights-dir firmware/weights/ \
  --seq-len 10 \
  --output-dir firmware/hls_output/ \
  --synth
```

### Key firmware arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--rnn-type` | `gru` | RNN variant: `gru`, `rnn`, or `rnn_relu` |
| `--seq-len` | `10` | Sequence length K (micro-steps per chunk) |
| `--precision` | `ap_fixed<16,6>` | Fixed-point format (16-bit, 6 integer bits) |
| `--reuse-factor` | `1` | 1 = fully parallel (min latency), higher = smaller FPGA footprint |
| `--clock-period` | `5` | Target clock period in ns (5 ns = 200 MHz) |
| `--fpga-part` | `xcu250-figd2104-2L-e` | Xilinx FPGA part (Alveo U250). Use `xcvu13p-flga2577-2-e` for CMS Phase-2 L1T |

### Firmware file structure

```
firmware/
├── README.md              # Detailed firmware documentation
├── config.yaml            # Default FPGA/synthesis parameters
├── extract_weights.py     # Step 1: PyTorch .pt → .npy weight files
├── unroll_gru.py          # GRU unroll → Keras Dense layers
├── unroll_rnn.py          # Simple RNN unroll → Keras Dense layers
├── convert_hls.py         # Step 2: weights → Keras → hls4ml → HLS C++
└── validate.py            # Step 3: numerical equivalence check
```

## Step 6 Generate Summary Plots
### Summary of different agents’ performance (default: MC)

```
python3 -m Control.summary --bkgType=MC --path Data/Trigger_food_MC.h5 \
--out outputs/SummaryPanels_MC.pdf

python3 -m Control.summary --bkgType=RealData --path Data/Trigger_food_Data.h5 \
--out outputs/SummaryPanels_Data.pdf --bins 7

```

