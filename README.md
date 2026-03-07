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

### Single training run on MC
```bash
conda run -n AutoTrig python RL/demo_single_trigger_grpo_as_feature_all_training.py \
  --run-ht --run-adt --save-models --max-chunks 148 \
  --lambda_1 0.25 --lambda_3 0.75 \
  --models-dir outputs/best_mc/models_mc \
  --outdir outputs/best_mc
```

### Hyperparameter sweep (grid search over lambda_1, lambda_3)
```bash
wandb sweep sweep_lambda_train80.yaml
wandb agent <SWEEP_ID>
```

### Validate on held-out 20% MC
```bash
conda run -n AutoTrig python RL/demo_single_trigger_grpo_as_feature_all_rollout.py \
  --input Data/Trigger_food_MC.h5 --control MC \
  --models-dir outputs/best_mc/models_mc \
  --run-ht --run-adt --skip-chunks 148 \
  --lambda_1 0.25 --lambda_3 0.75
```

### Rollout MC-trained models on full MC
```bash
conda run -n AutoTrig python RL/demo_single_trigger_grpo_as_feature_all_rollout.py \
  --input Data/Trigger_food_MC.h5 --control MC \
  --models-dir outputs/best_mc/models_mc \
  --run-ht --run-adt \
  --lambda_1 0.25 --lambda_3 0.75
```

### Rollout MC-trained models on CMS real data
```bash
conda run -n AutoTrig python RL/demo_single_trigger_grpo_as_feature_all_rollout.py \
  --input Data/Matched_data_2016_dim2.h5 --control RealData \
  --models-dir outputs/best_mc/models_mc \
  --run-ht --run-adt \
  --lambda_1 0.25 --lambda_3 0.75
```

### Train and rollout directly on CMS real data
```bash
# Train
conda run -n AutoTrig python RL/demo_single_trigger_grpo_as_feature_all_training.py \
  --input Data/Matched_data_2016_dim2.h5 --control RealData \
  --run-ht --run-adt --save-models \
  --lambda_1 0.25 --lambda_3 0.75 \
  --models-dir outputs/best_real/models_real

# Rollout
conda run -n AutoTrig python RL/demo_single_trigger_grpo_as_feature_all_rollout.py \
  --input Data/Matched_data_2016_dim2.h5 --control RealData \
  --models-dir outputs/best_real/models_real \
  --run-ht --run-adt \
  --lambda_1 0.25 --lambda_3 0.75
```

### Key RL arguments

| Argument | Description |
|----------|-------------|
| `--max-chunks N` | Use only first N chunks (148 = 80% train split) |
| `--skip-chunks N` | Skip first N chunks (148 = 20% validation split) |
| `--save-models` | Save trained model checkpoints |
| `--models-dir PATH` | Load/save models from PATH |
| `--run-ht` | Enable HT trigger alongside AD trigger |
| `--run-adt` | Enable ADT baseline |
| `--alpha` | tt-bar focus weight (default: 0.7) |
| `--lambda_1` | Background rate tracking reward weight (default: 0.5) |
| `--lambda_3` | Threshold movement penalty weight (default: 0.2) |

## Step 5 Generate Summary Plots
### Summary of different agents’ Performance (default:MC)

```
python3 -m Control.summary --bkgType=MC --path Data/Trigger_food_MC.h5 \
--out outputs/SummaryPanels_MC.pdf

python3 -m Control.summary --bkgType=RealData --path Data/Trigger_food_Data.h5 \
--out outputs/SummaryPanels_Data.pdf --bins 7

```

