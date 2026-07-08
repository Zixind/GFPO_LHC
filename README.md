# Learning to Trigger: Reinforcement Learning at the Large Hadron Collider

[![arXiv](https://img.shields.io/badge/arXiv-2606.23993-b31b1b.svg)](https://arxiv.org/abs/2606.23993)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Paper%20page-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/papers/2606.23993)

**Repository developer:** Zixin Ding ([zixin@uchicago.edu](mailto:zixin@uchicago.edu))

## News

- **2026-07** 🎉 Our paper has been accepted to the [ICML 2026 AI4Physics Workshop](https://ai4physics-workshop.github.io) as an **Oral**!

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

The RL pipeline trains adaptive trigger policies on Monte Carlo (MC) simulation
and deploys them on CMS real collision data. The MC dataset
(`Trigger_food_MC.h5`) contains ~185 chunks after the calibration window. We use
an 80/20 temporal split (chunks 0–147 train / 148–end held-out).

All experiments are reproduced by **three self-contained scripts**, one per
evaluation setting. Each runs all baselines over 3 seeds (42, 123, 456) and
writes a `paper_table.csv` per seed. **No setting overlaps training and
evaluation data.**

| # | Setting | Train on | Evaluate on | Script |
|---|---------|----------|-------------|--------|
| 1 | MC hold-out | first **80%** MC (chunks 0–147) | held-out **20%** MC (148–end) | `run_setting1_mc_holdout.sh` |
| 2 | Sim-to-real (frozen) | **full** MC | CMS Run 283408 (frozen) | `run_setting2_cms_deploy.sh` |
| 3 | Test-time training | **full** MC | CMS Run 283408 (online `--ttt`) | `run_setting3_cms_ttt.sh` |

```bash
# Setting 1 — train on 80% MC, freeze, report ONLY the held-out 20% MC
bash run_setting1_mc_holdout.sh

# Setting 2 — train on full MC, freeze, deploy on CMS real data (sim-to-real)
bash run_setting2_cms_deploy.sh

# Setting 3 — load full-MC checkpoints, adapt online on CMS (needs Setting 2 first)
bash run_setting3_cms_ttt.sh
```

**Baselines** (run by every script): `constant`, `pid`, `adt`, `dqn`, `dqn_f`,
`ppo`, `grpo`, `lgrpo`, `gfpo_f`, `gfpo_fr`, `cpo`, plus `spot` (DSPOT). Our
methods are **GFPO-F** and **GFPO-FR**.

Each script is a thin wrapper around two entry points — the training script
(`RL/demo_single_trigger_grpo_as_feature_all_training.py`) and the rollout
script (`RL/demo_single_trigger_grpo_as_feature_all_rollout_v2.py`). The core
commands are, for one seed:

```bash
# (Setting 1a) train on the first 80% MC and save checkpoints
python RL/demo_single_trigger_grpo_as_feature_all_training.py \
  --input Data/Trigger_food_MC.h5 --control MC --outdir outputs/mc_seed_42 \
  --seed 42 --run-ht --run-adt --baselines "adt,dqn,dqn_f,ppo,grpo,gfpo_f,gfpo_fr" \
  --max-chunks 148 --ht-step 2.0 --alpha 0.3 --lambda_1 0.25 \
  --group-size-sample 64 --group-size-keep 16 --save-models

# (Setting 1b) freeze, evaluate ALL baselines on the held-out 20% MC only
python RL/demo_single_trigger_grpo_as_feature_all_rollout_v2.py \
  --input Data/Trigger_food_MC.h5 --control MC --outdir outputs/mc_seed_42_eval_holdout \
  --models-dir outputs/mc_seed_42_all_MC/models_mc --load-models --eval-only \
  --seed 42 --run-ht --run-adt --start-chunk 148 \
  --baselines "constant,pid,adt,dqn,dqn_f,ppo,grpo,lgrpo,gfpo_f,gfpo_fr,cpo" \
  --ht-step 2.0 --alpha 0.3 --group-size-sample 64 --group-size-keep 16

# (Setting 2) full-MC training (drop --max-chunks); deploy frozen on CMS
python ...all_rollout_v2.py --input Data/Matched_data_2016_dim2.h5 --control RealData \
  --models-dir outputs/mc_seed_42_fulltrain_all_MC/models_mc --load-models --eval-only ...

# (Setting 3) same full-MC checkpoint, adapt online on CMS with --ttt
python ...all_rollout_v2.py --input Data/Matched_data_2016_dim2.h5 --control RealData \
  --models-dir outputs/mc_seed_42_fulltrain_all_MC/models_mc --load-models --ttt ...
```

### Key RL arguments

| Argument | Description |
|----------|-------------|
| `--max-chunks N` | Train on only the first N chunks (148 = 80% MC split; omit for full MC) |
| `--start-chunk N` | Begin evaluation at chunk N (148 = held-out 20%) |
| `--eval-only` | Frozen rollout — no policy updates (Settings 1, 2) |
| `--ttt` | Test-time training — online policy updates during deployment (Setting 3) |
| `--save-models` / `--load-models` | Save / load `.pt` checkpoints |
| `--models-dir PATH` | Directory of model `.pt` files |
| `--baselines "a,b,c"` | Comma-separated baselines to run |
| `--run-ht` / `--run-adt` | Enable the HT trigger / ADT baseline |
| `--alpha` | $t\bar{t}$ focus weight |
| `--lambda_1` | Background-rate-tracking vs. signal reward weight |
| `--lambda_2` | Threshold-movement (smoothness) penalty weight |
| `--control MC\|RealData` | Data source (MC simulation or CMS real data) |
| `--outdir PATH` | Output directory (`_all_MC` / `_all_RealData` suffix added automatically) |

## Citing this paper

If you use this code or build on this work, please cite ([arXiv:2606.23993](https://arxiv.org/abs/2606.23993)):

```bibtex
@misc{ding2026learning,
  title         = {Learning to Trigger: Reinforcement Learning at the Large Hadron Collider},
  author        = {Ding, Zixin and Emami, Shaghayegh and Salvi, Giovanna and Tosciri, Cecilia and Gandrakota, Abhijith and Ngadiuba, Jennifer and Tran, Nhan and Herwig, Christian and Miller, David W. and Chen, Yuxin},
  year          = {2026},
  eprint        = {2606.23993},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2606.23993}
}
```
