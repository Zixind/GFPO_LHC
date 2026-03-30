#!/bin/bash
# Experiment 3: Train on CMS Run, roll out on CMS Run
# ─────────────────────────────────────────────────────────────────────────────
# SPOT and AnomalyTransformer calibrate on the first --*-n-calib CMS chunks,
# then adapt on the remaining CMS chunks.  No MC data is used for calibration.
#
# RL agents   : still loaded from MC-trained models (RL never trained on CMS).
# SPOT        : calibrates GPD on first 50 CMS chunks, DSPOT on rest.
# AT          : trains transformer on first 50 CMS chunks, thresholds on rest.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

OUTDIR="outputs/exp3_cms_train_cms_adapt"
MODELS_DIR="outputs/demo_sing_grpo_as_feature_all_MC/models_mc"

echo "============================================================"
echo "Experiment 3: CMS-train + CMS-adapt  (SPOT / AT only)"
echo "Output  : ${OUTDIR}"
echo "Models  : ${MODELS_DIR}"
echo "============================================================"

conda run -n adaptive python RL/demo_single_trigger_grpo_as_feature_all_rollout_v2.py \
    --input      Data/Matched_data_2016_dim2.h5 \
    --control    RealData \
    --outdir     "${OUTDIR}" \
    --models-dir "${MODELS_DIR}" \
    --baselines  "constant,pid,spot,anomaly_transformer,sac" \
    --spot-n-calib  50 \
    --spot-window    5 \
    --at-n-calib    50 \
    --at-win-size   64 \
    --at-d-model    64 \
    --at-n-heads     4 \
    --at-e-layers    2 \
    --at-d-ff      128 \
    --at-epochs      3 \
    --at-lr        1e-4 \
    --run-ht \
    "$@"

echo ""
echo "Done. Results in ${OUTDIR}"
