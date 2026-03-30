#!/bin/bash
# Experiment 2: Train on MC, roll out on CMS Run
# ─────────────────────────────────────────────────────────────────────────────
# RL agents   : loaded from pre-trained MC models (train=False on CMS)
# SPOT        : calibration loaded from MC run (outputs/calib_mc/SPOT_AD.json)
# AT          : model loaded from MC run (outputs/calib_mc/AnomalyTransformer_AD.pt)
#
# PREREQUISITE: Run Experiment 1 first to produce calibration in outputs/calib_mc/
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

OUTDIR="outputs/exp2_mc_train_cms_adapt"
MODELS_DIR="outputs/demo_sing_grpo_as_feature_all_MC/models_mc"
CALIB_LOAD="outputs/calib_mc"

if [ ! -d "${CALIB_LOAD}" ]; then
    echo "ERROR: Calibration dir '${CALIB_LOAD}' not found."
    echo "       Run Experiment 1 first: bash run_exp1_mc_train_mc_adapt.sh"
    exit 1
fi

echo "============================================================"
echo "Experiment 2: MC-train + CMS-adapt"
echo "Output    : ${OUTDIR}"
echo "Models    : ${MODELS_DIR}"
echo "CalibLoad : ${CALIB_LOAD}"
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
    --load-calib-dir "${CALIB_LOAD}" \
    --run-ht \
    "$@"

echo ""
echo "Done. Results in ${OUTDIR}"
