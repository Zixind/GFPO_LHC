#!/bin/bash
# Experiment 1: Train on MC, adapt on MC
# ─────────────────────────────────────────────────────────────────────────────
# All methods (RL + SPOT + AnomalyTransformer) calibrate/train on MC chunks
# and are then evaluated on the remaining MC chunks.
#
# RL agents   : train online via the rollout script (train=True inside DQN/GRPO/etc.)
# SPOT        : calibrates GPD on first --spot-n-calib MC chunks, DSPOT on rest
# AT          : trains transformer on first --at-n-calib MC chunks, thresholds on rest
#
# Calibration is saved to outputs/calib_mc/ for use in Experiment 2.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

OUTDIR="outputs/exp1_mc_train_mc_adapt"
MODELS_DIR="outputs/demo_sing_grpo_as_feature_all_MC/models_mc"
CALIB_SAVE="outputs/calib_mc"

echo "============================================================"
echo "Experiment 1: MC-train + MC-adapt"
echo "Output  : ${OUTDIR}"
echo "Models  : ${MODELS_DIR}"
echo "CalibSave: ${CALIB_SAVE}"
echo "============================================================"

conda run -n adaptive python RL/demo_single_trigger_grpo_as_feature_all_rollout_v2.py \
    --input      Data/Trigger_food_MC.h5 \
    --control    MC \
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
    --save-calib-dir "${CALIB_SAVE}" \
    --run-ht \
    "$@"

echo ""
echo "Done. Results in ${OUTDIR}"
echo "Calibration saved to ${CALIB_SAVE} (use in Experiment 2)"
