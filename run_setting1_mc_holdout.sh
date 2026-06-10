#!/bin/bash
# =============================================================================
# SETTING 1 — Standard MC: train on the first 80% MC, evaluate on the last 20%.
# =============================================================================
# Protocol (no train/eval overlap; ONLY the held-out 20% is reported):
#   (a) train all trainable RL methods on chunks 0-147   (--max-chunks 148)
#   (b) train CPO separately (same 80% split) and copy its checkpoint in
#   (c) freeze, evaluate all baselines on the held-out 20% (--start-chunk 148
#       --eval-only): chunks 148-end
#   (d) DSPOT on the held-out 20% (in-window 5-chunk calibration)
#
# Reported source per seed:
#   outputs/mc_seed_<SEED>_eval_holdout_all_MC/tables/paper_table.csv  (11 baselines)
#   outputs/mc_seed_<SEED>_dspot_eval_all_MC/tables/paper_table.csv    (DSPOT)
#
# Usage:  bash run_setting1_mc_holdout.sh
# =============================================================================
set -euo pipefail

SEEDS=(42 123 456)
MC=Data/Trigger_food_MC.h5
TRAIN=RL/demo_single_trigger_grpo_as_feature_all_training.py
ROLLOUT=RL/demo_single_trigger_grpo_as_feature_all_rollout_v2.py

# Methods trained online by the training script (CPO trained separately below).
RL_TRAIN="adt,dqn,dqn_f,ppo,grpo,gfpo_f,gfpo_fr"
# All baselines reported on the held-out 20% (RL loaded from checkpoints).
EVAL_BASELINES="constant,pid,adt,dqn,dqn_f,ppo,grpo,lgrpo,gfpo_f,gfpo_fr,cpo"

COMMON="--ht-step 2.0 --alpha 0.3 --lambda_1 0.25 --group-size-sample 64 --group-size-keep 16"
CPO_FLAGS="--cpo-delta 0.03 --cpo-cost-limit 1.0 --cpo-cg-iters 10 --cpo-cg-damping 0.1 \
           --cpo-ls-steps 10 --cpo-ls-decay 0.8 --cpo-batch-min 128 --cpo-group-size 16 --cpo-lambda-1 0.25"

for SEED in "${SEEDS[@]}"; do
  echo "==================== SETTING 1 — seed ${SEED} ===================="

  # (a) train trainable RL methods on the FIRST 80% MC
  conda run -n adaptive python "${TRAIN}" \
      --input "${MC}" --control MC --outdir "outputs/mc_seed_${SEED}" --seed "${SEED}" \
      --run-ht --run-adt --baselines "${RL_TRAIN}" --max-chunks 148 \
      ${COMMON} --grpo-lr 1e-4 --dqn-lr 1e-4 --dqn-f-train-chunks 1 --dqn-f-eps 0.0 \
      --save-models

  # (b) train CPO on the same 80% split, then copy its checkpoints into the
  #     main models dir so the held-out eval can load every method at once.
  conda run -n adaptive python "${TRAIN}" \
      --input "${MC}" --control MC --outdir "outputs/mc_seed_${SEED}_cpo" --seed "${SEED}" \
      --run-ht --baselines "cpo" --max-chunks 148 ${COMMON} ${CPO_FLAGS} --save-models
  for T in AD HT; do
    mkdir -p "outputs/mc_seed_${SEED}_all_MC/models_mc/${T}"
    cp "outputs/mc_seed_${SEED}_cpo_all_MC/models_mc/${T}/CPO.pt" \
       "outputs/mc_seed_${SEED}_all_MC/models_mc/${T}/CPO.pt"
  done

  # (c) freeze, evaluate ALL baselines on the held-out 20% MC (chunks 148-end)
  conda run -n adaptive python "${ROLLOUT}" \
      --input "${MC}" --control MC --outdir "outputs/mc_seed_${SEED}_eval_holdout" \
      --models-dir "outputs/mc_seed_${SEED}_all_MC/models_mc" --load-models --eval-only \
      --seed "${SEED}" --run-ht --run-adt --baselines "${EVAL_BASELINES}" \
      --start-chunk 148 ${COMMON} ${CPO_FLAGS}

  # (d) DSPOT on the held-out 20% only (calibrate on first 5 held-out chunks)
  conda run -n adaptive python "${ROLLOUT}" \
      --input "${MC}" --control MC --outdir "outputs/mc_seed_${SEED}_dspot_eval" \
      --seed "${SEED}" --run-ht --baselines "spot" \
      --start-chunk 148 --spot-n-calib 5 --spot-window 5 --ht-step 2.0 --alpha 0.3
done

echo ""
echo "Setting 1 complete. Held-out 20% MC results:"
for SEED in "${SEEDS[@]}"; do
  echo "  outputs/mc_seed_${SEED}_eval_holdout_all_MC/tables/paper_table.csv"
  echo "  outputs/mc_seed_${SEED}_dspot_eval_all_MC/tables/paper_table.csv"
done
