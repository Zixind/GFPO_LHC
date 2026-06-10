#!/bin/bash
# =============================================================================
# SETTING 2 — Sim-to-real: train on FULL MC, freeze, deploy on CMS run data.
# =============================================================================
# Protocol (RL never trained on CMS; the frozen MC policy is rolled out on the
# real CMS run):
#   (a) train all trainable RL methods on the FULL MC trajectory (no --max-chunks)
#   (b) train CPO on full MC separately and copy its checkpoint in
#   (c) deploy frozen (--eval-only) on CMS Run 283408 real data, all baselines
#
# Reported source per seed:
#   outputs/cms_seed_<SEED>_all_RealData/tables/paper_table.csv
#
# Usage:  bash run_setting2_cms_deploy.sh
# =============================================================================
set -euo pipefail

SEEDS=(42 123 456)
MC=Data/Trigger_food_MC.h5
CMS=Data/Matched_data_2016_dim2.h5
TRAIN=RL/demo_single_trigger_grpo_as_feature_all_training.py
ROLLOUT=RL/demo_single_trigger_grpo_as_feature_all_rollout_v2.py

RL_TRAIN="adt,dqn,dqn_f,ppo,grpo,gfpo_f,gfpo_fr"
DEPLOY_BASELINES="constant,pid,adt,dqn,dqn_f,ppo,grpo,lgrpo,gfpo_f,gfpo_fr,cpo"

COMMON="--ht-step 2.0 --alpha 0.3 --lambda_1 0.25 --group-size-sample 64 --group-size-keep 16"
CPO_FLAGS="--cpo-delta 0.03 --cpo-cost-limit 1.0 --cpo-cg-iters 10 --cpo-cg-damping 0.1 \
           --cpo-ls-steps 10 --cpo-ls-decay 0.8 --cpo-batch-min 128 --cpo-group-size 16 --cpo-lambda-1 0.25"

for SEED in "${SEEDS[@]}"; do
  echo "==================== SETTING 2 — seed ${SEED} ===================="

  # (a) train trainable RL methods on the FULL MC trajectory (NO --max-chunks)
  conda run -n adaptive python "${TRAIN}" \
      --input "${MC}" --control MC --outdir "outputs/mc_seed_${SEED}_fulltrain" --seed "${SEED}" \
      --run-ht --run-adt --baselines "${RL_TRAIN}" \
      ${COMMON} --grpo-lr 1e-4 --dqn-lr 1e-4 --dqn-f-train-chunks 1 --dqn-f-eps 0.0 \
      --save-models

  # (b) train CPO on full MC, copy checkpoint into the full-train models dir
  conda run -n adaptive python "${TRAIN}" \
      --input "${MC}" --control MC --outdir "outputs/mc_seed_${SEED}_cpo_fulltrain" --seed "${SEED}" \
      --run-ht --baselines "cpo" ${COMMON} ${CPO_FLAGS} --save-models
  for T in AD HT; do
    mkdir -p "outputs/mc_seed_${SEED}_fulltrain_all_MC/models_mc/${T}"
    cp "outputs/mc_seed_${SEED}_cpo_fulltrain_all_MC/models_mc/${T}/CPO.pt" \
       "outputs/mc_seed_${SEED}_fulltrain_all_MC/models_mc/${T}/CPO.pt"
  done

  # (c) deploy the FROZEN full-MC policy on CMS real data (--eval-only, no --ttt)
  conda run -n adaptive python "${ROLLOUT}" \
      --input "${CMS}" --control RealData --outdir "outputs/cms_seed_${SEED}" \
      --models-dir "outputs/mc_seed_${SEED}_fulltrain_all_MC/models_mc" --load-models --eval-only \
      --seed "${SEED}" --run-ht --run-adt --baselines "${DEPLOY_BASELINES}" \
      ${COMMON} ${CPO_FLAGS}
done

echo ""
echo "Setting 2 complete. CMS sim-to-real (frozen) results:"
for SEED in "${SEEDS[@]}"; do
  echo "  outputs/cms_seed_${SEED}_all_RealData/tables/paper_table.csv"
done
