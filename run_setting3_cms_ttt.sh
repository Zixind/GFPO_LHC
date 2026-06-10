#!/bin/bash
# =============================================================================
# SETTING 3 — Test-time training (TTT): train on FULL MC, then adapt online on
#             CMS run data.
# =============================================================================
# Protocol: load the FULL-MC-trained checkpoints from Setting 2 and roll out on
# CMS Run 283408 with --ttt (online policy updates during deployment), for all
# baselines.
#
# PREREQUISITE: run_setting2_cms_deploy.sh (produces the full-MC checkpoints
#               outputs/mc_seed_<SEED>_fulltrain_all_MC/models_mc/).
#
# Reported source per seed:
#   outputs/ttt_seed_<SEED>_all_RealData/tables/paper_table.csv
#
# Usage:  bash run_setting3_cms_ttt.sh
# =============================================================================
set -euo pipefail

SEEDS=(42 123 456)
CMS=Data/Matched_data_2016_dim2.h5
ROLLOUT=RL/demo_single_trigger_grpo_as_feature_all_rollout_v2.py

TTT_BASELINES="constant,pid,adt,dqn,dqn_f,ppo,grpo,lgrpo,gfpo_f,gfpo_fr,cpo"

COMMON="--ht-step 2.0 --alpha 0.3 --lambda_1 0.25 --group-size-sample 64 --group-size-keep 16"
CPO_FLAGS="--cpo-delta 0.03 --cpo-cost-limit 1.0 --cpo-cg-iters 10 --cpo-cg-damping 0.1 \
           --cpo-ls-steps 10 --cpo-ls-decay 0.8 --cpo-batch-min 128 --cpo-group-size 16 --cpo-lambda-1 0.25"

for SEED in "${SEEDS[@]}"; do
  MODELS="outputs/mc_seed_${SEED}_fulltrain_all_MC/models_mc"
  if [ ! -d "${MODELS}" ]; then
    echo "ERROR: full-MC checkpoints not found at ${MODELS}"
    echo "       Run run_setting2_cms_deploy.sh first."
    exit 1
  fi

  echo "==================== SETTING 3 (TTT) — seed ${SEED} ===================="

  # Load full-MC checkpoint, adapt online on CMS real data (--ttt)
  conda run -n adaptive python "${ROLLOUT}" \
      --input "${CMS}" --control RealData --outdir "outputs/ttt_seed_${SEED}" \
      --models-dir "${MODELS}" --load-models --ttt \
      --seed "${SEED}" --run-ht --run-adt --baselines "${TTT_BASELINES}" \
      ${COMMON} ${CPO_FLAGS}
done

echo ""
echo "Setting 3 complete. CMS test-time-training results:"
for SEED in "${SEEDS[@]}"; do
  echo "  outputs/ttt_seed_${SEED}_all_RealData/tables/paper_table.csv"
done
