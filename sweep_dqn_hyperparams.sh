#!/bin/bash
# DQN hyperparameter sweep to maximize h->4b signal efficiency
# Key params: alpha (signal mix), lambda_1 (rate vs signal), beta (move penalty)
set -e

SCRIPT="RL/DQN_Ht_AS_feature.py"
BASE_OUT="outputs/dqn_sweep"

# Configs: (name, alpha, beta, lambda_1)
# Baseline: alpha=0.4, beta=0.2, lambda_1=0.25
# Strategy: lower alpha = more h->4b focus, lower lambda_1 = more signal focus

declare -a CONFIGS=(
    "baseline,0.4,0.2,0.25"
    "alpha02,0.2,0.2,0.25"
    "alpha01,0.1,0.2,0.25"
    "alpha02_lam15,0.2,0.2,0.15"
    "alpha01_lam15,0.1,0.2,0.15"
    "alpha02_beta01,0.2,0.1,0.25"
    "alpha01_beta01,0.1,0.1,0.25"
    "alpha02_lam15_beta01,0.2,0.1,0.15"
)

for cfg in "${CONFIGS[@]}"; do
    IFS=',' read -r name alpha beta lam1 <<< "$cfg"
    outdir="${BASE_OUT}/${name}"
    echo "========================================="
    echo "Running: $name (alpha=$alpha, beta=$beta, lambda_1=$lam1)"
    echo "Output: $outdir"
    echo "========================================="
    conda run -n adaptive python "$SCRIPT" \
        --outdir "$outdir" \
        --alpha "$alpha" \
        --beta "$beta" \
        --lambda_1 "$lam1" \
        2>&1 | tail -5
    echo ""
done

echo "All done. Compare sht/sas plots across ${BASE_OUT}/*/."
