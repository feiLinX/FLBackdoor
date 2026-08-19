#!/usr/bin/env bash
# Sweep DomainNet with resnet50 across robust aggregations under CDLS with
# prediction-KL donor selection and model poisoning. For each aggregation, sweep
# four (nbyz, bd_partition) settings. DomainNet uses 224x224 inputs, so
# batch_size=64, aug_mult=10 (few raw images per client), and auto_aug are set.
set -euo pipefail

PYTHON=/scratch/jmh8504/envs/jz/bin/python
SCRIPT="$(dirname "$0")/main.py"

MODELS=(mobilenetv2)
AGGREGATIONS=(krum flame foolsgold bnguard deepsight)
KRUM_M=15   # multi-krum survivors (ignored by other aggregations)

# (nbyz, bd_partition) settings to sweep
NBYZ_LIST=(2)
BD_PARTITION_LIST=(0.3)

# model-poisoning hyper-parameters
MP_STEALTH_LAMBDA=1e-3   # weight of the ||w - w_global||^2 stealth regularizer
MP_SCALE=2.0             # constrain-and-scale boost (capped at benign median norm)

for model in "${MODELS[@]}"; do
    for agg in "${AGGREGATIONS[@]}"; do
        for i in "${!NBYZ_LIST[@]}"; do

            nbyz="${NBYZ_LIST[$i]}"
            bd_partition="${BD_PARTITION_LIST[$i]}"

            echo "Running dataset=domain, model=${model}, aggregation=${agg}, krum_m=${KRUM_M}, adv=CDLS, bd_distance=pred_kl, model_poison=true, nbyz=${nbyz}, bd_partition=${bd_partition}, aug_mult=10, batch_size=64, auto_aug=true"
            "$PYTHON" "$SCRIPT" \
                --dataset domain \
                --model "$model" \
                --aggregation "$agg" \
                --krum_m "$KRUM_M" \
                --nbyz "$nbyz" \
                --bd_partition "$bd_partition" \
                --adv_type CDLS \
                --bd_target_label 0 \
                --bd_distance pred_kl \
                --bd_model_poison \
                --bd_stealth_lambda "$MP_STEALTH_LAMBDA" \
                --bd_scale "$MP_SCALE" \
                --aug_mult 10 \
                --batch_size 64 \
                --auto_aug

        done
    done
done
