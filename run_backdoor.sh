#!/usr/bin/env bash
# Sweep model architectures and robust-aggregation defenses on digits.
# For each (model, aggregation) pair run two configs:
#   1. clean baseline  -> report clean test accuracy
#   2. CDLS backdoor with pred_kl distance + model poisoning
set -euo pipefail

PYTHON=/scratch/jmh8504/envs/jz/bin/python
SCRIPT="$(dirname "$0")/main.py"

MODELS=(mobilenetv2 resnet18)
AGGREGATIONS=(foolsgold deepsight bnguard)

# model-poisoning hyper-parameters, applied only on the model_poison=true runs
MP_STEALTH_LAMBDA=1e-3   # weight of the ||w - w_global||^2 stealth regularizer
MP_SCALE=2.0             # constrain-and-scale boost (capped at benign median norm)

for model in "${MODELS[@]}"; do
    for aggregation in "${AGGREGATIONS[@]}"; do

        echo "Running model=${model}, aggregation=${aggregation}, clean baseline (clean test accuracy)"
        "$PYTHON" "$SCRIPT" \
            --model "$model" \
            --dataset digits \
            --aggregation "$aggregation" \
            --nrounds 70 \
            --adv_type None \
            --bd_clean_baseline

        echo "Running model=${model}, aggregation=${aggregation}, adv=CDLS, bd_distance=pred_kl, model_poison=true"
        "$PYTHON" "$SCRIPT" \
            --model "$model" \
            --dataset digits \
            --aggregation "$aggregation" \
            --nbyz 2 \
            --bd_partition 0.5 \
            --nrounds 70 \
            --adv_type CDLS \
            --bd_target_label 0 \
            --bd_distance pred_kl \
            --bd_model_poison \
            --bd_stealth_lambda "$MP_STEALTH_LAMBDA" \
            --bd_scale "$MP_SCALE"

    done
done
