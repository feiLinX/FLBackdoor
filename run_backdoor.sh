#!/usr/bin/env bash
# Sweep model architectures, Byzantine-client counts, and backdoor partitions
# using FedAvg with prediction KL distance and model poisoning.
set -euo pipefail

PYTHON=/scratch/jmh8504/envs/jz/bin/python
SCRIPT="$(dirname "$0")/main.py"

MODELS=(resnet34)
NBYZ_VALUES=(2 4)
PARTITIONS=(0.1 0.3 0.5)

# model-poisoning hyper-parameters, applied only on the model_poison=true runs
MP_STEALTH_LAMBDA=1e-3   # weight of the ||w - w_global||^2 stealth regularizer
MP_SCALE=2.0             # constrain-and-scale boost (capped at benign median norm)

for model in "${MODELS[@]}"; do
    for nbyz in "${NBYZ_VALUES[@]}"; do
        for bd_partition in "${PARTITIONS[@]}"; do

            if [[ "$nbyz" == 2 && "$model" == resnet18 ]]; then
                    continue
            fi

            if [[ "$nbyz" == 4 && "$bd_partition" == 0.1 && "$model" == resnet18 ]]; then
                    continue
            fi

            echo "Running model=${model}, aggregation=fedavg, nbyz=${nbyz}, bd_distance=pred_kl, model_poison=true, bd_partition=${bd_partition}"
            "$PYTHON" "$SCRIPT" \
                --model "$model" \
                --dataset digits \
                --aggregation fedavg \
                --nrounds 70 \
                --adv_type CDLS \
                --nbyz "$nbyz" \
                --bd_target_label 0 \
                --bd_partition "$bd_partition" \
                --bd_distance pred_kl \
                --bd_model_poison

        done
    done
done
