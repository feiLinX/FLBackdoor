#!/usr/bin/env bash
# Sweep the cross-domain label-swap (CDLS) backdoor over
# models x nbyz x bd_partition, mirroring the notebook cell.
set -euo pipefail

PYTHON=/scratch/jmh8504/envs/jz/bin/python
SCRIPT="$(dirname "$0")/main.py"

MODELS=(mobilenetv2 resnet34 resnet18)
NBYZ=(4 2)
PARTITIONS=(0.1 0.3 0.5 1.0)

for model in "${MODELS[@]}"; do
    for nbyz in "${NBYZ[@]}"; do
        for partition in "${PARTITIONS[@]}"; do
            echo "Running model=${model}, nbyz=${nbyz}, partition=${partition}"
            "$PYTHON" "$SCRIPT" \
                --model "$model" \
                --dataset digits \
                --adv_type CDLS \
                --nbyz "$nbyz" \
                --bd_target_label 0 \
                --bd_partition "$partition"
        done
    done
done
