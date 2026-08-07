#!/usr/bin/env bash
# Sweep DomainNet across aggregations {fedavg, multi-krum (krum_m=15), flame} and
# models {mobilenetv2, resnet50} under CDLS with prediction-KL donor selection and
# model poisoning. DomainNet uses 224x224 inputs, so batch_size=32, aug_mult=10
# (few raw images per client), and auto_aug are set for these runs.
set -euo pipefail

PYTHON=/scratch/jmh8504/envs/jz/bin/python
SCRIPT="$(dirname "$0")/main.py"

MODELS=(resnet50)
AGGREGATIONS=(fedavg krum flame)
KRUM_M=15   # multi-krum survivors (ignored by fedavg/flame)

for model in "${MODELS[@]}"; do
    for agg in "${AGGREGATIONS[@]}"; do

        echo "Running dataset=office, model=${model}, aggregation=${agg}, krum_m=${KRUM_M}, aug_mult=10, batch_size=64, auto_aug=true"
        "$PYTHON" "$SCRIPT" \
            --dataset office \
            --model "$model" \
            --aggregation "$agg" \
            --krum_m "$KRUM_M" \
            --nrounds 100 \
            --adv_type None \
            --aug_mult 10 \
            --batch_size 64 \
            --auto_aug \
            --adv_type None

    done
done
