#!/usr/bin/env bash
# Test the SoDa (OOD + self-reference) backdoor on Digits-5 using MobileNetV2,
# across robust aggregations {fedavg, multi-krum (krum_m=15), foolsgold,
# deepsight, flame, bnguard}. SoDa draws poison images from an OOD dataset and
# applies its self-reference regularizer inside fedavg_local (no separate
# model-poisoning step).
set -euo pipefail

PYTHON=/scratch/jmh8504/envs/jz/bin/python
SCRIPT="$(dirname "$0")/main.py"

MODEL=mobilenetv2
AGGREGATIONS=(fedavg krum foolsgold deepsight flame bnguard)
KRUM_M=10   # multi-krum survivors (ignored by other aggregations)

# SoDa self-reference regularizer weights
SODA_L2=0.1
SODA_COS=100.0

for aggregation in "${AGGREGATIONS[@]}"; do

    echo "Running dataset=digits, model=${MODEL}, aggregation=${aggregation}, krum_m=${KRUM_M}, adv=SoDa"
    "$PYTHON" "$SCRIPT" \
        --lr 5e-4 \
        --wd 1e-5 \
        --model "$MODEL" \
        --dataset digits \
        --aggregation "$aggregation" \
        --krum_m "$KRUM_M" \
        --nbyz 2 \
        --bd_partition 0.3 \
        --nrounds 70 \
        --adv_type SoDa \
        --bd_target_label 0 \
        --soda_l2 "$SODA_L2" \
        --soda_cos "$SODA_COS"

done
