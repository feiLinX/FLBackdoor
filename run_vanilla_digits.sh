#!/usr/bin/env bash
# Test the Vanilla (Bagdasaryan et al. 2020) pattern-trigger backdoor with
# model replacement on Digits-5 using MobileNetV2, across robust aggregations
# {fedavg, multi-krum (krum_m=15), foolsgold, deepsight, flame, bnguard}.
set -euo pipefail

PYTHON=/scratch/jmh8504/envs/jz/bin/python
SCRIPT="$(dirname "$0")/main.py"

MODEL=mobilenetv2
AGGREGATIONS=(fedavg krum foolsgold deepsight flame bnguard)
KRUM_M=10   # multi-krum survivors (ignored by other aggregations)

# Vanilla uses uncapped model replacement (boosted by the update-scaling factor)
MP_SCALE=2.0

for aggregation in "${AGGREGATIONS[@]}"; do

    echo "Running dataset=digits, model=${MODEL}, aggregation=${aggregation}, krum_m=${KRUM_M}, adv=Vanilla, model_replacement=true"
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
        --adv_type Vanilla \
        --bd_target_label 0 \
        --bd_model_poison \
        --bd_scale "$MP_SCALE"

done
