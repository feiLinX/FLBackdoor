#!/usr/bin/env bash
# Sweep robust-aggregation defenses on digits with mobilenetv2 under CDLS
# (pred_kl donor selection). For each aggregation, sweep model_poison={false,true}
# and bd_partition={0.1,0.5} with nbyz=1.
set -euo pipefail

PYTHON=/scratch/jmh8504/envs/jz/bin/python
SCRIPT="$(dirname "$0")/main.py"

MODEL=mobilenetv2
AGGREGATIONS=(fedavg krum flame foolsgold deepsight bnguard)
KRUM_M=15   # multi-krum survivors (ignored by other aggregations)

BD_PARTITION_LIST=(0.1 0.5)

# model-poisoning hyper-parameters, applied only on the model_poison=true runs
MP_STEALTH_LAMBDA=1e-3   # weight of the ||w - w_global||^2 stealth regularizer
MP_SCALE=2.0             # constrain-and-scale boost (capped at benign median norm)

for aggregation in "${AGGREGATIONS[@]}"; do
    for bd_partition in "${BD_PARTITION_LIST[@]}"; do

        echo "Running model=${MODEL}, aggregation=${aggregation}, krum_m=${KRUM_M}, adv=CDLS, bd_distance=pred_kl, model_poison=false, nbyz=1, bd_partition=${bd_partition}"
        "$PYTHON" "$SCRIPT" \
            --model "$MODEL" \
            --dataset digits \
            --aggregation "$aggregation" \
            --krum_m "$KRUM_M" \
            --nbyz 1 \
            --bd_partition "$bd_partition" \
            --nrounds 70 \
            --adv_type CDLS \
            --bd_target_label 0 \
            --bd_distance pred_kl

        echo "Running model=${MODEL}, aggregation=${aggregation}, krum_m=${KRUM_M}, adv=CDLS, bd_distance=pred_kl, model_poison=true, nbyz=1, bd_partition=${bd_partition}"
        "$PYTHON" "$SCRIPT" \
            --model "$MODEL" \
            --dataset digits \
            --aggregation "$aggregation" \
            --krum_m "$KRUM_M" \
            --nbyz 1 \
            --bd_partition "$bd_partition" \
            --nrounds 70 \
            --adv_type CDLS \
            --bd_target_label 0 \
            --bd_distance pred_kl \
            --bd_model_poison \
            --bd_stealth_lambda "$MP_STEALTH_LAMBDA" \
            --bd_scale "$MP_SCALE"

    done
done
