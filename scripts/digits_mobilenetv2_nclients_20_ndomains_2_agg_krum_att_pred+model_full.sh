#!/usr/bin/env bash
# Digits-5 | model=MobileNetV2 | nclients=20 | num_domains (k)=2 | nbyz=2 | bd_partition=0.3 | aggregation=krum
# CDLS option: CDLS pred_kl + full model poisoning (stealth reg lambda=1e-3 + constrain-and-scale scale=2.0)
# Runs all C(5,2)=10 client-domain combinations in sequence; reports each combination's final ACC / ASR and their mean
set -euo pipefail

PYTHON="${PYTHON:-python}"
SCRIPT="$(dirname "$0")/../main.py"

"$PYTHON" "$SCRIPT" \
    --dataset digits \
    --model mobilenetv2 \
    --lr 5e-4 \
    --wd 1e-5 \
    --aggregation krum \
    --nclients 20 \
    --nrounds 35 \
    --epochs 5 \
    --krum_m 15 \
    --num_domains 2 \
    --adv_type CDLS \
    --nbyz 2 \
    --bd_target_label 0 \
    --bd_partition 0.3 \
    --bd_distance pred_kl \
    --bd_model_poison \
    --bd_stealth_lambda 1e-3 \
    --bd_scale 2.0 \
    --log_file_name digits_mobilenetv2_nclients_20_ndomains_2_agg_krum_att_pred+model_full
