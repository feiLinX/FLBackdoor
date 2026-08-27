#!/usr/bin/env bash
# Digits-5 | model=MobileNetV2 | nclients=20 | aggregation=krum
# CDLS option: CDLS with pred_kl donor selection + data poisoning only (no model poisoning)
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
    --nrounds 30 \
    --epochs 5 \
    --krum_m 15 \
    --adv_type CDLS \
    --nbyz 2 \
    --bd_target_label 0 \
    --bd_partition 0.3 \
    --bd_distance pred_kl \
    --log_file_name digits_mobilenetv2_nclients_20_agg_krum_att_pred+data
