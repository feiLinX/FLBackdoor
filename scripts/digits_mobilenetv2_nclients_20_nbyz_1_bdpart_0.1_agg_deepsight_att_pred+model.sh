#!/usr/bin/env bash
# Digits-5 | model=MobileNetV2 | nclients=20 | nbyz=1 | bd_partition=0.1 | aggregation=deepsight
# CDLS option: CDLS pred_kl + full model poisoning (stealth reg lambda=1e-3 + constrain-and-scale scale=2.0)
set -euo pipefail

PYTHON="${PYTHON:-python}"
SCRIPT="$(dirname "$0")/../main.py"

"$PYTHON" "$SCRIPT" \
    --dataset digits \
    --model mobilenetv2 \
    --lr 5e-4 \
    --wd 1e-5 \
    --aggregation deepsight \
    --nclients 20 \
    --nrounds 35 \
    --epochs 5 \
    --krum_m 15 \
    --adv_type CDLS \
    --nbyz 1 \
    --bd_target_label 0 \
    --bd_partition 0.1 \
    --bd_distance pred_kl \
    --bd_model_poison \
    --bd_stealth_lambda 1e-3 \
    --bd_scale 2.0 \
    --log_file_name digits_mobilenetv2_nclients_20_nbyz_1_bdpart_0.1_agg_deepsight_att_pred+model
