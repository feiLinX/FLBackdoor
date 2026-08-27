#!/usr/bin/env bash
# Digits-5 | model=MobileNetV2 | nclients=20 | aggregation=flame
# CDLS option: none (clean baseline, CDLS disabled)
set -euo pipefail

PYTHON="${PYTHON:-python}"
SCRIPT="$(dirname "$0")/../main.py"

"$PYTHON" "$SCRIPT" \
    --dataset digits \
    --model mobilenetv2 \
    --lr 5e-4 \
    --wd 1e-5 \
    --aggregation flame \
    --nclients 20 \
    --nrounds 30 \
    --epochs 5 \
    --krum_m 15 \
    --adv_type None \
    --log_file_name digits_mobilenetv2_nclients_20_agg_flame_att_none
