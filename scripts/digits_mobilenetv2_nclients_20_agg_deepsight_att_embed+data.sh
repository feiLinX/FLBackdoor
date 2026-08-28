#!/usr/bin/env bash
# Digits-5 | model=MobileNetV2 | nclients=20 | aggregation=deepsight
# CDLS option: CDLS with embed_kl (SimCLR encoder) donor selection + data poisoning only
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
    --nbyz 2 \
    --bd_target_label 0 \
    --bd_partition 0.3 \
    --bd_distance embed_kl \
    --log_file_name digits_mobilenetv2_nclients_20_agg_deepsight_att_embed+data
