#!/usr/bin/env bash
# DomainNet | model=MobileNetV2 | nclients=20 | aggregation=fedavg
# CDLS option: none (clean baseline, CDLS disabled)
set -euo pipefail

PYTHON="${PYTHON:-python}"
SCRIPT="$(dirname "$0")/../main.py"

"$PYTHON" "$SCRIPT" \
    --dataset domain \
    --model mobilenetv2 \
    --lr 1e-2 \
    --wd 5e-4 \
    --aggregation fedavg \
    --nclients 20 \
    --nrounds 30 \
    --epochs 5 \
    --krum_m 15 \
    --batch_size 64 \
    --aug_mult 10 \
    --auto_aug \
    --adv_type None \
    --log_file_name domain_mobilenetv2_nclients_20_agg_fedavg_att_none
