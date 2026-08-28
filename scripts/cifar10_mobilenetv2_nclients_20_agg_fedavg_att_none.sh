#!/usr/bin/env bash
# CIFAR-10 | model=MobileNetV2 | nclients=20 | aggregation=fedavg
# CDLS option: none (clean baseline, CDLS disabled)
set -euo pipefail

PYTHON="${PYTHON:-python}"
SCRIPT="$(dirname "$0")/../main.py"

"$PYTHON" "$SCRIPT" \
    --dataset cifar10 \
    --model mobilenetv2 \
    --lr 5e-2 \
    --wd 5e-4 \
    --aggregation fedavg \
    --nclients 20 \
    --nrounds 35 \
    --epochs 5 \
    --krum_m 15 \
    --adv_type None \
    --log_file_name cifar10_mobilenetv2_nclients_20_agg_fedavg_att_none
