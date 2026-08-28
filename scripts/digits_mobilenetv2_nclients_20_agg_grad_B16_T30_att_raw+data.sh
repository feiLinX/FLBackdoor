#!/usr/bin/env bash
# Digits-5 | model=MobileNetV2 | nclients=20 | aggregation=grad (def_num_recon=16, def_recon_iters=30)
# CDLS option: CDLS raw_kl donor selection + data poisoning only (no model poisoning)
set -euo pipefail

PYTHON="${PYTHON:-python}"
SCRIPT="$(dirname "$0")/../main.py"

"$PYTHON" "$SCRIPT" \
    --dataset digits \
    --model mobilenetv2 \
    --lr 5e-4 \
    --wd 1e-5 \
    --aggregation grad \
    --nclients 20 \
    --nrounds 35 \
    --epochs 5 \
    --krum_m 15 \
    --def_num_recon 16 \
    --def_recon_iters 30 \
    --adv_type CDLS \
    --nbyz 2 \
    --bd_target_label 0 \
    --bd_partition 0.3 \
    --bd_distance raw_kl \
    --log_file_name digits_mobilenetv2_nclients_20_agg_grad_B16_T30_att_raw+data
