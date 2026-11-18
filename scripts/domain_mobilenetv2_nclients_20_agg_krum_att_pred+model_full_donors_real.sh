#!/usr/bin/env bash
# DomainNet | model=MobileNetV2 | nclients=20 | aggregation=krum (multi-krum, krum_m=15)
# CDLS option: CDLS with pred_kl donor selection + full model poisoning (stealth reg + constrain-and-scale)
# Adversary domains: victim clipart + 1 donor domain (real); SimCLR pretrained on clipart, real
set -euo pipefail

PYTHON="${PYTHON:-python}"
SCRIPT="$(dirname "$0")/../main.py"

"$PYTHON" "$SCRIPT" \
    --dataset domain \
    --model mobilenetv2 \
    --lr 1e-2 \
    --wd 4e-5 \
    --aggregation krum \
    --nclients 20 \
    --nrounds 80 \
    --epochs 5 \
    --krum_m 15 \
    --batch_size 64 \
    --aug_mult 10 \
    --auto_aug \
    --adv_type CDLS \
    --nbyz 2 \
    --bd_target_label 0 \
    --bd_partition 0.3 \
    --bd_donor_domains real \
    --bd_distance pred_kl \
    --bd_model_poison \
    --bd_stealth_lambda 1e-3 \
    --bd_scale 2.0 \
    --log_file_name domain_mobilenetv2_nclients_20_agg_krum_att_pred+model_full_donors_real
