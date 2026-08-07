#!/usr/bin/env bash
# Test GRAID against all CDLS variants on Digits: the three donor-distance KLs
# {raw_kl, embed_kl, pred_kl} x {without, with model poisoning}. GRAID inherits
# main.py's defaults (def_num_recon=32, def_recon_iters=100, def_recon_every=3, ...).
set -euo pipefail

PYTHON=/scratch/jmh8504/envs/jz/bin/python
SCRIPT="$(dirname "$0")/main.py"

MODELS=(mobilenetv2)
DISTANCES=(raw_kl embed_kl pred_kl)
MODEL_POISON=(false true)

# model-poisoning hyper-parameters, applied only on the model_poison=true runs
MP_STEALTH_LAMBDA=1e-3   # weight of the ||w - w_global||^2 stealth regularizer
MP_SCALE=2.0             # constrain-and-scale boost (capped at benign median norm)

for model in "${MODELS[@]}"; do
    for distance in "${DISTANCES[@]}"; do
        for model_poison in "${MODEL_POISON[@]}"; do

            echo "Running dataset=digits, model=${model}, aggregation=graid, bd_distance=${distance}, model_poison=${model_poison}"

            CMD=("$PYTHON" "$SCRIPT"
                --model "$model"
                --dataset digits
                --aggregation graid
                --nrounds 70
                --adv_type CDLS
                --nbyz 2
                --bd_target_label 0
                --bd_distance "$distance")

            if [[ "$model_poison" == true ]]; then
                CMD+=(--bd_model_poison
                      --bd_stealth_lambda "$MP_STEALTH_LAMBDA"
                      --bd_scale "$MP_SCALE")
            fi

            "${CMD[@]}"

        done
    done
done
