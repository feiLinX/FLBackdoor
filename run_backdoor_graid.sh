#!/usr/bin/env bash
# Test GRAID against the two trigger-backdoor model-poisoning baselines on Digits:
# PGD (Attack of the Tails, Wang et al. 2020) and Neurotoxin (Zhang et al. 2022),
# each x {without, with model poisoning}. GRAID inherits main.py's defaults
# (def_num_recon=32, def_recon_iters=100, def_recon_every=3, ...).
set -euo pipefail

PYTHON=/scratch/jmh8504/envs/jz/bin/python
SCRIPT="$(dirname "$0")/main.py"

MODELS=(mobilenetv2)
ATTACKS=(PGD Neurotoxin)
MODEL_POISON=(false true)

# ---- shared pattern-trigger backdoor settings (PGD & Neurotoxin) ----
NBYZ=2                  # number of malicious clients (the first nbyz clients)
BD_TARGET_LABEL=0       # trigger target class
BD_PARTITION=0.5        # fraction of a malicious client's non-target samples that get the trigger
BD_TRIGGER_SIZE=5       # side length (px) of the square bottom-right trigger patch
BD_TRIGGER_VALUE=255    # pixel value stamped for the trigger

# ---- PGD-specific (Attack of the Tails) ----
PGD_EPS=1.0             # L2 radius of the ball around w_global the malicious weights are projected into each step

# ---- Neurotoxin-specific ----
NEURO_MASK_RATIO=0.1    # fraction of top benign-gradient coordinates the attacker freezes

# ---- model-poisoning (constrain-and-scale), applied only on model_poison=true runs ----
MP_SCALE=2.0            # constrain-and-scale boost (capped at benign median update norm)

for model in "${MODELS[@]}"; do
    for attack in "${ATTACKS[@]}"; do
        for model_poison in "${MODEL_POISON[@]}"; do

            echo "Running dataset=digits, model=${model}, aggregation=graid, attack=${attack}, model_poison=${model_poison}"

            # --pgd_eps / --neuro_mask_ratio are both passed every run; each attack
            # simply ignores the one that does not apply to it.
            CMD=("$PYTHON" "$SCRIPT"
                --model "$model"
                --dataset digits
                --aggregation graid
                --nrounds 70
                --adv_type "$attack"
                --nbyz "$NBYZ"
                --bd_target_label "$BD_TARGET_LABEL"
                --bd_partition "$BD_PARTITION"
                --bd_trigger_size "$BD_TRIGGER_SIZE"
                --bd_trigger_value "$BD_TRIGGER_VALUE"
                --pgd_eps "$PGD_EPS"
                --neuro_mask_ratio "$NEURO_MASK_RATIO")

            if [[ "$model_poison" == true ]]; then
                CMD+=(--bd_model_poison
                      --bd_scale "$MP_SCALE")
            fi

            "${CMD[@]}"

        done
    done
done
