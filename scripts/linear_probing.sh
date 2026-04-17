#!/bin/bash
#
# Step 3 — Linear probing: freeze pre-trained backbone, train single 91-class linear probe.
# Early stopping on val detection F1. Saves best classifier checkpoint.
# Evaluates: multi-class (all 91), detection (binary collapse), classification (vulnerable-only).
#
set -e
cd "$(dirname "$0")/.."

PYTHONPATH=src:$PYTHONPATH uv run python src/linear_probing.py \
    --output_dir=saved_models \
    --dataset=vcldata \
    --pretrain_checkpoint=pretrain_vul_model \
    --to_checkpoint=probe_unified \
    --pretrain_code_model_name=microsoft/codebert-base \
    --pretrain_text_model_name=roberta-base \
    --code_length 512 \
    --hidden_size 768 \
    --train_batch_size 256 \
    --eval_batch_size 512 \
    --learning_rate 2e-4 \
    --max_grad_norm 1.0 \
    --num_workers 8 \
    --epochs 30 \
    --patience 5 \
    --seed 123456 \
    --do_train \
    --do_test 2>&1 | tee "saved_models/linear_probing_unified.log"
