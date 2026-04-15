#!/bin/bash
#
# Step 2 — Pre-training: trains on pretrain set, saves checkpoint at the end.
# No validation / no early stopping.
#
set -e
cd "$(dirname "$0")/.."

PYTHONPATH=src:$PYTHONPATH uv run python src/pretrain.py \
    --output_dir=saved_models \
    --dataset=vcldata \
    --to_checkpoint=pretrain_vul_model \
    --pretrain_code_model_name=microsoft/codebert-base \
    --pretrain_text_model_name=roberta-base \
    --epochs 30 \
    --code_length 512 \
    --hidden_size 768 \
    --train_batch_size 256 \
    --eval_batch_size 512 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --num_workers 8 \
    --seed 123456 \
    --do_train 2>&1 | tee saved_models/pretrain.log
