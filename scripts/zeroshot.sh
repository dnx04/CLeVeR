#!/bin/bash
#
# Step 4 — Zero-shot evaluation: pre-trained model only, no fine-tuning.
# Runs unified evaluation (detection + classification) in a single pass.
#
set -e
cd "$(dirname "$0")/.."

PYTHONPATH=src:$PYTHONPATH uv run python src/zeroshot.py \
    --from_pretrain_checkpoint=saved_models/pretrain_vul_model.bin \
    --pretrain_code_model_name=microsoft/codebert-base \
    --pretrain_text_model_name=roberta-base \
    --code_length 512 \
    --hidden_size 768 \
    --eval_batch_size 512 \
    --seed 123456 \
    2>&1 | tee saved_models/zeroshot.log
