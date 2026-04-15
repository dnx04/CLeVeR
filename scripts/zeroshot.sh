#!/bin/bash
#
# Step 4 — Zero-shot evaluation: pre-trained model only, no fine-tuning.
# --do_test: binary detection (security vs vulnerability)
# --do_test_cls: 10-class CWE classification
#
set -e
cd "$(dirname "$0")/.."

PYTHONPATH=src:$PYTHONPATH uv run python src/zeroshot.py \
    --output_dir=saved_models \
    --dataset=vcldata \
    --from_checkpoint=pretrain_vul_model \
    --pretrain_code_model_name=microsoft/codebert-base \
    --pretrain_text_model_name=roberta-base \
    --code_length 512 \
    --hidden_size 768 \
    --eval_batch_size 512 \
    --seed 123456 \
    --do_test \
    --do_test_cls 2>&1 | tee saved_models/zeroshot.log
