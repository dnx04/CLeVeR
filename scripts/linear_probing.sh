#!/bin/bash
#
# Step 3 — Linear probing: freeze pre-trained backbone, train linear probe on train set.
# Early stopping on val set. Saves best classifier checkpoint.
#
# Usage:
#   ./linear_probing.sh detection      # binary vulnerability detection
#   ./linear_probing.sh classification  # 10-class CWE classification
#
set -e
cd "$(dirname "$0")/.."

TASK="${1:-detection}"

if [[ "$TASK" == "detection" ]]; then
    TASK_FLAG="--detection"
    PROBE_DIR="probe_detect"
elif [[ "$TASK" == "classification" ]]; then
    TASK_FLAG="--classification"
    PROBE_DIR="probe_classify"
else
    echo "Error: task must be 'detection' or 'classification', got '$TASK'"
    exit 1
fi

PYTHONPATH=src:$PYTHONPATH uv run python src/linear_probing.py \
    --output_dir=saved_models \
    --dataset=vcldata \
    --pretrain_checkpoint=pretrain_vul_model \
    --to_checkpoint="$PROBE_DIR" \
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
    --do_test \
    $TASK_FLAG 2>&1 | tee "saved_models/linear_probing_${TASK}.log"
