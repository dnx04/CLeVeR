#!/bin/bash
#
# Step 1 — Data preprocessing: raw JSONL -> stratified train/val/test splits.
# Outputs to dataset/<dataset_name>/.
#
set -e
cd "$(dirname "$0")/.."

PYTHONPATH=src:$PYTHONPATH uv run python src/data_preprocess.py \
    --dataset=dataset/vcldata.pkl \
    --dataset_name=vcldata \
    --seed 220703
