# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CLeVeR is a multi-modal contrastive learning framework for vulnerability code representation, published at ACL 2025 Findings. It uses dual RobertaModel encoders (code + text) with adapters and cross-attention to learn aligned vulnerability representations under supervision of vulnerability descriptions.

## Commands

### Data Collection (Step 0)
```bash
cd dataset
python collect_data.py          # Collect vulnerability data from SARD
python extract_functions.py     # Extract and filter functions via regex
cd ..
```

### Preprocessing (Step 1)
```bash
sh preprocess.sh                # Runs data_preprocess.py, outputs dataset_train.pkl/test.pkl
```

### Pre-training (Step 2)
```bash
sh train.sh                     # Runs clever.py with --do_train --do_test
```

### Fine-tuning via Linear Probing (Step 3-1)
```bash
sh linear_probe_detect.sh       # Vulnerability detection (binary) or classification (10 CWE types)
```

### Zero-shot Inference (Step 3-2)
```bash
sh zero_shot_detect.sh          # Runs clever.py with --do_test or --do_test_cls
```

### Direct Python invocations (with key args)
```bash
# Preprocess
python data_preprocess.py --dataset=dataset/dataset.jsonl --dataset_name=vcldata \
    --pretrain_text_model_name=pretrain_text_model --pretrain_code_model_name=pretrain_code_model

# Pre-train
python clever.py --output_dir=saved_models --dataset=vcldata --do_train --do_test \
    --pretrain_text_model_name=... --pretrain_code_model_name=... \
    --from_checkpoint=... --to_checkpoint=...

# Linear probe (detection)
python linear_probe.py --do_train --do_test --dataset=vcltestdata \
    --pretrain_checkpoint=pretrain_vul_model --to_checkpoint=probe_...
```

## Architecture

### Model Components (model.py)
- **ContrastiveModel**: Main model with `code_encoder` (CodeEncoder) and `desc_encoder` (DescriptionEncoder)
- **CodeEncoder**: RobertaModel backbone + CodeAdapter + CrossAttention + DescriptionClassifier. Uses CLS token representation refined via cross-attention with description query
- **DescriptionEncoder**: RobertaModel backbone + DescriptionAdapter
- **CodeAdapter**: Self-attention + FFN + LayerNorm for code representation refinement
- **DescriptionAdapter**: FFN + LayerNorm for text representation refinement
- **CrossAttention**: Multi-head attention (8 heads) for code-description alignment
- **LinearProbe**: Simple FC layer for downstream classification (frozen backbone + trainable probe)

### Loss Function (model.py)
- `info_nce_loss`: Contrastive loss between code and description embeddings
- Training combines info_nce_loss + alpha * description_loss (alpha=0.7)

### Flag Semantics in ContrastiveModel.forward()
- `"train"`: Returns contrastive loss, expects all inputs (func, description, source, sink)
- `"test"`: Returns code+description embeddings for similarity comparison
- `"probe"`: Returns code representation only (for linear probing)

## Data Flow

1. Raw data: `dataset/dataset.jsonl` → JSON with fields: func, name, cwe_id, source, sink, reason, label
2. Preprocessed: `preprocessed_data/{dataset_name}_train.pkl`, `_test.pkl` → list of ExampleFeature
3. Datasets (dataset.py):
   - `TrainData`: Returns 8 tensors (func tokens, desc tokens, source tokens, sink tokens)
   - `DetectionTestData`: Binary detection with description_0="security", description_1="vulnerability"
   - `ClassificationTestData`: 10-class CWE classification with predefined description templates
   - `DetectionProbeData`/`ClassificationProbeData`: Code-only for linear probing

## CWE Types (Classification)
`["78", "121", "122", "129", "190", "284", "390", "400", "416", "476"]`
Maps to: OS Command Injection, Stack-based Buffer Overflow, Heap-based Buffer Overflow, Improper Validation of Array Index, Integer Overflow, Improper Access Control, Detection of Error Condition Without Action, Uncontrolled Resource Consumption, Use After Free, NULL Pointer Dereference

## Key Paths
- Preprocessed data: `preprocessed_data/`
- Saved models: `saved_models/`
- HuggingFace model dirs: `pretrain_code_model`, `pretrain_text_model`
- Raw dataset: `dataset/dataset.jsonl`
