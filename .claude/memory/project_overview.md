---
name: project_overview
description: CLeVeR multi-modal contrastive learning for vulnerability detection (ACL 2025 Findings)
type: project
originSessionId: 6712eff8-1101-44f7-a540-bfd44b3bdbfc
lastUpdated: 2026-04-17
---

# CLeVeR Project Overview

## Paper
**CLeVeR: Multi-modal Contrastive Learning for Vulnerability Code Representation**
- Published: ACL 2025 Findings
- GitHub: https://github.com/yoimiya-nlp/CLeVeR

## Key Results (from paper)
| Dataset | Task | F1 |
|---|---|---|
| VCLData-ft | Detection | 96.53% |
| SynData | Detection | 98.03% |
| RealData | Detection | 72.82% |
| VCLData-ft | Classification | 80.34% (Weighted-F1) |
| RealData | Zero-shot Detection | 65.89% |

## Architecture (model.py)

### Components
1. **ContrastiveModel**: Main model with `code_encoder` (CodeEncoder) + `desc_encoder` (DescriptionEncoder)
2. **CodeEncoder**: RobertaModel + CodeAdapter + CrossAttention + DescriptionClassifier
   - CodeAdapter: self-attention + FFN + LayerNorm (refines CLS token only, not full sequence)
   - CrossAttention: MHA where description is query, code is key/value
   - DescriptionClassifier (FFN): acts as Description Simulator
3. **DescriptionEncoder**: RobertaModel + DescriptionAdapter (FFN + LN)
4. **LinearProbe**: `nn.Linear(768, num_classes)` — frozen backbone + trainable classifier

### Key Hyperparameters
- `alpha = 0.7` — weight for description loss in pre-training
- `lamda_0 = 0.25` — source/sink weight in description fusion
- `lambda_1 = 0.2` — CLS vs refined representation weight
- `temperature = 0.1` — for InfoNCE loss

### Pre-training Loss
```
L_pre-train = L_info + alpha * L_desc
```

### Description Representation
```
D = 0.25 * source_cls + 0.25 * sink_cls + 0.5 * reason_cls
```

### Vulnerability Code Representation
```
V = 0.2 * cls + 0.8 * refined_representation
```

## Dataset: VCLData
- **File**: `dataset/vcldata.jsonl` (280,034 samples, 283MB)
- Fields: `func`, `name`, `label`, `cwe_id`, `source`, `sink`, `reason`
- **Label semantics**: `label` = ground truth (0=safe, 1=vulnerable)
- **CWE semantics**: `cwe_id` on safe samples = what was fixed (metadata, NOT active vuln)
- **Unlabeled vuln**: `cwe_id="None"` AND `label=1` = vulnerable but no specific CWE type

## CWE Frequency Threshold
- `CWE_FREQUENCY_THRESHOLD = 50`
- **89 specific CWE types** pass the threshold
- **22 rare CWE types** excluded from train/val/test but included in pretrain

## Class Index Layout (91 total)
```
0          : safe (cwe_id=None/"None", label=0)
1          : vulnerable but not CWE-labeled (cwe_id="None", label=1) — "unlabeled vuln"
2 .. 90    : specific CWE types (CWE_LIST[i] → class i+2)
```

### CWE2INT Index Offset = 2
- CWE_LIST[0] (CWE-114) → class index 2
- Index 0 = safe, Index 1 = unlabeled-vuln, Index 2-90 = specific CWEs

## MITRE CWE Names
- `src/cwe_names.json` — full MITRE CWE v4.14 catalog (963 entries)

## FLAW Comments in Source Code
- SARD source code contains `/* FLAW: text */` or `/* POTENTIAL FLAW: text */` comments
- ~52.5% of samples have FLAW comments
- FLAW in SAFE (label=0) samples describes the BAD counterpart, NOT the current safe function
- FLAW in VULNERABLE (label=1) samples describes the actual vulnerability

## Scripts

### Preprocessing
```bash
sh scripts/preprocess.sh
```

### Pre-training
```bash
sh scripts/train.sh
```

### Fine-tuning via Linear Probe
```bash
sh scripts/linear_probing.sh
```

### Zero-shot Evaluation
```bash
sh scripts/zeroshot.sh
```

## Known Implementation Gaps vs Paper

1. **CrossAttention**: Paper uses C[1:] (all tokens except CLS) as key/value. Implementation uses full code_hidden_states.
2. **CodeAdapter**: Paper applies MHA over full code sequence. Implementation applies self-attention to CLS token only.
3. **No early stopping**: Training runs all epochs even when F1 plateaus.
4. **Linear probe learning rate**: Paper doesn't specify; implementation uses 2e-4 (10x pre-training LR).
