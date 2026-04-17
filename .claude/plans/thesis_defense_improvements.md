# CLeVeR Improvements — Thesis Defense

## Context

**Dataset:** `dataset/vcldata.jsonl` (280,034 samples, SARD) — ORIGINAL data with generic SARD reasons
**IMPORTANT:** `dataset/vcldata.pkl` is DEPRECATED. Do not use it. All reason enhancement work on the pickle was abandoned. We are starting fresh from the JSONL.

**Problem 1:** Model uses CLS-only pooling → no line-level information → cannot localize vulnerabilities.
**Problem 2:** InfoNCE treats all negatives equally → suboptimal when 65.7% of descriptions are generic noise → addressed via Asymmetric Loss + Hard Negative Mining.
**Problem 3:** Multi-epoch training without knowledge transfer → later epochs forget early epochs → addressed via Progressive Self-Distillation with EMA teacher.
**Problem 4:** 65.7% of descriptions are generic/noisy → down-weighting loses signal → LLM-Augmented Description Generation (Gemma 4) + UniXcoder quality gate.

**Step 0 (CURRENT):** Run UniXcoder similarity analysis on vcldata.jsonl to establish baseline statistics.
**Critical gap:** No quantitative metric to verify that LLM-generated descriptions are actually better than the original generic ones. Need code-reasoning similarity comparison via UniXcoder.

---

## CRITICAL: LLM Description Quality Metric (Pre-requisite for all description work)

### The Problem

LLM-generated descriptions may look fluent but could be **worse** than original generic descriptions at representing the actual code semantics. Example from testing:
- LLM output: "The code performs a `strncat` operation into the `data` buffer..." → Gemma guessed CWE-120 (not in valid 89 CWEs)
- Original generic: "This function first connect_socket Read data..." (meaningful but generic)

Without a metric, we cannot determine whether to **accept or reject** LLM output.

### Proposed Method: UniXcoder Code-Reasoning Similarity

Use **UniXcoder** (a state-of-the-art code language model) to encode both code and description, then compare cosine similarity.

**Algorithm:**
```python
def sym(code_snippet: str, description: str) -> float:
    tokens = tokenizer(code, return_tensors="pt", padding=True, truncation=True, max_length=512)
    emb = model(**tokens).last_hidden_state[:, 0, :]
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return (code_emb * text_emb).sum(dim=-1).item()
```

### Quality Gate
```python
old_sim = sym(code, old_reason)
new_sim = sym(code, new_reason)
accept = new_sim > old_sim + 0.05
```

---

## Improvement 1: Token-Level Contrastive Learning with Line-Aware Aggregation

### Problem Statement

`CodeEncoder.forward()` (model.py:97-98) discards all token-level information.

### Proposed Method: Line-Aware Token Pooling

- Parse line boundaries from raw code
- Per-line representation via mean pooling of token embeddings
- Line-level contrastive alignment with FLAW comment supervision

### Ablation Switch
`--line_level`

---

## Improvement 2: Asymmetric Loss with Hard Negative Mining

### Problem Statement

InfoNCE treats all negatives equally — easy negatives dilute gradient from hard negatives.

### Proposed Method

- Asymmetric loss: down-weight easy negatives (gamma_neg < 1)
- Hard negative margin: penalize max negative similarity

### Ablation Switch
`--loss_type {nce,asl}`

---

## Improvement 3: Progressive Self-Distillation with EMA Teacher

### Problem Statement

Multi-epoch training without knowledge transfer → later epochs forget early epochs.

### Proposed Method

- EMA teacher: `θ_teacher = μ * θ_teacher + (1-μ) * θ_student`
- Per-step distillation loss: `L_total = (1-α) * L_contrastive + α * L_kd`

### Ablation Switch
`--self_distill --ema_decay 0.999`

---

## Improvement 4: LLM-Augmented Description Generation (Gemma 4)

### 3-Pass Pipeline

1. **Pass 1:** Generate description (Gemma)
2. **Pass 2:** Classify CWE ID (Gemma, unlabeled only)
3. **Pass 3:** UniXcoder quality gate — accept if `sym(code, new) > sym(code, old) + 0.05`

### Ablation Switch
`--use_augmented_descriptions {original,gemma_accepted,gemma_all}`

---

## Ablation 5: UniXcoder as Backbone Encoder

Replace `microsoft/codebert-base` with `microsoft/unixcoder-base` in CodeEncoder.

### Ablation Switch
`--code_encoder {codebert,unixcoder}`

---

## Summary

| # | Improvement | Ablation Switch |
|---|-----------|-----------------|
| 1 | Line-aware token pooling | `--line_level` |
| 2 | Asymmetric loss + hard negative margin | `--loss_type {nce,asl}` |
| 3 | Progressive Self-Distillation with EMA teacher | `--self_distill --ema_decay` |
| 4 | LLM-Augmented Description Generation + UniXcoder quality gate | `--use_augmented_descriptions {original,gemma_accepted,gemma_all}` |
| 5 | UniXcoder as Backbone Encoder | `--code_encoder {codebert,unixcoder}` |

---

## Current Pipeline Status (2026-04-17)

### DONE
- [x] vcldata.pkl ABANDONED — back to `dataset/vcldata.jsonl` (original SARD data)
- [x] `dataset/analyze_symmetry.py` written with `--n-samples` and `--seed 220703` args

### CURRENT TASK (Step 0)
Run on server (GPU required for speed):
```bash
uv run python dataset/analyze_symmetry.py --n-samples 50000 --seed 220703
```

### REMAINING TASKS

#### Step 1: Implement LLM Enhancement with Quality Gate
1. For each problematic sample (generic reason):
   - Pass 1: Gemma generates specific description
   - Pass 2: Gemma classifies CWE ID (unlabeled vuln only)
   - Pass 3: UniXcoder quality gate — accept if `sym(code, new) > sym(code, old) + 0.05`
2. Update `reason` field (and `cwe_id` for unlabeled vuln if valid)
3. Save enhanced dataset as new JSONL

#### Step 2: Split enhanced dataset
- Stratified split: 80% pretrain, 10% val, 10% test
- Exclude rare CWEs (<50 samples) from val/test
- Generate `dataset/vcldata/vcldata_{pretrain,train,val,test}.pkl`

#### Step 3: Implement all 5 improvements + ablations

### Files to Modify / Create

| File | Changes |
|------|---------|
| `dataset/llm_enhance_reasons.py` | 3-pass pipeline: generate → CWE-classify → UniXcoder quality gate |
| `src/data_preprocess.py` | Split logic for ≥50 CWE threshold |
| `src/model.py` | `asymmetric_contrastive_loss()`, `EMATeacher`, `LineEncoder`, `--code_encoder` |
| `src/data.py` | `LineLevelData`, dynamic `CWE2INT` mapping, `NUM_CLASSES=91` |
| `src/pretrain.py` | All ablation flags |
| `src/zeroshot.py` | Dynamic 90-CWE descriptions, `--code_encoder` |
| `src/linear_probing.py` | `num_classes=91`, unified evaluation metrics |
| New `src/line_detect.py` | Line-level evaluation script |