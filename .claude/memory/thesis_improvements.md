---
name: thesis_improvements
description: 5 planned improvements for CLeVeR thesis defense with ablation switches
type: project
originSessionId: 6712eff8-1101-44f7-a540-bfd44b3bdbfc
lastUpdated: 2026-04-17
---

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

LLM-generated descriptions may look fluent but could be **worse** than original generic descriptions at representing the actual code semantics.

### Proposed Method: UniXcoder Code-Reasoning Similarity

Use **UniXcoder** to encode both code and description, then compare cosine similarity.

**Quality Gate:** `new_sim > old_sim + 0.05`

---

## 5 Improvements

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
- `dataset/analyze_symmetry.py` written

### CURRENT TASK (Step 0)
```bash
uv run python dataset/analyze_symmetry.py --n-samples 50000 --seed 220703
```

### REMAINING TASKS
1. Implement LLM Enhancement with Quality Gate
2. Split enhanced dataset
3. Implement all 5 improvements + ablations
