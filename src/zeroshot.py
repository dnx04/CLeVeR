# coding=utf-8
"""CLeVeR zero-shot evaluation — unified detection and classification in a single pass per sample."""

from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler
from transformers import RobertaTokenizer
from tqdm import tqdm

from model import ContrastiveModel

logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


# ---------------------------------------------------------------------------
# Import unified descriptions and CWE mapping from data.py
# UNIFIED_DESCS[0] = safe, UNIFIED_DESCS[1:] = CWE type descriptions
# CWE_LIST = list of CWE IDs in index order (CWE_LIST[i] -> class i+1)
# ---------------------------------------------------------------------------
from data import UNIFIED_DESCS, CWE_LIST, CWE2INT, NUM_CLASSES


def pre_encode_descriptions(model, text_tokenizer, device, max_length=64):
    """
    Pre-encode all unified description texts through the description encoder + adapter.
    Returns a tensor of shape (NUM_CLASSES, hidden) — 1 safe + len(CWE_LIST) CWE types.
    """
    inputs = text_tokenizer(
        UNIFIED_DESCS,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Encode all descriptions in one batched forward pass through desc_encoder
    with torch.no_grad():
        outputs = model.desc_encoder.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )
        cls_tokens = outputs.last_hidden_state[:, 0, :]  # (NUM_CLASSES, hidden)
        desc_embeddings = model.desc_encoder.description_adapter(
            cls_tokens
        )  # (NUM_CLASSES, hidden)

    return desc_embeddings  # (NUM_CLASSES, hidden)


def encode_code(model, func_input_ids, func_attention_mask, device):
    """
    Encode a batch of code functions through the code encoder.
    Returns a tensor of shape (batch, hidden).
    """
    with torch.no_grad():
        outputs = model.code_encoder.encoder(
            input_ids=func_input_ids.to(device),
            attention_mask=func_attention_mask.to(device),
        )
        cls = outputs.last_hidden_state[:, 0, :]  # (batch, hidden)
        cls = model.code_encoder.code_adapter(cls)  # (batch, hidden)
        # description_classifier(projection) refines the CLS for comparison with desc embeddings
        code_refined = model.code_encoder.description_classifier(cls)  # (batch, hidden)
        code_repr = 0.2 * cls + 0.8 * code_refined  # (batch, hidden)
    return code_repr


def evaluate_both(args, model, code_tokenizer, text_tokenizer, flag="test"):
    """
    Unified zero-shot evaluation: single 90-class prediction per sample (0=safe, 1-89=CWE types).

    Pre-computes all NUM_CLASSES description embeddings once.
    For each sample: encode code once, compute cosine similarity against all descriptions.
    Prediction = argmax over all classes.

    Metrics derived:
      - Unified F1: standard multi-class F1 over all 90 classes
      - Detection F1: binary F1 (collapse to safe=0, vulnerable=1)
      - Classification F1: CWE-type F1 only on vulnerable ground-truth samples
    """
    from data import UnifiedData

    # ---------------------------------------------------------------------------
    # 1. Pre-encode all unified description embeddings (done once)
    # ---------------------------------------------------------------------------
    all_desc_embeddings = pre_encode_descriptions(model, text_tokenizer, args.device)
    # all_desc_embeddings shape: (NUM_CLASSES, hidden)

    # ---------------------------------------------------------------------------
    # 2. Load test data using UnifiedData (unified CWE label: 0=safe, 1-90=CWE)
    # ---------------------------------------------------------------------------
    test_dataset = UnifiedData(code_tokenizer, text_tokenizer, args, flag=flag)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(
        test_dataset,
        sampler=test_sampler,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    model.eval()

    all_preds = []
    all_trues = []
    all_sims = []  # for top-k and confidence analysis

    for batch in tqdm(test_dataloader, desc="Zero-shot unified"):
        (func_input_ids, func_attention_mask, _, _, label) = [
            x.to(args.device) for x in batch
        ]

        # Encode code once per sample
        code_repr = encode_code(model, func_input_ids, func_attention_mask, args.device)
        # code_repr: (batch, hidden)

        # Compute cosine similarity against all unified descriptions
        # all_desc_embeddings: (NUM_CLASSES, hidden) -> unsqueeze to (1, NUM_CLASSES, hidden)
        # code_repr: (batch, hidden) -> unsqueeze to (batch, 1, hidden)
        sims = torch.nn.functional.cosine_similarity(
            code_repr.unsqueeze(1),  # (batch, 1, hidden)
            all_desc_embeddings.unsqueeze(0),  # (1, NUM_CLASSES, hidden)
            dim=-1,  # reduce over hidden dim
        )  # shape: (batch, NUM_CLASSES)

        # Unified prediction: argmax over all classes
        preds = sims.argmax(dim=-1)  # (batch,) — class index 0-90

        all_preds.append(preds.cpu().numpy())
        all_trues.append(label.cpu().numpy())
        all_sims.append(sims.cpu().numpy())

    all_preds = np.concatenate(all_preds, 0)
    all_trues = np.concatenate(all_trues, 0)
    all_sims = np.concatenate(all_sims, 0)  # (N, NUM_CLASSES)

    from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

    # ---------------------------------------------------------------------------
    # Metric 1: Multi-class F1 over all 91 classes (macro and weighted)
    # ---------------------------------------------------------------------------
    multiclass_result = {
        "accuracy": accuracy_score(all_trues, all_preds),
        "recall_macro": recall_score(
            all_trues, all_preds, average="macro", zero_division=0
        ),
        "precision_macro": precision_score(
            all_trues, all_preds, average="macro", zero_division=0
        ),
        "f1_macro": f1_score(all_trues, all_preds, average="macro", zero_division=0),
        "recall_weighted": recall_score(
            all_trues, all_preds, average="weighted", zero_division=0
        ),
        "precision_weighted": precision_score(
            all_trues, all_preds, average="weighted", zero_division=0
        ),
        "f1_weighted": f1_score(
            all_trues, all_preds, average="weighted", zero_division=0
        ),
    }

    # ---------------------------------------------------------------------------
    # Metric 2: Detection F1 — collapse to binary (safe=0 vs vulnerable=1)
    # ---------------------------------------------------------------------------
    detect_trues = (all_trues != 0).astype(int)  # 0 if safe, 1 if vulnerable
    detect_preds = (all_preds != 0).astype(
        int
    )  # 0 if predicted safe, 1 if predicted CWE
    detection_result = {
        "accuracy": accuracy_score(detect_trues, detect_preds),
        "recall": recall_score(detect_trues, detect_preds),
        "precision": precision_score(detect_trues, detect_preds),
        "f1": f1_score(detect_trues, detect_preds),
    }

    # ---------------------------------------------------------------------------
    # Metric 3: Classification F1 — only on vulnerable ground-truth samples
    # Ground-truth safe samples (label=0) are excluded.
    # A correct prediction requires: both truth AND prediction are CWE types
    # (not None/safe), and the CWE type matches.
    # ---------------------------------------------------------------------------
    vuln_mask = all_trues != 0  # only vulnerable ground-truth samples
    if vuln_mask.sum() > 0:
        classification_result = {
            "accuracy": accuracy_score(all_trues[vuln_mask], all_preds[vuln_mask]),
            "recall_weighted": recall_score(
                all_trues[vuln_mask],
                all_preds[vuln_mask],
                average="weighted",
                zero_division=0,
            ),
            "precision_weighted": precision_score(
                all_trues[vuln_mask],
                all_preds[vuln_mask],
                average="weighted",
                zero_division=0,
            ),
            "f1_weighted": f1_score(
                all_trues[vuln_mask],
                all_preds[vuln_mask],
                average="weighted",
                zero_division=0,
            ),
            "num_vulnerable_samples": int(vuln_mask.sum()),
        }
    else:
        classification_result = {"num_vulnerable_samples": 0}

    logger.info(
        "***** Zero-shot unified (%s) — num_classes=%d *****", flag, len(UNIFIED_DESCS)
    )
    logger.info(
        "  Total samples: %d, Vulnerable samples: %d",
        len(all_trues),
        int(vuln_mask.sum()),
    )

    logger.info("***** Multi-class results (all %d classes) *****", NUM_CLASSES)
    for key in sorted(multiclass_result.keys()):
        logger.info("  %s = %s", key, str(round(multiclass_result[key], 4)))

    logger.info("***** Detection results (binary: safe vs vulnerable) *****")
    for key in sorted(detection_result.keys()):
        logger.info("  %s = %s", key, str(round(detection_result[key], 4)))

    logger.info("***** Classification results (CWE types, vulnerable-only) *****")
    for key in sorted(classification_result.keys()):
        logger.info("  %s = %s", key, str(round(classification_result[key], 4)))

    # ---------------------------------------------------------------------------
    # Statistics for loss function improvement
    # ---------------------------------------------------------------------------
    from sklearn.metrics import confusion_matrix

    # Per-CWE stats (for loss function design: which classes need more attention)
    cwe_stats = {}
    for cwe in set(all_trues):
        mask = all_trues == cwe
        if mask.sum() == 0:
            continue
        cwe_preds = all_preds[mask]
        correct = (cwe_preds == cwe).sum()
        total = mask.sum()
        cwe_stats[int(cwe)] = {"total": int(total), "correct": int(correct), "acc": correct / total if total > 0 else 0}

    # Most confused CWE pairs (for pairwise loss / curriculum learning)
    conf_matrix = confusion_matrix(all_trues, all_preds, labels=list(range(NUM_CLASSES)))
    confused_pairs = []
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            if i != j and conf_matrix[i, j] >= 3:
                confused_pairs.append((i, j, int(conf_matrix[i, j])))
    confused_pairs.sort(key=lambda x: -x[2])

    # Detection confusion (safe vs vulnerable)
    detect_cm = confusion_matrix(detect_trues, detect_preds)
    tn, fp, fn, tp = detect_cm.ravel() if detect_cm.size == 4 else (0, 0, 0, 0)

    # Top-k accuracy (for confidence-based loss weighting)
    top_k_accs = {}
    for k in [1, 3, 5]:
        top_k_correct = 0
        top_k_total = 0
        for idx, true_cwe in enumerate(all_trues):
            top_k_preds = np.argsort(all_sims[idx])[-k:]
            if true_cwe in top_k_preds:
                top_k_correct += 1
            top_k_total += 1
        top_k_accs[k] = top_k_correct / top_k_total if top_k_total > 0 else 0

    stats_result = {
        "per_cwe_stats": cwe_stats,
        "top_confused_pairs": confused_pairs[:20],
        "detection_cm": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "top_k_accuracy": {k: round(v, 4) for k, v in top_k_accs.items()},
        "num_classes": NUM_CLASSES,
        "num_safe_test": int((all_trues == 0).sum()),
        "num_vuln_test": int((all_trues != 0).sum()),
    }

    logger.info("***** Statistics for loss function design *****")
    logger.info("  Safe test samples: %d, Vulnerable test samples: %d", stats_result["num_safe_test"], stats_result["num_vuln_test"])
    logger.info("  Detection TN=%d FP=%d FN=%d TP=%d", tn, fp, fn, tp)
    if confused_pairs:
        logger.info("  Top confused pairs (true->pred, count):")
        for true_cwe, pred_cwe, count in confused_pairs[:10]:
            logger.info("    CWE-%d -> CWE-%d: %d misclassifications", true_cwe, pred_cwe, count)
    print("Statistics:", stats_result)

    # Always print all results (unified evaluation gives all metrics at once)
    print("Multiclass results:", multiclass_result)
    print("Detection results:", detection_result)
    print("Classification results:", classification_result)

    return {
        "multiclass": multiclass_result,
        "detection": detection_result,
        "classification": classification_result,
        "stats": stats_result,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--from_pretrain_checkpoint",
        default=None,
        type=str,
        required=True,
        help="Path to the pretrained model checkpoint to evaluate",
    )
    parser.add_argument("--pretrain_text_model_name", default="roberta-base", type=str)
    parser.add_argument(
        "--pretrain_code_model_name", default="microsoft/codebert-base", type=str
    )
    parser.add_argument("--code_length", default=512, type=int)
    parser.add_argument("--hidden_size", default=768, type=int)
    parser.add_argument("--eval_batch_size", default=256, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.warning("device: %s, n_gpu: %s", device, args.n_gpu)

    set_seed(args)

    code_tokenizer = RobertaTokenizer.from_pretrained(args.pretrain_code_model_name)
    text_tokenizer = RobertaTokenizer.from_pretrained(args.pretrain_text_model_name)

    model = ContrastiveModel(args)
    pretrain_ckpt_path = args.from_pretrain_checkpoint
    model.load_state_dict(torch.load(pretrain_ckpt_path))
    logger.info("Loaded pretrained model from %s", pretrain_ckpt_path)
    model.to(device)

    # Unified evaluation: both tasks in one pass
    results = evaluate_both(args, model, code_tokenizer, text_tokenizer, flag="test")

    return results


if __name__ == "__main__":
    main()
