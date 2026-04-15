# coding=utf-8
"""CLeVeR zero-shot evaluation — detection and classification in a single pass per sample."""

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
# Pre-computed description texts and CWE list
# ---------------------------------------------------------------------------
DETECTION_DESCS = [
    "This is a security function.",
    "This is a vulnerability function.",
]

CWE_LIST = ["78", "121", "122", "129", "190", "284", "390", "400", "416", "476"]
CLASSIFICATION_DESCS = [
    "A vulnerability of OS Command Injection.",
    "A vulnerability of Stack-based Buffer Overflow.",
    "A vulnerability of Heap-based Buffer Overflow.",
    "A vulnerability of Improper Validation of Array Index.",
    "A vulnerability of Integer Overflow.",
    "A vulnerability of Improper Access Control.",
    "A vulnerability of Detection of Error Condition Without Action.",
    "A vulnerability of Uncontrolled Resource Consumption.",
    "A vulnerability of Use After Free.",
    "A vulnerability of NULL Pointer Dereference.",
]


def pre_encode_descriptions(model, text_tokenizer, device, max_length=64):
    """
    Pre-encode all description texts through the description encoder + adapter.
    Returns a tensor of shape (12, hidden) — 2 detection + 10 classification.
    """
    all_descs = DETECTION_DESCS + CLASSIFICATION_DESCS
    inputs = text_tokenizer(
        all_descs,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    # Encode all 12 descriptions in one batched forward pass through desc_encoder
    with torch.no_grad():
        # desc_encoder = RobertaModel + DescriptionAdapter
        outputs = model.desc_encoder.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_tokens = outputs.last_hidden_state[:, 0, :]          # (12, hidden)
        desc_embeddings = model.desc_encoder.description_adapter(cls_tokens)  # (12, hidden)

    return desc_embeddings  # (12, hidden)


def encode_code(model, func_input_ids, func_attention_mask, device):
    """
    Encode a batch of code functions through the code encoder.
    Returns a tensor of shape (batch, hidden).
    """
    with torch.no_grad():
        outputs = model.code_encoder.encoder(
            input_ids=func_input_ids.to(device),
            attention_mask=func_attention_mask.to(device)
        )
        cls = outputs.last_hidden_state[:, 0, :]               # (batch, hidden)
        cls = model.code_encoder.code_adapter(cls)              # (batch, hidden)
        # description_classifier(projection) refines the CLS for comparison with desc embeddings
        code_refined = model.code_encoder.description_classifier(cls)  # (batch, hidden)
        code_repr = 0.2 * cls + 0.8 * code_refined           # (batch, hidden)
    return code_repr


def evaluate_both(args, model, code_tokenizer, text_tokenizer, flag='test'):
    """
    Unified zero-shot evaluation: detection AND classification in one pass per sample.

    Pre-computes all 12 description embeddings once.
    For each sample: encode code once, then compute cosine similarity against all 12 descriptions.
    Results are split: first 2 similarities -> detection, last 10 -> classification.
    """
    from data import DetectionTestData

    # ---------------------------------------------------------------------------
    # 1. Pre-encode all 12 description embeddings (done once, ~negligible time)
    # ---------------------------------------------------------------------------
    all_desc_embeddings = pre_encode_descriptions(model, text_tokenizer, args.device)
    # all_desc_embeddings shape: (12, hidden)
    # detection_descs[0:2], classification_descs[2:12]

    # ---------------------------------------------------------------------------
    # 2. Load test data using DetectionTestData (has both binary label and CWE label)
    # ---------------------------------------------------------------------------
    test_dataset = DetectionTestData(code_tokenizer, text_tokenizer, args, flag=flag)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(
        test_dataset, sampler=test_sampler, batch_size=args.eval_batch_size,
        num_workers=args.num_workers, pin_memory=True
    )

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    model.eval()

    d_preds, d_trues = [], []
    c_preds, c_trues = [], []

    for batch in tqdm(test_dataloader, desc="Zero-shot (both)"):
        (func_input_ids, func_attention_mask,
         _, _, _, _, label) = [x.to(args.device) for x in batch]

        # Encode code once per sample
        code_repr = encode_code(model, func_input_ids, func_attention_mask, args.device)
        # code_repr: (batch, hidden)

        # Compute cosine similarity against all 12 descriptions at once
        # all_desc_embeddings: (12, hidden) -> unsqueeze to (1, 12, hidden)
        # code_repr: (batch, hidden) -> unsqueeze to (batch, 1, hidden)
        sims = torch.nn.functional.cosine_similarity(
            code_repr.unsqueeze(1),          # (batch, 1, hidden)
            all_desc_embeddings.unsqueeze(0), # (1, 12, hidden)
            dim=-1                            # reduce over hidden dim
        )  # shape: (batch, 12)

        # Detection: argmax over first 2 similarities
        detect_sim = sims[:, :2]              # (batch, 2)
        dp = detect_sim.argmax(dim=-1)        # (batch,)

        # Classification: argmax over last 10 similarities
        class_sim = sims[:, 2:]               # (batch, 10)
        cp = class_sim.argmax(dim=-1)         # (batch,)

        d_preds.append(dp.cpu().numpy())
        d_trues.append(label.cpu().numpy())
        c_preds.append(cp.cpu().numpy())
        c_trues.append(label.cpu().numpy())

    d_preds = np.concatenate(d_preds, 0)
    d_trues = np.concatenate(d_trues, 0)
    c_preds = np.concatenate(c_preds, 0)
    c_trues = np.concatenate(c_trues, 0)

    from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

    detection_result = {
        "accuracy": accuracy_score(d_trues, d_preds),
        "recall": recall_score(d_trues, d_preds),
        "precision": precision_score(d_trues, d_preds),
        "f1": f1_score(d_trues, d_preds),
    }

    classification_result = {
        "accuracy": accuracy_score(c_trues, c_preds),
        "recall": recall_score(c_trues, c_preds, average='weighted'),
        "precision": precision_score(c_trues, c_preds, average='weighted'),
        "f1": f1_score(c_trues, c_preds, average='weighted'),
    }

    logger.info("***** Zero-shot detection results (%s) *****", flag)
    for key in sorted(detection_result.keys()):
        logger.info("  %s = %s", key, str(round(detection_result[key], 4)))

    logger.info("***** Zero-shot classification results (%s) *****", flag)
    for key in sorted(classification_result.keys()):
        logger.info("  %s = %s", key, str(round(classification_result[key], 4)))

    return {"detection": detection_result, "classification": classification_result}


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", default=None, type=str, required=True)
    parser.add_argument("--dataset", default=None, type=str, required=True)
    parser.add_argument("--from_checkpoint", default=None, type=str,
                        help="Pre-trained model checkpoint to load")
    parser.add_argument("--to_checkpoint", default=None, type=str)
    parser.add_argument("--pretrain_text_model_name", default="roberta-base", type=str)
    parser.add_argument("--pretrain_code_model_name", default="microsoft/codebert-base", type=str)
    parser.add_argument("--code_length", default=512, type=int)
    parser.add_argument("--hidden_size", default=768, type=int)
    parser.add_argument("--eval_batch_size", default=256, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--do_test", action='store_true', help="Run zero-shot detection")
    parser.add_argument("--do_test_cls", action='store_true', help="Run zero-shot classification")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    logger.warning("device: %s, n_gpu: %s", device, args.n_gpu)

    set_seed(args)

    code_tokenizer = RobertaTokenizer.from_pretrained(args.pretrain_code_model_name)
    text_tokenizer = RobertaTokenizer.from_pretrained(args.pretrain_text_model_name)

    model = ContrastiveModel(args)
    if args.from_checkpoint:
        ckpt_path = os.path.join(args.output_dir, args.from_checkpoint, 'model.bin')
        model.load_state_dict(torch.load(ckpt_path))
        logger.info("Loaded model from %s", ckpt_path)
    model.to(device)

    # Unified evaluation: both tasks in one pass
    results = evaluate_both(args, model, code_tokenizer, text_tokenizer, flag='test')

    if args.do_test:
        print("Detection results:", results['detection'])
    if args.do_test_cls:
        print("Classification results:", results['classification'])

    return results


if __name__ == "__main__":
    main()
