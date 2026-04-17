# coding=utf-8
"""CLeVeR linear probing script — freezes pre-trained backbone, trains linear probe on train set."""

from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.optim import AdamW
from tqdm import tqdm

from model import ContrastiveModel, LinearProbe
from data import NUM_CLASSES, CWE_LIST

logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


# ---------------------------------------------------------------------------
# DataLoader builders
# ---------------------------------------------------------------------------
def build_probe_dataloader(
    args, code_tokenizer, text_tokenizer, flag, batch_size, task
):
    """Build a DataLoader for probe data.
    task: 'detection' (binary) or 'classification' (unified CWE)
    Both use ClassificationProbeData which returns unified CWE labels (0=safe, 1-90=CWE).
    """
    from data import ClassificationProbeData

    dataset = ClassificationProbeData(code_tokenizer, text_tokenizer, args, flag=flag)

    sampler = (
        SequentialSampler(dataset)
        if flag in ("val", "test")
        else RandomSampler(dataset)
    )
    return DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def encode_code_batch(model, func_input_ids, func_attention_mask, device):
    """Encode a batch of code functions through the frozen backbone. Returns (batch, hidden)."""
    with torch.no_grad():
        outputs = model.code_encoder.encoder(
            input_ids=func_input_ids.to(device),
            attention_mask=func_attention_mask.to(device),
        )
        cls = outputs.last_hidden_state[:, 0, :]
        cls = model.code_encoder.code_adapter(cls)
        code_refined = model.code_encoder.description_classifier(cls)
        code_repr = 0.2 * cls + 0.8 * code_refined
    return code_repr


def extract_features(model, dataloader, device):
    """Extract code representations and labels from a dataloader."""
    model.eval()
    features, labels = [], []
    for batch in tqdm(dataloader, desc="Extracting features"):
        func_input_ids, func_attention_mask, label = [x.to(device) for x in batch]
        code_repr = encode_code_batch(
            model, func_input_ids, func_attention_mask, device
        )
        features.append(code_repr.cpu())
        labels.append(label.cpu())
    return torch.cat(features, dim=0), torch.cat(labels, dim=0)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_unified(classifier, features, labels, device):
    """
    Unified evaluation computing all three metric groups from a single prediction pass.

    Labels: 0 = safe, 1-90 = CWE types (unified CWE mapping)
    Predictions: argmax from a classifier head trained on NUM_CLASSES classes.

    Returns:
        multiclass_result: multi-class F1 over all 91 classes
        detection_result: binary F1 (safe vs vulnerable)
        classification_result: F1 on vulnerable samples only
    """
    classifier.eval()
    y_preds, y_trues = [], []

    dataset = torch.utils.data.TensorDataset(features, labels)
    loader = DataLoader(dataset, batch_size=512)

    with torch.no_grad():
        for feat, lbl in loader:
            feat, lbl = feat.to(device), lbl.to(device)
            outputs = classifier(feat)
            y_preds.append(outputs.argmax(dim=-1).cpu().numpy())
            y_trues.append(lbl.cpu().numpy())

    y_preds = np.concatenate(y_preds, 0)
    y_trues = np.concatenate(y_trues, 0)

    from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

    # Metric 1: Multi-class F1 over all NUM_CLASSES
    multiclass_result = {
        "accuracy": accuracy_score(y_trues, y_preds),
        "recall_macro": recall_score(
            y_trues, y_preds, average="macro", zero_division=0
        ),
        "precision_macro": precision_score(
            y_trues, y_preds, average="macro", zero_division=0
        ),
        "f1_macro": f1_score(y_trues, y_preds, average="macro", zero_division=0),
        "recall_weighted": recall_score(
            y_trues, y_preds, average="weighted", zero_division=0
        ),
        "precision_weighted": precision_score(
            y_trues, y_preds, average="weighted", zero_division=0
        ),
        "f1_weighted": f1_score(y_trues, y_preds, average="weighted", zero_division=0),
    }

    # Metric 2: Detection F1 — collapse to binary (safe=0, vulnerable=1)
    detect_trues = (y_trues != 0).astype(int)
    detect_preds = (y_preds != 0).astype(int)
    detection_result = {
        "accuracy": accuracy_score(detect_trues, detect_preds),
        "recall": recall_score(detect_trues, detect_preds),
        "precision": precision_score(detect_trues, detect_preds),
        "f1": f1_score(detect_trues, detect_preds),
    }

    # Metric 3: Classification F1 — only on vulnerable ground-truth samples
    vuln_mask = y_trues != 0
    if vuln_mask.sum() > 0:
        classification_result = {
            "accuracy": accuracy_score(y_trues[vuln_mask], y_preds[vuln_mask]),
            "recall_weighted": recall_score(
                y_trues[vuln_mask],
                y_preds[vuln_mask],
                average="weighted",
                zero_division=0,
            ),
            "precision_weighted": precision_score(
                y_trues[vuln_mask],
                y_preds[vuln_mask],
                average="weighted",
                zero_division=0,
            ),
            "f1_weighted": f1_score(
                y_trues[vuln_mask],
                y_preds[vuln_mask],
                average="weighted",
                zero_division=0,
            ),
            "num_vulnerable_samples": int(vuln_mask.sum()),
        }
    else:
        classification_result = {"num_vulnerable_samples": 0}

    return multiclass_result, detection_result, classification_result


def save_classifier(args, classifier):
    output_dir = os.path.join(args.output_dir, args.to_checkpoint)
    os.makedirs(output_dir, exist_ok=True)
    torch.save(classifier.state_dict(), os.path.join(output_dir, "classifier.bin"))
    logger.info("Saved classifier to %s/classifier.bin", output_dir)


def save_features(args, split, task, features, labels):
    """Cache extracted features to disk to avoid re-extraction on subsequent runs."""
    cache_dir = os.path.join(args.output_dir, args.to_checkpoint, "features")
    os.makedirs(cache_dir, exist_ok=True)
    torch.save(features, os.path.join(cache_dir, f"{split}_{task}_features.pt"))
    torch.save(labels, os.path.join(cache_dir, f"{split}_{task}_labels.pt"))
    logger.info("Cached %s features to %s", split, cache_dir)


def load_features(args, split, task):
    """Load cached features if available."""
    f_path = os.path.join(
        args.output_dir, args.to_checkpoint, "features", f"{split}_{task}_features.pt"
    )
    l_path = os.path.join(
        args.output_dir, args.to_checkpoint, "features", f"{split}_{task}_labels.pt"
    )
    if os.path.exists(f_path) and os.path.exists(l_path):
        logger.info("Loading cached %s features from %s", split, f_path)
        return torch.load(f_path), torch.load(l_path)
    return None, None


# ---------------------------------------------------------------------------
# Training loop with early stopping
# ---------------------------------------------------------------------------
def train_with_early_stopping(
    args,
    classifier,
    train_features,
    train_labels,
    val_features,
    val_labels,
    device,
    num_classes,
):
    """
    Train linear probe with early stopping on detection F1 (binary collapse of 91-class).
    Uses detection F1 as the early stopping signal since it reflects both detection
    and classification quality across all classes.
    """
    train_ds = torch.utils.data.TensorDataset(train_features, train_labels)
    train_loader = DataLoader(train_ds, batch_size=args.train_batch_size, shuffle=True)

    optimizer = AdamW(
        classifier.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    best_f1, epochs_no_improve, stop = 0.0, 0, False

    for epoch in range(args.epochs):
        classifier.train()
        total_loss = 0
        for feat, lbl in tqdm(train_loader, desc=f"Epoch {epoch}"):
            feat, lbl = feat.to(device), lbl.to(device)
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(classifier(feat), lbl)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), args.max_grad_norm)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = round(total_loss / len(train_loader), 5)
        _, val_detect_results, val_class_results = evaluate_unified(
            classifier, val_features, val_labels, device
        )

        # Use detection F1 as early stopping signal
        val_f1 = val_detect_results["f1"]
        logger.info(
            "Epoch %d — loss %.5f — val detection f1 %.4f — val classification f1_weighted %.4f",
            epoch,
            avg_loss,
            val_f1,
            val_class_results["f1_weighted"],
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            epochs_no_improve = 0
            logger.info("  Best val F1: %s — saving checkpoint", round(best_f1, 4))
            save_classifier(args, classifier)
        else:
            epochs_no_improve += 1
            logger.info("  No improvement (%d/%d)", epochs_no_improve, args.patience)
            if epochs_no_improve >= args.patience:
                logger.info("Early stopping at epoch %d", epoch)
                stop = True
                break

    logger.info("Training complete. Best val F1: %s", round(best_f1, 4))
    return best_f1, stop


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", default=None, type=str, required=True)
    parser.add_argument("--dataset", default=None, type=str, required=True)
    parser.add_argument("--pretrain_checkpoint", default=None, type=str)
    parser.add_argument(
        "--from_checkpoint",
        default=None,
        type=str,
        help="Load a previously saved classifier for evaluation",
    )
    parser.add_argument(
        "--to_checkpoint",
        default=None,
        type=str,
        help="Folder to save/load classifier checkpoint",
    )
    parser.add_argument("--pretrain_text_model_name", default="roberta-base", type=str)
    parser.add_argument(
        "--pretrain_code_model_name", default="microsoft/codebert-base", type=str
    )
    parser.add_argument("--code_length", default=512, type=int)
    parser.add_argument("--hidden_size", default=768, type=int)

    # Unified: single 91-class classifier handles both detection (binary collapse) and classification
    parser.add_argument("--do_train", action="store_true", help="Train linear probe")
    parser.add_argument(
        "--do_test",
        action="store_true",
        help="Evaluate on test set (both detection and classification)",
    )

    # Hyperparams
    parser.add_argument("--train_batch_size", default=256, type=int)
    parser.add_argument("--eval_batch_size", default=512, type=int)
    parser.add_argument("--learning_rate", default=2e-4, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)

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

    from transformers import RobertaTokenizer

    code_tokenizer = RobertaTokenizer.from_pretrained(args.pretrain_code_model_name)
    text_tokenizer = RobertaTokenizer.from_pretrained(args.pretrain_text_model_name)

    # Load frozen backbone
    backbone = ContrastiveModel(args)
    if args.pretrain_checkpoint:
        ckpt_path = os.path.join(args.output_dir, args.pretrain_checkpoint, "model.bin")
        backbone.load_state_dict(torch.load(ckpt_path))
        logger.info("Loaded backbone from %s", ckpt_path)
    backbone.to(device)
    for p in backbone.parameters():
        p.requires_grad = False
    backbone.eval()

    # ---------------------------------------------------------------------------
    # Always use unified CWE labels (0=safe, 1-90=CWE types)
    # A single 91-class classifier handles both detection and classification.
    # ---------------------------------------------------------------------------
    num_classes = NUM_CLASSES  # 90

    # Feature extraction and caching (unified, no task branching)
    train_features = val_features = test_features = None
    train_labels = val_labels = test_labels = None

    # Try to load cached features first
    for split, needed in [
        ("train", args.do_train),
        ("val", args.do_train),
        ("test", args.do_test),
    ]:
        if needed:
            f, lbl = load_features(args, split, "unified")
            if f is not None:
                if split == "train":
                    train_features, train_labels = f, lbl
                elif split == "val":
                    val_features, val_labels = f, lbl
                else:
                    test_features, test_labels = f, lbl

    # Extract missing features
    if args.do_train and train_features is None:
        loader = build_probe_dataloader(
            args,
            code_tokenizer,
            text_tokenizer,
            "train",
            args.train_batch_size,
            "unified",
        )
        train_features, train_labels = extract_features(backbone, loader, device)
        save_features(args, "train", "unified", train_features, train_labels)

    if args.do_train and val_features is None:
        loader = build_probe_dataloader(
            args, code_tokenizer, text_tokenizer, "val", args.eval_batch_size, "unified"
        )
        val_features, val_labels = extract_features(backbone, loader, device)
        save_features(args, "val", "unified", val_features, val_labels)

    if args.do_test and test_features is None:
        loader = build_probe_dataloader(
            args,
            code_tokenizer,
            text_tokenizer,
            "test",
            args.eval_batch_size,
            "unified",
        )
        test_features, test_labels = extract_features(backbone, loader, device)
        save_features(args, "test", "unified", test_features, test_labels)

    # ---------------------------------------------------------------------------
    # Training
    # ---------------------------------------------------------------------------
    if args.do_train:
        classifier = LinearProbe(input_dim=768, num_classes=num_classes)
        if args.from_checkpoint:
            ckpt_path = os.path.join(
                args.output_dir, args.from_checkpoint, "classifier.bin"
            )
            classifier.load_state_dict(torch.load(ckpt_path))
            logger.info("Loaded classifier from %s", ckpt_path)
        classifier.to(device)

        best_f1, _ = train_with_early_stopping(
            args,
            classifier,
            train_features,
            train_labels,
            val_features,
            val_labels,
            device,
            num_classes,
        )

        # If NOT evaluating test afterwards, we're done
        if not args.do_test:
            return

    # ---------------------------------------------------------------------------
    # Test evaluation: single 91-class classifier evaluates both detection and classification
    # ---------------------------------------------------------------------------
    if args.do_test:
        classifier = LinearProbe(input_dim=768, num_classes=num_classes)
        ckpt_path = os.path.join(args.output_dir, args.to_checkpoint, "classifier.bin")
        if os.path.exists(ckpt_path):
            classifier.load_state_dict(torch.load(ckpt_path))
            logger.info("Loaded classifier from %s", ckpt_path)
        else:
            logger.warning("No classifier found at %s — skipping test eval", ckpt_path)
            return
        classifier.to(device)
        classifier.eval()

        multiclass_result, detection_result, classification_result = evaluate_unified(
            classifier, test_features, test_labels, device
        )

        logger.info("***** Test results — num_classes=%d *****", NUM_CLASSES)
        logger.info("--- Multi-class (all %d classes) ---", NUM_CLASSES)
        for k, v in multiclass_result.items():
            logger.info("  %s = %s", k, str(round(v, 4)))
        print("Multiclass results:", multiclass_result)

        logger.info(
            "--- Detection (binary: safe vs vulnerable — collapsed from 91 classes) ---"
        )
        for k, v in detection_result.items():
            logger.info("  %s = %s", k, str(round(v, 4)))
        print("Detection results:", detection_result)

        logger.info(
            "--- Classification (CWE types — vulnerable ground-truth samples only) ---"
        )
        for k, v in classification_result.items():
            logger.info("  %s = %s", k, str(round(v, 4)))
        print("Classification results:", classification_result)

    logger.info("Done.")


if __name__ == "__main__":
    main()
