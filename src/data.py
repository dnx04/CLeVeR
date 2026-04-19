import os
import logging
import pickle
import json
import torch
from torch.utils.data import Dataset

from class_def import CWE_LIST, CWE2INT, NUM_CLASSES, DATASET_NAME

logger = logging.getLogger(__name__)


def get_dataset_path(dataset_name, split):
    """Returns path to dataset pickle file at dataset/<name>/<name>_<split>.pkl"""
    return f"dataset/{dataset_name}/{dataset_name}_{split}.pkl"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def is_safe_cwe(cwe_id):
    """Returns True if cwe_id represents a safe (non-vulnerable) sample."""
    return cwe_id is None or str(cwe_id) == "None"


_CWE_NAMES = json.load(open("src/cwe_names.json"))


def get_cwe_description(cwe_id):
    """Get a description template for a given CWE ID."""
    if cwe_id in _CWE_NAMES:
        return f"A vulnerability of {_CWE_NAMES[cwe_id]}."
    return f"A vulnerability of CWE-{cwe_id}."


# Unified description list for zero-shot: CWE type descriptions only (no safe class)
# Prediction: if max_sim < threshold -> safe (label 0), else -> CWE type
UNIFIED_DESCS = [get_cwe_description(cwe) for cwe in CWE_LIST]


def _load_examples(args, flag):
    """Load examples from pickle, filtering rare CWEs for non-pretrain splits."""
    path = get_dataset_path(args.dataset, flag)
    with open(path, "rb") as f:
        examples = pickle.load(f)

    if "pretrain" not in flag:
        # Keep: safe (cwe_id=None/"None", label=0) OR
        #        specific CWE (cwe_id in CWE2INT)
        def keep_example(ex):
            cwe_str = str(ex.cwe_id)
            lbl = int(ex.label)
            if is_safe_cwe(ex.cwe_id) or lbl == 0:
                return True  # safe
            return cwe_str in CWE2INT  # specific CWE

        examples = [ex for ex in examples if keep_example(ex)]
    return examples


# ---------------------------------------------------------------------------
# Dataset classes
# ---------------------------------------------------------------------------


class TrainData(Dataset):
    def __init__(self, code_tokenizer, text_tokenizer, args, flag=""):
        self.examples = []
        self.args = args
        self.code_tokenizer = code_tokenizer
        self.text_tokenizer = text_tokenizer
        self.code_max_length = 512
        self.text_max_length = 64

        self.examples = _load_examples(args, flag)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        func = self.examples[item].func
        description = self.examples[item].description
        source = self.examples[item].source
        sink = self.examples[item].sink

        func_input = self.code_tokenizer(
            func,
            max_length=self.code_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        description_input = self.text_tokenizer(
            description,
            max_length=self.text_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        source_input = self.text_tokenizer(
            source,
            max_length=self.text_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        sink_input = self.text_tokenizer(
            sink,
            max_length=self.text_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return (
            func_input["input_ids"].squeeze(0),
            func_input["attention_mask"].squeeze(0),
            description_input["input_ids"].squeeze(0),
            description_input["attention_mask"].squeeze(0),
            source_input["input_ids"].squeeze(0),
            source_input["attention_mask"].squeeze(0),
            sink_input["input_ids"].squeeze(0),
            sink_input["attention_mask"].squeeze(0),
        )


class UnifiedData(Dataset):
    """
    Unified dataset for both zero-shot evaluation and linear probing.
    - Zero-shot: uses description tokens + cosine similarity
    - Linear probing: ignores description tokens, uses only code + label

    Class index layout (NUM_CLASSES total):
      0  : safe (cwe_id=None/"None", label=0)
      1+ : specific CWE types (cwe_id in CWE2INT)
    """

    def __init__(self, code_tokenizer, text_tokenizer, args, flag=""):
        self.examples = _load_examples(args, flag)
        self.code_tokenizer = code_tokenizer
        self.text_tokenizer = text_tokenizer
        self.code_max_length = 512
        self.text_max_length = 64

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        func = self.examples[item].func
        cwe_id = self.examples[item].cwe_id
        label = int(self.examples[item].label)
        cwe_str = str(cwe_id)

        # Safe samples: label = -1 (not a CWE class)
        # CWE samples: label = CWE2INT[cwe_str] (0-97, where 0=CWE_LIST[0]=CWE-114)
        if is_safe_cwe(cwe_id) or label == 0:
            unified_label = -1  # special marker for safe (not a CWE class)
        else:
            unified_label = CWE2INT.get(cwe_str, -1)  # -1 if unknown CWE

        func_input = self.code_tokenizer(
            func,
            max_length=self.code_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Zero-shot needs all description tokens; linear probing ignores them
        desc_ids_list, desc_mask_list = [], []
        for desc in UNIFIED_DESCS:
            d = self.text_tokenizer(
                desc,
                max_length=self.text_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            desc_ids_list.append(d["input_ids"].squeeze(0))
            desc_mask_list.append(d["attention_mask"].squeeze(0))

        return (
            func_input["input_ids"].squeeze(0),
            func_input["attention_mask"].squeeze(0),
            torch.stack(desc_ids_list),
            torch.stack(desc_mask_list),
            torch.tensor(unified_label),
        )
