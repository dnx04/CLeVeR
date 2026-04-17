from torch.utils.data import Dataset
import logging
import pickle
import torch
import json
from collections import Counter
import os

logger = logging.getLogger(__name__)


def get_dataset_path(dataset_name, split):
    """Returns path to dataset pickle file at dataset/<name>/<name>_<split>.pkl"""
    return f"dataset/{dataset_name}/{dataset_name}_{split}.pkl"


# ---------------------------------------------------------------------------
# Dynamic CWE mapping: built before any Dataset classes to avoid forward refs
# Only includes CWEs with >= 50 samples in the full dataset
# ---------------------------------------------------------------------------
CWE_FREQUENCY_THRESHOLD = 50


def _build_cwe_mapping():
    """
    Scan all pickle files to find CWEs with >= 50 samples total.
    Returns (cwe_list, cwe2int) where cwe2int maps CWE string -> 2-based index
    (0 = safe/None, 1 = vulnerable without a labeled CWE type).

    Class index layout (91 total):
      0          : safe (cwe_id=None or cwe_id="None", label=0)
      1          : vulnerable but not CWE-labeled (cwe_id="None", label=1)
      2 .. 90    : specific CWE types (CWE_LIST[i] -> class i+2)
    """
    cwe_counter = Counter()
    for split in ["pretrain", "train", "val", "test"]:
        path = f"dataset/vcldata/vcldata_{split}.pkl"
        if not os.path.exists(path):
            continue
        with open(path, "rb") as f:
            data = pickle.load(f)
        for x in data:
            if x.cwe_id is not None:
                cwe_counter[str(x.cwe_id)] += 1

    valid_cwes = sorted(
        [
            cwe
            for cwe, cnt in cwe_counter.items()
            if cnt >= CWE_FREQUENCY_THRESHOLD and cwe != "None"
        ]
    )
    # CWE index 0 = safe, index 1 = unlabeled vuln, index 2+ = specific CWE
    cwe2int = {cwe: idx + 2 for idx, cwe in enumerate(valid_cwes)}
    return valid_cwes, cwe2int


CWE_LIST, CWE2INT = _build_cwe_mapping()
# 91 total: index 0 = safe, index 1 = unlabeled vuln, indices 2-90 = specific CWEs
NUM_CLASSES = len(CWE_LIST) + 2


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


# Unified description list for zero-shot: index 0 = safe, index 1 = unlabeled vuln, index 2+ = CWE types
SAFE_DESCRIPTION = "This is a secure function with no vulnerability."
UNLABELED_VULN_DESCRIPTION = "A vulnerable function with no specific CWE classification."
UNIFIED_DESCS = (
    [SAFE_DESCRIPTION, UNLABELED_VULN_DESCRIPTION]
    + [get_cwe_description(cwe) for cwe in CWE_LIST]
)


def _load_examples(args, flag):
    """Load examples from pickle, filtering rare CWEs for non-pretrain splits."""
    path = get_dataset_path(args.dataset, flag)
    with open(path, "rb") as f:
        examples = pickle.load(f)

    if "pretrain" not in flag:
        # Keep: safe (cwe_id=None/"None", label=0) OR
        #        unlabeled vuln (cwe_id="None", label=1) OR
        #        specific CWE (cwe_id in CWE2INT)
        def keep_example(ex):
            cwe_str = str(ex.cwe_id)
            lbl = int(ex.label)
            if is_safe_cwe(ex.cwe_id) or lbl == 0:
                return True  # safe
            if cwe_str == "None" and lbl == 1:
                return True  # unlabeled vuln
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


class DetectionProbeData(Dataset):
    """Used for feature extraction. Returns code tokens + binary label (0=safe, 1=vulnerable)."""

    def __init__(self, code_tokenizer, text_tokenizer, args, flag=""):
        self.examples = _load_examples(args, flag)
        self.code_tokenizer = code_tokenizer
        self.text_tokenizer = text_tokenizer
        self.code_max_length = 512

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        func = self.examples[item].func
        label = int(self.examples[item].label)
        func_input = self.code_tokenizer(
            func,
            max_length=self.code_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return (
            func_input["input_ids"].squeeze(0),
            func_input["attention_mask"].squeeze(0),
            torch.tensor(label),
        )


class ClassificationProbeData(Dataset):
    """Used for feature extraction. Returns code tokens + unified CWE label.

    Class index layout (91 total):
      0  : safe (cwe_id=None/"None", label=0)
      1  : unlabeled vuln (cwe_id="None", label=1)
      2+ : specific CWE types (cwe_id in CWE2INT)
    """

    def __init__(self, code_tokenizer, text_tokenizer, args, flag=""):
        self.examples = _load_examples(args, flag)
        self.code_tokenizer = code_tokenizer
        self.text_tokenizer = text_tokenizer
        self.code_max_length = 512

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        func = self.examples[item].func
        cwe_id = self.examples[item].cwe_id
        label = int(self.examples[item].label)
        cwe_str = str(cwe_id)

        if cwe_str == "None" and label == 1:
            # Vulnerable but no specific CWE type
            cwe_label = 1
        elif is_safe_cwe(cwe_id) or label == 0:
            # Safe
            cwe_label = 0
        else:
            # Specific CWE type
            cwe_label = CWE2INT.get(cwe_str, 0)

        func_input = self.code_tokenizer(
            func,
            max_length=self.code_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return (
            func_input["input_ids"].squeeze(0),
            func_input["attention_mask"].squeeze(0),
            torch.tensor(cwe_label),
        )


class UnifiedTestData(Dataset):
    """
    Used for zero-shot evaluation. Returns code tokens + all description tokens + unified CWE label.

    Class index layout (91 total):
      0  : safe (cwe_id=None/"None", label=0)
      1  : unlabeled vuln (cwe_id="None", label=1)
      2+ : specific CWE types (cwe_id in CWE2INT)
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

        if cwe_str == "None" and label == 1:
            unified_label = 1
        elif is_safe_cwe(cwe_id) or label == 0:
            unified_label = 0
        else:
            unified_label = CWE2INT.get(cwe_str, 0)

        func_input = self.code_tokenizer(
            func,
            max_length=self.code_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

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
            desc_ids_list,
            desc_mask_list,
            torch.tensor(unified_label),
        )
