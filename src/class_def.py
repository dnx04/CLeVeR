"""
CLeVeR class definitions — CWE label mapping and NUM_CLASSES.
Dynamically built from dataset at import time.
Can be overridden by writing a pre-computed mapping to avoid scanning all pickles.
"""

from collections import Counter
import os
import pickle
from generate_example import ExampleFeature

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATASET_NAME = "vcldata"  # dataset/vcldata/vcldata_*.pkl
CWE_FREQUENCY_THRESHOLD = 50  # minimum samples to include a CWE class

# ---------------------------------------------------------------------------
# Class index layout
#   0 .. N-1  : specific CWE types (CWE_LIST[i] -> class i)
# ---------------------------------------------------------------------------

def build_cwe_mapping():
    """Scan all pickle files to build CWE->class mapping dynamically."""
    cwe_counter = Counter()
    for split in ["pretrain", "train", "val", "test"]:
        path = f"dataset/{DATASET_NAME}/{DATASET_NAME}_{split}.pkl"
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
    # CWE index 0 = first CWE type, no safe class in mapping
    cwe2int = {cwe: idx for idx, cwe in enumerate(valid_cwes)}
    return valid_cwes, cwe2int


# Build mapping at import time
CWE_LIST, CWE2INT = build_cwe_mapping()

# Total classes: 98 CWE types (0-97), no safe class in mapping
NUM_CLASSES = len(CWE_LIST)

print(f"[class.py] Loaded {NUM_CLASSES} CWE types (class 0-{NUM_CLASSES-1}), safe is not a class")