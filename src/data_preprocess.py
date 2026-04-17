from torch.utils.data import Dataset
import logging
import os
from tqdm import tqdm
import random
import numpy as np
import torch
import pickle
import argparse
from sklearn.model_selection import train_test_split
from collections import Counter


logger = logging.getLogger(__name__)


CWE_FREQUENCY_THRESHOLD = 50


class Preprocess(Dataset):
    def __init__(self, args, file_path=""):
        self.examples = []
        dataset_file = file_path

        # Load pickle containing list of ExampleFeature objects
        with open(dataset_file, "rb") as f:
            dataset = pickle.load(f)

        random.shuffle(dataset)

        # Separate rare CWE (< threshold) vs common CWE (>= threshold) — single pass
        cwe_counter = Counter()
        rare_data, common_data, safe_data = [], [], []
        for ex in dataset:
            if ex.cwe_id is not None and str(ex.cwe_id) != "None":
                cwe_counter[str(ex.cwe_id)] += 1

        for ex in dataset:
            cwe_str = str(ex.cwe_id)
            if cwe_str == "None" or ex.cwe_id is None:
                safe_data.append(ex)
            elif cwe_counter[cwe_str] >= CWE_FREQUENCY_THRESHOLD:
                common_data.append(ex)
            else:
                rare_data.append(ex)

        logger.info(f"Total: {len(dataset)}, Safe: {len(safe_data)}, Common: {len(common_data)}, Rare: {len(rare_data)}")

        # Rare CWE + safe samples -> pretrain only
        pretrain_data = rare_data + safe_data

        # Common CWE: 80% pretrain, 20% downstream (train/val/test)
        common_labels = [str(ex.cwe_id) for ex in common_data]
        pretrain_common, downstream_data = train_test_split(
            common_data, test_size=0.2, stratify=common_labels, random_state=args.seed
        )
        pretrain_data.extend(pretrain_common)

        # Downstream: 70% train, 10% val, 20% test
        downstream_labels = [str(ex.cwe_id) for ex in downstream_data]
        train_data, test_data = train_test_split(
            downstream_data,
            test_size=0.2,
            stratify=downstream_labels,
            random_state=args.seed,
        )
        train_labels = [str(ex.cwe_id) for ex in train_data]
        train_data, val_data = train_test_split(
            train_data, test_size=0.125, stratify=train_labels, random_state=args.seed
        )  # 0.125 of 80% = 10% of total

        print("pretrain_data: ", len(pretrain_data))
        print("train_data: ", len(train_data))
        print("val_data: ", len(val_data))
        print("test_data: ", len(test_data))

        # Data is already ExampleFeature objects from pickle
        out_dir = "dataset/" + args.dataset_name
        os.makedirs(out_dir, exist_ok=True)

        self.pretrain_examples = pretrain_data
        with open(
            out_dir + "/" + args.dataset_name + "_pretrain.pkl",
            "wb",
        ) as f:
            pickle.dump(self.pretrain_examples, f)

        self.train_examples = train_data
        with open(
            out_dir + "/" + args.dataset_name + "_train.pkl",
            "wb",
        ) as f:
            pickle.dump(self.train_examples, f)

        self.val_examples = val_data
        with open(
            out_dir + "/" + args.dataset_name + "_val.pkl",
            "wb",
        ) as f:
            pickle.dump(self.val_examples, f)

        self.test_examples = test_data
        with open(
            out_dir + "/" + args.dataset_name + "_test.pkl",
            "wb",
        ) as f:
            pickle.dump(self.test_examples, f)

        self.examples = self.pretrain_examples  # for __len__ compatibility

        for idx, example in enumerate(self.examples[:2]):
            logger.info("*** Example ***")
            logger.info("idx: {}".format(example.idx))
            logger.info("func: {}".format(example.func))
            logger.info("label: {}".format(example.label))
            logger.info("source: {}".format(example.source))
            logger.info("sink: {}".format(example.sink))
            logger.info("description: {}".format(example.description))
            logger.info("cwe_id: {}".format(example.cwe_id))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        default="dataset/vcldata.pkl",
        type=str,
        required=True,
        help="the dataset pickle file path",
    )
    parser.add_argument(
        "--dataset_name", default="vcldata", type=str, required=True, help="the dataset name"
    )

    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )

    args = parser.parse_args()
    set_seed(args)
    Preprocess(args, file_path=args.dataset)
