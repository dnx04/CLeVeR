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


logger = logging.getLogger(__name__)


class Preprocess(Dataset):
    def __init__(self, args, file_path=""):
        self.examples = []
        dataset_file = file_path

        # Load pickle containing list of ExampleFeature objects
        with open(dataset_file, "rb") as f:
            dataset = pickle.load(f)

        random.shuffle(dataset)

        # Split: 80% pre-train, 20% downstream
        # Downstream 20% split: fine-tune 70%, val 10%, test 20%
        labels = [x.label for x in dataset]

        pretrain_data, downstream_data = train_test_split(
            dataset, test_size=0.2, stratify=labels, random_state=args.seed
        )

        downstream_labels = [x.label for x in downstream_data]
        train_data, test_data = train_test_split(
            downstream_data,
            test_size=0.2,
            stratify=downstream_labels,
            random_state=args.seed,
        )
        train_labels = [x.label for x in train_data]
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
        print("pretrain_examples: ", len(self.pretrain_examples))

        self.train_examples = train_data
        with open(
            out_dir + "/" + args.dataset_name + "_train.pkl",
            "wb",
        ) as f:
            pickle.dump(self.train_examples, f)
        print("train_examples: ", len(self.train_examples))

        self.val_examples = val_data
        with open(
            out_dir + "/" + args.dataset_name + "_val.pkl",
            "wb",
        ) as f:
            pickle.dump(self.val_examples, f)
        print("val_examples: ", len(self.val_examples))

        self.test_examples = test_data
        with open(
            out_dir + "/" + args.dataset_name + "_test.pkl",
            "wb",
        ) as f:
            pickle.dump(self.test_examples, f)
        print("test_examples: ", len(self.test_examples))

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
