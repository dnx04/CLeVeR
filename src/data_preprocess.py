from torch.utils.data import Dataset
import logging
from tqdm import tqdm
import json
import random
import numpy as np
import torch
import pickle
from generate_example import generate_description
import argparse
from sklearn.model_selection import train_test_split


logger = logging.getLogger(__name__)


class Preprocess(Dataset):
    def __init__(self, args, file_path=""):
        self.examples = []
        dataset_file = file_path

        dataset = []
        idx = 0
        with open(dataset_file) as f:
            for line in f:
                line = line.strip()
                js = json.loads(line)
                js["idx"] = str(idx)
                dataset.append(js)
                idx += 1

        random.shuffle(dataset)

        # Split: 80% pre-train, 20% downstream
        # Downstream 20% split: fine-tune 70%, val 10%, test 20%
        labels = [x.get("label", "0") for x in dataset]

        pretrain_data, downstream_data = train_test_split(
            dataset, test_size=0.2, stratify=labels, random_state=args.seed
        )

        downstream_labels = [x.get("label", "0") for x in downstream_data]
        train_data, test_data = train_test_split(
            downstream_data,
            test_size=0.2,
            stratify=downstream_labels,
            random_state=args.seed,
        )
        train_labels = [x.get("label", "0") for x in train_data]
        train_data, val_data = train_test_split(
            train_data, test_size=0.125, stratify=train_labels, random_state=args.seed
        )  # 0.125 of 80% = 10% of total

        print("pretrain_data: ", len(pretrain_data))
        print("train_data: ", len(train_data))
        print("val_data: ", len(val_data))
        print("test_data: ", len(test_data))

        self.pretrain_examples = [
            generate_description(x)
            for x in tqdm(pretrain_data, total=len(pretrain_data))
        ]
        with open(
            "dataset/" + args.dataset_name + "/" + args.dataset_name + "_pretrain.pkl",
            "wb",
        ) as f:
            pickle.dump(self.pretrain_examples, f)
        print("pretrain_examples: ", len(self.pretrain_examples))

        self.train_examples = [
            generate_description(x) for x in tqdm(train_data, total=len(train_data))
        ]
        with open(
            "dataset/" + args.dataset_name + "/" + args.dataset_name + "_train.pkl",
            "wb",
        ) as f:
            pickle.dump(self.train_examples, f)
        print("train_examples: ", len(self.train_examples))

        self.val_examples = [
            generate_description(x) for x in tqdm(val_data, total=len(val_data))
        ]
        with open(
            "dataset/" + args.dataset_name + "/" + args.dataset_name + "_val.pkl", "wb"
        ) as f:
            pickle.dump(self.val_examples, f)
        print("val_examples: ", len(self.val_examples))

        self.test_examples = [
            generate_description(x) for x in tqdm(test_data, total=len(test_data))
        ]
        with open(
            "dataset/" + args.dataset_name + "/" + args.dataset_name + "_test.pkl", "wb"
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
        default=None,
        type=str,
        required=True,
        help="the dataset txt file path",
    )
    parser.add_argument(
        "--dataset_name", default=None, type=str, required=True, help="the dataset name"
    )

    parser.add_argument(
        "--pretrain_text_model_name",
        default=None,
        type=str,
        help="The model checkpoint for weights initialization.",
    )
    parser.add_argument(
        "--pretrain_code_model_name",
        default=None,
        type=str,
        help="The model checkpoint for weights initialization.",
    )

    parser.add_argument(
        "--program_language",
        default="",
        type=str,
        help="Optional pretrained config name or path if not the same as model_name_or_path",
    )

    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )

    args = parser.parse_args()
    set_seed(args)
    # text_config = RobertaConfig.from_pretrained(args.pretrain_text_model_name)
    # text_tokenizer = RobertaTokenizer.from_pretrained(args.pretrain_text_model_name)
    Preprocess(args, file_path=args.dataset)
