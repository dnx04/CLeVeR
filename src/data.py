from torch.utils.data import Dataset
import logging
import torch
import pickle

logger = logging.getLogger(__name__)


def get_dataset_path(dataset_name, split):
    """Returns path to dataset pickle file at dataset/<name>/<name>_<split>.pkl"""
    return f'dataset/{dataset_name}/{dataset_name}_{split}.pkl'


class TrainData(Dataset):
    def __init__(self, code_tokenizer, text_tokenizer, args, flag=''):
        self.examples = []
        self.args = args
        self.code_tokenizer = code_tokenizer
        self.text_tokenizer = text_tokenizer
        self.code_max_length = 512
        self.text_max_length = 64

        if 'pretrain' in flag:
            path = get_dataset_path(args.dataset, 'pretrain')
            with open(path, 'rb') as f:
                self.examples = pickle.load(f)
        elif 'train' in flag:
            path = get_dataset_path(args.dataset, 'train')
            with open(path, 'rb') as f:
                self.examples = pickle.load(f)
        elif 'val' in flag:
            path = get_dataset_path(args.dataset, 'val')
            with open(path, 'rb') as f:
                self.examples = pickle.load(f)
        elif 'test' in flag:
            path = get_dataset_path(args.dataset, 'test')
            with open(path, 'rb') as f:
                self.examples = pickle.load(f)
        else:
            print("file_path error!")

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
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        func_input_ids = func_input['input_ids'].squeeze(0)
        func_attention_mask = func_input['attention_mask'].squeeze(0)

        description_input = self.text_tokenizer(
            description,
            max_length=self.text_max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        description_input_ids = description_input['input_ids'].squeeze(0)
        description_attention_mask = description_input['attention_mask'].squeeze(0)

        source_input = self.text_tokenizer(
            source,
            max_length=self.text_max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        source_input_ids = source_input['input_ids'].squeeze(0)
        source_attention_mask = source_input['attention_mask'].squeeze(0)

        sink_input = self.text_tokenizer(
            sink,
            max_length=self.text_max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        sink_input_ids = sink_input['input_ids'].squeeze(0)
        sink_attention_mask = sink_input['attention_mask'].squeeze(0)

        return (func_input_ids, func_attention_mask, description_input_ids, description_attention_mask,
                source_input_ids, source_attention_mask, sink_input_ids, sink_attention_mask)


class DetectionTestData(Dataset):
    def __init__(self, code_tokenizer, text_tokenizer, args, flag=''):
        self.examples = []
        self.args = args
        self.code_tokenizer = code_tokenizer
        self.text_tokenizer = text_tokenizer
        self.code_max_length = 512
        self.text_max_length = 64

        if 'pretrain' in flag:
            path = get_dataset_path(args.dataset, 'pretrain')
            with open(path, 'rb') as f:
                self.examples = pickle.load(f)
        elif 'train' in flag:
            path = get_dataset_path(args.dataset, 'train')
            with open(path, 'rb') as f:
                self.examples = pickle.load(f)
        elif 'val' in flag:
            path = get_dataset_path(args.dataset, 'val')
            with open(path, 'rb') as f:
                self.examples = pickle.load(f)
        elif 'test' in flag:
            path = get_dataset_path(args.dataset, 'test')
            with open(path, 'rb') as f:
                self.examples = pickle.load(f)
        else:
            print("file_path error!")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        func = self.examples[item].func
        label = int(self.examples[item].label)
        description_0 = "This is a security function."
        description_1 = "This is a vulnerability function."

        func_input = self.code_tokenizer(
            func,
            max_length=self.code_max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        func_input_ids = func_input['input_ids'].squeeze(0)
        func_attention_mask = func_input['attention_mask'].squeeze(0)

        description_0_input = self.text_tokenizer(
            description_0,
            max_length=self.text_max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        description_0_input_ids = description_0_input['input_ids'].squeeze(0)
        description_0_attention_mask = description_0_input['attention_mask'].squeeze(0)

        description_1_input = self.text_tokenizer(
            description_1,
            max_length=self.text_max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        description_1_input_ids = description_1_input['input_ids'].squeeze(0)
        description_1_attention_mask = description_1_input['attention_mask'].squeeze(0)

        return (func_input_ids, func_attention_mask, description_0_input_ids, description_0_attention_mask,
                description_1_input_ids, description_1_attention_mask, torch.tensor(label))


cwe_list = ["78", "121", "122", "129", "190",
            "284", "390", "400", "416", "476"]
int_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
cwe2int = dict(zip(cwe_list, int_list))


class ClassificationTestData(Dataset):
    def __init__(self, code_tokenizer, text_tokenizer, args, flag=''):
        self.examples = []
        self.args = args
        self.code_tokenizer = code_tokenizer
        self.text_tokenizer = text_tokenizer
        self.code_max_length = 512
        self.text_max_length = 64

        if 'pretrain' in flag:
            path = get_dataset_path(args.dataset, 'pretrain')
            with open(path, 'rb') as f:
                self.examples = pickle.load(f)
        elif 'train' in flag:
            path = get_dataset_path(args.dataset, 'train')
            with open(path, 'rb') as f:
                self.examples = pickle.load(f)
        elif 'val' in flag:
            path = get_dataset_path(args.dataset, 'val')
            with open(path, 'rb') as f:
                self.examples = pickle.load(f)
        elif 'test' in flag:
            path = get_dataset_path(args.dataset, 'test')
            with open(path, 'rb') as f:
                self.examples = pickle.load(f)
        else:
            print("file_path error!")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        func = self.examples[item].func
        cwe_id = self.examples[item].cwe_id
        description_list = ["A vulnerability of OS Command Injection.",
                            "A vulnerability of Stack-based Buffer Overflow.",
                            "A vulnerability of Heap-based Buffer Overflow.",
                            "A vulnerability of Improper Validation of Array Index.",
                            "A vulnerability of Integer Overflow.",
                            "A vulnerability of Improper Access Control.",
                            "A vulnerability of Detection of Error Condition Without Action.",
                            "A vulnerability of Uncontrolled Resource Consumption.",
                            "A vulnerability of Use After Free.",
                            "A vulnerability of NULL Pointer Dereference."]
        cwe_label = cwe2int[cwe_id]

        func_input = self.code_tokenizer(
            func,
            max_length=self.code_max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        func_input_ids = func_input['input_ids'].squeeze(0)
        func_attention_mask = func_input['attention_mask'].squeeze(0)

        description_input_ids_list = []
        description_attention_mask_list = []
        for description in description_list:
            description_input = self.text_tokenizer(
                description,
                max_length=self.text_max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            description_input_ids = description_input['input_ids'].squeeze(0)
            description_attention_mask = description_input['attention_mask'].squeeze(0)
            description_input_ids_list.append(description_input_ids)
            description_attention_mask_list.append(description_attention_mask)
        description_input_ids_list = torch.stack(description_input_ids_list)
        description_attention_mask_list = torch.stack(description_attention_mask_list)

        return (func_input_ids, func_attention_mask, description_input_ids_list,
                description_attention_mask_list, torch.tensor(cwe_label))


class DetectionProbeData(Dataset):
    def __init__(self, code_tokenizer, text_tokenizer, args, flag=''):
        self.examples = []
        self.args = args
        self.code_tokenizer = code_tokenizer
        self.text_tokenizer = text_tokenizer
        self.code_max_length = 512
        self.text_max_length = 64

        if 'pretrain' in flag:
            path = get_dataset_path(args.dataset, 'pretrain')
            with open(path, 'rb') as f:
                self.examples = pickle.load(f)
        elif 'train' in flag:
            path = get_dataset_path(args.dataset, 'train')
            with open(path, 'rb') as f:
                self.examples = pickle.load(f)
        elif 'val' in flag:
            path = get_dataset_path(args.dataset, 'val')
            with open(path, 'rb') as f:
                self.examples = pickle.load(f)
        elif 'test' in flag:
            path = get_dataset_path(args.dataset, 'test')
            with open(path, 'rb') as f:
                self.examples = pickle.load(f)
        else:
            print("file_path error!")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        func = self.examples[item].func
        label = int(self.examples[item].label)

        func_input = self.code_tokenizer(
            func,
            max_length=self.code_max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        func_input_ids = func_input['input_ids'].squeeze(0)
        func_attention_mask = func_input['attention_mask'].squeeze(0)

        return func_input_ids, func_attention_mask, torch.tensor(label)


class ClassificationProbeData(Dataset):
    def __init__(self, code_tokenizer, text_tokenizer, args, flag=''):
        self.examples = []
        self.args = args
        self.code_tokenizer = code_tokenizer
        self.text_tokenizer = text_tokenizer
        self.code_max_length = 512
        self.text_max_length = 64

        if 'pretrain' in flag:
            path = get_dataset_path(args.dataset, 'pretrain')
            with open(path, 'rb') as f:
                self.examples = pickle.load(f)
        elif 'train' in flag:
            path = get_dataset_path(args.dataset, 'train')
            with open(path, 'rb') as f:
                self.examples = pickle.load(f)
        elif 'val' in flag:
            path = get_dataset_path(args.dataset, 'val')
            with open(path, 'rb') as f:
                self.examples = pickle.load(f)
        elif 'test' in flag:
            path = get_dataset_path(args.dataset, 'test')
            with open(path, 'rb') as f:
                self.examples = pickle.load(f)
        else:
            print("file_path error!")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        func = self.examples[item].func
        cwe_id = self.examples[item].cwe_id
        if cwe_id not in cwe2int:
            print(f"cwe_id {cwe_id} not found in dictionary.")
            cwe_label = 0
        else:
            cwe_label = cwe2int[cwe_id]

        func_input = self.code_tokenizer(
            func,
            max_length=self.code_max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        func_input_ids = func_input['input_ids'].squeeze(0)
        func_attention_mask = func_input['attention_mask'].squeeze(0)

        return func_input_ids, func_attention_mask, torch.tensor(cwe_label)