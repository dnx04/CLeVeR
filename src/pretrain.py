# coding=utf-8
"""CLeVeR pre-training script — trains on pretrain set, saves checkpoint at the end."""

from __future__ import absolute_import, division, print_function

import argparse
from typing import cast
import logging
import os
import random
import numpy as np
import torch
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader, RandomSampler
from transformers import get_linear_schedule_with_warmup, RobertaTokenizer
from torch.optim import AdamW
from tqdm import tqdm

from model import ContrastiveModel
from data import TrainData

logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, code_tokenizer, text_tokenizer):
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.train_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    args.max_steps = args.epochs * len(train_dataloader)
    args.warmup_steps = args.max_steps // 5
    model.to(args.device)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.max_steps
    )

    scaler = GradScaler(device="cuda")

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    logger.info("***** Running pre-training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info("  Total optimization steps = %d", args.max_steps)

    model.zero_grad()

    for idx in range(args.epochs):
        bar = tqdm(train_dataloader, total=len(train_dataloader), desc=f"Epoch {idx}")
        tr_num = 0
        train_loss = 0.0

        for step, batch in enumerate(bar):
            (
                func_input_ids,
                func_attention_mask,
                description_input_ids,
                description_attention_mask,
                source_input_ids,
                source_attention_mask,
                sink_input_ids,
                sink_attention_mask,
            ) = [x.to(args.device) for x in batch]
            model.train()
            with autocast(device_type="cuda"):
                loss = model(
                    func_input_ids,
                    func_attention_mask,
                    description_input_ids,
                    description_attention_mask,
                    source_input_ids,
                    source_attention_mask,
                    sink_input_ids,
                    sink_attention_mask,
                    "train",
                )

            if args.n_gpu > 1:
                loss = loss.mean()

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_num += 1
            train_loss += loss.item()
            bar.set_description(
                "epoch {} loss {}".format(idx, round(train_loss / tr_num, 5))
            )

            if (step + 1) % args.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

        logger.info("Epoch %d complete — avg loss: %.5f", idx, train_loss / tr_num)

    # Save final checkpoint
    logger.info("Training complete — saving checkpoint...")
    checkpoint_path = args.save_checkpoint
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    model_to_save = model.module if hasattr(model, "module") else model
    torch.save(model_to_save.state_dict(), checkpoint_path)
    logger.info("Saved checkpoint to %s", checkpoint_path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", default=None, type=str, required=True)
    parser.add_argument(
        "--save_checkpoint",
        default=None,
        type=str,
        required=True,
        help="Path to save the trained model checkpoint",
    )
    parser.add_argument("--code_length", default=512, type=int)
    parser.add_argument("--pretrain_text_model_name", default="roberta-base", type=str)
    parser.add_argument(
        "--pretrain_code_model_name", default="microsoft/codebert-base", type=str
    )
    parser.add_argument("--train_batch_size", default=256, type=int)
    parser.add_argument("--eval_batch_size", default=512, type=int)
    parser.add_argument("--hidden_size", default=768, type=int)
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--warmup_steps", default=0, type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=1)

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
    total_num = sum(p.numel() for p in model.parameters())
    print("Total parameters: ", total_num)
    logger.info("Training parameters: %s", args)

    train_dataset = TrainData(code_tokenizer, text_tokenizer, args, flag="pretrain")
    train(args, train_dataset, model, code_tokenizer, text_tokenizer)


if __name__ == "__main__":
    main()
