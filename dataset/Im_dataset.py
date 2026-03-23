import json

from torch.utils.data import Dataset
import torch
import os
import random
from datasets import load_dataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# 这里设置tokennizers的并行性为false，避免报错


class PretrainDataset(Dataset):
    def __init__(self,data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset("json", data_files=data_path, split="train")

    def __len__(self):
        return len(self.samples)

    # 下面的get item返回的应该是模型能够直接使用的数据
    # 首先我们先从sample中得到对应的文本
    # 然后再用tokennizer进行tokenize
    # 然后再添加EOS/BOS
    # 记得加Pad,然后编写对应的label（右迁，但是hf默认实现了）
    # 另外记得编写mask,来防止pad参加loss 和 attention 的计算
    def __getitem__(self, index):
        sample = self.samples[index]
        pad_token_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id
        )

        tokens = self.tokenizer(
            str(sample["text"]),
            add_special_tokens=False,
            max_length=max(self.max_length - 2, 1),
            truncation=True,
        ).input_ids

        if self.tokenizer.bos_token_id is not None:
            tokens = [self.tokenizer.bos_token_id] + tokens
        if self.tokenizer.eos_token_id is not None:
            tokens = tokens + [self.tokenizer.eos_token_id]

        input_ids = tokens[: self.max_length]
        input_ids = input_ids + [pad_token_id] * (self.max_length - len(input_ids))
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = (input_ids != pad_token_id).to(torch.long)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }



