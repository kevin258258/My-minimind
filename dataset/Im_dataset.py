import os
import random

import torch
from datasets import load_dataset
from torch.utils.data import Dataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_json_dataset(data_path):
    cache_dir = os.environ.get("HF_DATASETS_CACHE", "/tmp/hf_datasets_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return load_dataset(
        "json",
        data_files=data_path,
        split="train",
        cache_dir=cache_dir,
    )


def pre_processing_chat(conversations, add_system_ratio=0.2):
    # tool-use 数据完整保留，不自动插入 system
    if any(conv.get("tools") for conv in conversations):
        return conversations

    system_prompts = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是minimind，一个小巧但有用的语言模型。",
        "你是一个专业的AI助手，请提供有价值的回答。",
        "你是minimind，请尽力帮助用户解决问题。",
        "你是一个可靠的AI，请给出准确的回答。",
        "You are a helpful AI assistant.",
        "You are minimind, a lightweight intelligent assistant.",
        "You are a friendly chatbot. Please answer the user's questions carefully.",
        "You are a knowledgeable AI. Try your best to provide accurate information.",
        "You are minimind, a small but useful language model.",
    ]

    if conversations and conversations[0].get("role") != "system":
        if random.random() < add_system_ratio:
            return [
                {"role": "system", "content": random.choice(system_prompts)}
            ] + conversations
    return conversations


def post_processing_chat(prompt_content, empty_think_ratio=0.2):
    if (
        "<think>\n\n</think>\n\n" in prompt_content
        and random.random() > empty_think_ratio
    ):
        prompt_content = prompt_content.replace("<think>\n\n</think>\n\n", "")
    return prompt_content


class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_json_dataset(data_path)

    def __len__(self):
        return len(self.samples)

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


class SFTDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_json_dataset(data_path)
        self.pad_token_id = self._get_pad_token_id()
        self.assistant_prefix_ids = self.tokenizer(
            f"{self.tokenizer.bos_token}assistant\n",
            add_special_tokens=False,
        ).input_ids
        self.turn_suffix_ids = self.tokenizer(
            f"{self.tokenizer.eos_token}\n",
            add_special_tokens=False,
        ).input_ids

    def _get_pad_token_id(self):
        if self.tokenizer.pad_token_id is not None:
            return self.tokenizer.pad_token_id
        if getattr(self.tokenizer, "eos_token_id", None) is not None:
            return self.tokenizer.eos_token_id
        raise ValueError("Tokenizer must define pad_token_id or eos_token_id")

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        messages = list(conversations)
        tools = None
        if messages:
            tools = messages[0].get("tools") or messages[0].get("function")

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools,
        )

    def generate_labels(self, input_ids):
        labels = [-100] * len(input_ids)
        prefix_len = len(self.assistant_prefix_ids)
        suffix_len = len(self.turn_suffix_ids)

        i = 0
        while i < len(input_ids):
            if input_ids[i : i + prefix_len] == self.assistant_prefix_ids:
                start = i + prefix_len
                end = start
                while end < len(input_ids):
                    if input_ids[end : end + suffix_len] == self.turn_suffix_ids:
                        end += suffix_len
                        break
                    end += 1

                if end > len(input_ids):
                    end = len(input_ids)

                for j in range(start, min(end, len(input_ids))):
                    labels[j] = input_ids[j]
                i = end
                continue
            i += 1

        return labels

    def __getitem__(self, index):
        sample = self.samples[index]
        conversations = pre_processing_chat(sample["conversations"])
        prompt = self.create_chat_prompt(conversations)
        prompt = post_processing_chat(prompt)

        input_ids = self.tokenizer(
            prompt,
            add_special_tokens=False,
            max_length=self.max_length,
            truncation=True,
        ).input_ids
        labels = self.generate_labels(input_ids)

        attention_mask = [1] * len(input_ids)
        pad_len = self.max_length - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [self.pad_token_id] * pad_len
            labels = labels + [-100] * pad_len
            attention_mask = attention_mask + [0] * pad_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }
