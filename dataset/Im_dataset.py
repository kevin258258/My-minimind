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

    def _tokenize_chat(self, conversations):
        if not conversations:
            return []
        prompt = self.create_chat_prompt(conversations)
        prompt = post_processing_chat(prompt)
        return self.tokenizer(prompt, add_special_tokens=False).input_ids

    def generate_labels(self, conversations, input_ids):
        labels = [-100] * len(input_ids)

        # 对每个 assistant 消息，比较“拼接前后”token边界来定位可学习区间，
        # 避免和具体 chat_template 的特殊token格式硬编码耦合。
        for i, msg in enumerate(conversations):
            if msg.get("role") != "assistant":
                continue

            prev_ids = self._tokenize_chat(conversations[:i])
            curr_ids = self._tokenize_chat(conversations[: i + 1])

            start = min(len(prev_ids), len(input_ids))
            end = min(len(curr_ids), len(input_ids))
            for j in range(start, end):
                labels[j] = input_ids[j]

        return labels

    def __getitem__(self, index):
        sample = self.samples[index]
        conversations = pre_processing_chat(sample["conversations"])
        input_ids = self._tokenize_chat(conversations)
        labels = self.generate_labels(conversations, input_ids)

        if len(input_ids) > self.max_length:
            input_ids = input_ids[: self.max_length]
            labels = labels[: self.max_length]

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


class DPODataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_json_dataset(data_path)
        self.pad_token_id = self._get_pad_token_id()

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

    def _tokenize_chat(self, conversations):
        if not conversations:
            return []
        prompt = self.create_chat_prompt(conversations)
        prompt = post_processing_chat(prompt)
        return self.tokenizer(prompt, add_special_tokens=False).input_ids

    def generate_labels(self, conversations, input_ids):
        labels = [-100] * len(input_ids)
        for i, msg in enumerate(conversations):
            if msg.get("role") != "assistant":
                continue

            prev_ids = self._tokenize_chat(conversations[:i])
            curr_ids = self._tokenize_chat(conversations[: i + 1])
            start = min(len(prev_ids), len(input_ids))
            end = min(len(curr_ids), len(input_ids))
            for j in range(start, end):
                labels[j] = input_ids[j]
        return labels

    def _build_features(self, conversations):
        # DPO中不做随机system增强，避免chosen/rejected对的噪声不一致
        input_ids = self._tokenize_chat(conversations)
        labels = self.generate_labels(conversations, input_ids)

        if len(input_ids) > self.max_length:
            input_ids = input_ids[: self.max_length]
            labels = labels[: self.max_length]

        attention_mask = [1] * len(input_ids)
        pad_len = self.max_length - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [self.pad_token_id] * pad_len
            labels = labels + [-100] * pad_len
            attention_mask = attention_mask + [0] * pad_len

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long),
        )

    def __getitem__(self, index):
        sample = self.samples[index]
        chosen = sample["chosen"]
        rejected = sample["rejected"]

        c_input_ids, c_labels, c_attention_mask = self._build_features(chosen)
        r_input_ids, r_labels, r_attention_mask = self._build_features(rejected)

        return {
            "chosen_input_ids": c_input_ids,
            "chosen_labels": c_labels,
            "chosen_attention_mask": c_attention_mask,
            "rejected_input_ids": r_input_ids,
            "rejected_labels": r_labels,
            "rejected_attention_mask": r_attention_mask,
        }
