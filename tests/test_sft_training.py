import json
import tempfile
import unittest
from pathlib import Path

import torch


class TokenResult:
    def __init__(self, input_ids):
        self.input_ids = input_ids


class StubTokenizer:
    bos_token = "<s>"
    eos_token = "</s>"
    pad_token_id = 0

    def __call__(
        self,
        text,
        add_special_tokens=False,
        max_length=None,
        truncation=False,
    ):
        input_ids = [ord(ch) + 10 for ch in text]
        if truncation and max_length is not None:
            input_ids = input_ids[:max_length]
        return TokenResult(input_ids)

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=False,
        tools=None,
    ):
        assert not tokenize
        parts = []
        for message in messages:
            parts.append(
                f"{self.bos_token}{message['role']}\n{message['content']}{self.eos_token}\n"
            )
        if add_generation_prompt:
            parts.append(f"{self.bos_token}assistant\n")
        return "".join(parts)


class SFTDatasetTests(unittest.TestCase):
    def test_sft_dataset_returns_batch_dict_and_masks_non_assistant_tokens(self):
        from dataset.Im_dataset import SFTDataset

        sample = {
            "conversations": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "2+2?"},
                {"role": "assistant", "content": "4"},
            ]
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "sft.jsonl"
            data_path.write_text(json.dumps(sample, ensure_ascii=False) + "\n")

            dataset = SFTDataset(
                str(data_path),
                tokenizer=StubTokenizer(),
                max_length=128,
            )

            item = dataset[0]

        self.assertEqual(set(item.keys()), {"input_ids", "labels", "attention_mask"})
        self.assertEqual(item["input_ids"].shape, item["labels"].shape)
        self.assertEqual(item["input_ids"].shape, item["attention_mask"].shape)

        labels = item["labels"].tolist()
        visible_ids = [token for token in labels if token != -100]
        assistant_text_ids = [ord(ch) + 10 for ch in "<s>assistant\n4</s>\n"]

        self.assertEqual(visible_ids, assistant_text_ids)


class TrainSFTScriptTests(unittest.TestCase):
    def test_build_parser_uses_sft_defaults(self):
        from trainer import train_sft

        parser = train_sft.build_parser()
        args = parser.parse_args([])

        self.assertEqual(args.save_weight, "sft")
        self.assertEqual(args.from_weight, "pretrain")
        self.assertTrue(args.data_path.endswith("dataset/sft_mini_512.jsonl"))
        self.assertEqual(args.max_seq_len, 512)

    def test_init_model_loads_legacy_self_attn_checkpoint_keys(self):
        from model.model import HollowStoneMindConfig, HollowStoneMindForCausalLM
        from trainer.trainer_utils import init_model

        config = HollowStoneMindConfig(hidden_size=32, num_hidden_layers=1)
        source_model = HollowStoneMindForCausalLM(config)
        source_state = source_model.state_dict()
        legacy_state = {
            key.replace(".attention.", ".self_attn."): value
            for key, value in source_state.items()
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            weight_path = Path(tmpdir) / "legacy_32.pth"
            torch.save(legacy_state, weight_path)

            loaded_model, _ = init_model(
                config,
                from_weight="legacy",
                tokenizer_path="model",
                save_dir=tmpdir,
                device="cpu",
            )

        self.assertTrue(
            torch.equal(
                loaded_model.model.layers[0].attention.q_proj.weight,
                source_model.model.layers[0].attention.q_proj.weight,
            )
        )


if __name__ == "__main__":
    unittest.main()
