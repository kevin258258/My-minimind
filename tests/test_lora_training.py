import copy
import tempfile
import unittest

import torch
from torch import nn


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 4, bias=False)
        self.k_proj = nn.Linear(4, 4, bias=False)
        self.ffn = nn.Linear(4, 6, bias=False)

    def forward(self, x):
        return self.q_proj(x) + self.k_proj(x)


class LoRATests(unittest.TestCase):
    def test_apply_lora_keeps_initial_forward_and_only_wraps_target_modules(self):
        from model.model_lora import LoRALinear, apply_lora

        model = ToyModel()
        x = torch.randn(2, 4)
        baseline = model(x)

        apply_lora(model, rank=2, target_modules=("q_proj",))

        self.assertIsInstance(model.q_proj, LoRALinear)
        self.assertIsInstance(model.k_proj, nn.Linear)
        self.assertIsInstance(model.ffn, nn.Linear)

        adapted = model(x)
        self.assertTrue(torch.allclose(baseline, adapted))

    def test_save_and_load_lora_restores_adapter_weights(self):
        from model.model_lora import apply_lora, load_lora, save_lora

        base_model = ToyModel()
        source_model = copy.deepcopy(base_model)
        apply_lora(source_model, rank=2, target_modules=("q_proj",))

        with torch.no_grad():
            source_model.q_proj.lora_A.weight.fill_(0.5)
            source_model.q_proj.lora_B.weight.fill_(0.25)

        x = torch.randn(2, 4)
        expected = source_model(x)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/adapter.pth"
            save_lora(source_model, path)

            target_model = copy.deepcopy(base_model)
            apply_lora(target_model, rank=2, target_modules=("q_proj",))
            load_lora(target_model, path)

            actual = target_model(x)

        self.assertTrue(torch.allclose(expected, actual))

    def test_mark_only_lora_trainable_freezes_base_weights(self):
        from model.model_lora import apply_lora, mark_only_lora_trainable

        model = ToyModel()
        apply_lora(model, rank=2, target_modules=("q_proj",))
        lora_params = mark_only_lora_trainable(model)

        self.assertTrue(lora_params)
        self.assertTrue(all(param.requires_grad for param in lora_params))
        self.assertFalse(model.q_proj.base_layer.weight.requires_grad)
        self.assertFalse(model.k_proj.weight.requires_grad)
        self.assertFalse(model.ffn.weight.requires_grad)


class TrainLoRAScriptTests(unittest.TestCase):
    def test_build_parser_uses_repo_defaults(self):
        from trainer import train_lora

        parser = train_lora.build_parser()
        args = parser.parse_args([])

        self.assertEqual(args.lora_name, "lora_medical")
        self.assertEqual(args.from_weight, "sft")
        self.assertTrue(args.data_path.endswith("dataset/lora_medical.jsonl"))
        self.assertEqual(args.max_seq_len, 512)


if __name__ == "__main__":
    unittest.main()
