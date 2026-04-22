import tempfile
import unittest
from pathlib import Path

import torch


class DPOTests(unittest.TestCase):
    def test_sequence_log_probs_ignores_masked_labels(self):
        from trainer.train_dpo import sequence_log_probs

        logits = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0],
                    [5.0, 0.0, 0.0],
                    [0.0, 5.0, 0.0],
                    [0.0, 0.0, 5.0],
                ]
            ]
        )
        labels = torch.tensor([[0, 0, -100, 2]])

        logps = sequence_log_probs(logits, labels)

        expected = (
            torch.log_softmax(logits[:, 0, :], dim=-1)[0, 0]
            + torch.log_softmax(logits[:, 2, :], dim=-1)[0, 2]
        )
        self.assertTrue(torch.allclose(logps, expected.unsqueeze(0)))

    def test_build_reference_model_loads_legacy_self_attn_checkpoint_keys(self):
        from model.model import HollowStoneMindConfig, HollowStoneMindForCausalLM
        from trainer.train_dpo import build_reference_model

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

            ref_model = build_reference_model(
                config,
                reference_weight="legacy",
                save_dir=tmpdir,
                device="cpu",
            )

        self.assertTrue(
            torch.equal(
                ref_model.model.layers[0].attention.q_proj.weight,
                source_model.model.layers[0].attention.q_proj.weight,
            )
        )
        self.assertFalse(any(p.requires_grad for p in ref_model.parameters()))


class TrainDPOScriptTests(unittest.TestCase):
    def test_build_parser_uses_repo_defaults(self):
        from trainer import train_dpo

        parser = train_dpo.build_parser()
        args = parser.parse_args([])

        self.assertEqual(args.save_weight, "dpo")
        self.assertEqual(args.from_weight, "sft")
        self.assertEqual(args.reference_weight, "sft")
        self.assertTrue(args.data_path.endswith("dataset/dpo.jsonl"))


if __name__ == "__main__":
    unittest.main()
