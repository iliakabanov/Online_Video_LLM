"""Минимальные проверки режимов GT, padding, CLI и causal-инвариантности UHead."""
import unittest

import torch

from .config import FEATURE_DIMS, UHeadConfig
from .data import LabelledItem, collate
from .labels import make_targets
from .model import UHead
from .train import binary_loss, build_parser


class TargetTests(unittest.TestCase):
    def test_hard_values_and_length(self):
        target = make_targets(7, 3, "hard")
        self.assertEqual(target.shape, (7,))
        self.assertEqual(set(target.tolist()), {0.0, 1.0})
        self.assertEqual(target.tolist(), [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])

    def test_step_window_and_short_video(self):
        target = make_targets(8, 5, "step", step_window=3)
        self.assertEqual(set(target.tolist()), {0.0, 0.5, 1.0})
        self.assertEqual(target.tolist(), [0.0, 0.0, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0])
        self.assertEqual(make_targets(2, 0, "step", step_window=5).tolist(), [1.0, 1.0])

    def test_sigmoid_value_monotonicity_and_target_zero(self):
        target = make_targets(8, 3, "sigmoid", sigmoid_slope=0.65)
        self.assertAlmostEqual(float(target[3]), 0.9, places=6)
        self.assertTrue(bool((torch.from_numpy(target[1:]) > torch.from_numpy(target[:-1])).all()))
        self.assertAlmostEqual(float(make_targets(1, 0, "sigmoid")[0]), 0.9, places=6)


class PipelineTests(unittest.TestCase):
    @staticmethod
    def _config():
        return UHeadConfig(projection_dims={"lm": 8, "attention": 8, "entropy": 4, "vision": 4, "confidence": 4},
                           model_dim=16, nhead=4, num_layers=1, dropout=0.0)

    @staticmethod
    def _features(batch=2, steps=5):
        return {name: torch.randn(batch, steps, dim) for name, dim in FEATURE_DIMS.items()}

    def test_padding_does_not_affect_loss(self):
        logits = torch.tensor([[0.2, -0.3, 100.0], [0.5, 100.0, 100.0]])
        targets = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
        padding = torch.tensor([[False, False, True], [False, True, True]])
        reference = binary_loss(logits, targets, padding)
        logits[padding] = -100.0
        self.assertTrue(torch.allclose(reference, binary_loss(logits, targets, padding)))

    def test_target_lengths_match_feature_sequences(self):
        rows = {name: torch.zeros(4, dim).numpy() for name, dim in FEATURE_DIMS.items()}
        target = make_targets(4, 2, "hard")
        _, targets, padding, items = collate([(rows, target, LabelledItem(0, target, 2, 1))])
        self.assertEqual(targets.shape, (1, 4))
        self.assertFalse(bool(padding.any()))
        self.assertEqual(items[0].target_pair, 2)

    def test_binary_shape_and_causal_invariance(self):
        model = UHead(self._config()).eval()
        features = self._features(batch=1)
        changed = {name: value.clone() for name, value in features.items()}
        for value in changed.values():
            value[:, 3:] = torch.randn_like(value[:, 3:])
        with torch.no_grad():
            before, after = model(features), model(changed)
        self.assertEqual(before.shape, (1, 5))
        torch.testing.assert_close(before[:, :3], after[:, :3], rtol=0, atol=1e-6)

    def test_all_tag_modes_are_accepted_by_cli(self):
        parser = build_parser()
        for tags in ("hard", "step", "sigmoid"):
            args = parser.parse_args(["--tags", tags, "--output", "results"])
            self.assertEqual(args.tags, tags)


if __name__ == "__main__":
    unittest.main()
