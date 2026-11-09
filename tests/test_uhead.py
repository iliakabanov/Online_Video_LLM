import torch
import numpy as np

from online_vllm.train.loss import focal_loss
from uhead.config import FEATURE_DIMS, UHeadConfig
from uhead.model import UHead
from online_vllm.models.heads import StateHead
from uhead.train import load_checkpoint, save_checkpoint
from uhead.train import _pad_probabilities, binary_loss, split_by_video
from uhead.data import LabelledItem, confidence_features, feature_rows


def small_config():
    return UHeadConfig(
        projection_dims={"lm": 8, "attention": 8, "entropy": 4, "vision": 4, "confidence": 4},
        model_dim=16, nhead=4, num_layers=1, dropout=0.0, temporal_detector="transformer",
    )


def small_compact_config():
    return UHeadConfig(
        projection_dims={"lm": 8, "attention": 8, "entropy": 4, "vision": 4, "confidence": 4},
        model_dim=16, nhead=4, num_layers=1, dropout=0.0, temporal_detector="compact",
        compact_model_dim=8, compact_nhead=2, compact_num_layers=1, compact_ffn_dim=12,
    )


def features(batch=2, steps=4):
    return {name: torch.randn(batch, steps, dim) for name, dim in FEATURE_DIMS.items()}


def test_default_architecture_dimensions():
    config = UHeadConfig()
    assert config.fusion_dim == 736
    model = UHead(config).eval()
    with torch.no_grad():
        logits = model(features(batch=1, steps=2))
    assert logits.shape == (1, 2)


def test_forward_backward_padding_and_zero_variance_normalization():
    model = UHead(small_config())
    stats = {name: (torch.zeros(dim), torch.zeros(dim)) for name, dim in FEATURE_DIMS.items()}
    model.set_normalization(stats)
    x = features()
    padding = torch.tensor([[False, False, False, False], [False, False, True, True]])
    target = torch.tensor([[0., 1., 1., 1.], [0., 1., 0., 0.]])
    logits = model(x, padding)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits[~padding], target[~padding])
    loss.backward()
    assert logits.shape == (2, 4)
    assert torch.isfinite(logits).all() and torch.isfinite(loss)


def test_eval_split_keeps_every_video_in_one_half():
    from types import SimpleNamespace
    videos = [f"v{i}" for i in range(7) for _ in range(i % 3 + 1)]  # 7 видео по 1-3 вопроса
    fs = SimpleNamespace(records=[SimpleNamespace(video=v) for v in videos])
    items = [LabelledItem(i, np.zeros(1, dtype=np.float32), 0, 0) for i in range(len(videos))]
    validation, test, info = split_by_video(fs, items, seed=0)
    assert len(info["validation_videos"]) == 3 and len(info["test_videos"]) == 4
    assert not set(info["validation_videos"]) & set(info["test_videos"])
    assert sorted(item.index for item in validation + test) == list(range(len(videos)))
    assert {videos[item.index] for item in validation} == set(info["validation_videos"])
    assert split_by_video(fs, items, seed=0)[2] == info  # разбиение воспроизводимо


def test_constant_train_columns_are_only_centered():
    model = UHead(small_config())
    stats = {name: (torch.zeros(dim), torch.full((dim,), 2.0)) for name, dim in FEATURE_DIMS.items()}
    stats["lm"][1][:3] = 0.0
    model.set_normalization(stats)
    std = model.fusion_block.lm_std
    assert std[:3].tolist() == [1.0, 1.0, 1.0] and bool((std[3:] == 2.0).all())


def test_causal_invariance_and_prefix():
    model = UHead(small_config()).eval()
    x = features(batch=1, steps=5)
    changed = {name: value.clone() for name, value in x.items()}
    for value in changed.values():
        value[:, 3:] = torch.randn_like(value[:, 3:]) * 100
    with torch.no_grad():
        original = model(x)
        future_changed = model(changed)
        prefix = model.forward_prefix(x, 3)
    torch.testing.assert_close(original[:, :3], future_changed[:, :3], rtol=0, atol=1e-6)
    torch.testing.assert_close(original[:, :3], prefix, rtol=0, atol=1e-6)


def test_compact_transformer_is_causal_and_uses_its_bottleneck():
    model = UHead(small_compact_config()).eval()
    assert model.temporal_adapter.in_features == 16
    assert model.temporal_adapter.out_features == 8
    assert model.classifier.in_features == 8
    x = features(batch=1, steps=5)
    changed = {name: value.clone() for name, value in x.items()}
    for value in changed.values():
        value[:, 3:] = torch.randn_like(value[:, 3:]) * 100
    with torch.no_grad():
        original, future_changed = model(x), model(changed)
    torch.testing.assert_close(original[:, :3], future_changed[:, :3], rtol=0, atol=1e-6)


def test_default_compact_temporal_block_matches_mlp_head_capacity():
    compact = UHead(UHeadConfig(groups=["entropy"], temporal_detector="compact"))
    compact_temporal = sum(parameter.numel() for name, parameter in compact.named_parameters()
                           if not name.startswith("fusion_block."))
    assert compact_temporal == 263_037


def test_compact_and_mlp_capacities_differ_by_132_for_every_ablation():
    ablations = ("lm", "attention", "entropy", "lm_attention", "lm_entropy", "attention_entropy",
                 "lm_attention_entropy")
    for ablation in ablations:
        groups = ablation.split("_")
        compact = UHead(UHeadConfig(groups=groups, temporal_detector="compact"))
        mlp = StateHead(UHeadConfig(groups=groups, temporal_detector="mlp"))
        compact_parameters = sum(parameter.numel() for parameter in compact.parameters())
        mlp_parameters = sum(parameter.numel() for parameter in mlp.parameters())
        assert mlp_parameters - compact_parameters == 132


def test_checkpoint_restores_normalization(tmp_path):
    config = small_config()
    model = UHead(config).eval()
    stats = {name: (torch.full((dim,), 2.0), torch.full((dim,), 3.0)) for name, dim in FEATURE_DIMS.items()}
    model.set_normalization(stats)
    save_checkpoint(model, config, _TrainConfig(), tmp_path)
    checkpoint = torch.load(tmp_path / "uhead.pt", map_location="cpu", weights_only=True)
    assert checkpoint["temporal_detector"] == "transformer"
    assert not any("backbone" in key or "qwen" in key for key in checkpoint["model_state"])
    restored = load_checkpoint(tmp_path).eval()
    x = features(batch=1)
    with torch.no_grad():
        torch.testing.assert_close(model(x), restored(x))
    for name in FEATURE_DIMS:
        torch.testing.assert_close(getattr(restored.fusion_block, f"{name}_mean"), stats[name][0])


def test_compact_checkpoint_restores_as_compact_model(tmp_path):
    config = small_compact_config()
    model = UHead(config).eval()
    save_checkpoint(model, config, _TrainConfig(), tmp_path)
    restored = load_checkpoint(tmp_path).eval()
    assert restored.config.temporal_detector == "compact"
    assert restored.temporal_adapter.out_features == config.compact_model_dim


def test_statehead_mlp_has_same_fusion_and_no_future_access():
    transformer_config = small_config()
    mlp_config = UHeadConfig.from_dict({**transformer_config.__dict__, "temporal_detector": "mlp"})
    model = StateHead(mlp_config).eval()
    assert isinstance(model.fusion_block, type(UHead(transformer_config).fusion_block))
    x = features(batch=1, steps=5)
    changed = {name: value.clone() for name, value in x.items()}
    for value in changed.values():
        value[:, 3:] = torch.randn_like(value[:, 3:]) * 100
    with torch.no_grad():
        original, future_changed = model(x), model(changed)
    assert original.shape == (1, 5)
    torch.testing.assert_close(original[:, :3], future_changed[:, :3], rtol=0, atol=1e-6)


def test_binary_loss_supports_patch_and_sequence_means():
    logits = torch.tensor([[0.0, 4.0, 0.0], [0.0, 0.0, 0.0]])
    targets = torch.zeros_like(logits)
    padding = torch.tensor([[False, True, True], [False, False, False]])
    patch = binary_loss(logits, targets, padding, "patch_mean")
    sequence = binary_loss(logits, targets, padding, "sequence_mean")
    per = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    expected_patch = (per[0, 0] + per[1].sum()) / 4
    expected_sequence = (per[0, 0] + per[1].mean()) / 2
    torch.testing.assert_close(patch, expected_patch)
    torch.testing.assert_close(sequence, expected_sequence)


def test_ragged_probabilities_are_right_padded_only_for_evaluation():
    padded, mask = _pad_probabilities([torch.tensor([.1, .9]).numpy(), torch.tensor([.2, .3, .8]).numpy()])
    assert padded.shape == mask.shape == (2, 3)
    assert mask.tolist() == [[True, True, False], [True, True, True]]


def test_feature_rows_does_not_read_unselected_groups():
    class FakeFeatures:
        def __init__(self):
            self.calls = []
            self.meta = {"max_choices": 4}
        def question(self, name, index):
            self.calls.append(name)
            if name != "probe_logprob":
                raise AssertionError(f"Неожиданно прочитана группа {name}")
            return torch.zeros(3, 15).numpy()
        def video(self, index):
            raise AssertionError("Неожиданно прочитана vision группа")
    fs = FakeFeatures()
    rows = feature_rows(fs, 0, 3, ["confidence"])
    assert list(rows) == ["confidence"]
    assert rows["confidence"].shape == (3, FEATURE_DIMS["confidence"])
    assert fs.calls == ["probe_logprob"]


def test_confidence_representation_keeps_choice_availability_and_is_finite():
    raw = torch.tensor([
        [-1.0, -2.0, float("nan"), float("nan"), *range(11)],
        [-3.0, -4.0, -5.0, -6.0, *range(11)],
    ]).numpy().astype("float32")
    transformed = confidence_features(raw, max_choices=4)
    assert np.isnan(raw[:, :4]).any()  # Raw cache не мутируется.
    assert transformed.shape == (2, FEATURE_DIMS["confidence"])
    assert np.isfinite(transformed).all()
    np.testing.assert_array_equal(transformed[0, 4:8], [1, 1, 0, 0])
    np.testing.assert_array_equal(transformed[1, 4:8], [1, 1, 1, 1])

    config = UHeadConfig(groups=["confidence"], projection_dims={"lm": 8, "attention": 8, "entropy": 4,
                  "vision": 4, "confidence": 4}, model_dim=8, nhead=2, num_layers=1, dropout=0.0)
    model = UHead(config)
    values = torch.from_numpy(transformed)[None]
    mean, std = values[0].mean(0), values[0].std(0)
    model.set_normalization({"confidence": (mean, std)})
    logits = model({"confidence": values})
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits))
    assert torch.isfinite(logits).all() and torch.isfinite(loss)


class _TrainConfig:
    def save(self, path):
        path.write_text("seed: 0\n", encoding="utf-8")
