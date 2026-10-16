"""Offline smoke tests: real tiny Qwen saved and loaded through from_pretrained, real PEFT."""
from unittest.mock import patch

import pytest
import torch

from online_vllm.models import (
    BackboneConfig, LoRAConfig, ModelConfig, StateHeadConfig,
    build_model, load_checkpoint, save_checkpoint,
)


def settings(path, classes=None):
    return ModelConfig(
        backbone=BackboneConfig(model_id=path, dtype="float32"),
        lora=LoRAConfig(r=2, alpha=4, dropout=0, modules=["q_proj", "v_proj"], layers=[0]),
        head=StateHeadConfig(classes=classes or ["silent", "prepare", "speak"], hidden_dims=[16], dropout=0),
    )


def video_inputs():
    # Two frames -> one temporal patch, 2x2 spatial patches -> one visual token.
    ids = torch.tensor([[5, 6, 62, 61, 63]])
    return dict(input_ids=ids, attention_mask=torch.ones_like(ids), mm_token_type_ids=(ids == 61).long() * 2,
                pixel_values_videos=torch.randn(4, 24),
                video_grid_thw=torch.tensor([[1, 2, 2]]), second_per_grid_ts=[1.0])


def test_load_video_backward_generate_and_reload(tiny_base, tmp_path):
    model = build_model(settings(tiny_base), local_files_only=True)
    inputs = video_inputs()
    model.train()
    with patch.object(model.qwen.lm_head, "forward", side_effect=AssertionError("LM head in state path")):
        logits = model(inputs).state_logits
        assert logits.shape == (1, 3)
        torch.nn.functional.cross_entropy(logits, torch.tensor([2])).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for n, p in model.named_parameters() if "lora_" in n)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.state_head.parameters())
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    assert all("layers.0." in n for n, p in model.backbone.named_parameters() if p.requires_grad)
    torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=0.1).step()
    model.eval()
    with patch.object(model.state_head, "forward", side_effect=AssertionError("MLP in language path")):
        assert model(inputs, mode="language").logits.shape == (1, 5, 64)
        answer = model.generate_answer(inputs, min_new_tokens=2, max_new_tokens=2, do_sample=False)
        assert answer.shape == (1, 2)
        cached_answer = model.generate_answer(
            inputs, min_new_tokens=2, max_new_tokens=2, do_sample=False, use_cache=True,
        )
        torch.testing.assert_close(answer, cached_answer)
    expected = model.predict_state(inputs).state_logits
    save_checkpoint(model, tmp_path / "checkpoint")
    restored = load_checkpoint(tmp_path / "checkpoint", local_files_only=True).eval()
    torch.testing.assert_close(restored.predict_state(inputs).state_logits, expected)


@pytest.mark.parametrize("padding", ["left", "right"])
def test_last_non_padding_and_class_count(tiny_base, padding):
    model = build_model(settings(tiny_base, ["wait", "answer"]), local_files_only=True).eval()
    single = dict(input_ids=torch.tensor([[5, 6]]), attention_mask=torch.tensor([[1, 1]]))
    ids, mask = ([0, 5, 6], [0, 1, 1]) if padding == "left" else ([5, 6, 0], [1, 1, 0])
    batch = dict(input_ids=torch.tensor([ids, [7, 8, 9]]),
                 attention_mask=torch.tensor([mask, [1, 1, 1]]))
    result = model.predict_state(batch).state_logits
    assert result.shape == (2, 2)
    torch.testing.assert_close(result[:1], model.predict_state(single).state_logits)
    with pytest.raises(ValueError, match="non-padding"):
        model(dict(input_ids=torch.tensor([[0]]), attention_mask=torch.tensor([[0]])))


def test_8bit_needs_cuda(tiny_base):
    config = settings(tiny_base)
    config.backbone.quantization = "8bit"
    with pytest.raises(ValueError, match="CUDA"):
        build_model(config, local_files_only=True, device="cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="bitsandbytes int8 needs CUDA")
def test_8bit_language_model_only(tiny_base):
    import bitsandbytes as bnb

    config = settings(tiny_base)
    config.lora.enabled = False
    exact = build_model(config, local_files_only=True, device="cuda").eval()
    config.backbone.quantization = "8bit"
    model = build_model(config, local_files_only=True, device="cuda").eval()
    int8 = [n for n, m in model.qwen.named_modules() if isinstance(m, bnb.nn.Linear8bitLt)]
    assert int8 and all(n.startswith("model.language_model.") for n in int8)
    assert not any(isinstance(m, bnb.nn.Linear8bitLt) for m in model.qwen.model.visual.modules())
    assert next(model.state_head.parameters()).dtype == torch.float32
    inputs = {k: v.cuda() if torch.is_tensor(v) else v for k, v in video_inputs().items()}
    mask = torch.zeros_like(inputs["input_ids"], dtype=torch.bool)
    mask[0, 3] = True
    with torch.no_grad():
        a, b = model.decision_features(inputs, mask).float(), exact.decision_features(inputs, mask).float()
    assert torch.isfinite(a).all() and torch.nn.functional.cosine_similarity(a, b).item() > 0.9


def test_bad_lora_target_fails(tiny_base):
    config = settings(tiny_base)
    config.lora.modules = ["missing_projection"]
    with pytest.raises(ValueError, match="Unmatched LoRA"):
        build_model(config, local_files_only=True)


def test_head_only_checkpoint(tiny_base, tmp_path):
    config = settings(tiny_base)
    config.lora.enabled = False
    model = build_model(config, local_files_only=True).eval()
    assert all(n.startswith("state_head.") for n, p in model.named_parameters() if p.requires_grad)
    inputs = video_inputs()
    expected = model.predict_state(inputs).state_logits
    save_checkpoint(model, tmp_path)
    restored = load_checkpoint(tmp_path, local_files_only=True).eval()
    torch.testing.assert_close(restored.predict_state(inputs).state_logits, expected)
