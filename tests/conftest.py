import pytest


@pytest.fixture
def builder():
    """Real Qwen processor around a toy word-level tokenizer; 2x2 patches, 2 frames per temporal patch."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import (
        PreTrainedTokenizerFast, Qwen2_5_VLProcessor, Qwen2VLImageProcessor, Qwen2VLVideoProcessor,
    )

    from online_vllm.data.inputs import InputBuilder

    tokens = ["<unk>", "<pad>", "<|image_pad|>", "<|video_pad|>",
              "<|vision_start|>", "<|vision_end|>", "question", "assistant", "<|im_end|>", "green"]
    tokenizer = Tokenizer(WordLevel({t: i for i, t in enumerate(tokens)}, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="<unk>", pad_token="<pad>", eos_token="<|im_end|>",
        additional_special_tokens=tokens[2:6],
    )
    template = (
        "{{ messages[0]['content'][0]['text'] }} "
        "<|vision_start|><|video_pad|><|vision_end|>"
        "{% if add_generation_prompt %} assistant{% endif %}"
    )
    processor = Qwen2_5_VLProcessor(
        tokenizer=tokenizer,
        image_processor=Qwen2VLImageProcessor(),
        video_processor=Qwen2VLVideoProcessor(
            size={"shortest_edge": 16, "longest_edge": 16},
            patch_size=2, temporal_patch_size=2, merge_size=2,
        ),
        chat_template=template,
    )
    return InputBuilder(processor)


@pytest.fixture(scope="session")
def tiny_base(tmp_path_factory):
    import torch
    from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

    torch.manual_seed(7)
    config = Qwen2_5_VLConfig(
        text_config=dict(
            vocab_size=64, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=128,
            rope_parameters={"rope_type": "default", "mrope_section": [1, 1, 2]},
        ),
        vision_config=dict(
            depth=1, hidden_size=16, intermediate_size=32, num_heads=2,
            patch_size=2, spatial_merge_size=2, temporal_patch_size=2,
            out_hidden_size=32, window_size=8, fullatt_block_indexes=[0],
        ),
        image_token_id=60, video_token_id=61,
        vision_start_token_id=62, vision_end_token_id=63,
    )
    base = Qwen2_5_VLForConditionalGeneration(config)
    base.generation_config.pad_token_id = 0
    base.generation_config.eos_token_id = 2
    path = tmp_path_factory.mktemp("tiny_qwen")
    base.save_pretrained(path)
    return str(path)
