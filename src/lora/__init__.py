"""Isolated LoRA SFT utilities for VideoQA.

It reads only the established CLEVRER-Proactive group-1 JSONL split; no new
dataset schema or split is invented here.
"""

from .lora import (
    TIME_POINTS,
    ChoiceEvaluationExample,
    ClevrerLoRASFTDataset,
    ClevrerLoRASplitConfig,
    ClevrerLoRAValidationProvider,
    LoRASFTConfig,
    PrefixSelection,
    ValidationProvider,
    assert_lora_isolation,
    build_clevrer_lora_data,
    clevrer_lora_test_provider,
    collate_sft,
    configure_gradient_checkpointing,
    evaluate_validation,
    load_lora_adapter,
    masked_sft_loss,
    prepare_lora_sft_model,
    sample_prefix,
    save_lora_adapter,
    split_validation_videos,
    train_lora_sft,
)

__all__ = [
    "TIME_POINTS", "ChoiceEvaluationExample", "ClevrerLoRASFTDataset", "ClevrerLoRASplitConfig",
    "ClevrerLoRAValidationProvider", "LoRASFTConfig", "PrefixSelection", "ValidationProvider", "assert_lora_isolation",
    "build_clevrer_lora_data", "clevrer_lora_test_provider", "collate_sft", "configure_gradient_checkpointing",
    "evaluate_validation", "load_lora_adapter", "masked_sft_loss", "prepare_lora_sft_model", "sample_prefix",
    "save_lora_adapter", "split_validation_videos", "train_lora_sft",
]
