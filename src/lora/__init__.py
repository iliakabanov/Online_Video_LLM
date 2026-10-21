"""Isolated LoRA SFT utilities for VideoQA.

It reads only the established CLEVRER-Proactive group-1 JSONL split; no new
dataset schema or split is invented here.
"""

from .lora import (
    ChoiceEvaluationExample,
    ClevrerLoRASFTDataset,
    ClevrerLoRASplitConfig,
    ClevrerLoRAValidationProvider,
    LoRASFTConfig,
    PrefixSelection,
    ValidationProvider,
    assert_lora_isolation,
    build_clevrer_lora_data,
    collate_sft,
    configure_gradient_checkpointing,
    evaluate_validation,
    masked_sft_loss,
    prepare_lora_sft_model,
    sample_prefix,
    save_lora_adapter,
    train_lora_sft,
)

__all__ = [
    "ChoiceEvaluationExample", "ClevrerLoRASFTDataset", "ClevrerLoRASplitConfig", "ClevrerLoRAValidationProvider",
    "LoRASFTConfig", "PrefixSelection", "ValidationProvider", "assert_lora_isolation", "build_clevrer_lora_data",
    "collate_sft", "configure_gradient_checkpointing", "evaluate_validation", "masked_sft_loss",
    "prepare_lora_sft_model", "sample_prefix", "save_lora_adapter", "train_lora_sft",
]
