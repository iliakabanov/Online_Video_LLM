"""Optional smoke run with the configured pretrained 3B weights and synthetic frames.

Unlike the offline pytest smoke, this downloads the model on first use.
python scripts/smoke_model.py --device cuda
"""
import argparse
import logging

import numpy as np
import torch

from online_vllm.data.inputs import InputBuilder, VideoPrefix
from online_vllm.models import ModelConfig, build_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/model/qwen_3b_state.yaml")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    config = ModelConfig.load(args.config)
    if args.device == "cpu":
        config.backbone.dtype = "float32"
    model = build_model(config, local_files_only=args.local_files_only).to(args.device).eval()
    builder = InputBuilder.from_config(model.config, local_files_only=args.local_files_only)
    frames = np.zeros((2, 56, 56, 3), dtype=np.uint8)
    inputs = builder.build_observation("What is visible?", VideoPrefix(frames, fps=2), t=0.5)
    inputs = inputs.to(args.device)
    logits = model.predict_state(inputs).state_logits
    answer_ids = model.generate_answer(inputs, max_new_tokens=4, do_sample=False)
    assert logits.shape == (1, len(config.head.classes)) and torch.isfinite(logits).all()
    print("State logits:", logits.cpu().tolist())
    print("Answer:", builder.processor.batch_decode(answer_ids, skip_special_tokens=True))
    print("Resolved revision:", model.config.backbone.revision)


if __name__ == "__main__":
    main()
