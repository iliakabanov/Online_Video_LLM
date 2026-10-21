"""Accuracy of a LoRA SFT run on its test half: the half of the LoRA validation videos that training never saw
(split.json of the run). Same prefixes as the validation during training (start / target / answerable / end), at the
training frame size, overall and per family. --adapter none evaluates the base Qwen on the same questions.

python scripts/evaluate_lora_sft.py --run outputs/lora_sft --adapter best
python scripts/evaluate_lora_sft.py --run outputs/lora_sft --adapter none     # the same test half, without LoRA
"""
import argparse
import json
import os
from dataclasses import replace
from pathlib import Path

import torch

import _paths  # noqa: F401  (src/ on the path)
from lora import TIME_POINTS, clevrer_lora_test_provider, evaluate_validation, load_lora_adapter
from online_vllm.data.inputs import InputBuilder
from online_vllm.models import ModelConfig, build_model

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="output directory of train_lora_sft.py (model.yaml, split.json)")
    parser.add_argument("--adapter", choices=("best", "last", "none"), default="best")
    parser.add_argument("--data-root", default=os.environ.get("DATA_ROOT", REPO / "data"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", help="metrics JSON; default <run>/test_<adapter>.json")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    run = Path(args.run)
    config = ModelConfig.load(run / "model.yaml")
    if args.adapter == "none":
        config = replace(config, lora=replace(config.lora, enabled=False))
    model = build_model(config, local_files_only=args.local_files_only, device=args.device)
    if args.adapter != "none":
        load_lora_adapter(model, run / args.adapter)
    model.eval()
    builder = InputBuilder.from_config(model.config, local_files_only=args.local_files_only)
    provider, split = clevrer_lora_test_provider(run, data_root=args.data_root, builder=builder)
    width, height = split.train_frame_size
    result = evaluate_validation(model, provider, resolutions=((width, height),), device=args.device,
                                 desc=f"Test, adapter {args.adapter}")[f"{width}x{height}"]
    predictions = result.pop("predictions")

    metadata = {"run": str(run), "adapter": args.adapter, "split": "test", "questions": len(provider.records),
                "videos": len({record.video for record in provider.records})}
    adapter_metadata = run / args.adapter / "sft_metadata.json"
    if adapter_metadata.exists():
        metadata["adapter_metadata"] = json.loads(adapter_metadata.read_text(encoding="utf-8"))
    output = Path(args.output) if args.output else run / f"test_{args.adapter}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"metadata": metadata, **result}, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    output.with_name(f"{output.stem}_predictions.json").write_text(
        json.dumps(predictions, allow_nan=False) + "\n", encoding="utf-8")
    print(f"test, adapter {args.adapter}, {metadata['questions']} questions / {metadata['videos']} videos: " +
          ", ".join(f"{point} {100 * result[f'{point}_micro_accuracy']:.2f}%" for point in TIME_POINTS) +
          f" -> {output}")


if __name__ == "__main__":
    main()
