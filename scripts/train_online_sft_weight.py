"""Online SFT (scripts/train_online_sft.py) as a reference run, with only the weight of the answer loss changed:
loss = slot loss + answer_weight * answer loss.

Everything else is taken from the reference run (online.json, sft_config.json): --init-run and --init-adapter (null:
a fresh LoRA on the base model, --init-adapter none), slot targets (--tags, --step-window, --sigmoid-slope), --taus,
--selection, the training videos (the reference must not use --skip-offline-videos or --train-videos), epochs, lr, batch
size, gradient accumulation, gradient checkpointing and seed. The optimizer settings that train_online_sft.py has no
flag for (weight_decay, betas, grad_clip) must equal its defaults, otherwise the script stops. The weight is chosen on
validation by scripts/choose_answer_weight.py.

    python scripts/train_online_sft_weight.py --answer-weight 0.5
    python scripts/train_online_sft_weight.py --reference outputs/listen_sft_10ep --answer-weight 0.25 \\
        --output outputs/listen_sft_10ep_aw0.25
"""
import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import _paths  # noqa: F401  (src/ on the path)
from lora.lora import LoRASFTConfig

REPO = Path(__file__).resolve().parents[1]


def reference_command(reference: Path, answer_weight: float, output: str, epochs: int | None) -> list[str]:
    run = json.loads((reference / "online.json").read_text(encoding="utf-8"))
    sft = json.loads((reference / "sft_config.json").read_text(encoding="utf-8"))
    split = json.loads((reference / "split.json").read_text(encoding="utf-8"))
    if run.get("resumed_from"):
        raise SystemExit(f"{reference} continues another run ({run['resumed_from']}): give the original run")
    online_train = split.get("online_train", {})
    if online_train.get("skipped_offline_videos"):
        raise SystemExit(f"{reference} was trained with --skip-offline-videos: not supported here")
    defaults = asdict(LoRASFTConfig())
    for key in ("weight_decay", "betas", "grad_clip"):
        saved, default = sft[key], defaults[key]
        if key == "betas":
            saved, default = list(saved), list(default)
        if saved != default:
            raise SystemExit(f"{reference}: {key} = {sft[key]} differs from the default {defaults[key]}, "
                             "train_online_sft.py has no flag for it")
    objective = run["objective"]
    init_adapter = run.get("init_adapter")
    command = [sys.executable, str(REPO / "scripts" / "train_online_sft.py"),
               "--init-run", run["init_run"],
               "--init-adapter", "none" if init_adapter is None else Path(init_adapter).name,
               "--tags", objective["tags"], "--step-window", str(objective["step_window"]),
               "--sigmoid-slope", str(objective["sigmoid_slope"]), "--answer-weight", f"{answer_weight:g}",
               "--taus", *[f"{tau:g}" for tau in run["taus"]], "--selection", run["selection"],
               "--epochs", str(sft["epochs"] if epochs is None else epochs), "--lr", f"{sft['learning_rate']:g}",
               "--batch-size", str(sft["batch_size"]), "--gradient-accumulation", str(sft["gradient_accumulation"]),
               "--gradient-checkpointing" if sft["gradient_checkpointing"] else "--no-gradient-checkpointing",
               "--seed", str(sft["seed"]), "--output", output]
    if online_train.get("videos") is not None:
        command += ["--train-videos", str(online_train["videos"])]
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", type=Path, default=Path("outputs/listen_sft_10ep"),
                        help="online run whose settings are repeated (default outputs/listen_sft_10ep)")
    parser.add_argument("--answer-weight", type=float, required=True)
    parser.add_argument("--output", help="default <reference>_aw<answer_weight>")
    parser.add_argument("--epochs", type=int, help="default: as the reference")
    parser.add_argument("--dry-run", action="store_true", help="print the command of train_online_sft.py and exit")
    args = parser.parse_args()
    if args.answer_weight < 0:
        parser.error("--answer-weight must be >= 0")

    output = args.output or f"{args.reference}_aw{args.answer_weight:g}"
    command = reference_command(args.reference, args.answer_weight, output, args.epochs)
    print(shlex.join(command), flush=True)
    if not args.dry_run:
        sys.exit(subprocess.call(command))


if __name__ == "__main__":
    main()
