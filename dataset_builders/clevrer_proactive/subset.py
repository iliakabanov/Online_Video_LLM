"""A first training subset of <split>_balanced.json: group 1 only, no cause-effect questions, no "no" that is known only
at the end of the video. Written in both formats: <PROCESSED>/<name>.json (generator format, frames) and
<PROCESSED>/<name>.jsonl (common schema, as export.py).

Dropped:
  - group 2 (an answer per interval);
  - cause_collision, cause_exit;
  - will_collide, will_exit, will_start entirely: their "no" is known only at the end, and without it every answer is "yes";
  - clevrer_exist answered only at the end (resolved_at_end);
  - official queries about the "last" object (to collide with X / that enters / that exits): the generator checks
    finality for exist only, so these are labelled answerable at the latest such event so far, while a viewer can know
    it is the last one only at the end of the video. Every other official query is final at its start_frame.
In what is left an answer appears at an event inside the video and does not change.

Each subset is also halved by video, in scene_index order (README §11): <name>_lora.jsonl, the first half of the
videos, for fine-tuning Qwen with LoRA; <name>_features.jsonl, the second half, for the features of the readiness
heads. All questions of a video stay in one half, so a fine-tuned model never gives features of a video it learned.

    python -m dataset_builders.clevrer_proactive.subset   # -> train_group1.*, val_group1.*, *_lora.jsonl, *_features.jsonl
"""
import json
import re

from online_vllm.data.schema import save_jsonl

from .balance import stats
from .export import convert
from .paths import PROCESSED

DROP_FAMILIES = {"cause_collision", "cause_exit", "will_collide", "will_exit", "will_start"}
SPLITS = {"train": "train_group1", "validation": "val_group1"}
HALVES = ("lora", "features")  # first and second half of the videos


def asks_last(q):
    """Official query whose program orders events with 'last' (the official templates always say it in the text)."""
    return q["family"].startswith("clevrer_query") and re.search(r"\blast\b", q["question"]) is not None


def keep(q):
    return (q["group"] == 1 and q["family"] not in DROP_FAMILIES and not q.get("resolved_at_end")
            and not asks_last(q))


def halves(records):
    """The questions of the first and of the second half of the videos by scene_index (the first half is the
    smaller one for an odd number of videos), each in its original order."""
    scenes = {}
    for r in records:
        scenes.setdefault(r.video, r.meta["scene_index"])
    videos = sorted(scenes, key=scenes.__getitem__)
    first = set(videos[:len(videos) // 2])
    return [r for r in records if r.video in first], [r for r in records if r.video not in first]


def main():
    for split, name in SPLITS.items():
        qs = json.loads((PROCESSED / f"{split}_balanced.json").read_text(encoding="utf-8"))
        kept = [q for q in qs if keep(q)]
        stats(kept, name)
        (PROCESSED / f"{name}.json").write_text(json.dumps(kept, ensure_ascii=False, indent=1), encoding="utf-8")
        records = [convert(q) for q in kept]
        for r in records:
            r.check()
        save_jsonl(records, PROCESSED / f"{name}.jsonl")
        print(f"{len(qs)} -> {len(kept)} questions -> {PROCESSED / name}.json, .jsonl")
        for half, part in zip(HALVES, halves(records)):
            save_jsonl(part, PROCESSED / f"{name}_{half}.jsonl")
            print(f"  {half}: {len({r.video for r in part})} videos, {len(part)} questions -> {name}_{half}.jsonl")


if __name__ == "__main__":
    main()
