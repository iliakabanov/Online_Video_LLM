"""Convert <PROCESSED>/<split>_balanced.json (frames) into the common schema (seconds): <PROCESSED>/<split>.jsonl.

Frame k covers [k / 25, (k + 1) / 25) s, so an interval [start_frame, end_frame] becomes
[start_frame / 25, (end_frame + 1) / 25); accept_frames -> accept_window = accept_frames / 25.

    python -m dataset_builders.clevrer_proactive.export --split train
"""
import argparse
import json
import os

from online_vllm.data.schema import Interval, Record, save_jsonl

from .paths import DATA_ROOT, PROCESSED, video_path

FPS, N_FRAMES = 25, 128
META = ("group", "family", "answer_type", "source", "scene_index", "resolved_at_end", "cause_kind")


def video_rel(q):
    """The video relative to DATA_ROOT, where it actually is (see paths.video_path)."""
    return os.path.relpath(video_path(q["split"], q["video"]), DATA_ROOT).replace(os.sep, "/")


def convert(q):
    return Record(
        id=q["id"], dataset="clevrer_proactive", split=q["split"], video=video_rel(q),
        duration=N_FRAMES / FPS, question=q["question"], choices=q["choices"],
        ask_time=q["ask_frame"] / FPS, accept_window=q["accept_frames"] / FPS,
        answers=[Interval(a["start_frame"] / FPS, (a["end_frame"] + 1) / FPS, a["answer"]) for a in q["answers"]],
        meta={"fps": FPS, **{k: q[k] for k in META if k in q}})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    args = ap.parse_args()

    qs = json.loads((PROCESSED / f"{args.split}_balanced.json").read_text(encoding="utf-8"))
    records = [convert(q) for q in qs]
    for r in records:
        r.check()
    save_jsonl(records, PROCESSED / f"{args.split}.jsonl")
    print(f"{len(records)} records -> {PROCESSED / f'{args.split}.jsonl'}")


if __name__ == "__main__":
    main()
