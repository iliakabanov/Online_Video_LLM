"""Convert <PROCESSED>/<split>_balanced.json into the common schema: <PROCESSED>/<split>.jsonl.

Both are in frames of the source video (25 fps) with inclusive intervals; the common schema adds fps, num_frames and
the video path relative to DATA_ROOT, and moves dataset-specific fields into meta.

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
        fps=FPS, num_frames=N_FRAMES, question=q["question"], choices=q["choices"],
        ask_frame=q["ask_frame"],
        answers=[Interval(a["start_frame"], a["end_frame"], a["answer"]) for a in q["answers"]],
        meta={k: q[k] for k in META if k in q})


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
