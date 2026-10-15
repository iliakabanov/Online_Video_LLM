"""Convert <PROCESSED>/<split>_balanced.json (frames) into the common schema (seconds): <PROCESSED>/<split>.jsonl.

Frame k covers [k / 25, (k + 1) / 25) s, so an interval [start_frame, end_frame] becomes
[start_frame / 25, (end_frame + 1) / 25); accept_frames -> accept_window = accept_frames / 25.

    python -m dataset_builders.clevrer_proactive.export --split train
"""
import argparse
import json
import os

from online_vllm.data.schema import Interval, Record, save_jsonl

from .paths import DATA_ROOT, PROCESSED, RAW

FPS, N_FRAMES = 25, 128
META = ("group", "family", "answer_type", "source", "scene_index", "resolved_at_end", "cause_kind")


def convert(q, video_rel):
    return Record(
        id=q["id"], dataset="clevrer_proactive", split=q["split"], video=f"{video_rel}/{q['video']}",
        duration=N_FRAMES / FPS, question=q["question"], choices=q["choices"],
        ask_time=q["ask_frame"] / FPS, accept_window=q["accept_frames"] / FPS,
        answers=[Interval(a["start_frame"] / FPS, (a["end_frame"] + 1) / FPS, a["answer"]) for a in q["answers"]],
        meta={"fps": FPS, **{k: q[k] for k in META if k in q}})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    args = ap.parse_args()

    qs = json.loads((PROCESSED / f"{args.split}_balanced.json").read_text(encoding="utf-8"))
    video_rel = os.path.relpath(RAW / "videos", DATA_ROOT).replace(os.sep, "/")
    records = [convert(q, video_rel) for q in qs]
    for r in records:
        r.check()
    save_jsonl(records, PROCESSED / f"{args.split}.jsonl")
    print(f"{len(records)} records -> {PROCESSED / f'{args.split}.jsonl'}")


if __name__ == "__main__":
    main()
