"""Common record format of all proactive-QA datasets, the only thing training and evaluation see.

Times are frame indices of the source video; `fps` gives seconds (frame k is time k / fps). A question is asked at
`ask_frame`; `answers` are consecutive intervals [start_frame, end_frame] (both inclusive) with the answer that is
correct on those frames. A single interval reaching the last frame means the answer appears once and does not change;
several intervals mean the answer changes over time.

    {"id": ..., "dataset": "clevrer_proactive", "split": "train", "video": "raw/clevrer/videos/video_00001.mp4",
     "fps": 25, "num_frames": 128, "question": ..., "choices": [...], "ask_frame": 0,
     "answers": [{"start_frame": 34, "end_frame": 127, "answer": "green"}], "meta": {...}}

`video` is relative to DATA_ROOT; `meta` keeps dataset-specific fields (group, family, answer type ...).
"""
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator


@dataclass
class Interval:
    start_frame: int
    end_frame: int
    answer: str


@dataclass
class Record:
    id: str
    dataset: str
    split: str
    video: str
    fps: float
    num_frames: int
    question: str
    choices: list[str]
    ask_frame: int
    answers: list[Interval]
    meta: dict = field(default_factory=dict)

    def answer_at(self, frame: int):
        """The correct answer on this frame, None before the first interval (not answerable yet)."""
        for a in self.answers:
            if a.start_frame <= frame <= a.end_frame:
                return a.answer
        return None

    def check(self):
        """Invariants every dataset must satisfy; raises ValueError."""
        if not self.answers:
            raise ValueError(f"{self.id}: no answers")
        if any(a.answer not in self.choices for a in self.answers):
            raise ValueError(f"{self.id}: answer not among the choices")
        if len(set(self.choices)) != len(self.choices):
            raise ValueError(f"{self.id}: duplicate choices")
        for a, b in zip(self.answers, self.answers[1:]):
            if b.start_frame != a.end_frame + 1:
                raise ValueError(f"{self.id}: intervals are not consecutive")
        if any(a.start_frame > a.end_frame for a in self.answers) or self.answers[0].start_frame < self.ask_frame \
                or self.answers[-1].end_frame >= self.num_frames:
            raise ValueError(f"{self.id}: bad interval bounds")


def to_dict(r: Record) -> dict:
    return asdict(r)


def from_dict(d: dict) -> Record:
    return Record(**{**d, "answers": [Interval(**a) for a in d["answers"]]})


def save_jsonl(records, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(to_dict(r), ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> Iterator[Record]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield from_dict(json.loads(line))
