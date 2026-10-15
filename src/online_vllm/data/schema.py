"""Common record format of all proactive-QA datasets, the only thing training and evaluation see.

Times are in seconds (datasets differ in fps). A question is asked at `ask_time`; `answers` are consecutive intervals
[start, end) with the answer that is correct during that interval; an answer counts as on time if given within
`accept_window` seconds after the start of its interval. A single interval reaching the end of the video means the
answer appears once and does not change; several intervals mean the answer changes over time.

    {"id": ..., "dataset": "clevrer_proactive", "split": "train", "video": "raw/clevrer/videos/video_00001.mp4",
     "duration": 5.12, "question": ..., "choices": [...], "ask_time": 0.0, "accept_window": 0.48,
     "answers": [{"start": 1.36, "end": 5.12, "answer": "green"}], "meta": {...}}

`video` is relative to DATA_ROOT; `meta` keeps dataset-specific fields (group, family, answer type, source fps ...).
"""
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator


@dataclass
class Interval:
    start: float
    end: float
    answer: str


@dataclass
class Record:
    id: str
    dataset: str
    split: str
    video: str
    duration: float
    question: str
    choices: list[str]
    ask_time: float
    accept_window: float
    answers: list[Interval]
    meta: dict = field(default_factory=dict)

    def answer_at(self, t: float):
        """The correct answer at time t, None before the first interval (not answerable yet)."""
        for a in self.answers:
            if a.start <= t < a.end:
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
            if abs(a.end - b.start) > 1e-6:
                raise ValueError(f"{self.id}: intervals are not consecutive")
        if any(a.start >= a.end for a in self.answers) or self.answers[0].start < self.ask_time - 1e-6:
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
