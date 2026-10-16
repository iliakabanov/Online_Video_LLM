"""Invariants of the built CLEVRER-Proactive files (skipped if a file has not been built yet)."""
import json

import pytest

from dataset_builders.clevrer_proactive.paths import PROCESSED
from online_vllm.data.schema import load_jsonl

N_CHOICES = {"yesno": 2, "material": 2, "shape": 3}


@pytest.mark.parametrize("split", ["train", "validation"])
def test_balanced(split):
    path = PROCESSED / f"{split}_balanced.json"
    if not path.exists():
        pytest.skip(f"{path} not built")
    qs = json.loads(path.read_text(encoding="utf-8"))
    assert len({q["id"] for q in qs}) == len(qs)
    for q in qs:
        assert len(q["choices"]) == N_CHOICES.get(q["answer_type"], 4), q["id"]
        assert len(set(q["choices"])) == len(q["choices"]), q["id"]
        assert all(a["answer"] in q["choices"] for a in q["answers"]), q["id"]
        iv = q["answers"]
        assert all(b["start_frame"] == a["end_frame"] + 1 for a, b in zip(iv, iv[1:])), q["id"]
        assert iv[-1]["end_frame"] == 127, q["id"]
        assert q["ask_frame"] == 0
        if q["group"] == 1:
            assert len(iv) == 1 and iv[0]["start_frame"] >= 25, q["id"]
        else:
            assert len(iv) >= 2 and any(a["start_frame"] >= 25 for a in iv[1:]), q["id"]


@pytest.mark.parametrize("split", ["train", "validation"])
def test_common_schema(split):
    path = PROCESSED / f"{split}.jsonl"
    if not path.exists():
        pytest.skip(f"{path} not built")
    n = 0
    for r in load_jsonl(path):
        r.check()
        assert r.answers[-1].end_frame == r.num_frames - 1
        n += 1
    assert n > 0
