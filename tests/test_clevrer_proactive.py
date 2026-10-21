"""Invariants of the built CLEVRER-Proactive files (skipped if a file has not been built yet)."""
import json

import pytest

from dataset_builders.clevrer_proactive.paths import PROCESSED
from dataset_builders.clevrer_proactive.subset import HALVES, SPLITS, asks_last, keep
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


def test_last_queries_are_not_kept():
    q = {"group": 1, "family": "clevrer_query_shape", "question": "What shape is the last object to collide with the cube?"}
    assert asks_last(q) and not keep(q)
    assert keep({**q, "question": "What shape is the first object to collide with the cube?"})
    # Only official queries are filtered: the word alone does not drop a new-family question.
    assert not asks_last({**q, "family": "first_hit", "question": "Which object collides last?"})


@pytest.mark.parametrize("name", SPLITS.values())
def test_group1_subset_matches_keep(name):
    path = PROCESSED / f"{name}.json"
    if not path.exists():
        pytest.skip(f"{path} not built")
    qs = json.loads(path.read_text(encoding="utf-8"))
    dropped = [q["id"] for q in qs if not keep(q)]
    assert not dropped, f"{len(dropped)} questions fail subset.keep (e.g. {dropped[:3]}); rerun subset"


@pytest.mark.parametrize("name", SPLITS.values())
def test_lora_and_features_halves_split_the_videos(name):
    paths = [PROCESSED / f"{name}_{half}.jsonl" for half in HALVES]
    if not all(path.exists() for path in paths):
        pytest.skip(f"halves of {name} not built")
    whole = list(load_jsonl(PROCESSED / f"{name}.jsonl"))
    lora, features = (list(load_jsonl(path)) for path in paths)
    assert sorted(r.id for r in lora + features) == sorted(r.id for r in whole)  # every question, once
    lora_videos, feature_videos = {r.video for r in lora}, {r.video for r in features}
    assert not lora_videos & feature_videos  # no video in both halves
    assert len(feature_videos) - len(lora_videos) in (0, 1)
    assert max(r.meta["scene_index"] for r in lora) < min(r.meta["scene_index"] for r in features)
