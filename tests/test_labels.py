"""Patch labels on the frames the model sees (pure Python, no model)."""
import pytest

from online_vllm.data.labels import PREPARE, SILENT, SPEAK, plan
from online_vllm.data.schema import Interval, Record


def record(start, num_frames=128, answers=None):
    return Record(id="r", dataset="d", split="train", video="v.mp4", fps=25, num_frames=num_frames,
                  question="q", choices=["a", "b"], ask_frame=0,
                  answers=answers or [Interval(start, num_frames - 1, "a")])


def test_every_frame():
    # S = 39, w = 10: S + w = 49 is the last frame of patch (48, 49).
    p = plan(record(39), stride=1, delay=10)
    assert p.num_frames == 50 and p.answer == "a"
    assert p.labels == [SILENT] * 19 + [PREPARE] * 5 + [SPEAK]


def test_speak_moves_to_the_patch_end():
    # S + w = 48 is the first frame of patch (48, 49): speak only after the whole patch.
    p = plan(record(38), stride=1, delay=10)
    assert p.num_frames == 50
    assert p.labels == [SILENT] * 19 + [PREPARE] * 5 + [SPEAK]


def test_stride_in_model_frames():
    # stride 3: S = 39 -> model frame 13; w = 4 -> 17, the end of patch (16, 17).
    p = plan(record(39), stride=3, delay=4)
    assert p.num_frames == 18
    assert p.labels == [SILENT] * 6 + [PREPARE] * 2 + [SPEAK]
    # S = 40 -> model frame ceil(40 / 3) = 14 (source frame 42); 14 + 4 = 18 -> patch (18, 19).
    p = plan(record(40), stride=3, delay=4)
    assert p.num_frames == 20
    assert p.labels == [SILENT] * 7 + [PREPARE] * 2 + [SPEAK]


def test_no_delay():
    p = plan(record(39), stride=1, delay=0)
    assert p.labels == [SILENT] * 19 + [SPEAK]


def test_speak_beyond_the_video_is_dropped():
    assert plan(record(120), stride=1, delay=10) is None
    assert plan(record(117), stride=1, delay=10).num_frames == 128


def test_group2_rejected():
    r = record(0, answers=[Interval(0, 33, "a"), Interval(34, 127, "b")])
    with pytest.raises(ValueError, match="group-1"):
        plan(r, stride=1, delay=10)
