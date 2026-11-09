"""State labels of a group-1 record on the frames the model sees.

The model sees every k-th source frame (model frame i = source frame i * k). The vision encoder packs `temporal_patch`
consecutive model frames into one patch, so a decision is possible only after the last frame of a patch, and a patch
is labelled by its last model frame e. With S the first model frame at which the answer is known and w the delay:

    e <  S                silent
    S <= e < S + w        prepare   (the answer is visible; deliberate delay for confidence)
    the first e >= S + w  speak     the sequence ends with this patch, the answer follows

Frames after the speak patch are not part of the training sequence.
"""
from dataclasses import dataclass

from .schema import Record

SILENT, PREPARE, SPEAK = "silent", "prepare", "speak"


@dataclass
class Plan:
    num_frames: int     # model frames in the sequence, a whole number of patches
    labels: list[str]   # one per patch
    answer: str


def whole_patches(record: Record, stride: int, temporal_patch: int = 2) -> int:
    """Number of whole temporal patches in the full video as the model sees it; a trailing partial patch is dropped."""
    return -(-record.num_frames // stride) // temporal_patch


def plan(record: Record, stride: int, delay: int, temporal_patch: int = 2) -> Plan | None:
    """Labels per patch, or None if the speak patch does not fit into the video."""
    if len(record.answers) != 1:
        raise ValueError(f"{record.id}: only group-1 records (a single answer interval) are supported")
    if record.ask_frame != 0:
        raise ValueError(f"{record.id}: the question must be asked at frame 0")
    start = -(-record.answers[0].start_frame // stride)  # first model frame i with i * stride >= S
    speak = start + delay
    patch = max(0, -(-(speak - temporal_patch + 1) // temporal_patch))  # first patch whose last frame >= speak
    num_frames = (patch + 1) * temporal_patch
    if patch + 1 > whole_patches(record, stride, temporal_patch):
        return None
    labels = []
    for p in range(patch + 1):
        end = p * temporal_patch + temporal_patch - 1
        labels.append(SPEAK if p == patch else SILENT if end < start else PREPARE)
    return Plan(num_frames, labels, record.answers[0].answer)
