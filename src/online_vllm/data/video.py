"""Reading a video as a stream of frames sampled at a fixed rate."""
from pathlib import Path

import imageio_ffmpeg
import numpy as np


def read_frames(path: Path, fps: float, max_side: int | None = None):
    """Frames at t = i / fps (i = 0, 1, ...) as (times [N], frames [N, H, W, 3] uint8)."""
    vf = f"fps={fps}:round=up" + (f",scale='if(gt(iw,ih),{max_side},-2)':'if(gt(iw,ih),-2,{max_side})'" if max_side else "")
    reader = imageio_ffmpeg.read_frames(str(path), output_params=["-vf", vf])
    meta = next(reader)
    w, h = meta["size"]
    frames = [np.frombuffer(b, np.uint8).reshape(h, w, 3) for b in reader]
    return np.arange(len(frames)) / fps, np.stack(frames) if frames else np.zeros((0, h, w, 3), np.uint8)


def read_every(path: Path, stride: int = 1, size: tuple[int, int] | None = None):
    """Source frames 0, stride, 2 * stride, ... as uint8 [N, H, W, 3], resized to size = (width, height) if given."""
    reader = imageio_ffmpeg.read_frames(str(path), output_params=["-vf", f"scale={size[0]}:{size[1]}"] if size else None)
    meta = next(reader)
    w, h = size or meta["size"]
    frames = [np.frombuffer(b, np.uint8).reshape(h, w, 3) for i, b in enumerate(reader) if i % stride == 0]
    return np.stack(frames) if frames else np.zeros((0, h, w, 3), np.uint8)
