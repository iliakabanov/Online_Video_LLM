"""Непрерывные бинарные метки готовности для временных рядов пар кадров."""
from typing import Literal

import numpy as np


TagMode = Literal["hard", "step", "sigmoid"]


def make_targets(length: int, target_pair: int, tags: TagMode, step_window: int = 5,
                 sigmoid_slope: float = 0.65) -> np.ndarray:
    """Возвращает вещественные метки формы [P] без обрезания последовательности после target_pair."""
    if length < 1:
        raise ValueError("length must be positive")
    if not 0 <= target_pair < length:
        raise ValueError("target_pair must belong to the sequence")
    if step_window < 0:
        raise ValueError("step_window must be nonnegative")
    if sigmoid_slope <= 0:
        raise ValueError("sigmoid_slope must be positive")
    positions = np.arange(length, dtype=np.float32)
    if tags == "hard":
        return (positions >= target_pair).astype(np.float32)
    if tags == "step":
        result = np.zeros(length, dtype=np.float32)
        result[max(0, target_pair - step_window):target_pair] = 0.5
        result[target_pair:] = 1.0
        return result
    if tags == "sigmoid":
        values = sigmoid_slope * (positions - target_pair) + np.log(9.0)
        return (1.0 / (1.0 + np.exp(-values))).astype(np.float32)
    raise ValueError(f"Unknown tags mode: {tags}")
