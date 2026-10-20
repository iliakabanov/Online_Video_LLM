"""Адаптер кешированных признаков, потоковая нормализация и padding батчей."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.utils.data import Dataset

from online_vllm.data.labels import plan
from .config import FEATURE_DIMS
from .labels import TagMode, make_targets

if TYPE_CHECKING:
    from online_vllm.train.features import FeatureSet


CONFIDENCE_SLOTS = 4


@dataclass(frozen=True)
class LabelledItem:
    index: int
    target: np.ndarray
    target_pair: int
    visible_pair: int


def labelled(fs: FeatureSet, config, tags: TagMode, step_window: int, sigmoid_slope: float) -> tuple[list[LabelledItem], int]:
    """Создаёт бинарные GT для всех доступных пар; plan задаёт target_pair и видимость ответа."""
    items, dropped = [], 0
    for i, record in enumerate(fs.records):
        state_plan = plan(record, config.video.frame_stride, config.labels.speak_delay, fs.meta["temporal_patch"])
        if state_plan is None:
            dropped += 1
        else:
            target_pair = len(state_plan.labels) - 1
            length = fs.patches[i]
            if target_pair >= length:
                raise ValueError(f"{record.id}: target_pair is absent from extracted features")
            visible_pair = next(j for j, value in enumerate(state_plan.labels) if value != "silent")
            items.append(LabelledItem(i, make_targets(length, target_pair, tags, step_window, sigmoid_slope),
                                     target_pair, visible_pair))
    return items, dropped


def confidence_features(raw: np.ndarray, max_choices: int) -> np.ndarray:
    """Делает cached probe конечным, сохраняя NaN-маску недоступных вариантов.

    Raw ``probe_logprob`` намеренно остаётся неизменённым: первые
    ``max_choices`` столбцов содержат log P букв и NaN после конца списка
    вариантов. В detector подаются значения с NaN→0 и отдельная бинарная
    маска доступности; поэтому ноль не смешивает отсутствующий вариант с
    действительным значением.
    """
    if raw.ndim != 2 or not 0 < max_choices < raw.shape[1] or max_choices > CONFIDENCE_SLOTS:
        raise ValueError("Некорректная форма probe_logprob или max_choices")
    letters, tail = raw[:, :max_choices], raw[:, max_choices:]
    valid = np.isfinite(letters)
    if not np.isfinite(tail).all():
        raise ValueError("В top-logprob или entropy probe_logprob обнаружен NaN/inf")
    values = np.zeros((len(raw), CONFIDENCE_SLOTS), dtype=np.float32)
    availability = np.zeros_like(values)
    values[:, :max_choices] = np.where(valid, letters, 0.0)
    availability[:, :max_choices] = valid
    return np.concatenate([values, availability, tail.astype(np.float32, copy=False)], axis=1)


def feature_rows(fs: FeatureSet, index: int, n: int, groups: list[str]) -> dict[str, np.ndarray]:
    """Возвращает исходные признаки первых n пар без изменения порядка координат."""
    if not groups or len(groups) != len(set(groups)) or set(groups) - set(FEATURE_DIMS):
        raise ValueError("groups должны быть непустым списком уникальных известных групп")
    loaders = {
        "lm": lambda: [fs.question(name, index)[:n] for name in ("lm_last", "lm_mean", "probe_hidden")],
        "attention": lambda: [fs.question(name, index)[:n] for name in ("att_last", "att_mean", "att_probe")],
        "entropy": lambda: [fs.question("att_entropy", index)[:n]],
        "vision": lambda: [fs.video(index)[:n]],
        "confidence": lambda: [confidence_features(fs.question("probe_logprob", index)[:n], fs.meta["max_choices"])],
    }
    arrays = {name: loaders[name]() for name in groups}
    result = {name: np.concatenate([x.reshape(n, -1) for x in arrays[name]], axis=1).astype(np.float32)
              for name in groups}
    for name, array in result.items():
        if array.shape != (n, FEATURE_DIMS[name]):
            raise ValueError(f"{name}: expected {(n, FEATURE_DIMS[name])}, got {array.shape}; incompatible extraction")
    return result


class ReadinessDataset(Dataset):
    """Одна запись датасета — одна отдельная видеопоследовательность, не объединённая с другими."""
    def __init__(self, fs: FeatureSet, items: list[LabelledItem], groups: list[str]):
        self.fs, self.items, self.groups = fs, items, groups

    def __len__(self):
        return len(self.items)

    def __getitem__(self, position):
        item = self.items[position]
        n = len(item.target)
        return feature_rows(self.fs, item.index, n, self.groups), item.target, item


def collate(batch):
    """Дополняет только временную ось; True в padding_mask означает недействительную позицию."""
    rows, targets, items = zip(*batch)
    max_length = max(len(x) for x in targets)
    features = {name: torch.zeros(len(batch), max_length, value.shape[1], dtype=torch.float32)
                for name, value in rows[0].items()}
    target = torch.zeros((len(batch), max_length), dtype=torch.float32)
    padding_mask = torch.ones(len(batch), max_length, dtype=torch.bool)
    for b, (values, labels) in enumerate(zip(rows, targets)):
        n = len(labels)
        target[b, :n] = torch.from_numpy(labels)
        padding_mask[b, :n] = False
        for name, value in values.items():
            features[name][b, :n] = torch.from_numpy(value)
    return features, target, padding_mask, items


def fit_normalization(dataset: ReadinessDataset) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Потоковые train-only mean/std в float32 без материализации всего набора в памяти."""
    count = {name: 0 for name in dataset.groups}
    total = {name: np.zeros(FEATURE_DIMS[name], dtype=np.float64) for name in dataset.groups}
    square = {name: np.zeros(FEATURE_DIMS[name], dtype=np.float64) for name in dataset.groups}
    for item in dataset.items:
        values = feature_rows(dataset.fs, item.index, len(item.target), dataset.groups)
        for name, x in values.items():
            x = x.astype(np.float64, copy=False)
            count[name] += len(x)
            total[name] += x.sum(0)
            square[name] += np.square(x).sum(0)
    stats = {}
    for name in dataset.groups:
        if not count[name]:
            raise ValueError(f"No non-padding rows for {name}")
        mean = total[name] / count[name]
        variance = np.maximum(square[name] / count[name] - np.square(mean), 0.0)
        stats[name] = (torch.from_numpy(mean.astype(np.float32)), torch.from_numpy(np.sqrt(variance).astype(np.float32)))
    return stats
