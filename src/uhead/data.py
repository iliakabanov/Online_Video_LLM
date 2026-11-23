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


def labelled(fs: FeatureSet, config, tags: TagMode, step_window: int, sigmoid_slope: float,
             gaussian_sigma: float = 2.0) -> tuple[list[LabelledItem], int]:
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
            items.append(LabelledItem(i, make_targets(length, target_pair, tags, step_window, sigmoid_slope,
                                                      gaussian_sigma),
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


GROUP_ARRAYS = {"lm": ("lm_last", "lm_mean", "probe_hidden"), "attention": ("att_last", "att_mean", "att_probe"),
                "entropy": ("att_entropy",), "vision": ("vis_mean",), "confidence": ("probe_logprob",),
                "listen_token_logit": ("listen_token_logit",)}
# Группы с выхода языковой модели (после её последнего слоя): при max_layer меньше числа слоёв их нет.
OUTPUT_GROUPS = ("confidence", "listen_token_logit", "last_hidden")


def _depth(meta: dict, max_layer: int | None) -> int:
    """Число первых слоёв языковой модели, признаки которых видит голова (max_layer; None — все)."""
    layers = int(meta["layers"])
    if max_layer is None:
        return layers
    if not 1 <= max_layer <= layers:
        raise ValueError(f"max_layer должен лежать в 1..{layers}, получено {max_layer}")
    return int(max_layer)


def lm_layer_indices(meta: dict, max_layer: int | None = None) -> list[int]:
    """Индексы извлечённых слоёв (meta lm_layers, нумерация с 1) не глубже max_layer — оси слоя массивов группы lm."""
    depth = _depth(meta, max_layer)
    return [i for i, layer in enumerate(meta["lm_layers"]) if layer <= depth]


def _limit(name: str, array: np.ndarray, meta: dict, max_layer: int | None) -> np.ndarray:
    """Массив name [пары, ...] только со слоями 1..max_layer: у lm — ось извлечённых слоёв, у внимания и энтропии —
    столбцы голов, которые идут по слоям подряд (столбец = (слой − 1) · heads + голова)."""
    if max_layer is None:
        return array
    if name in GROUP_ARRAYS["lm"]:
        return array[:, lm_layer_indices(meta, max_layer)]
    columns = _depth(meta, max_layer) * int(meta["heads"])
    if name in GROUP_ARRAYS["attention"]:
        return array[:, :columns]
    if name == "att_entropy":
        return array[:, :, :columns]
    return array


def feature_dims(meta: dict, max_layer: int | None = None) -> dict[str, int]:
    """Размеры групп в этих признаках (meta.json): у признаков онлайн-модели attention шире (группа <listen>) и есть
    listen_token_logit; группы, массивов которых нет, не входят. max_layer — только слои языковой модели 1..max_layer:
    lm — извлечённые слои не глубже него (нет таких — группы нет), attention и entropy — головы этих слоёв, группы с
    выхода модели (OUTPUT_GROUPS) — только при всех слоях; vision от слоёв языковой модели не зависит."""
    shapes = {name: tuple(_limit(name, np.empty((1, *shape), dtype=np.uint8), meta, max_layer).shape[1:])
              for name, shape in meta["shapes"].items()}
    dims = {name: sum(int(np.prod(shapes[a])) for a in arrays) for name, arrays in GROUP_ARRAYS.items()
            if all(a in shapes for a in arrays)}
    dims = {name: size for name, size in dims.items() if size > 0}
    if "confidence" in dims:  # буквы дополнены до CONFIDENCE_SLOTS и индикатор доступности каждой (confidence_features)
        dims["confidence"] += 2 * CONFIDENCE_SLOTS - meta["max_choices"]
    if "lm_last" in shapes and meta.get("layers") in meta.get("lm_layers", []):
        dims["last_hidden"] = int(meta["shapes"]["lm_last"][1])  # последний слой в позиции решения
    if max_layer is not None and _depth(meta, max_layer) < int(meta["layers"]):
        for name in OUTPUT_GROUPS:
            dims.pop(name, None)
    return dims


def feature_rows(fs: FeatureSet, index: int, n: int, groups: list[str],
                 max_layer: int | None = None) -> dict[str, np.ndarray]:
    """Возвращает исходные признаки первых n пар без изменения порядка координат (max_layer — feature_dims)."""
    if not groups or len(groups) != len(set(groups)) or set(groups) - set(FEATURE_DIMS):
        raise ValueError("groups должны быть непустым списком уникальных известных групп")
    dims = feature_dims(fs.meta, max_layer)
    missing = set(groups) - set(dims)
    if missing:
        depth = "" if max_layer is None else f" при слоях 1..{max_layer}"
        raise ValueError(f"В этих признаках{depth} нет групп {sorted(missing)} (есть {sorted(dims)})")

    def question(name):
        return _limit(name, fs.question(name, index)[:n], fs.meta, max_layer)

    loaders = {
        "lm": lambda: [question(name) for name in ("lm_last", "lm_mean", "probe_hidden")],
        "attention": lambda: [question(name) for name in ("att_last", "att_mean", "att_probe")],
        "entropy": lambda: [question("att_entropy")],
        "vision": lambda: [fs.video(index)[:n]],
        "confidence": lambda: [confidence_features(fs.question("probe_logprob", index)[:n], fs.meta["max_choices"])],
        "listen_token_logit": lambda: [fs.question("listen_token_logit", index)[:n]],
        # Последний слой (после финальной нормализации) на последнем визуальном токене пары.
        "last_hidden": lambda: [fs.question("lm_last", index)[:n, fs.meta["lm_layers"].index(fs.meta["layers"])]],
    }
    arrays = {name: loaders[name]() for name in groups}
    result = {name: np.concatenate([x.reshape(n, -1) for x in arrays[name]], axis=1).astype(np.float32)
              for name in groups}
    for name, array in result.items():
        if array.shape != (n, dims[name]):
            raise ValueError(f"{name}: expected {(n, dims[name])}, got {array.shape}; incompatible extraction")
    return result


class ReadinessDataset(Dataset):
    """Одна запись датасета — одна отдельная видеопоследовательность, не объединённая с другими. max_layer — признаки
    только слоёв языковой модели 1..max_layer (feature_dims)."""
    def __init__(self, fs: FeatureSet, items: list[LabelledItem], groups: list[str], max_layer: int | None = None):
        self.fs, self.items, self.groups, self.max_layer = fs, items, groups, max_layer

    def __len__(self):
        return len(self.items)

    def __getitem__(self, position):
        item = self.items[position]
        n = len(item.target)
        return feature_rows(self.fs, item.index, n, self.groups, self.max_layer), item.target, item


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
    dims = feature_dims(dataset.fs.meta, dataset.max_layer)
    count = {name: 0 for name in dataset.groups}
    total = {name: np.zeros(dims[name], dtype=np.float64) for name in dataset.groups}
    square = {name: np.zeros(dims[name], dtype=np.float64) for name in dataset.groups}
    for item in dataset.items:
        values = feature_rows(dataset.fs, item.index, len(item.target), dataset.groups, dataset.max_layer)
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
