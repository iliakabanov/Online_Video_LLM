"""Модели без дообучения на CLEVRER-Proactive: что они отвечают в четыре момента видео и, у онлайн-модели, когда сами
готовы ответить. Описание эксперимента — eval/ZERO_SHOT.md.

Системы (SYSTEMS):
    qwen_offline     Qwen2.5-VL-3B: вопрос с вариантами, затем видео одним входом;
    mmduet2_offline  MMDuet2 на том же входе;
    mmduet2_online   MMDuet2 в своём формате (data/turns.py): ход пользователя — пара кадров, ход ассистента —
                     «NO REPLY».
Обе модели видят одни и те же кадры. Ответ везде — буква с наибольшей вероятностью среди вариантов вопроса в позиции,
с которой ассистент начал бы отвечать (проба, train/FEATURES.md); у онлайн-модели там же читается P(NO REPLY).

Один проход на вопрос даёт ответы на всех парах видео: офлайн — проба после каждой пары поверх KV-кэша
(lora.probe_letter_logprobs), онлайн — начало каждого хода ассистента. Замеры — пары first, target, mid и end
(checkpoint_pairs); остальные пары дают кривые.
"""
from __future__ import annotations

import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from ..data.inputs import LETTERS, InputBuilder
from ..data.labels import plan, whole_patches
from ..data.schema import Record
from ..data.turns import TurnsBuilder
from ..models.model import attention_kernels
from .answers import AnswerTrace, GRID, INK, INK_2, MUTED, answer_curves, choice_probabilities
from .metrics import _binary_curves

SYSTEMS = {"qwen_offline": "Qwen/Qwen2.5-VL-3B-Instruct", "mmduet2_offline": "wangyueqian/MMDuet2",
           "mmduet2_online": "wangyueqian/MMDuet2"}
CHECKPOINTS = ("first", "target", "mid", "end")
CHECKPOINT_NAMES = {"first": "первая пара", "target": "ответ стал виден", "mid": "после события", "end": "конец видео"}
# Первые три слота категориальной палитры (как в eval/readiness_plot.py), по порядку систем.
SYSTEM_COLORS = ("#2a78d6", "#eb6834", "#1baf7a")


# ----------------------------------------------------------------------------------------------- выборка и замеры

def checkpoint_pairs(record: Record, num_pairs: int, seed: int, stride: int = 1, temporal_patch: int = 2) -> dict | None:
    """Пары замеров вопроса или None, если вопрос не подходит.

    first — первая пара; target — первая пара, последний кадр которой не раньше начала ответа (speak_delay = 0);
    mid — случайная пара после target вплоть до последней, своя у каждого вопроса и не зависящая от порядка вопросов;
    end — последняя пара. Нужна хотя бы одна пара после target, иначе mid совпал бы с ней."""
    state_plan = plan(record, stride, 0, temporal_patch)
    if state_plan is None:
        return None
    target = len(state_plan.labels) - 1
    if target > num_pairs - 2:
        return None
    mid = random.Random(f"{seed}:{record.id}").randint(target + 1, num_pairs - 1)
    return {"first": 0, "target": target, "mid": mid, "end": num_pairs - 1}


def sample_questions(records, videos: int, seed: int, stride: int = 1, temporal_patch: int = 2) -> list[Record]:
    """По одному вопросу на видео из `videos` случайных видео (вопросы одного видео связаны между собой, так
    доверительные интервалы по вопросам честные). Видео и вопрос выбираются детерминированно по seed."""
    by_video = defaultdict(list)
    for r in records:
        if checkpoint_pairs(r, whole_patches(r, stride, temporal_patch), seed, stride, temporal_patch) is not None:
            by_video[r.video].append(r)
    order = sorted(by_video)
    if len(order) < videos:
        raise ValueError(f"Подходящих видео {len(order)}, нужно {videos}")
    random.Random(seed).shuffle(order)
    return [random.Random(f"{seed}:{v}").choice(sorted(by_video[v], key=lambda r: r.id)) for v in order[:videos]]


def letter_token_ids(tokenizer, count: int = len(LETTERS)) -> torch.Tensor:
    ids = [tokenizer(letter, add_special_tokens=False).input_ids for letter in LETTERS[:count]]
    if any(len(i) != 1 for i in ids):
        raise ValueError("Буквы вариантов должны быть одним токеном")
    return torch.tensor([i[0] for i in ids])


# ------------------------------------------------------------------------------------------------ проходы моделей

@torch.inference_mode()
def offline_pass(model, builder: InputBuilder, prompt: str, frames: np.ndarray, fps: float, letter_ids: torch.Tensor,
                 device) -> dict:
    """Офлайн: log P букв вариантов (по всему словарю) после каждой пары [P, K], одним проходом."""
    from lora.lora import probe_letter_logprobs  # lora зависит от online_vllm, не наоборот
    from ..train.features import layout
    inputs = builder._encode(prompt, frames, fps)
    lay = layout(inputs, prompt, builder.processor.tokenizer)
    letters = probe_letter_logprobs(model, inputs, lay, range(lay.num_pairs), letter_ids, device=device)
    return {"letters": letters.cpu().numpy()}


@torch.inference_mode()
def online_pass(model, builder: TurnsBuilder, prompt: str, frames: np.ndarray, letter_ids: torch.Tensor, device) -> dict:
    """Онлайн: на каждом ходе log P букв [T, K], log P(NO REPLY) [T] и то, начал бы жадный ответ с «NO» [T]."""
    turns = builder.build(prompt, frames)
    qwen = model.qwen.model
    inputs = {k: v.to(device) for k, v in turns.inputs.items() if k in
              ("input_ids", "attention_mask", "mm_token_type_ids", "pixel_values", "image_grid_thw")}
    qwen.rope_deltas = None
    with attention_kernels():
        hidden = qwen(**inputs, use_cache=False, return_dict=True).last_hidden_state[0]
    n = len(turns.silence)
    rows = turns.decisions.to(device)[:, None] + torch.arange(n, device=device)          # [T, n]: решение и «NO RE»
    log_probs = model.qwen.lm_head(hidden[rows]).float().log_softmax(-1)                  # [T, n, vocab]
    silence = torch.tensor(turns.silence, device=device)
    no_reply = log_probs.gather(-1, silence.expand(len(rows), n)[..., None])[..., 0].sum(1)
    first = log_probs[:, 0]
    return {"letters": first[:, letter_ids.to(device)].cpu().numpy(), "silence": no_reply.cpu().numpy(),
            "no_top1": (first.argmax(-1) == silence[0]).cpu().numpy()}


# ------------------------------------------------------------------------------------------------------- метрики

def bootstrap_ci(values, seed: int = 0, resamples: int = 2000) -> list[float]:
    """95% перцентильный интервал среднего по вопросам."""
    values = np.asarray(values, dtype=float)
    if not len(values):
        return [None, None]
    means = values[np.random.default_rng(seed).integers(0, len(values), (resamples, len(values)))].mean(1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def mcnemar(first, second) -> dict:
    """Точный двусторонний тест McNemar для парных булевых исходов: b — верно только в first, c — только в second."""
    first, second = np.asarray(first, bool), np.asarray(second, bool)
    b, c = int((first & ~second).sum()), int((~first & second).sum())
    n = b + c
    p = 1.0 if not n else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)
    return {"only_first": b, "only_second": c, "p_value": p}


class Answers:
    """Ответы одного вопроса одной системы по парам."""

    def __init__(self, row: dict):
        self.row = row
        self.k = row["choices"]
        self.letters = np.asarray(row["letters"], dtype=np.float64)[:, :self.k]
        self.probabilities = choice_probabilities(self.letters, self.k)                     # [P, K]
        self.correct = row["correct"]
        self.silence = None if row.get("silence") is None else np.asarray(row["silence"], dtype=np.float64)
        self.no_top1 = None if row.get("no_top1") is None else np.asarray(row["no_top1"], dtype=bool)

    @property
    def is_correct(self) -> np.ndarray:
        return self.probabilities.argmax(1) == self.correct

    def at(self, pair: int) -> dict:
        p = self.probabilities[pair]
        top = np.sort(p)[::-1]
        out = {"correct": bool(p.argmax() == self.correct), "p_correct": float(p[self.correct]),
               "p_answer": float(top[0]), "margin": float(top[0] - top[1]),
               "entropy": float(-(p * np.log(np.clip(p, 1e-12, 1))).sum() / math.log(self.k)),
               "letter_mass": float(np.exp(self.letters[pair]).sum()), "chance": 1 / self.k}
        if self.silence is not None:
            out.update({"p_no_reply": float(np.exp(self.silence[pair])),
                        "self_reply": bool(not self.no_top1[pair])})
        return out


def _summarize(points: list[dict], seed: int) -> dict:
    mean = lambda key: float(np.mean([p[key] for p in points]))
    correct = [p["correct"] for p in points]
    out = {"questions": len(points), "accuracy": float(np.mean(correct)), "accuracy_ci95": bootstrap_ci(correct, seed),
           "chance": mean("chance"), "p_correct": mean("p_correct"), "p_answer": mean("p_answer"),
           "margin": mean("margin"), "entropy": mean("entropy"),
           "letter_mass_median": float(np.median([p["letter_mass"] for p in points]))}
    if "p_no_reply" in points[0]:
        quiet = np.array([p["p_no_reply"] for p in points])
        replies = [p["self_reply"] for p in points]
        out.update({"p_no_reply_quartiles": [float(q) for q in np.percentile(quiet, [25, 50, 75])],
                    "self_reply_rate": float(np.mean(replies)),
                    "self_reply_correct_rate": float(np.mean([r and c for r, c in zip(replies, correct)]))})
    return out


def _groups(rows) -> dict:
    """Разрезы: тип ответа, семейство, квартиль момента появления ответа."""
    starts = np.array([r["start_frame"] for r in rows])
    edges = np.percentile(starts, [25, 50, 75])
    quartile = lambda s: f"Q{int(np.searchsorted(edges, s, side='right')) + 1}"
    return {"answer_type": [r["answer_type"] for r in rows], "family": [r["family"] for r in rows],
            "start_quartile": [quartile(s) for s in starts],
            "start_quartile_edges_frames": edges.tolist()}


def system_metrics(rows: list[dict], seed: int = 0) -> dict:
    """Метрики одной системы: по замерам, прирост от first к target, AUROC уверенности, разрезы."""
    answers = [Answers(r) for r in rows]
    at = {c: [a.at(r["pairs"][c]) for a, r in zip(answers, rows)] for c in CHECKPOINTS}
    result = {"checkpoints": {c: _summarize(at[c], seed) for c in CHECKPOINTS}}
    first, target = [p["correct"] for p in at["first"]], [p["correct"] for p in at["target"]]
    result["gain_target_vs_first"] = {"accuracy": float(np.mean(target) - np.mean(first)),
                                      **mcnemar(target, first)}
    # Растёт ли уверенность, когда ответ становится виден: пары (first: 0, target: 1) по всем вопросам.
    scores = np.array([p["p_answer"] for p in at["first"]] + [p["p_answer"] for p in at["target"]])
    labels = np.r_[np.zeros(len(rows)), np.ones(len(rows))].astype(np.int64)
    result["confidence_auroc_target_vs_first"] = _binary_curves(labels, scores)[0]
    groups = _groups(rows)
    result["start_quartile_edges_frames"] = groups.pop("start_quartile_edges_frames")
    result["breakdown"] = {}
    for name, keys in groups.items():
        result["breakdown"][name] = {
            key: {c: {"questions": int(sum(k == key for k in keys)),
                      "accuracy": float(np.mean([p["correct"] for p, k in zip(at[c], keys) if k == key]))}
                  for c in CHECKPOINTS}
            for key in sorted(set(keys))}
    return result


def compare_systems(runs: dict[str, list[dict]]) -> dict:
    """Попарно на каждом замере: доля совпадающих ответов и McNemar по верности (вопросы — общие у всех систем)."""
    names = list(runs)
    ids = [r["id"] for r in runs[names[0]]]
    for name in names[1:]:
        if [r["id"] for r in runs[name]] != ids:
            raise ValueError(f"{name}: другие вопросы или другой порядок, чем у {names[0]}")
    answers = {name: [Answers(r) for r in rows] for name, rows in runs.items()}
    result = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            pair = {}
            for c in CHECKPOINTS:
                pa = [x.probabilities[r["pairs"][c]].argmax() for x, r in zip(answers[a], runs[a])]
                pb = [x.probabilities[r["pairs"][c]].argmax() for x, r in zip(answers[b], runs[b])]
                ca = [x.is_correct[r["pairs"][c]] for x, r in zip(answers[a], runs[a])]
                cb = [x.is_correct[r["pairs"][c]] for x, r in zip(answers[b], runs[b])]
                pair[c] = {"agreement": float(np.mean(np.equal(pa, pb))),
                           "accuracy_difference": float(np.mean(ca) - np.mean(cb)), **mcnemar(ca, cb)}
            result[f"{a} vs {b}"] = pair
    return result


def traces(rows: list[dict]) -> list[AnswerTrace]:
    """Ответы по парам в формате eval/answers.py; при speak_delay = 0 видимая пара совпадает с целевой."""
    return [AnswerTrace(Answers(r).probabilities, r["correct"], r["pairs"]["target"], r["pairs"]["target"])
            for r in rows]


def readiness_predictions(rows: list[dict], metadata: dict) -> dict:
    """Онлайн-система в формате eval/export.py: «готовность» = 1 − P(NO REPLY) на каждой паре и верность ответа по
    буквам там же. Его читают eval/thresholds.py и eval/readiness_plot.py."""
    predictions = []
    for r in rows:
        a = Answers(r)
        if a.silence is None:
            raise ValueError(f"{r['id']}: нет P(NO REPLY), это не онлайн-система")
        predictions.append({"record_id": r["id"], "probabilities": (1 - np.exp(a.silence)).clip(0, 1).tolist(),
                            "target_pair": r["pairs"]["target"], "visible_pair": r["pairs"]["target"],
                            "fps": r["fps"], "answer_correct": a.is_correct.tolist()})
    return {"metadata": {"model": metadata.get("model"), "split": "zero_shot", "tags": "1 - P(NO REPLY)",
                         "temporal_patch": metadata["frames_per_pair"], "frame_stride": metadata["stride"]},
            "predictions": predictions}


# ------------------------------------------------------------------------------------------------------- графики

def plot_systems(curves: dict[str, dict], path: str | Path, title: str):
    """Точность и P(правильный) по сдвигу относительно целевой пары у всех систем на одних осях."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    chance = next(iter(curves.values()))["levels"]["chance"]
    for ax, (key, panel) in zip(axes, (("accuracy", "Точность ответа"), ("p_correct", "P(правильный ответ)"))):
        for (name, result), color in zip(curves.items(), SYSTEM_COLORS):
            points = [row for row in result["curve"] if row["count"]]
            ax.plot([row["offset"] for row in points], [row[key] for row in points], color=color, linewidth=2,
                    marker="o", markersize=3.5, markeredgecolor="white", markeredgewidth=0.8, label=name, zorder=3)
        ax.axhline(chance, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=2)
        ax.text(ax.get_xlim()[1], chance, " угадывание", color=INK_2, fontsize=9, va="bottom", ha="right")
        ax.axvline(0, color=MUTED, linewidth=1, zorder=1)
        ax.text(0, 0.01, " ответ стал виден", color=INK_2, fontsize=9, va="bottom", ha="left")
        ax.set_ylim(0, 1.02)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.set_xlabel("сдвиг относительно целевой пары, пары (1 пара = 0,08 с)", color=INK_2, fontsize=9)
        ax.set_title(panel, color=INK, fontsize=11, loc="left")
        ax.grid(axis="y", color=GRID, linewidth=1)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=INK_2, labelsize=9)
        ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="lower right")
    fig.suptitle(title, color=INK_2, fontsize=10, x=0.01, ha="left")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def curves_of(rows: list[dict], offsets=range(-30, 31)) -> dict:
    return answer_curves(traces(rows), offsets)
