"""95% bootstrap-интервалы test-метрик детекторов готовности при порогах τ, выбранных на validation (как в
scripts/test_report.py, online_vllm/eval/report.py test_report).

Повторная выборка — видео test с возвращением (столько же видео, сколько в test); видео входит со всеми своими
вопросами, поэтому число вопросов в выборке немного меняется. Видео вопроса — префикс record_id до типа вопроса
(validation_12500_clevrer_query_material_0 -> validation_12500). Интервал перцентильный (2.5% и 97.5%), пороги
фиксированы с validation; выборки видео одни и те же для всех систем (у всех систем test — одни вопросы).

Метрики: AUROC по всем парам и в окнах |k - t*| <= 2 и <= 5, доля ответов вовремя (on_time_0_4: 0..4 пары после целевой) при
τ критерия, средняя задержка непреждевременных ответов (пары после t*) при τ ограничений R ≤ 5% / 10% и A ≥ 90%.

python scripts/test_bootstrap.py --output results/test_bootstrap.json \\
    --system "SB-3B" results/separate/sb3b/predictions_validation.json results/separate/sb3b/predictions_test_best.json
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

import _paths  # noqa: F401  (src/ on the path)
from online_vllm.eval.metrics import constraint_key
from online_vllm.eval.report import _aligned, combined, questions, test_report, two_detector_report

LATENCY_KEYS = {"latency_R5_A90": constraint_key(0.05, 0.90), "latency_R10_A90": constraint_key(0.10, 0.90)}
VIDEO = re.compile(r"^([a-z]+_\d+)_")


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def video_of(record_id: str) -> str:
    match = VIDEO.match(record_id)
    if not match:
        raise ValueError(f"Не удалось выделить видео из record_id {record_id!r}")
    return match.group(1)


class Pairs:
    """Пары всех вопросов, отсортированные по оценке один раз: AUROC выборки — взвешенный (вес пары = сколько раз
    её вопрос попал в выборку), что совпадает с AUROC на выборке с повторами (с учётом равных оценок)."""

    def __init__(self, scores, labels, question):
        order = np.argsort(-scores, kind="stable")
        self.y, self.question = labels[order].astype(float), question[order]
        score = scores[order]
        self.ends = np.r_[np.flatnonzero(score[:-1] != score[1:]), len(score) - 1]

    def auroc(self, weights: np.ndarray) -> float:
        w = weights[self.question]
        tp, fp = np.cumsum(w * self.y)[self.ends], np.cumsum(w * (1 - self.y))[self.ends]
        if tp[-1] == 0 or fp[-1] == 0:
            return float("nan")
        return float(np.trapezoid(np.r_[0.0, tp / tp[-1]], np.r_[0.0, fp / fp[-1]]))


def first_fire(probabilities, n: int, tau: float) -> tuple[int, bool]:
    hits = np.flatnonzero(np.asarray(probabilities[:n], dtype=float) > tau)
    return (int(hits[0]), True) if len(hits) else (n - 1, False)


def _rows(payload: dict, order: list[str]) -> list[dict]:
    rows = {row["record_id"]: row for row in payload["predictions"]}
    return [rows[record_id] for record_id in order]


class System:
    """report — test_report или two_detector_report; test — предсказания test, по которым считаются AUROC и on_time
    (для пары — оценки s_k правила при τ1 критерия); latency_tests — то же для каждой задержки (для пары — при τ1,
    выбранном для её ограничений), по умолчанию test."""

    def __init__(self, name: str, report: dict, test: dict, order: list[str], latency_tests: dict | None = None):
        self.name, self.report = name, report
        latency_tests = latency_tests or {}
        rows = _rows(test, order)
        scores, labels, question, offsets = [], [], [], []
        for q, row in enumerate(rows):
            p = np.asarray(row["probabilities"], dtype=float)
            n, target = len(p), int(row["target_pair"])
            scores.append(p); labels.append(np.arange(n) >= target)
            question.append(np.full(n, q)); offsets.append(np.arange(n) - target)
        scores, labels = np.concatenate(scores), np.concatenate(labels)
        question, offsets = np.concatenate(question), np.concatenate(offsets)
        self.all_pairs = Pairs(scores, labels, question)
        near = np.abs(offsets) <= 2
        self.pm2 = Pairs(scores[near], labels[near], question[near])
        near = np.abs(offsets) <= 5
        self.pm5 = Pairs(scores[near], labels[near], question[near])

        # По вопросам при τ критерия: ответ вовремя (0..4 пары после t*; пропуск — не вовремя).
        tau = self.report["tau"]
        self.on_time = np.zeros(len(rows))
        for q, row in enumerate(rows):
            k, fired = first_fire(row["probabilities"], len(row["probabilities"]), tau)
            self.on_time[q] = fired and 0 <= k - row["target_pair"] <= 4
        # Задержка при τ ограничений (operating_point): ответ на первой паре с p_k > τ, иначе на последней;
        # средняя задержка k − t* по непреждевременным ответам.
        self.latency = {}
        for name, key in LATENCY_KEYS.items():
            chosen = self.report["constrained_latency"][key]["validation"]
            if not chosen["feasible"]:
                self.latency[name] = None
                continue
            timely, delay = np.zeros(len(rows)), np.zeros(len(rows))
            key_rows = _rows(latency_tests[name], order) if name in latency_tests else rows
            for q, row in enumerate(key_rows):
                k, _ = first_fire(row["probabilities"], len(row["probabilities"]), chosen["tau"])
                timely[q] = k >= row["target_pair"]
                delay[q] = max(k - row["target_pair"], 0)
            self.latency[name] = (timely, delay)

    def metrics(self, weights: np.ndarray) -> dict:
        result = {"auroc": self.all_pairs.auroc(weights), "auroc_pm2": self.pm2.auroc(weights),
                  "auroc_pm5": self.pm5.auroc(weights),
                  "on_time_0_4": float(weights @ self.on_time / weights.sum())}
        for name, value in self.latency.items():
            if value is None:
                result[name] = None
                continue
            timely, delay = value
            count = weights @ timely
            result[name] = float(weights @ (timely * delay) / count) if count else float("nan")
        return result

    def reference(self) -> dict:
        """Точечные значения из test_report — для сверки с весами 1."""
        classification, at = self.report["classification"], self.report["at_tau"]
        result = {"auroc": classification["auroc"], "auroc_pm2": classification["auroc_pm2"],
                  "auroc_pm5": classification["auroc_pm5"],
                  "on_time_0_4": at["on_time_0_4"]}
        for name, key in LATENCY_KEYS.items():
            point = self.report["constrained_latency"][key]["test_at_validation_tau"]
            result[name] = None if point is None else point["latency_mean"]
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--system", nargs=3, action="append", default=[], metavar=("NAME", "VALIDATION", "TEST"))
    parser.add_argument("--pair", nargs=6, action="append", default=[],
                        metavar=("NAME", "MODE", "VAL_P", "TEST_P", "VAL_Q", "TEST_Q"),
                        help="два детектора одной модели (как в scripts/test_report.py): and | or, пара порогов "
                             "с validation (online_vllm/eval/report.py two_detector_report)")
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", help="JSON с точечными значениями и интервалами")
    args = parser.parse_args()

    if not args.system and not args.pair:
        parser.error("нужен хотя бы один --system или --pair")
    systems, reference, order = [], None, None

    def check(name, test_payload):
        nonlocal reference, order
        current = questions(test_payload)
        if reference is None:
            reference, order = (name, current), [row["record_id"] for row in test_payload["predictions"]]
        elif current != reference[1]:
            raise SystemExit(f"Test «{name}» — другие вопросы или целевые пары, чем у «{reference[0]}»")

    for name, validation, test in args.system:
        test_payload = _load(test)
        check(name, test_payload)
        systems.append(System(name, test_report(_load(validation), test_payload, "on_time_0_4"), test_payload, order))
        print(f"загружено: {name}")
    for name, mode, val_p, test_p, val_q, test_q in args.pair:
        test_first, test_second = _load(test_p), _load(test_q)
        check(name, test_first)
        report = two_detector_report(_load(val_p), test_first, _load(val_q), test_second, mode, "on_time_0_4")
        second_scores = _aligned(test_first, test_second)
        latency_tests = {key: combined(test_first, second_scores, report["constrained_latency"][constraint]["tau1"],
                                       mode) for key, constraint in LATENCY_KEYS.items()}
        systems.append(System(name, report, combined(test_first, second_scores, report["tau1"], mode), order,
                              latency_tests))
        print(f"загружено: {name} (τ1 = {report['tau1']:.4g}, τ2 = {report['tau']:.4g})")
    reference = reference[1]

    videos, video_index = np.unique([video_of(record_id) for record_id in order], return_inverse=True)
    rng = np.random.default_rng(args.seed)
    draws = rng.integers(0, len(videos), size=(args.resamples, len(videos)))
    question_counts = []
    samples = {system.name: [] for system in systems}
    for draw in draws:
        video_weights = np.bincount(draw, minlength=len(videos)).astype(float)
        weights = video_weights[video_index]  # вес вопроса = сколько раз выбрано его видео
        question_counts.append(int(weights.sum()))
        for system in systems:
            samples[system.name].append(system.metrics(weights))

    output = {"resamples": args.resamples, "seed": args.seed, "videos": int(len(videos)), "questions": len(order),
              "questions_per_resample": {"min": min(question_counts), "median": float(np.median(question_counts)),
                                         "max": max(question_counts)}, "systems": {}}
    print(f"\ntest: {len(videos)} видео, {len(order)} вопросов; {args.resamples} выборок видео с возвращением "
          f"(вопросов в выборке {min(question_counts)}..{max(question_counts)}); 95% перцентильный интервал")
    keys = ["auroc", "auroc_pm2", "auroc_pm5", "on_time_0_4", *LATENCY_KEYS]
    for system in systems:
        point, check = system.metrics(np.ones(len(order))), system.reference()
        for key in keys:
            if point[key] is not None and not np.isclose(point[key], check[key], rtol=0, atol=1e-9):
                raise SystemExit(f"{system.name}: {key} = {point[key]} не совпадает с test_report {check[key]}")
        entry = {"tau": system.report["tau"],
                 "latency_tau": {name: system.report["constrained_latency"][key]["validation"].get("tau")
                                 for name, key in LATENCY_KEYS.items()}}
        for key in keys:
            if point[key] is None:
                entry[key] = None
                continue
            values = np.array([s[key] for s in samples[system.name]], dtype=float)
            low, high = np.nanpercentile(values, [2.5, 97.5])
            entry[key] = {"value": point[key], "ci95": [float(low), float(high)]}
        output["systems"][system.name] = entry

    def cell(entry, scale=1.0, digits=3):
        if entry is None:
            return "-"
        v, (lo, hi) = entry["value"] * scale, entry["ci95"]
        return f"{v:.{digits}f} [{lo * scale:.{digits}f}, {hi * scale:.{digits}f}]"

    width = max(len(s.name) for s in systems)
    print(f"{'система':{width}s} | {'AUROC':^21s} | {'AUROC±2':^21s} | {'AUROC±5':^21s} | {'On-time, %':^18s} | "
          f"{'Latency A≥90% P≤5%':^18s} | {'Latency A≥90% P≤10%':^18s}")
    for name, e in output["systems"].items():
        print(f"{name:{width}s} | {cell(e['auroc'])} | {cell(e['auroc_pm2'])} | {cell(e['auroc_pm5'])} | "
              f"{cell(e['on_time_0_4'], 100, 1)} | {cell(e['latency_R5_A90'], 1, 2)} | {cell(e['latency_R10_A90'], 1, 2)}")
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"-> {args.output}")


if __name__ == "__main__":
    main()
