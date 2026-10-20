# Causal UHead для готовности к ответу

Модуль обучает альтернативу `StateHead` только на заранее сохранённых признаках. Qwen2.5-VL и backbone в процессе не загружаются. Источник архитектурного стиля — [`UncertaintyHead`](https://github.com/IINemo/llm-uncertainty-head/blob/main/luh/heads/uncertainty_head.py): двухслойная projection MLP с `LayerNorm` и `GELU`, затем `TransformerEncoder` и classifier.

Изменения для streaming video readiness:

- пять независимых групп признаков одной пары кадров нормализуются статистиками только train split и проектируются раздельно;
- полученный временной ряд проходит **causal** Transformer: позиция `t` не может обращаться к парам после `t`;
- padding mask не позволяет смешивать разные видео и исключает padding из loss;
- classifier выдаёт один бинарный logit готовности для каждой пары кадров.

## Размерности конфигурации по умолчанию

| Группа | Вход | Выход |
| --- | ---: | ---: |
| LM | 24576 | 256 |
| Attention | 12672 | 256 |
| Entropy | 1728 | 64 |
| Vision | 5120 | 128 |
| Confidence | 19 | 32 |

После конкатенации получается `736`, fusion даёт `512`, а итоговые logits имеют форму `[B, P]`.

## Режимы GT

`python -m uhead.train` требует явный `--tags`:

- `hard`: 0 до целевой пары и 1 начиная с неё;
- `step`: 0.5 в окне `--step-window` (по умолчанию 5) перед целью;
- `sigmoid`: гладкий переход с наклоном `--sigmoid-slope` (по умолчанию 0.65) и значением 0.9 в целевой паре.

Все режимы используют все доступные пары видео, включая пары после target pair. Loss — `binary_cross_entropy_with_logits` только по непустым позициям. Результаты автоматически сохраняются в подкаталоге режима (`hard`, `step` или `sigmoid`), а параметры разметки — в `tags.json` checkpoint-а.

## Запуск

Из корня репозитория (пакет `uhead` лежит в `src/`):

```bash
PYTHONPATH=src python -m uhead.train --tags hard --output outputs/group1_uhead --device cuda
PYTHONPATH=src python -m uhead.train --tags step --step-window 5 --output outputs/group1_uhead --device cuda
PYTHONPATH=src python -m uhead.train --tags sigmoid --sigmoid-slope 0.65 --output outputs/group1_uhead --device cuda
```

Аргумент `--uhead` задаёт архитектурный YAML, а `--train` — существующий тренировочный YAML с путями к shards и порогами. Для ablation удалите ненужные имена из `groups` в `configs/uhead/default.yaml`.

## Validation и test

- Признаки `features.eval` делятся **по видео** (seed из конфига) на две равные части: validation и test. Вопросы одного видео всегда в одной части; списки видео — в `split.json`. `data.eval_samples` здесь не используется.
- После каждой эпохи в `metrics.jsonl` пишутся средний по батчам train loss и метрики validation; сохраняется `last.pt`, а при наименьшем validation loss — `best.pt`.
- После обучения `best.pt` оценивается на validation при каждом пороге `eval.thresholds`. Итог — `results.json` (лучшая эпоха, классификационные метрики, онлайн-метрики по каждому порогу) и вероятности `predictions_validation.json` (формат `online_vllm.eval.export`).
- Test обучение не трогает. По `results.json` выберите порог и оцените test отдельно:

```bash
PYTHONPATH=src python -m uhead.evaluate --run outputs/group1_uhead/hard --threshold 0.7 [--weights last.pt]
```

Результат — `<run>/test_best_tau0.7.json` (loss, классификационные и онлайн-метрики при этом пороге) и вероятности `predictions_test_best_tau0.7.json`.

В каталоге запуска лежат `best.pt` и `last.pt` (веса, включая normalizer buffers), `uhead.yaml`, `train.yaml`, `tags.json`, `experiment.json`. Загрузка: `from uhead.train import load_checkpoint; model = load_checkpoint(".../hard")` (берётся `best.pt`; `last.pt` — через `filename="last.pt"`); `tags = model.tag_settings`.
