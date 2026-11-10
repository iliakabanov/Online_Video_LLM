# Онлайн-SFT: управляющие токены `<listen>` / `<speak>`

Qwen, дообученная офлайн отвечать на вопрос по префиксу видео (`scripts/train_lora_sft.py`), учится сама решать,
когда отвечать. После каждой пары кадров она пишет управляющий токен: `<listen>` — смотреть дальше, `<speak>` — ответить
сейчас. Код — [online.py](online.py), запуск — `scripts/train_online_sft.py` и `scripts/evaluate_online_sft.py`,
тесты — `tests/test_online_sft.py`.

## Модель

- Лучший офлайн-адаптер (`--init-run`, `--init-adapter best`) вливается в веса Qwen.
- Поверх обучается новая LoRA с теми же настройками (`model.yaml` офлайн-прогона: q/k/v/o всех слоёв, r = 32).
- `<listen>` и `<speak>` добавляются в токенизатор как специальные токены. Их строки эмбеддингов инициализируются
  средним эмбеддингом слов «listen» / «speak» и обучаются (`trainable_token_indices` из PEFT). В 3B эмбеддинги и
  `lm_head` связаны, поэтому одна строка служит обоим. Остальные веса заморожены.

## Обучающая последовательность

Один вопрос — один проход:

```
основная   [вопрос <|vision_start|>] [пара 0] ▢ [пара 1] ▢ … [пара P−1] ▢
ветки      [<speak> <|vision_end|><|im_end|>\n<|im_start|>assistant\n] → буква ответа   (после пары s и после P−1)
```

| позиция | на входе | цель |
|---|---|---|
| слот ▢ после пары k < t* | `<listen>` | `<listen>` |
| слот ▢ после пары k ≥ t* | `<listen>` | `<speak>` |
| последний токен ветки | — | буква правильного ответа |

- t* — целевая пара: первая пара, последний кадр которой показывает ответ (`speak_delay` = 0, как в офлайн-SFT).
- Цель слота k предсказывается из последнего визуального токена пары k, то есть до самого слота. Поэтому то, что
  стоит на входе слота, на его собственную цель не влияет. `<listen>` на входе повторяет реальную историю политики,
  которая ещё не ответила.
- **Ветки ответа** вместо отдельных проходов. Через маску внимания ветка после пары s видит основную
  последовательность до последнего визуального токена пары s, без `<listen>` слота s, и свои токены. Основная
  последовательность ветки не видит. Это то же, что отдельная последовательность, оборванная `<speak>` после пары s
  (`test_branch_equals_a_separate_sequence_without_its_own_listen`). При обучении веток две: s ~ U(t*, P−1), заново
  каждую эпоху, и конец видео P−1.
- **Потеря:** слоты дают −(y·log P(`<speak>`) + (1−y)·log P(`<listen>`)) по всему словарю, среднее по слотам. Ответ
  даёт кросс-энтропию буквы, среднее по веткам. Итог: слоты + `--answer-weight` × ответ.
  `--tags hard` даёт y ∈ {0, 1}, то есть обычную CE на токен. `step` и `sigmoid` дают мягкие y, как у UHead
  (`uhead/labels.py`). Мягкая цель по всему словарю одновременно стягивает массу на два токена и задаёт их соотношение.
- **3D-позиции как в офлайне.** Визуальные токены получают позиции непрерывного видео (`models/positions.py`). Слот
  после пары k стоит на позиции max + 1, там же, где у офлайн-модели начинался хвост шаблона. Ветка продолжает с этой
  позиции. До слота k модель видит ровно те позиции, что при офлайн-обучении на префиксе до пары k.
- Длина последовательности: ~64 × 55 + вопрос + 2 ветки ≈ 3,7 тыс. токенов при 252×168.

## Инференс

`OnlineSession` — потоковый цикл с KV-кэшем:

1. Вопрос проходит один раз и попадает в кэш.
2. Пара k: её 54 визуальных токена дописываются в кэш одним проходом.
   p_k = softmax(logit `<speak>`, logit `<listen>`)[`<speak>`] на последнем из них.
3. Если p_k > τ, дописываются `<speak>` и хвост шаблона и читается буква ответа. Иначе дописывается `<listen>`.
4. Видео кончилось без `<speak>`: `<speak>` дописывается сразу после последней пары, без `<listen>` перед ним (как
   ветка конца видео при обучении), и это считается пропуском (missed).

τ = 0,5 — «родной» режим: `<speak>` вероятнее `<listen>`. Визуальные токены пары берутся из кэша энкодера или
считаются `encode_pair`. Тест `test_streaming_session_equals_one_pass` проверяет, что сессия даёт те же p_k и ответы,
что один проход оценки.

## Запуск

```bash
# Нужен офлайн-прогон train_lora_sft.py: model.yaml, split.json, best/. Данные и кэш энкодера берутся из него же.
python scripts/train_online_sft.py --init-run outputs/lora_sft --output outputs/online_sft --tags hard
python scripts/train_online_sft.py --init-run outputs/lora_sft --output outputs/online_sft_sigmoid --tags sigmoid
python scripts/evaluate_online_sft.py --run outputs/online_sft --adapter best         # test-половина
```

- Видео train и половины validation/test те же, что у офлайн-прогона: `split.json` переносится,
  `--validation-videos` не меняется. Поэтому test-метрики сравнимы с офлайн-моделью.
- Остальные флаги как у `train_lora_sft.py`: `--epochs`, `--lr`, `--batch-size`, `--gradient-accumulation`,
  `--gradient-checkpointing`, `--train-videos`. `--init-adapter none` начинает с базовой Qwen.
- **Память:** обучаемая последовательность — всё видео, около 3,7 тыс. токенов, длиннее средних офлайн-префиксов.
  Если батч 2 не помещается, используйте `--batch-size 1 --gradient-accumulation 4` или `--gradient-checkpointing`.

## Что пишется в `--output`

| файл | что |
|---|---|
| `best/`, `last/` | адаптер: новая LoRA и изменение двух строк токенов. `best` — по наименьшему validation loss |
| `online.json` | офлайн-адаптер, id токенов, цели слотов, пороги: всё, чтобы пересобрать модель (`load_online_run`) |
| `model.yaml`, `split.json`, `sft_config.json` | настройки модели, данные и половины validation/test, оптимизация |
| `validation_metrics.jsonl` | по эпохе (0 — до обучения): loss (слоты, ответ), P(`<speak>`)+P(`<listen>`) на слотах, точность ответа в начале / на t* / в s / в конце, онлайн-метрики при каждом τ |
| `predictions_validation_epoch_<n>.json` | p_k и верность ответа на каждой паре, в формате `eval/export.py` |
| `test_best.json`, `predictions_test_best.json` | то же на test-половине (`evaluate_online_sft.py`) |

Файлы предсказаний читают общие инструменты:

```bash
PYTHONPATH=src python -m online_vllm.eval.thresholds --predictions outputs/online_sft/predictions_test_best.json \
    --output outputs/online_sft/thresholds_test.json --threshold-start 0.05 --threshold-stop 0.95 --threshold-step 0.05
PYTHONPATH=src python -m online_vllm.eval.readiness_plot --predictions outputs/online_sft/predictions_test_best.json \
    --output outputs/online_sft/readiness_curves.json --plot outputs/online_sft/readiness_curves.png
```
