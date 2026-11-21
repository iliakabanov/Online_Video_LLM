# Отдельный модуль решения (SB, в стиле StreamBridge)

Отдельная VLM смотрит поток и решает, когда будить отвечающую модель. Отвечает всегда офлайн-модель M_off, поэтому
сравнение с головами на признаках M_off (A) касается только тайминга. Код — [decision.py](decision.py), запуск —
`scripts/train_separate.py`, `scripts/evaluate_separate.py`, весь эксперимент — `experiment_launcher/separate/run.sh`,
тесты — `tests/test_separate.py`.

```
[вопрос] [пара 0] <act> [пара 1] <act> … [пара 63] <act>
                  p_0           p_1             p_63
```

Общее для обеих моделей:

- `<act>` — новый специальный токен, всегда один и тот же; обучается его строка эмбеддинга. История не зависит от прошлых
  решений, поэтому p_k одного прохода совпадают с потоковым инференсом (`DecisionSession`,
  `test_streaming_session_equals_one_pass`), а p_k не зависит от последующих пар
  (`test_one_pass_equals_the_video_cut_after_every_pair`).
- Пара — два исходных кадра (0,08 с), как у голов и онлайн-модели: те же t* и точки решения.
- LoRA с настройками M_off: q/k/v/o всех слоёв языковой модели, r 32, α 64; обучается с нуля.
- Голова — архитектура детектора «MLP last_hidden» (`StateHead`, `temporal_detector: mlp`, группа `last_hidden`,
  `configs/uhead/default.yaml`) на скрытом состоянии последнего слоя в позиции `<act>`, без нормализации по train.
- Потери: BCE с логитами по всем слотам батча, без весов классов (как у голов). Метки hard / step / sigmoid.

## SB-0.5B: LLaVA-OneVision-0.5B

`configs/model/llava_ov_0.5b_separate.yaml` (`llava-hf/llava-onevision-qwen2-0.5b-ov-hf`: Qwen2-0.5B, 24 слоя,
скрытое состояние 896, SigLIP 384×384).

- Кадры 384×384. SigLIP кодирует каждый кадр отдельно; пулинг 2×2 LLaVA-OV даёт 196 токенов на кадр (14×14;
  `--tokens-side` меньше 14 — дополнительный билинейный пулинг до side×side).
- **Кэш хранит признаки SigLIP до проектора** (1152 на токен), проектор работает в прямом проходе. По умолчанию он
  заморожен; `--train-projector` обучает его целиком вместе с LoRA (копия PEFT `modules_to_save`, начинается с
  предобученного, сохраняется в адаптере; lr как у LoRA, `--lr`). У LLaVA-OV пулинг идёт после проектора, здесь — до:
  иначе в кэше пришлось бы держать все 27×27 = 729 токенов кадра (~430 ГБ на 2000 видео вместо ~29 ГБ при 7×7). Пулинг
  тот же линейный, но по другую сторону нелинейности проектора — небольшое отличие от родной LLaVA-OV, которое
  обучаемый проектор и LoRA компенсируют.
- Пара — оба кадра (`--frames-per-pair 2`, 392 токена) или только последний (`1`: он и определяет целевую пару).
- Промпт — ход пользователя из шаблона самой LLaVA-OV (`<|im_start|>user …`), но вопрос перед видео: родной шаблон
  ставит видео первым, а в потоке вопрос задаётся в начале. `image_newline`, который LLaVA-OV ставит после всего видео,
  не используется: у потока нет конца. Позиции одномерные.
- Кэш энкодера свой: `data/cache/separate_vision/<модель>_f<кадров на пару>_t<side>_vision`.

| вариант | токенов на пару | длина примера | кэш на 2000 видео |
|---|---|---|---|
| 2 кадра × 196 (родное число токенов) | 392 | ~25 тыс. | ~115 ГБ |
| 1 кадр × 196 | 196 | ~12,6 тыс. | ~58 ГБ |
| **2 кадра × 49 (основной, по умолчанию в `run.sh`)** | 98 | ~6,3 тыс. | ~29 ГБ |
| 1 кадр × 49 | 49 | ~3,2 тыс. | ~14 ГБ |

Основной вариант — оба кадра пары по 49 токенов, `<act>` после пары. Энкодер Qwen склеивает два кадра пары в один
токен (54 на пару), SigLIP кодирует кадры по отдельности, поэтому кадры и число токенов одновременно не уравнять. Здесь
уравнены кадры (модуль видит всё, что видит M_off) и разрешение токена (сетка 7×7 против 9×6 у Qwen); токенов вдвое
больше, но по вычислениям SB-0.5B всё равно дешевле SB-3B (линейные слои ~в 3 раза, внимание примерно столько же).
Пулинг 7×7 сильнее родного для LLaVA-OV (14×14), сдвиг признаков учит LoRA. Абляции: только последний кадр
(`TRAIN_ARGS="--frames-per-pair 1"`), родные 196 токенов на кадр (`--tokens-side 14`).

## SB-3B: Qwen2.5-VL-3B

`configs/model/qwen_3b_lora_sft.yaml` без адаптера M_off. Раскладка онлайн-модели (`lora/online.py`) с `<listen>`,
заменённым на `<act>`: промпт офлайн-входа, видео одним входом с 3D-позициями непрерывного видео, слот после пары k —
на позиции max + 1. Кадры 252×168, 54 токена на пару, ~3,6 тыс. токенов на пример; кэш энкодера LoRA-прогонов.

## Данные — те же, что у голов `offline_10ep`

| часть | вопросы |
|---|---|
| train | первые 1000 видео `train_group1_features.jsonl` с целевой парой: 6436 вопросов |
| validation | вопросы `results/lora/mlp/last_hidden_mlp_off10/predictions_validation.json`: 3277 вопросов, 500 видео |
| test | вопросы `…/predictions_test_best_tau0.5.json`: 3229 вопросов, 500 видео |

Ответы M_off на каждой паре (`answer_correct`) берутся из тех же файлов предсказаний, поэтому онлайн-метрики — это пара
«SB решает, M_off отвечает».

## Обучение

AdamW, постоянные lr: LoRA и `<act>` — 1e-4, weight decay 0.01 (как M_off и M_on); голова — 5e-4, weight decay 0 (как
головы). Батч 2 × накопление 2, 10 эпох, `best/` — по наибольшему AUROC в окне |k − t*| ≤ 2 на validation (как головы).
До обучения и после каждой эпохи — оценка validation: `validation_metrics.jsonl`, `predictions_validation_epoch_<n>.json`.

```bash
# пробный прогон: 3 видео, 20 вопросов validation, 1 эпоха — память, скорость, что всё пишется
python scripts/train_separate.py --model-config configs/model/llava_ov_0.5b_separate.yaml \
    --output outputs/separate/smoke --train-videos 3 --validation-questions 20 --epochs 1 --overwrite
CUDA_VISIBLE_DEVICES=0 MODEL=llava bash experiment_launcher/separate/run.sh          # SB-0.5B: обучение, test, таблица
CUDA_VISIBLE_DEVICES=0 MODEL=llava TRAIN_ARGS="--train-projector" RUN=outputs/separate/sb05b_proj \
    bash experiment_launcher/separate/run.sh                                         # с обучаемым проектором
CUDA_VISIBLE_DEVICES=0 MODEL=llava TRAIN_ARGS="--frames-per-pair 1" RUN=outputs/separate/sb05b_f1 \
    bash experiment_launcher/separate/run.sh
CUDA_VISIBLE_DEVICES=0 MODEL=qwen TRAIN_ARGS="--gradient-checkpointing" bash experiment_launcher/separate/run.sh  # SB-3B
```

## Продолжение прерванного обучения

```bash
python scripts/train_separate.py --output outputs/separate/sb05b --resume              # с last/ до конца
python scripts/train_separate.py --output outputs/separate/sb05b --resume --epochs 15  # продлить готовый прогон
```

`run.sh` делает это сам: если в каталоге прогона есть `separate.json`, но нет `complete`, тот же запуск продолжает
обучение. С `--resume` настройки берутся из файлов прогона (`model.yaml`, `separate.json`, `split.json`,
`sft_config.json`); из флагов действуют только `--epochs`, `--device`, `--data-root`, `--num-workers`,
`--encoder-frames`, `--eval-batch-size`.

- Из `last/` восстанавливаются адаптер (LoRA, `<act>`, проектор, голова), состояние AdamW и генераторов случайных
  чисел, номер шага; лучшая эпоха пересчитывается по `validation_metrics.jsonl`. Батчи эпохи задаются её номером,
  поэтому прогон заканчивается так же, как непрерывный (`test_an_interrupted_run_resumes_as_one_run`).
- Эпоха считается законченной, когда записан её `last/` (он пишется рядом и подменяется целиком). Всё, что успела
  оставить недоделанная эпоха, — строка метрик, шаги, предсказания, `epoch_<n>/`, `best/` — отбрасывается, и эпоха
  проходится заново.
- Без `last/` (не закончена ни одна эпоха) прогон начинается заново в том же каталоге; полностью обученный (`complete`)
  не трогается.

## Что пишется

| где | что |
|---|---|
| `outputs/separate/<прогон>/` | `best/`, `last/` (+ `optimizer.pt`, `rng.pt` для `--resume`), `epoch_<n>/`: LoRA, строка `<act>`, проектор (с `--train-projector`), `head.safetensors`; `separate.json`, `model.yaml`, `split.json` (данные и настройки кэша), `sft_config.json`, метрики и предсказания; `complete` — все эпохи обучены |
| `results/separate/<прогон>/` | всё, кроме весов: метрики по эпохам, предсказания validation и test, `test_best.json` (метрики test, τ с validation, затраты: параметры, мс на решение, пик памяти) |
| `results/separate/test_report.json` | test-таблица: бейзлайны по уверенности, A (last_hidden и все группы), B (если есть её оценки на половинах голов), все прогоны SB |
