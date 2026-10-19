# Обучение

Модель смотрит видео по мере поступления кадров и после каждой пары кадров решает, что делать: молчать (`silent`), готовиться (`prepare`) или отвечать (`speak`). На `speak` она генерирует ответ, и работа с этим вопросом заканчивается. Модель: Qwen2.5-VL-3B + LoRA + `StateHead`, см. [models/README.md](../models/README.md).

## Что готово

| часть | статус | где |
|---|---|---|
| конфиг обучения | готово | `configs/train/group1.yaml`, `train/config.py` |
| разметка кадров | готово | `data/labels.py` |
| сборка последовательности, датасет, батч | готово | `data/inputs.py`, `data/dataset.py` |
| режим модели `decisions`: состояния на всех позициях решений за один проход | готово | `models/model.py` |
| извлечение признаков, обучение головы на них, онлайн-оценка с τ | готово | `train/features.py`, `train/head.py` |
| focal loss, сквозной цикл через базу (`scripts/train.py`) | готово | `train/loss.py`, `train/loop.py` |
| языковая потеря на ответе (`loss.lm_weight > 0`) | **нет** | |
| инференс с порогом τ | **нет** | `runtime/controller.py` |

Сейчас обучается только `StateHead` поверх замороженной базы (`configs/model/qwen_3b_head.yaml`, без LoRA). Основной путь — через сохранённые признаки (раздел «Признаки»). Сквозной цикл `scripts/train.py` прогоняет 3B на каждом шаге, поэтому пригодится для LoRA. Разделы, описывающие ещё не сделанное, помечены отдельно.

## Данные

Первый этап — только **группа 1**: ответ появляется один раз внутри видео и больше не меняется. Обучающее подмножество строит `dataset_builders/clevrer_proactive/subset.py` → `train_group1.jsonl`, `val_group1.jsonl` (что отброшено и почему — в docstring `subset.py`). Записи с несколькими интервалами ответа (группа 2) датасет не принимает.

Время в записях указано в кадрах исходного видео (25 fps, 128 кадров): `answers[0].start_frame` — кадр S, с которого ответ уже известен. Формат записи описан в `data/schema.py`.

## Какие кадры видит модель

- Модель видит каждый k-й исходный кадр (`video.frame_stride`). Кадр модели `i` — это исходный кадр `i·k`.
- Кадры заранее уменьшаются до `video.frame_size` (по умолчанию 252×168). Процессор их размер не меняет, поэтому стороны должны быть кратны 28.
- Визуальный энкодер Qwen кодирует видео **парами кадров** (`temporal_patch_size = 2`). Пара 252×168 даёт 18×12 = 216 патчей ViT, после слияния 2×2 это **54 токена** для языковой модели.
- Решение можно принять только после целой пары, поэтому метки ставятся на пары.

## Разметка

S переводится в кадры модели: первый кадр модели не раньше исходного S. Пусть `w = labels.speak_delay` (тоже в кадрах модели). Каждая пара размечается по своему **последнему** кадру `e`:

| условие | метка | смысл |
|---|---|---|
| e < S | `silent` | ответа ещё нет |
| S ≤ e < S + w | `prepare` | ответ уже виден, модель намеренно выжидает w кадров, чтобы отвечать увереннее |
| первая пара с e ≥ S + w | `speak` | пора отвечать |

На паре `speak` видео в обучающей последовательности обрывается, кадры после неё не используются. Если пара `speak` не помещается в видео, вопрос отбрасывается (`ProactiveDataset.dropped`).

Пример: k = 1, S = 39, w = 10. Граница 49 — конец пары (48, 49):

```
пары, кончающиеся на кадрах 1, 3, …, 37   → silent   (19)
пары, кончающиеся на кадрах 39, …, 47     → prepare  (5)
пара (48, 49)                              → speak    (1)
```

Если S + w приходится на первый кадр пары, `speak` сдвигается на её конец, то есть на один кадр позже.

## Обучающая последовательность

Один вопрос — одна последовательность — **один forward pass** за шаг:

```
<|im_start|>user
<вопрос>
Options:
A. <вариант>
B. <вариант>
…
Answer with the letter of the correct option.<|vision_start|>
  [пара 0]   ← метка на последнем визуальном токене пары
  [пара 1]   ←
  …
  [пара speak]
<|vision_end|><|im_end|>
<|im_start|>assistant
<буква правильного варианта><|im_end|>      ← только при loss.lm_weight > 0
```

`InputBuilder.build_sequence` возвращает входы процессора, включая `mm_token_type_ids`, и две разметки по позициям (`-100` — позиция не участвует в потере). Без `mm_token_type_ids` Qwen молча строит для видео обычные 1D-позиции вместо 3D, поэтому модель такие входы отклоняет. Разметки:

- `state_labels` — номер класса на последнем визуальном токене каждой пары;
- `lm_labels` — токены ответа и EOS, по соглашению HF (метка j — это токен j, сдвиг делает потеря).

Один проход по всей последовательности эквивалентен онлайн-просмотру. Визуальный энкодер Qwen2.5-VL не смешивает разные пары, а языковая модель причинная, поэтому позиция решения на паре p не видит кадров после p. Это проверяет `tests/test_sequence.py::test_hidden_states_do_not_see_later_patches`.

На реальных данных (252×168, k = 1, w = 10) последовательность занимает 1,4–2,4 тысячи токенов.

## Признаки

Набор признаков (несколько слоёв, внимание, проба «ответь сейчас», визуальный энкодер), его объём, формат хранения и выбор входов головы описаны в [FEATURES.md](FEATURES.md). Ниже — общая идея.

База заморожена, поэтому её скрытые состояния на позициях решений от эпохи к эпохе не меняются. `scripts/extract_features.py` один раз прогоняет базу по каждому вопросу и сохраняет признаки. `scripts/train_head.py` затем учит голову только на них: 3B в этом цикле не загружается, эпоха идёт минуты.

- **Признаки берутся по всему видео,** а не до `speak`: вопрос и все целые пары кадров (для CLEVRER 64 пары, ~3500 токенов). Признак на паре p после такого прохода совпадает с тем, что модель видит онлайн после пары p (`tests/test_features.py`). Поэтому:
  - разметку (`labels.speak_delay`), веса классов, γ и всё в `optim` можно менять без повторного извлечения;
  - признаки после пары `speak` дают онлайн-оценку с порогом τ.
- **Повторное извлечение нужно** при смене `video.*` (кадры) или модели: признаки сохраняют эти настройки в `meta.json`, и обучение головы сверяет их с конфигом.
- **Объём:** ~76 КиБ на пару, ~337 ГБ для `train_group1` и ~169 ГБ для `val_group1` (разбивка по группам — в FEATURES.md).
- **Прерывание не страшно:** повторный запуск той же команды продолжит с последнего сохранённого видео (файл `done`).

Онлайн-оценка: для каждого вопроса модель «отвечает» на первой паре, где P(speak) > τ. Исход:

| исход | когда сработало |
|---|---|
| `premature` | до того, как ответ виден (пара `silent`) |
| `early` | ответ виден, но раньше пары `speak` (`prepare`) |
| `on_target` | на паре `speak` |
| `late` | позже |
| `missed` | ни разу за видео |

Плюс медианная задержка в исходных кадрах относительно пары `speak`: отрицательная — раньше, положительная — позже.

## Шаг обучения

1. На признаках (`train_head.py`): батч из `batch_size` вопросов — их признаки до пары `speak`, разметка по текущему `speak_delay`. Через базу (`train.py`): батч последовательностей (паддинг справа) один раз проходит через модель в режиме `decisions`, и логиты состояния берутся на всех позициях, где `state_labels != -100`.
2. Потеря — focal loss, усреднённая по всем позициям решений батча:
   ```
   focal = −α_c · (1 − p_c)^γ · log p_c
   ```
   В последовательности много `silent`, несколько `prepare` и один `speak`. Focal loss и веса классов `α_c` компенсируют этот дисбаланс.
3. Градиенты накапливаются `grad_accum` проходов, затем clip, шаг AdamW и шаг расписания lr (линейный warmup, затем косинус до 0).

Проход через замороженную базу идёт без градиента и без хранения активаций: голове они не нужны, поэтому gradient checkpointing не требуется. `StateHead` хранится в float32 при любой точности базы, чтобы маленькие обновления не терялись при округлении до bf16.

Раз в `eval_every` шагов считаются loss, точность и полнота по каждому классу на фиксированной случайной выборке `eval_samples` вопросов из eval. Раз в `save_every` шагов сохраняется чекпоинт. Режим `decisions` даёт на каждой позиции те же логиты, что модель выдала бы онлайн на префиксе до этой позиции: это проверяет `tests/test_train.py::test_decisions_match_online_prefixes`.

**Языковая потеря *(в планах)*:** `+ λ · CE(ответ, lm_labels)`, где `lm_head` применяется только к позициям ответа. Пока `loss.lm_weight` должен быть 0.

## Инференс *(в планах)*

Кадры подаются парами и дописываются в KV-cache. На последнем токене каждой пары считается P(`speak`). Как только P(`speak`) > τ, дописываются `<|vision_end|><|im_end|>\n<|im_start|>assistant\n` и генерируется ответ, после чего обработка вопроса заканчивается. Модель в момент решения видит ровно то же, что видела при обучении.

## Конфиги

Модель: `configs/model/qwen_3b_head.yaml`, то есть Qwen2.5-VL-3B без LoRA с головой `[silent, prepare, speak]`. Номер класса в метках — индекс в `head.classes`, а `class_alpha` должен называть ровно эти классы.

Обучение: `configs/train/group1.yaml`.

```yaml
data:
  train: processed/clevrer_proactive/train_group1.jsonl   # относительно DATA_ROOT
  eval: processed/clevrer_proactive/val_group1.jsonl
  eval_samples: 500        # фиксированная случайная выборка eval для оценки по ходу обучения; null — весь eval
video:
  frame_stride: 1          # модель видит каждый k-й исходный кадр
  frame_size: [252, 168]   # [ширина, высота], кратно 28
labels:
  speak_delay: 10          # w в кадрах модели
loss:
  focal_gamma: 2.0         # γ
  class_alpha: {silent: 0.25, prepare: 1.0, speak: 1.0}   # α_c
  lm_weight: 0.0           # λ; пока должен быть 0
optim:
  lr: 1.0e-3
  weight_decay: 0.01
  warmup_steps: 50         # шагов оптимизатора линейного warmup, дальше косинус до 0
  epochs: 1
  max_steps: null          # ограничение на число шагов оптимизатора
  batch_size: 1            # последовательностей на один проход
  grad_accum: 16           # проходов на один шаг оптимизатора
  grad_clip: 1.0
  num_workers: 2           # воркеры DataLoader, декодирующие видео
  log_every: 10
  eval_every: 200
  save_every: 200
seed: 0
```

## Запуск

### 1. Окружение

Нужен Python ≥ 3.10 и torch с CUDA, затем:

```bash
pip install -r requirements.txt     # transformers 5.3.x, peft 0.18.x
```

### 2. Данные

```bash
VIDEOS=all bash scripts/prepare_clevrer_proactive.sh   # вопросы, аннотации и 18,6 ГБ видео train + validation
python -m dataset_builders.clevrer_proactive.subset    # -> train_group1.jsonl, val_group1.jsonl
```

### 3. Тесты

```bash
python -m pytest -q   # разметка, сборка последовательности, модель на маленькой случайной Qwen (без сети)
```

### 4. Проверка последовательностей на реальных данных

Скачивает только процессор Qwen (токенизатор и конфиги), без весов модели:

```python
from pathlib import Path
from transformers import AutoProcessor
from online_vllm.data.dataset import ProactiveDataset
from online_vllm.data.inputs import InputBuilder
from online_vllm.train.config import TrainConfig

builder = InputBuilder(AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct"))
ds = ProactiveDataset(Path("data/processed/clevrer_proactive/train_group1.jsonl"), Path("data"), builder,
                      TrainConfig.load("configs/train/group1.yaml"), ["silent", "prepare", "speak"])
x = ds[0]
print(len(ds), ds.dropped, x["input_ids"].shape, x["video_grid_thw"])
print(builder.processor.tokenizer.decode(x["input_ids"][0, -12:]))
```

Классы передаются в порядке `head.classes` из `configs/model/qwen_3b_state.yaml`: номер класса в метках — это его индекс в этом списке.

### 5. Извлечение признаков

Пошаговый план для сервера с A100 и выгрузкой на Google Drive — [FEATURES.md, раздел «Запуск на сервере»](FEATURES.md#запуск-на-сервере): ресурсы, rclone, пробный прогон, пилот, полное извлечение, наблюдение и сбои.

Это единственный шаг, где нужна 3B. На GPU меньше ~10 ГБ используйте `configs/model/qwen_3b_head_int8.yaml`: языковая часть в int8 (bitsandbytes), ~4,7 ГБ видеопамяти. Признаки тогда получаются от квантованной базы: голова должна работать на той же базе, и для таких признаков нужен отдельный каталог. Кроме того, LLM.int8 выбирает столбцы-выбросы по всей последовательности, поэтому причинность выполняется лишь приближённо; для обучения годятся признаки bf16. Windows при нехватке видеопамяти не падает, а выгружает лишнее в оперативную память, и всё идёт в десятки раз медленнее; поэтому скрипт ограничивает PyTorch 95% видеопамяти.

### 6. Обучение головы

```bash
python scripts/train_head.py --output outputs/group1_head --device cuda
```

Сквозное обучение через базу (для LoRA, нужна `batch_size` 1–2) запускается так:
```bash
python scripts/train.py --model configs/model/qwen_3b_head.yaml --train configs/train/group1.yaml \
    --output outputs/group1_e2e --device cuda
```

В `--output` появятся:
- `train.yaml` — конфиг обучения;
- `metrics.jsonl` — строка на каждое логирование и оценку: `loss`, `lr`, `eval_loss`, `eval_accuracy`, `eval_recall_<класс>`, `eval_confusion[истинный][предсказанный]`. У `train_head.py` есть ещё `online_<τ>_<исход>` и `online_<τ>_median_delay_frames`. Последняя оценка (`"final": true`) идёт по всему eval, промежуточные — по `eval_samples` вопросам;
- `step_N/` и `final/` — чекпоинты (`model.yaml`, `trainable.pt`, `train.yaml`), загружаются через `load_checkpoint` как полная модель.

**Где брать вектор для головы.** Голова учится на последнем визуальном токене пары. Нынешний `OnlineController` классифицирует по последнему токену шаблона (`assistant`), то есть по другой позиции. При реализации инференса с τ его нужно перевести на позицию решения.

## Файлы

```
configs/train/group1.yaml        данные, кадры, разметка, потеря, оптимизация
configs/model/qwen_3b_head.yaml  замороженная Qwen2.5-VL-3B + StateHead, без LoRA
scripts/extract_features.py      извлечение признаков (один раз, нужна 3B)
scripts/train_head.py            обучение головы на признаках
scripts/train.py                 сквозное обучение через базу
src/online_vllm/train/features.py  извлечение и чтение признаков
src/online_vllm/train/head.py    обучение головы на признаках, онлайн-оценка с τ
src/online_vllm/train/config.py  TrainConfig: загрузка и проверка конфига
src/online_vllm/train/loss.py    focal loss, веса классов
src/online_vllm/train/loop.py    цикл обучения, оценка, чекпоинты
src/online_vllm/models/model.py  режим decisions
src/online_vllm/data/labels.py   plan(): метки пар silent / prepare / speak
src/online_vllm/data/video.py    read_every(): каждый k-й кадр с уменьшением
src/online_vllm/data/inputs.py   InputBuilder.build_sequence(): последовательность и разметка по позициям
src/online_vllm/data/dataset.py  ProactiveDataset, collate
tests/test_labels.py, tests/test_sequence.py, tests/test_train.py, tests/test_features.py
```
