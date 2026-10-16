# Обучение

Модель смотрит видео по мере поступления кадров и после каждой пары кадров решает, что делать: молчать (`silent`), готовиться (`prepare`) или отвечать (`speak`). На `speak` она генерирует ответ, и работа с этим вопросом заканчивается. Модель: Qwen2.5-VL-3B + LoRA + `StateHead`, см. [models/README.md](../models/README.md).

## Что готово

| часть | статус | где |
|---|---|---|
| конфиг обучения | готово | `configs/train/group1.yaml`, `train/config.py` |
| разметка кадров | готово | `data/labels.py` |
| сборка последовательности, датасет, батч | готово | `data/inputs.py`, `data/dataset.py` |
| режим модели «один проход → состояния + ответ» | **нет** | `models/model.py` |
| focal loss, цикл обучения, `scripts/train.py` | **нет** | `train/` |
| инференс с порогом τ | **нет** | `runtime/controller.py` |

Разделы ниже описывают и готовые части, и те, что ещё в планах; второе помечено отдельно.

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
Options: <вариант>, <вариант>, …<|vision_start|>
  [пара 0]   ← метка на последнем визуальном токене пары
  [пара 1]   ←
  …
  [пара speak]
<|vision_end|><|im_end|>
<|im_start|>assistant
<правильный ответ><|im_end|>      ← только при loss.lm_weight > 0
```

`InputBuilder.build_sequence` возвращает входы процессора и две разметки по позициям (`-100` — позиция не участвует в потере):

- `state_labels` — номер класса на последнем визуальном токене каждой пары;
- `lm_labels` — токены ответа и EOS, по соглашению HF (метка j — это токен j, сдвиг делает потеря).

Один проход по всей последовательности эквивалентен онлайн-просмотру. Визуальный энкодер Qwen2.5-VL не смешивает разные пары, а языковая модель причинная, поэтому позиция решения на паре p не видит кадров после p. Это проверяет `tests/test_sequence.py::test_hidden_states_do_not_see_later_patches`.

На реальных данных (252×168, k = 1, w = 10) последовательность занимает 1,4–2,4 тысячи токенов.

## Потери *(в планах)*

```
L = focal(state_logits на позициях решений, state_labels) + λ · CE(ответ, lm_labels)
focal = −α_c · (1 − p_c)^γ · log p_c
```

- В последовательности много `silent`, несколько `prepare` и один `speak`. Focal loss и веса классов `α_c` компенсируют этот дисбаланс.
- `lm_head` применяется только к позициям ответа, а не ко всему словарю на всех позициях.
- Обучаются LoRA и `StateHead`, база заморожена.

## Инференс *(в планах)*

Кадры подаются парами и дописываются в KV-cache. На последнем токене каждой пары считается P(`speak`). Как только P(`speak`) > τ, дописываются `<|vision_end|><|im_end|>\n<|im_start|>assistant\n` и генерируется ответ, после чего обработка вопроса заканчивается. Модель в момент решения видит ровно то же, что видела при обучении.

## Конфиг: `configs/train/group1.yaml`

```yaml
video:
  frame_stride: 1          # модель видит каждый k-й исходный кадр
  frame_size: [252, 168]   # [ширина, высота], кратно 28
labels:
  speak_delay: 10          # w в кадрах модели
loss:
  focal_gamma: 2.0         # γ
  class_alpha: {silent: 0.25, prepare: 1.0, speak: 1.0}   # α_c
  lm_weight: 1.0           # λ; 0 — ответ не добавляется, учится только состояние
```

## Запуск

### 1. Окружение

Нужен Python ≥ 3.10 и torch с CUDA, затем:

```bash
pip install -e ".[train,dev]"     # transformers 5.3.x, peft 0.18.x
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

### 5. Обучение *(в планах)*

`scripts/train.py` с конфигами модели и обучения ещё не написан.

## Файлы

```
configs/train/group1.yaml        параметры кадров, разметки и потерь
src/online_vllm/train/config.py  TrainConfig: загрузка и проверка конфига
src/online_vllm/data/labels.py   plan(): метки пар silent / prepare / speak
src/online_vllm/data/video.py    read_every(): каждый k-й кадр с уменьшением
src/online_vllm/data/inputs.py   InputBuilder.build_sequence(): последовательность и разметка по позициям
src/online_vllm/data/dataset.py  ProactiveDataset, collate
tests/test_labels.py, tests/test_sequence.py
```
