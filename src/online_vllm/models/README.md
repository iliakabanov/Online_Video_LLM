# Qwen с классификатором состояния

Одна `Qwen2.5-VL-3B-Instruct`, LoRA и отдельная `StateHead`.

- `forward(inputs, mode="state")`: мультимодальная база → вектор последней непустой позиции → MLP. LM head не вызывается.
- `forward(inputs, mode="language")`: языковые логиты для обучения генерации. MLP не вызывается.
- `predict_state(inputs)`: классификация без градиентов, требует `eval()`.
- `generate_answer(inputs, ...)`: только новые token IDs ответа, без MLP; требует `eval()`.

Последний токен — буквально последняя непустая позиция подготовленного входа. Это может быть служебный токен chat template; выделенного токена решения нет. `InputBuilder` использует стандартный Qwen chat template с `add_generation_prompt=True`, чтобы тот же вход подходил для начала ответа. Внутри пользовательского сообщения порядок — вопрос, затем видео. История диалога не добавляется. Для другого формата можно передать собственные подготовленные входы напрямую.

`VideoPrefix` принимает RGB uint8 `[N,H,W,3]` с равномерной выборкой от времени 0. `InputBuilder` обрезает кадры по `t` **до** визуальной обработки, отключает повторную выборку кадров и передаёт FPS. Для нерегулярных временных меток понадобится отдельная реализация подготовки входа. Минимальная версия builder/controller работает с одним видео; модель принимает уже подготовленные батчи. Классификация поддерживает оба направления padding, генерация требует левый padding.

`OnlineController.observe()` классифицирует снимок входа и при `speak` запускает генерацию на том же снимке. Новые кадры во время ответа не поступают. После возврата можно вызвать `observe()` с новым `t`. История ответов не хранится, подавление повторных ответов задаётся внешней `should_answer` политикой. По умолчанию `prepare` не запускает генерацию. Контроллер синхронный, один экземпляр модели не рассчитан на параллельные вызовы.

## Настройки и обучение

Конфиг: `configs/model/qwen_3b_state.yaml`. Индекс класса определяется порядком `head.classes`. `hidden_dims: []` даёт линейную голову. `lora.layers: null` выбирает все слои, список — конкретные индексы. Компоненты LoRA: `language_model`, `vision_encoder`, `merger`; имена модулей должны соответствовать выбранному компоненту (например, vision attention использует `qkv`, `proj`). Резолвер проверяет модули и печатает точные пути через logging. Исходные веса заморожены; обучаются LoRA и MLP. `lora.enabled: false` позволяет обучать только голову.

```python
import torch
from online_vllm.models import ModelConfig, build_model, save_checkpoint

model = build_model(ModelConfig.load("configs/model/qwen_3b_state.yaml"))
model.train()
# inputs — результат processor, labels — метки состояния [B].
# logits = model(inputs, mode="state").state_logits
# loss = torch.nn.functional.cross_entropy(logits, labels)
# loss.backward()
```

Loss, создание меток `prepare/speak`, optimizer и расписание обучения остаются снаружи модели. Для language-обучения можно передать `labels` внутри inputs, используя стандартную causal-LM loss Qwen. Во время генерации MLP не определяет завершение ответа: используются EOS и ограничения генерации.

По умолчанию генерация пересчитывает префикс. `use_cache=True` разрешает KV cache только внутри одного ответа; между независимыми вызовами cache не передаётся.

`save_checkpoint(model, path)` сохраняет `model.yaml` и `trainable.pt` с LoRA/MLP. `load_checkpoint(path)` загружает исходную базу и эти веса; optimizer не сохраняется. На первой загрузке Hub revision разрешается в commit SHA и попадает в конфиг чекпоинта. Для всех экспериментов следует закрепить этот SHA в общем YAML. Локальный путь базовой модели должен оставаться доступен. Если используется изменённый processor/chat template, его следует сохранять отдельно через `processor.save_pretrained()`.

## Проверка

```powershell
python -m pip install -e ".[train,dev]"
python -m pytest tests/test_model_structure.py tests/test_model_inputs.py -q
```

Offline smoke создаёт маленькую случайно инициализированную Qwen той же архитектуры, сохраняет её и **реально загружает через `from_pretrained`**. Проверяет видеовход, LoRA-градиенты, разделение голов, padding и восстановление обученных параметров. Отдельный тест использует настоящий processor с игрушечным tokenizer. Эти тесты не оценивают качество предобученной 3B.

Проверка именно 3B с синтетическими кадрами (скачивает веса, требует достаточно RAM/VRAM):

```powershell
python scripts/smoke_model.py --device cuda
# При уже скачанной модели:
python scripts/smoke_model.py --device cuda --local-files-only
```

На CPU скрипт использует float32. MLP изначально случайная, поэтому её предсказания до обучения не имеют содержательного смысла.
