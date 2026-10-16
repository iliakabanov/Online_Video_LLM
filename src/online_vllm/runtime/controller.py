from dataclasses import dataclass
from typing import Callable

import torch


@dataclass
class ObservationResult:
    state_logits: torch.Tensor
    class_id: int
    answer_ids: torch.Tensor | None


class OnlineController:
    """Synchronous observation/answer cycle; no dialogue history is retained.

    Call again with fresh video after answer completion. A custom should_answer
    policy can implement suppression of repeated answers without changing model inputs.
    """

    def __init__(self, model, input_builder, *, should_answer: Callable[[int], bool] | None = None):
        self.model = model
        self.input_builder = input_builder
        if should_answer is None:
            speak_id = model.config.head.classes.index("speak")
            should_answer = lambda class_id: class_id == speak_id
        self.should_answer = should_answer

    def observe(self, question, video, t, **generation_kwargs):
        inputs = self.input_builder.build_observation(question, video, t)
        inputs = inputs.to(self.model.qwen.get_input_embeddings().weight.device)
        logits = self.model.predict_state(inputs).state_logits
        class_id = int(logits.argmax(dim=-1).item())
        answer = None
        if self.should_answer(class_id):
            # The same prepared snapshot is used throughout generation; no new frames enter.
            generation_kwargs.setdefault("max_new_tokens", 128)
            answer = self.model.generate_answer(inputs, **generation_kwargs)
        return ObservationResult(logits, class_id, answer)
