"""Streaming input of MMDuet2 (Wang et al., arXiv 2512.06810): the video arrives as user turns of frames, and the
assistant answers after every turn, "NO REPLY" while it has nothing to say.

    system     SYSTEM_PROMPT
    user       <prompt>[frame 0][frame 1]       every frame is a separate image, encoded on its own
    assistant  NO REPLY
    user       [frame 2][frame 3]
    assistant  NO REPLY
    ...

Every assistant turn is filled with "NO REPLY", so one forward pass gives, at every turn, what the model sees online
at that turn as long as it stayed silent: the language model is causal and the vision encoder never mixes images.
The decision of a turn is read at the last token of its "<|im_start|>assistant\n": the next-token distribution there
starts the model's reply, and the "NO REPLY" tokens that follow give P(NO REPLY) in the same pass.

Images carry no time in their 3D positions (t is the same for all tokens of an image, text continues after the
largest position), so the model sees time only as the order of the turns.
"""
from dataclasses import dataclass

import numpy as np
import torch

# The system prompt of MMDuet2's proactive inference (proactive_eval/inference.py in the authors' code).
SYSTEM_PROMPT = ("You are a helpful assistant. Your task is to answer questions based on continuously incoming video "
                 "frames. Your responses should include information from the video since your last reply (if any). If "
                 "the information in this segment of the video cannot answer the question, output \"NO REPLY\".")
NO_REPLY = "NO REPLY"


@dataclass
class Turns:
    inputs: dict              # processor outputs: input_ids, attention_mask, mm_token_type_ids, pixel_values, ...
    decisions: torch.Tensor   # [T] last token of the assistant header of every turn
    silence: list[int]        # token ids of NO_REPLY; they follow every decision position


class TurnsBuilder:
    """One sequence per question: the prompt in the first user turn, then `frames_per_turn` frames per turn."""

    def __init__(self, processor, frames_per_turn: int = 2):
        if frames_per_turn < 1:
            raise ValueError("frames_per_turn must be positive")
        self.processor = processor
        self.frames_per_turn = frames_per_turn
        self.silence = processor.tokenizer(NO_REPLY, add_special_tokens=False).input_ids

    def messages(self, prompt: str, num_turns: int) -> list[dict]:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for k in range(num_turns):
            content = [{"type": "text", "text": prompt}] if k == 0 else []
            content += [{"type": "image"}] * self.frames_per_turn
            messages += [{"role": "user", "content": content}, {"role": "assistant", "content": NO_REPLY}]
        return messages

    def build(self, prompt: str, frames: np.ndarray) -> Turns:
        """frames: uint8 [N, H, W, 3] at their final size (multiples of patch x merge), N a whole number of turns."""
        ip = self.processor.image_processor
        factor = ip.patch_size * ip.merge_size
        if frames.ndim != 4 or frames.shape[1] % factor or frames.shape[2] % factor:
            raise ValueError(f"Frames must be [N, H, W, 3] with H and W multiples of {factor}, got {frames.shape}")
        if not len(frames) or len(frames) % self.frames_per_turn:
            raise ValueError(f"{len(frames)} frames are not whole turns of {self.frames_per_turn}")
        turns = len(frames) // self.frames_per_turn
        text = self.processor.apply_chat_template(self.messages(prompt, turns), tokenize=False)
        inputs = self.processor(text=[text], images=list(frames), return_tensors="pt", do_resize=False)
        ids = inputs["input_ids"][0]
        first_image = int((inputs["mm_token_type_ids"][0] == 1).nonzero()[0, 0])
        n = len(self.silence)
        # "NO REPLY" also appears (quoted) in the system prompt; the assistant turns are the ones after the first image.
        starts = (ids.unfold(0, n, 1) == torch.tensor(self.silence)).all(1).nonzero()[:, 0]
        starts = starts[starts > first_image]
        if len(starts) != turns:
            raise ValueError(f"Expected {turns} assistant turns, found {len(starts)}")
        return Turns(dict(inputs), starts - 1, list(self.silence))
