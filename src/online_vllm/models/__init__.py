"""Общие конфигурации и ленивый доступ к Qwen-зависимым компонентам.

Cached-feature StateHead/UHead не должны даже импортировать PEFT или Qwen.
"""
from .config import BackboneConfig, LoRAConfig, ModelConfig, StateHeadConfig

__all__ = ["build_model", "load_checkpoint", "save_checkpoint", "save_head_checkpoint", "BackboneConfig",
           "LoRAConfig", "ModelConfig", "StateHeadConfig", "StateOutput", "VideoStateModel"]


def __getattr__(name):
    if name == "build_model":
        from .builder import build_model
        return build_model
    if name in {"load_checkpoint", "save_checkpoint", "save_head_checkpoint"}:
        from .checkpoint import load_checkpoint, save_checkpoint, save_head_checkpoint
        return {"load_checkpoint": load_checkpoint, "save_checkpoint": save_checkpoint,
                "save_head_checkpoint": save_head_checkpoint}[name]
    if name in {"StateOutput", "VideoStateModel"}:
        from .model import StateOutput, VideoStateModel
        return {"StateOutput": StateOutput, "VideoStateModel": VideoStateModel}[name]
    raise AttributeError(name)
