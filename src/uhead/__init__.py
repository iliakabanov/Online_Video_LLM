"""Causal UHead для определения готовности ответить по кешированным видео-признакам."""

from .config import UHeadConfig
from .model import UHead

__all__ = ["UHead", "UHeadConfig"]
