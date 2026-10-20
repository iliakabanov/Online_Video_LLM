"""Единые метрики streaming-readiness, независимые от архитектуры детектора."""

from .metrics import evaluate_readiness, flatten_readiness_metrics, statehead_probabilities, uhead_probabilities

__all__ = ["evaluate_readiness", "flatten_readiness_metrics", "statehead_probabilities", "uhead_probabilities"]
