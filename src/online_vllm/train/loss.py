import torch
import torch.nn.functional as F


def class_weights(class_alpha: dict[str, float], classes: list[str]) -> torch.Tensor:
    """alpha from the config in the order of the model's head classes."""
    if set(class_alpha) != set(classes):
        raise ValueError(f"class_alpha must name exactly the head classes {classes}, got {sorted(class_alpha)}")
    return torch.tensor([float(class_alpha[c]) for c in classes])


def focal_loss(logits: torch.Tensor, target: torch.Tensor, gamma: float, alpha: torch.Tensor) -> torch.Tensor:
    """Mean of -alpha_c * (1 - p_c)^gamma * log p_c over positions; gamma = 0 and alpha = 1 give cross-entropy."""
    log_p = F.log_softmax(logits.float(), dim=-1).gather(1, target[:, None])[:, 0]
    return (-alpha.to(logits.device)[target] * (1 - log_p.exp()) ** gamma * log_p).mean()
