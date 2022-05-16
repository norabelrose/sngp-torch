from torch import Tensor
import torch


def bce_hessian(features: Tensor, logits: Tensor, weight_decay: float) -> Tensor:
    """Analytic Hessian of the binary cross-entropy loss wrt the final layer weights."""
    probs = logits.sigmoid()
    x = torch.sqrt(probs * (1 - probs)) * features
    return x.T @ x / len(features) + weight_decay * torch.eye(features.shape[-1])

def ce_hessian(features: Tensor, logits: Tensor, weight_decay: float) -> Tensor:
    """
    Analytic Hessian of the K-class cross entropy loss wrt the final layer weights
    *of the most probable class.*
    """
    highest_probs = logits.softmax(dim=-1).argmax(dim=-1)
    x = features * highest_probs * (1 - highest_probs) ** 2
    return 2 * x.T @ x / len(features) + weight_decay * torch.eye(features.shape[-1])

def mse_hessian(features: Tensor, _: Tensor, weight_decay: float) -> Tensor:
    """Analytic Hessian of the MSE loss wrt the final layer weights."""
    base = 2 * features.T @ features / len(features)
    return base + weight_decay * torch.eye(features.shape[-1])

FAST_HESSIANS = {'bce': bce_hessian, 'ce': ce_hessian, 'mse': mse_hessian}
