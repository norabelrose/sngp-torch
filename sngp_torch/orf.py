from torch import Tensor
import math
import torch


def orf_init(dest: Tensor) -> Tensor:
    """Initialize a parameter with orthogonal random features."""
    dest.copy_(orthogonal_random_features(*dest.shape))
    return dest


def orthogonal_random_features(n: int, m: int, device = None):
    """
    Sample a random matrix of size (n, m) with columns that are 'as orthogonal as possible'
    given the dimensions. If m > n, then the matrix will consist of square orthogonal tiles
    concatenated horizontally.
    """
    return math.sqrt(n) * torch.cat([
        orthogonal_square_matrix(n, device)
        for _ in range(math.ceil(m / n))
    ], dim=1)[:, :m]


def orthogonal_square_matrix(n: int, device = None) -> Tensor:
    gaussian = torch.randn(n, n, device=device)
    q, _ = torch.linalg.qr(gaussian)
    return q