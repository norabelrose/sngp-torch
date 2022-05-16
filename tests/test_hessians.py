from sngp_torch.hessians import bce_hessian, ce_hessian, mse_hessian
from torch.nn.functional import binary_cross_entropy_with_logits, mse_loss
import pytest
import torch


@pytest.mark.parametrize('pair', [
    (bce_hessian, binary_cross_entropy_with_logits),
    (mse_hessian, mse_loss)
])
def test_correctness(pair):
    fast_hess, loss = pair

    features = torch.randn(1000, 256)
    weights = torch.randn(256)
    labels = torch.empty(1000).bernoulli(0.5)
    logits = features @ weights[..., None]

    def loss_fn(w):
        logits = features @ w[..., None]
        return loss(logits.squeeze(), labels) + 0.5 * w.norm() ** 2
    
    # Test against PyTorch autograd
    th_hessian = torch.autograd.functional.hessian(loss_fn, weights)
    our_hessian = fast_hess(features, logits, 1.0)
    torch.testing.assert_allclose(th_hessian, our_hessian)