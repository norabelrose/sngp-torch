from sngp_torch import RandomFeatureGP
from sngp_torch.orf import orf_init
import pytest
import torch


@pytest.mark.parametrize('input_shape', [(8, 12, 64), (8, 6, 16, 32)])
@pytest.mark.parametrize('kernel_init', [orf_init, torch.nn.init.normal_])
def test_rbf_approx(input_shape: tuple[int, ...], kernel_init):
    """Tests if default random features approximate a RBF kernel matrix."""
    torch.manual_seed(0)

    rff_gp = RandomFeatureGP(
        input_shape[-1], 1, 'mse', num_rff=8192,
        rff_weight_init=kernel_init
    )
    x = torch.randn(input_shape)
    y = rff_gp.featurize(x).squeeze(-1)

    expected = torch.exp(-torch.cdist(x, x) ** 2 / 2)
    actual = y @ y.mT

    torch.testing.assert_allclose(actual, expected, atol=5e-2, rtol=1e-2)