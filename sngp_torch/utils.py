from torch import nn


def apply_spectral_norm(module: nn.Module, name: str = 'weight') -> nn.Module:
    """
    Recursively apply spectral normalization to all linear and conv layers in a module.
    """
    for child in module.modules():
        if isinstance(child, nn.Linear) or isinstance(child, nn.Conv2d):
            nn.utils.parametrizations.spectral_norm(child, name)
    
    return module