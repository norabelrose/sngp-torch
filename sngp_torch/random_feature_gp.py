from .hessians import FAST_HESSIANS
from .orf import orf_init
from contextlib import contextmanager
from functools import partial
from torch.distributions import MultivariateNormal, Normal
from torch import nn, Tensor
from typing import Callable, Generator, Literal, Optional, Union
import torch
import warnings


# Loss functions for which we have fast analytical Hessians wrt the final layer weights
FastHessianLoss = Literal['bce', 'ce', 'mse']

class RandomFeatureGP(nn.Module):
    """
    Drop-in replacement for the final `nn.Linear` layer in a network that uses a random
    feature-based approximation to a Gaussian process for uncertainty estimation.
    """
    def __init__(
            self,
            in_features: int,
            out_features: int,
            loss_fn: Optional[FastHessianLoss],
            *,
            cov_momentum: float = 1.0,
            cov_ridge_penalty: float = 1.0,
            device: Union[torch.device, None] = None,
            feature_activation: Callable[[Tensor], Tensor] = torch.cos,
            feature_bias_init: Callable[[Tensor], Tensor] = partial(nn.init.uniform_, b=2 * torch.pi),
            feature_weight_init: Callable[[Tensor], Tensor] = orf_init,
            kernel_amplitude: float = 1.0,
            layer_norm: bool = True,
            num_features: int = 1024,
            weight_decay: float = 0.0,
        ):
        super().__init__()
        assert loss_fn is None or loss_fn in FAST_HESSIANS, "`loss_fn` must be one of ('bce', 'ce', 'mse')."

        self.cov_momentum = cov_momentum
        self.cov_ridge_penalty = cov_ridge_penalty
        self.loss_fn = loss_fn
        self.num_features = num_features
        self.recording_covariance = False
        self.feature_activation = feature_activation
        self.feature_scale = kernel_amplitude * (2 / num_features) ** 0.5
        self.weight_decay = weight_decay

        if layer_norm:
            self.layer_norm = nn.LayerNorm(in_features, elementwise_affine=False)
        else:
            self.layer_norm = lambda x: x
        
        self.linear = nn.Linear(num_features, out_features)
        self.register_buffer('num_samples', torch.tensor(0))
        self.register_buffer('covariance_matrix', None)
        self.register_buffer('precision_matrix', None)

        self.register_buffer('feature_weight', torch.empty(num_features, in_features, device=device))
        self.register_buffer('feature_bias', torch.empty(num_features, device=device))
        feature_weight_init(self.feature_weight)
        feature_bias_init(self.feature_bias)

        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device))
        self.bias = nn.Parameter(torch.empty(out_features, device=device))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        nn.init.constant_(self.bias, 0)
    
    # Make Pylance type checking happy
    covariance_matrix: Optional[Tensor]
    num_samples: Tensor
    precision_matrix: Optional[Tensor]
    feature_weight: Tensor
    feature_bias: Tensor

    def featurize(self, hiddens: Tensor) -> Tensor:
        """Compute the random Fourier features for the given hidden states."""
        x = nn.functional.linear(hiddens, -self.feature_weight, self.feature_bias)
        return self.feature_scale * self.feature_activation(x)
    
    def forward(self, hiddens: Tensor):
        """Compute the MAP estimate of the GP posterior."""
        phi = self.featurize(hiddens)
        outputs = self.linear(phi)

        # Update covariance automatically if we're recording & using a supported loss function.
        if self.recording_covariance and isinstance(self.loss_fn, str):
            hess_fn = FAST_HESSIANS[self.loss_fn]
            self.update_precision(hess_fn(phi, outputs, self.weight_decay))
        
        return outputs
    
    def posterior(self, hiddens: Tensor, diag_only: bool = False) -> Union[MultivariateNormal, Normal]:
        """Compute Laplace approximation to the GP posterior."""
        assert self.covariance_matrix is not None, "Covariance matrix not computed yet."

        # Compute the posterior mean
        phi = self.featurize(hiddens)
        mean = self.linear(phi).squeeze(-1)  # Remove trailing 1-dim to prevent broadcasting

        # Add jitter to diagonal to ensure positive-definiteness
        # Technique and 1e-3 value taken from GPyTorch
        jitter = torch.eye(len(phi), device=phi.device) * 1e-3
        cov = phi @ self.covariance_matrix @ phi.mT + jitter
    
        if diag_only:
            # We know the args are valid and we don't want the performance hit of checking
            return Normal(mean, torch.linalg.cholesky(cov).diag(), validate_args=False)
        else:
            return MultivariateNormal(mean, cov, validate_args=False)
    
    @contextmanager
    def record_covariance(self) -> Generator['RandomFeatureGP', None, None]:
        """Context manager for recording the posterior covariance statistics."""
        self.recording_covariance = True

        yield self

        self.recording_covariance = False
        if self.precision_matrix is not None:
            # For exact estimation we need to divide by the number of samples.
            if self.cov_momentum == 1.0:
                self.precision_matrix /= self.num_samples
            
            # Use pseudo-inverse to handle edge cases where precision matrix is singular.
            self.covariance_matrix = torch.linalg.pinv(self.precision_matrix)
        else:
            warnings.warn("Exiting a `record_covariance` block without any covariance data.")
    
    def reset_covariance(self):
        self.num_samples.zero_()
        self.covariance_matrix = None
        self.precision_matrix = None
    
    def update_precision(self, batch_prec_matrix: Tensor):
        assert self.recording_covariance, "Must be in a `record_covariance` context."

        # Check if we need to initialize first
        if self.precision_matrix is None:
            device = self.feature_bias.device
            r = self.cov_ridge_penalty
            self.precision_matrix = torch.eye(self.num_features, device=device) * r
        
        if self.cov_momentum == 1.0:
            self.precision_matrix += batch_prec_matrix
        else:
            self.precision_matrix = (
                self.cov_momentum * self.precision_matrix +
                (1 - self.cov_momentum) * batch_prec_matrix
            )
        
        self.num_samples += 1
