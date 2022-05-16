from .hessians import FAST_HESSIANS
from contextlib import contextmanager
from functools import partial
from torch.distributions import MultivariateNormal, Normal
from torch import nn, Tensor
from typing import Callable, Generator, Literal
import torch
import warnings


# Loss functions for which we have fast analytical Hessians wrt the final layer weights.
# For other losses we use `torch.autograd.functional.hessian` with the experimental
# vectorize flag turned on, but this is still slower than the analytical Hessians.
FastHessianLoss = Literal['bce', 'ce', 'mse']

class RandomFeatureGP(nn.Module):
    """
    Drop-in replacement for the final `nn.Linear` layer in a network that uses a random
    Fourier feature-based approximation to a Gaussian process for uncertainty estimation.
    """
    def __init__(
            self,
            in_features: int,
            out_features: int,
            loss_fn: Callable[[Tensor, Tensor], Tensor] | FastHessianLoss,
            *,
            cov_momentum: float = 1.0,
            cov_ridge_penalty: float = 1.0,
            device: torch.device | None = None,
            kernel_amplitude: float = 1.0,
            layer_norm: bool = True,
            num_rff: int = 1024,
            rff_activation: Callable[[Tensor], Tensor] = torch.cos,
            rff_bias_init: Callable[[Tensor], Tensor] = partial(nn.init.uniform_, b=2 * torch.pi),
            rff_weight_init: Callable[[Tensor], Tensor] = nn.init.normal_,
            weight_decay: float = 0.0,
        ):
        super().__init__()
        assert callable(loss_fn) or loss_fn in FAST_HESSIANS,\
            "`loss_fn` must be a callable or one of ('bce', 'ce', 'mse')."

        self.cov_momentum = cov_momentum
        self.cov_ridge_penalty = cov_ridge_penalty
        self.loss_fn = loss_fn
        self.num_rff = num_rff
        self.recording_covariance = False
        self.rff_activation = rff_activation
        self.rff_scale = kernel_amplitude * (2 / num_rff) ** 0.5
        self.weight_decay = weight_decay

        if layer_norm:
            self.layer_norm = nn.LayerNorm(in_features, elementwise_affine=False)
        else:
            self.layer_norm = lambda x: x
        
        self.linear = nn.Linear(num_rff, out_features)
        self.register_buffer('num_samples', torch.tensor(0))
        self.register_buffer('covariance_matrix', None)
        self.register_buffer('precision_matrix', None)

        self.register_buffer('rff_weight', torch.empty(num_rff, in_features, device=device))
        self.register_buffer('rff_bias', torch.empty(num_rff, device=device))
        rff_weight_init(self.rff_weight)
        rff_bias_init(self.rff_bias)
    
    # Make Pylance type checking happy
    covariance_matrix: Tensor | None
    num_samples: Tensor
    precision_matrix: Tensor | None
    rff_weight: Tensor
    rff_bias: Tensor

    def featurize(self, hiddens: Tensor) -> Tensor:
        """Compute the random Fourier features for the given hidden states."""
        x = nn.functional.linear(hiddens, -self.rff_weight, self.rff_bias)
        return self.rff_scale * self.rff_activation(x)
    
    def forward(self, hiddens: Tensor):
        """Compute the MAP estimate of the GP posterior."""
        phi = self.featurize(hiddens)
        outputs = self.linear(phi)

        # Update covariance automatically if we're recording & using a supported loss function.
        if self.recording_covariance and isinstance(self.loss_fn, str):
            hess_fn = FAST_HESSIANS[self.loss_fn]
            self._update_precision(hess_fn(phi, outputs, self.weight_decay))
        
        return outputs
    
    def posterior(self, hiddens: Tensor, diag_only: bool = False) -> MultivariateNormal | Normal:
        """Compute Laplace approximation to the GP posterior."""
        assert self.covariance_matrix is not None, "Covariance matrix not computed yet."

        # Compute the posterior mean
        phi = self.featurize(hiddens)
        mean = self.linear(phi).squeeze(-1)  # Remove trailing 1-dim to prevent broadcasting

        # Add jitter to diagonal to ensure positive-definiteness
        # Technique and 1e-3 value taken from GPyTorch
        cov = phi @ self.covariance_matrix @ phi.mT + torch.eye(len(phi)) * 1e-3
    
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
    
    def update_covariance(self, outputs: Tensor, targets: Tensor):
        assert callable(self.loss_fn), "No need to call `update_covariance` for built-in loss functions."
        assert self.recording_covariance, "Enter a `record_covariance` block first."

        hess = torch.autograd.functional.hessian(
            partial(self.loss_fn, targets), outputs, vectorize=True
        )
        self._update_precision(hess)
    
    def reset_covariance(self):
        self.num_samples.zero_()
        self.covariance_matrix = None
        self.precision_matrix = None
    
    def _update_precision(self, batch_prec_matrix: Tensor):
        # Check if we need to initialize first
        if self.precision_matrix is None:
            self.precision_matrix = torch.eye(self.num_rff) * self.cov_ridge_penalty
        
        if self.cov_momentum == 1.0:
            self.precision_matrix += batch_prec_matrix
        else:
            self.precision_matrix = (
                self.cov_momentum * self.precision_matrix +
                (1 - self.cov_momentum) * batch_prec_matrix
            )
        
        self.num_samples += 1
