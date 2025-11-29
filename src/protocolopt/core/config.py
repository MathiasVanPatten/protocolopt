from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union
import torch

@dataclass
class ProtocolOptimizerConfig:
    """Configuration for the ProtocolOptimizer class.

    Attributes:
        spatial_dimensions: Number of spatial dimensions.
        time_steps: Number of simulation time steps.
        epochs: Number of training epochs.
        learning_rate: Learning rate for the optimizer.
        beta: Inverse temperature parameter.
        gamma: Friction coefficient (overridden by simulator if present).
        dt: Time step size (overridden by simulator if present).
        mass: Particle mass (overridden by simulator if present).
        grad_clip_max_norm: Maximum norm for gradient clipping.
        samples_per_well: Number of samples to draw per well (for MCMC).
        mcmc_warmup_ratio: Ratio of warmup steps for MCMC.
        mcmc_starting_spatial_bounds: Spatial bounds for starting positions.
        mcmc_chains_per_well: Number of MCMC chains per well.
        min_neff: Minimum effective sample size for MCMC.
        mcmc_num_samples: Total number of MCMC samples (if not per well).
        run_every_epoch: Whether to run MCMC every epoch.
        flow_epochs: Epochs for flow training.
        flow_batch_size: Batch size for flow training.
        flow_training_samples_per_well: Samples per well for flow training.
        num_samples: Number of samples (generic).
    """
    spatial_dimensions: int = 1
    time_steps: int = 1000
    epochs: int = 100
    learning_rate: float = 0.001
    beta: float = 1.0
    gamma: float = 1.0
    dt: float = 0.01
    mass: float = 1.0
    grad_clip_max_norm: float = 1.0

    # MCMC / Sampling Params
    samples_per_well: Optional[int] = None
    mcmc_warmup_ratio: float = 0.1
    mcmc_starting_spatial_bounds: Optional[torch.Tensor] = None
    mcmc_chains_per_well: int = 1
    min_neff: Optional[float] = None
    mcmc_num_samples: int = 5000
    run_every_epoch: bool = False

    # Flow Params
    flow_epochs: int = 300
    flow_batch_size: int = 256
    flow_training_samples_per_well: int = 500

    # Generic
    num_samples: int = 1000

    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> 'ProtocolOptimizerConfig':
        """Creates a ProtocolOptimizerConfig from a dictionary, ignoring unknown keys."""
        valid_keys = cls.__dataclass_fields__.keys()
        filtered_params = {k: v for k, v in params.items() if k in valid_keys}
        return cls(**filtered_params)
