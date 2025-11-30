from ..core.sampling import InitialConditionGenerator
from ..core.potential import Potential
from ..core.protocol import Protocol
from ..core.loss import Loss
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from pyro.infer.mcmc import MCMC, NUTS
import pyro
import torch
import math
import itertools
import pyro.distributions as dist
import pyro.distributions.transforms as T
from tqdm import tqdm
from typing import Optional, Dict, Any, List, Tuple

class McmcNuts(InitialConditionGenerator):
    """Initial condition generator using MCMC with the NUTS sampler."""

    def __init__(
        self,
        dt: float,
        gamma: float,
        mass: float,
        device: torch.device,
        spatial_dimensions: int = 1,
        time_steps: int = 1000,
        beta: float = 1.0,
        starting_bounds: Optional[torch.Tensor] = None,
        samples_per_well: Optional[int] = None,
        num_samples: int = 5000,
        chains_per_well: int = 1,
        warmup_ratio: float = 0.1,
        min_neff: Optional[float] = None,
        run_every_epoch: bool = False
    ) -> None:
        """Initializes McmcNuts.

        Args:
            dt (float): Time step.
            gamma (float): Friction coefficient.
            mass (float): Particle mass.
            device (torch.device): Calculation device.
            spatial_dimensions (int): Number of spatial dimensions.
            time_steps (int): Number of time steps.
            beta (float): 1/kT
            starting_bounds (torch.Tensor): Bounds for starting positions.
            samples_per_well (int): Samples per well.
            num_samples (int): Total samples if not per well.
            chains_per_well (int): Chains per well.
            warmup_ratio (float): Ratio of warmup steps.
            min_neff (float): Minimum effective sample size.
            run_every_epoch (bool): Whether to run sampling every epoch.
        """
        self.device = device
        self.spatial_dimensions = spatial_dimensions
        self.time_steps = time_steps
        self.warmup_ratio = warmup_ratio
        
        if starting_bounds is None:
             self.starting_bounds = torch.tensor([[-5, 5]] * self.spatial_dimensions, device=self.device)
        else:
             self.starting_bounds = starting_bounds
             
        self.chains_per_well = chains_per_well
        self.min_neff = min_neff
        self.samples_per_well = samples_per_well
        
        if self.samples_per_well is None:
            self.num_samples = num_samples
        else:
            self.num_samples = self.samples_per_well * self.chains_per_well * 2**self.spatial_dimensions
            
        self.beta = beta
        self.gamma = gamma
        self.dt = dt
        self.noise_sigma = torch.sqrt(torch.tensor(2 * self.gamma / self.beta))
        self.mass = mass
        self.starting_pos = None
        self.run_every_epoch = run_every_epoch
        
        if self.run_every_epoch:
            print("Warning: Running MCMC every epoch will be very slow and is not recommended for training. Please use the ConditionalFlowBoltzmannGenerator instead if your potential doesn't change at t0.")

        self.hparams = {
            'spatial_dimensions': self.spatial_dimensions,
            'time_steps': self.time_steps,
            'warmup_ratio': self.warmup_ratio,
            'starting_bounds': self.starting_bounds.tolist() if isinstance(self.starting_bounds, torch.Tensor) else self.starting_bounds,
            'chains_per_well': self.chains_per_well,
            'min_neff': self.min_neff,
            'samples_per_well': self.samples_per_well,
            'num_samples': self.num_samples,
            'beta': self.beta,
            'gamma': self.gamma,
            'dt': self.dt,
            'mass': self.mass,
            'run_every_epoch': self.run_every_epoch,
            'name': self.__class__.__name__
        }

    def _get_initial_velocities(self) -> torch.Tensor:
        """Generates initial velocities from the Maxwell-Boltzmann distribution.

        Returns:
            Initial velocities tensor. Shape: (Num_Samples, Spatial_Dim).
        """
        # P(v) = exp(-beta * E_k)
        # E_k = 1/2 * m * v^2
        # P(v) = exp(-beta * 1/2 * m * v^2)
        # generally P(v) = exp(-v^2 * 1/2 / sigma^2)
        # v^2 * 1/2 / sigma^2 = beta * 1/2 * m * v^2
        # sigma^2 = 1 / (beta * m)
        # sigma = sqrt(1 / (beta * m))

        # we draw from randn which produces N(0, 1) and scale by sigma to get N(0, sigma)
        var = 1 / (self.beta * self.mass)
        samples = torch.randn(self.num_samples, self.spatial_dimensions, device=self.device) * math.sqrt(var)
        return samples

    def _get_noise(self) -> torch.Tensor:
        """Generates noise for the simulation.

        Returns:
            Noise tensor. Shape: (Num_Samples, Spatial_Dim, Time_Steps).
        """
        samples = torch.randn(self.num_samples, self.spatial_dimensions, self.time_steps, device=self.device) * self.noise_sigma * math.sqrt(self.dt)
        return samples

    def _run_multichain_mcmc(self, potential: Potential, protocol: Protocol, loss: Loss) -> torch.Tensor:
        """Run MCMC sampling, either per-well or global depending on parameters.

        Args:
            potential: The potential energy object.
            protocol: The protocol object providing coefficients at t=0.
            loss: The loss object (used for midpoints if per-well sampling).

        Returns:
            Sampled positions. Shape: (Num_Samples, Spatial_Dim).
        """

        max_attempts = 5

        if self.samples_per_well is not None:
            # Per-well sampling mode
            well_bounds = self._partition_bounds_by_midpoints(loss)
            all_samples = []

            for well_idx, bounds in enumerate(well_bounds):

                samples_per_chain = self.samples_per_well // self.chains_per_well

                for attempt in range(max_attempts):
                    well_samples = []

                    # Run chains
                    for chain_id in range(self.chains_per_well):
                        sampler = MCMC(
                            NUTS(lambda: self._posterior_for_mcmc(bounds[:, 0], bounds[:, 1], potential, protocol)),
                            num_samples=samples_per_chain,
                            warmup_steps=int(self.warmup_ratio * samples_per_chain)
                        )
                        sampler.run()
                        well_samples.append(sampler.get_samples()['x'])

                    # Check Neff if required
                    current_samples = torch.cat(well_samples, dim=0)

                    if self.min_neff is None:
                        break

                    # we stack the chains together to vectorize the Neff check
                    chain_stack = torch.stack(well_samples, dim=0)
                    neff_per_dim = pyro.ops.stats.effective_sample_size(chain_stack, chain_dim=0, sample_dim=1)
                    min_observed_neff = neff_per_dim.min().item()

                    print(f"Well {well_idx} Attempt {attempt+1}: Neff = {min_observed_neff:.2f} (Target: {self.min_neff})")

                    if min_observed_neff >= self.min_neff:
                        break

                    # Need more samples
                    if attempt < max_attempts - 1:
                        ratio = self.min_neff / max(min_observed_neff, 1e-6)
                        # scale factor, bounded to avoid explosion but ensure progress
                        scale_factor = max(1.1, min(ratio, 5.0))
                        new_samples_per_chain = int(samples_per_chain * scale_factor)
                        print(f"  - Neff insufficient. Increasing samples per chain from {samples_per_chain} to {new_samples_per_chain}")
                        samples_per_chain = new_samples_per_chain
                    else:
                        print(f"  - WARNING: Max attempts reached for Well {well_idx}. Neff {min_observed_neff:.2f} < {self.min_neff}")

                # Print Neff statistics for the final samples of this well
                # Re-calculate since we might have broken out of loop
                chain_stack = torch.stack(well_samples, dim=0)
                neff_per_dim = pyro.ops.stats.effective_sample_size(chain_stack, chain_dim=0, sample_dim=1)
                min_neff = neff_per_dim.min().item()
                mean_neff = neff_per_dim.mean().item()
                print(f"  Well {well_idx} Stats: Min Neff: {min_neff:.2f}, Mean Neff: {mean_neff:.2f}")

                all_samples.append(current_samples)

            self.starting_pos = torch.cat(all_samples, dim=0)
        else:
            # Legacy mode: global sampling
            all_samples = []
            num_chains = self.chains_per_well
            samples_per_chain = self.num_samples // num_chains

            bounds_low = self.starting_bounds[:, 0]
            bounds_high = self.starting_bounds[:, 1]

            for attempt in range(max_attempts):
                chain_samples_list = []

                for chain_id in range(num_chains):
                    sampler = MCMC(
                        NUTS(lambda: self._posterior_for_mcmc(bounds_low, bounds_high, potential, protocol)),
                        num_samples=samples_per_chain,
                        warmup_steps=int(self.warmup_ratio * samples_per_chain)
                    )
                    sampler.run()
                    chain_samples_list.append(sampler.get_samples()['x'])

                # Check Neff
                if self.min_neff is None:
                    all_samples = chain_samples_list
                    break

                chain_stack = torch.stack(chain_samples_list, dim=0)
                neff_per_dim = pyro.ops.stats.effective_sample_size(chain_stack, chain_dim=0, sample_dim=1)
                min_observed_neff = neff_per_dim.min().item()

                print(f"Global Sampling Attempt {attempt+1}: Neff = {min_observed_neff:.2f} (Target: {self.min_neff})")

                if min_observed_neff >= self.min_neff:
                    all_samples = chain_samples_list
                    break

                if attempt < max_attempts - 1:
                    ratio = self.min_neff / max(min_observed_neff, 1e-6)
                    scale_factor = max(1.1, min(ratio, 5.0))
                    new_samples_per_chain = int(samples_per_chain * scale_factor)
                    print(f"  - Neff insufficient. Increasing samples per chain from {samples_per_chain} to {new_samples_per_chain}")
                    samples_per_chain = new_samples_per_chain
                else:
                    print(f"  - WARNING: Max attempts reached. Neff {min_observed_neff:.2f} < {self.min_neff}")
                    all_samples = chain_samples_list

            # Print final stats for global sampling
            if len(all_samples) > 0:
                 chain_stack = torch.stack(all_samples, dim=0)
                 neff_per_dim = pyro.ops.stats.effective_sample_size(chain_stack, chain_dim=0, sample_dim=1)
                 min_neff = neff_per_dim.min().item()
                 mean_neff = neff_per_dim.mean().item()
                 print(f"  MCMC Batch Stats: Min Neff: {min_neff:.2f}, Mean Neff: {mean_neff:.2f}")

            self.starting_pos = torch.cat(all_samples, dim=0)
        return self.starting_pos

    def _log_prob(self, state_vectors: torch.Tensor, potential: Potential, protocol: Protocol) -> torch.Tensor:
        #exp(-beta * U), boltzman distribution assumed for posterior
        coeff_at_t0 = protocol.get_protocol_tensor()[:, 0]
        return -self.beta * potential.potential_value(state_vectors, coeff_at_t0)

    def _posterior_for_mcmc(self, bounds_low: torch.Tensor, bounds_high: torch.Tensor, potential: Potential, protocol: Protocol) -> None:
        #makes it look like pyro need it to look
        x = pyro.sample("x", pyro.distributions.Uniform(bounds_low, bounds_high).to_event(1))
        pyro.factor("logp", self._log_prob(x.unsqueeze(-1), potential, protocol))

    def _partition_bounds_by_midpoints(self, loss: Loss) -> List[torch.Tensor]:
        """Partition the sampling bounds using midpoints to create per-well regions"""

        if not hasattr(loss, 'midpoints'):
            return [self.starting_bounds]

        midpoints = loss.midpoints
        bounds = self.starting_bounds

        dim_segments = []
        for dim_idx in range(self.spatial_dimensions):
            low = bounds[dim_idx, 0]
            high = bounds[dim_idx, 1]
            mid = midpoints[dim_idx]
            dim_segments.append([[low, mid], [mid, high]])
        well_bounds = []
        for combination in itertools.product(*dim_segments):
            well_bound = torch.stack([torch.tensor(seg, device=self.device) for seg in combination])
            well_bounds.append(well_bound)
        return well_bounds

    def generate_initial_conditions(self, potential: Potential, protocol: Protocol, loss: Loss) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generates initial conditions (positions, velocities, noise).

        Args:
            potential: The potential energy object.
            protocol: The protocol object.
            loss: The loss object.

        Returns:
            Tuple of (initial_pos, initial_vel, noise).
        """
        if self.run_every_epoch:
            initial_pos = self.starting_pos
        else:
            initial_pos = self._run_multichain_mcmc(potential, protocol, loss)
        initial_vel = self._get_initial_velocities()
        noise = self._get_noise()
        return initial_pos, initial_vel, noise
