from ..core.sampling import InitialConditionGenerator
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

class McmcNuts(InitialConditionGenerator):
    def __init__(self, params, device):
        self.device = device
        self.spatial_dimensions = params.get('spatial_dimensions', 1)
        self.time_steps = params.get('time_steps', 1000)
        self.mcmc_warmup_ratio = params.get('mcmc_warmup_ratio', 0.1)
        self.mcmc_starting_spatial_bounds = params.get('mcmc_starting_spatial_bounds', torch.tensor([[-5, 5]] * self.spatial_dimensions, device=self.device))
        self.mcmc_chains_per_well = params.get('mcmc_chains_per_well', 1)
        self.min_neff = params.get('min_neff', None)
        self.samples_per_well = params.get('samples_per_well', None)
        if self.samples_per_well is None:
            self.mcmc_num_samples = params.get('mcmc_num_samples', 5000)
        else:
            self.mcmc_num_samples = self.samples_per_well * self.mcmc_chains_per_well * 2**self.spatial_dimensions
        self.beta = params.get('beta', 1.0)
        self.gamma = params['gamma']
        self.dt = params['dt']
        self.noise_sigma = torch.sqrt(torch.tensor(2 * self.gamma / self.beta))
        self.mass = params['mass']
        self.starting_pos = None
        self.run_every_epoch = params.get('run_every_epoch', False)
        if self.run_every_epoch:
            print("Warning: Running MCMC every epoch will be very slow and is not recommended for training. Please use the ConditionalFlowBoltzmannGenerator instead if your potential doesn't change at t0.")

    def _get_initial_velocities(self):
        # P(v) = exp(-beta * E_k)
        # E_k = 1/2 * m * v^2
        # P(v) = exp(-beta * 1/2 * m * v^2)
        # generally P(v) = exp(-v^2 * 1/2 / sigma^2)
        # v^2 * 1/2 / sigma^2 = beta * 1/2 * m * v^2
        # sigma^2 = 1 / (beta * m)
        # sigma = sqrt(1 / (beta * m))

        # we draw from randn which produces N(0, 1) and scale by sigma to get N(0, sigma)
        var = 1 / (self.beta * self.mass)
        samples = torch.randn(self.mcmc_num_samples, self.spatial_dimensions, device=self.device) * math.sqrt(var)
        return samples

    def _get_noise(self):
        samples = torch.randn(self.mcmc_num_samples, self.spatial_dimensions, self.time_steps, device=self.device) * self.noise_sigma * math.sqrt(self.dt)
        return samples

    def _run_multichain_mcmc(self, potential, potential_model, loss):
        """Run MCMC sampling, either per-well or global depending on parameters"""

        max_attempts = 5

        if self.samples_per_well is not None:
            # Per-well sampling mode
            well_bounds = self._partition_bounds_by_midpoints(loss)
            all_samples = []

            for well_idx, bounds in enumerate(well_bounds):

                samples_per_chain = self.samples_per_well // self.mcmc_chains_per_well

                for attempt in range(max_attempts):
                    well_samples = []

                    # Run chains
                    for chain_id in range(self.mcmc_chains_per_well):
                        sampler = MCMC(
                            NUTS(lambda: self._posterior_for_mcmc(bounds[:, 0], bounds[:, 1], potential, potential_model)),
                            num_samples=samples_per_chain,
                            warmup_steps=int(self.mcmc_warmup_ratio * samples_per_chain)
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
            num_chains = self.mcmc_chains_per_well
            samples_per_chain = self.mcmc_num_samples // num_chains

            bounds_low = self.mcmc_starting_spatial_bounds[:, 0]
            bounds_high = self.mcmc_starting_spatial_bounds[:, 1]

            for attempt in range(max_attempts):
                chain_samples_list = []

                for chain_id in range(num_chains):
                    sampler = MCMC(
                        NUTS(lambda: self._posterior_for_mcmc(bounds_low, bounds_high, potential, potential_model)),
                        num_samples=samples_per_chain,
                        warmup_steps=int(self.mcmc_warmup_ratio * samples_per_chain)
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

    def _log_prob(self, state_vectors, potential, potential_model):
        #exp(-beta * U), boltzman distribution assumed for posterior
        coeff_at_t0 = potential_model.get_coeff_grid()[:, 0]
        return -self.beta * potential.potential_value(state_vectors, coeff_at_t0)

    def _posterior_for_mcmc(self, bounds_low, bounds_high, potential, potential_model):
        #makes it look like pyro need it to look
        x = pyro.sample("x", pyro.distributions.Uniform(bounds_low, bounds_high).to_event(1))
        pyro.factor("logp", self._log_prob(x.unsqueeze(-1), potential, potential_model))

    def _partition_bounds_by_midpoints(self, loss):
        """Partition the sampling bounds using midpoints to create per-well regions"""

        if not hasattr(loss, 'midpoints'):
            return [self.mcmc_starting_spatial_bounds]

        midpoints = loss.midpoints
        bounds = self.mcmc_starting_spatial_bounds

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

    def generate_initial_conditions(self, potential, potential_model, loss):
        if self.run_every_epoch:
            initial_pos = self.starting_pos
        else:
            initial_pos = self._run_multichain_mcmc(potential, potential_model, loss)
        initial_vel = self._get_initial_velocities()
        noise = self._get_noise()
        return initial_pos, initial_vel, noise
