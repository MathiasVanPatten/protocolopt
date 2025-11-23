from abc import ABC, abstractmethod
import torch
import pyro
from pyro.infer.mcmc import MCMC, NUTS
import math
import itertools


class InitialConditionGenerator(ABC):
    @abstractmethod
    def generate_initial_conditions(self, *args, **kwargs):
        #you should return initial positions, initial velocities, and noise
        pass


class MCMCNUTSInitialConditionGenerator(InitialConditionGenerator):
    def __init__(self, params, device, dt, gamma, mass):
        self.device = device
        self.spatial_dimensions = params.get('spatial_dimensions', 1)
        self.time_steps = params.get('time_steps', 1000)
        self.mcmc_warmup_ratio = params.get('mcmc_warmup_ratio', 0.1)
        self.mcmc_starting_spatial_bounds = params.get('mcmc_starting_spatial_bounds', torch.tensor([[-5, 5]] * self.spatial_dimensions, device=self.device))
        self.mcmc_chains_per_well = params.get('mcmc_chains_per_well', 1)
        self.samples_per_well = params.get('samples_per_well', None)
        if self.samples_per_well is None:
            self.mcmc_num_samples = params.get('mcmc_num_samples', 5000)
        else:
            self.mcmc_num_samples = self.samples_per_well * self.mcmc_chains_per_well * 2**self.spatial_dimensions
        self.beta = params.get('beta', 1.0)
        self.gamma = gamma
        self.dt = dt
        self.noise_sigma = torch.sqrt(torch.tensor(2 * self.gamma / self.beta))
        self.mass = mass

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
        samples = torch.randn(self.mcmc_num_samples, self.spatial_dimensions, self.time_steps, device=self.device) * self.noise_sigma * torch.sqrt(torch.tensor(self.dt))
        return samples

    def _run_multichain_mcmc(self, potential, potential_model, loss):
        """Run MCMC sampling, either per-well or global depending on parameters"""
        if self.samples_per_well is not None:
            # Per-well sampling mode
            well_bounds = self._partition_bounds_by_midpoints(loss)
            all_samples = []
            
            for well_idx, bounds in enumerate(well_bounds):
                well_samples = []
                samples_per_chain = self.samples_per_well // self.mcmc_chains_per_well
                
                for chain_id in range(self.mcmc_chains_per_well):
                    sampler = MCMC(
                        NUTS(lambda: self._posterior_for_mcmc(bounds[:, 0], bounds[:, 1], potential, potential_model)),
                        num_samples=samples_per_chain,
                        warmup_steps=int(self.mcmc_warmup_ratio * samples_per_chain)
                    )
                    sampler.run()
                    well_samples.append(sampler.get_samples()['x'])
                
                all_samples.append(torch.cat(well_samples, dim=0))
            
            self.starting_pos = torch.cat(all_samples, dim=0)
        else:
            # Legacy mode: global sampling
            all_samples = []
            num_chains = self.mcmc_chains_per_well
            samples_per_chain = self.mcmc_num_samples // num_chains
            
            bounds_low = self.mcmc_starting_spatial_bounds[:, 0]
            bounds_high = self.mcmc_starting_spatial_bounds[:, 1]
            
            for chain_id in range(num_chains):
                sampler = MCMC(
                    NUTS(lambda: self._posterior_for_mcmc(bounds_low, bounds_high, potential, potential_model)),
                    num_samples=samples_per_chain,
                    warmup_steps=int(self.mcmc_warmup_ratio * samples_per_chain)
                )
                sampler.run()
                all_samples.append(sampler.get_samples()['x'])
            
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
        initial_pos = self._run_multichain_mcmc(potential, potential_model, loss)
        initial_vel = self._get_initial_velocities()
        noise = self._get_noise()
        return initial_pos, initial_vel, noise