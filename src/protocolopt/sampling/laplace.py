from ..core.sampling import InitialConditionGenerator
import torch
import math
from torch.func import vmap, hessian
import pyro.distributions as dist

class LaplaceApproximation(InitialConditionGenerator):
    def __init__(self, params, centers, device):
        self.centers = centers
        self.device = device
        self.dt = params['dt']
        self.gamma = params['gamma']
        self.mass = params['mass']
        self.beta = params['beta']
        self.spatial_dimensions = params.get('spatial_dimensions', 1)
        self.time_steps = params.get('time_steps', 1000)
        self.mcmc_starting_spatial_bounds = params.get('mcmc_starting_spatial_bounds', None)

        if 'num_samples' in params:
            self.num_samples = params['num_samples']
        elif 'mcmc_num_samples' in params:
            self.num_samples = params['mcmc_num_samples']
        elif 'samples_per_well' in params:
             self.num_samples = params['samples_per_well'] * (2**self.spatial_dimensions)
        else:
             self.num_samples = 1000

        self.noise_sigma = torch.sqrt(torch.tensor(2 * self.gamma / self.beta))

    def _solve_landscape(self, potential, protocol):
        coeff_at_t0 = protocol.get_coeff_grid()[:, 0]
        def potential_kernel(x):
            return potential.potential_value(x, coeff_at_t0)

        batched_hessian_func = vmap(hessian(potential_kernel), in_dims = 0)

        hessian_at_centers = batched_hessian_func(self.centers) # Nwells x spatial_dimensions x spatial_dimensions
        sign, logabsdet = torch.linalg.slogdet(hessian_at_centers)
        if not (sign == 1).all():
            raise ValueError(f"Some hessians ({len(hessian_at_centers[sign == -1])}/{len(hessian_at_centers):.2f}%) are not positive definite, some of your bit centers are unstable")
        if not (logabsdet > math.log(1e-5)).all():
            raise ValueError(f"Some hessians ({len(logabsdet[logabsdet <= 0])}/{len(logabsdet):.2f}%) have too near zero determinant (1e-5 cutoff), some of your bit centers are unsuitable for this method. Request better handling of this if needed.")
        # Log P = -E - 0.5 * log|H|
        self.log_weights = -potential_kernel(self.centers) - 0.5 * logabsdet
        self.hessian_at_centers = hessian_at_centers

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
        samples = torch.randn(self.num_samples, self.spatial_dimensions, device=self.device) * math.sqrt(var)
        return samples

    def _get_noise(self):
        samples = torch.randn(self.num_samples, self.spatial_dimensions, self.time_steps, device=self.device) * self.noise_sigma * math.sqrt(self.dt)
        return samples

    def _get_initial_positions(self):
        with torch.no_grad():
            chosen_centers = dist.Categorical(self.log_weights).sample((self.num_samples,))
            center_locs = self.centers[chosen_centers]
            chosen_H = self.hessian_at_centers[chosen_centers]

            # Laplaces approximation, x ~ N(x_0, H^-1)
            # precision matrix will invert the hessian internally
            samples = dist.MultivariateNormal(center_locs, precision_matrix = chosen_H).sample()
            return samples

    def generate_initial_conditions(self, potential, protocol, loss):
        if not hasattr(self, 'log_weights'):
            print("Solving landscape for Laplace approximation...")
            self._solve_landscape(potential, protocol)
            print("Landscape solved")

        initial_pos = self._get_initial_positions()
        initial_vel = self._get_initial_velocities()
        noise = self._get_noise()
        return initial_pos, initial_vel, noise
