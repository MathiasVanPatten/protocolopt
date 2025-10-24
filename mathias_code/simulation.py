#definition of the free paramters, defintion of the static setup
import torch
import pyro
from pyro.infer.mcmc import MCMC, NUTS
import math

#TODO In theory now the ability to form a full trajectory tensor with all of the potential, protocol, and sim engine
# chain should work, in the next session we need to tie them together so they pass what's needed between
# and set up the actual epoch training loop in simulation which probably will be renamed to allow for 
# a full training loop

class Simulation:
    def __init__(self, potential, sim_engine, params) -> None:
        #need some kind of initial conditions
        #need some kind of parameters input
        #need some kind of starting conditions filter, let's just let it be x for now
        self.potential = potential
        self.sim_engine = sim_engine
        self.device = self.potential.potential_model.device
        self.spatial_dimensions = params.get('spatial_dimensions', 1)

        self.time_steps = params.get('time_steps', 1000)
        self.steps_per_spatial = params.get('steps_per_spatial', 100)

        self.mcmc_num_samples = params.get('mcmc_num_samples', 5000)
        self.mcmc_warmup_ratio = params.get('mcmc_warmup_ratio', 0.1)
        self.mcmc_starting_spatial_bounds = params.get('mcmc_starting_spatial_bounds', torch.tensor([[-5, 5]] * self.spatial_dimensions), device = self.device)
        
        self.beta = params.get('beta', 1.0)
        self.gamma = self.sim_engine.gamma
        self.dt = self.sim_engine.dt
        

        self.epochs = params.get('epochs', 100)
        self.learning_rate = params.get('learning_rate', 0.001)
        if self.potential.potential_model.fixed_starting:
            self.mcmc_sampler = MCMC(NUTS(self._posterior_for_mcmc), num_samples=self.mcmc_num_samples, warmup_steps=self.mcmc_warmup_ratio * self.mcmc_num_samples)
            self.mcmc_sampler.run()
    
    def _log_prob(self, state_vectors):
        #exp(-beta * U), boltzman distribution assumed for posterior
        coeff_at_t0 = self.potential.coeff_grid[:, 0]
        return -self.beta * self.potential.potential_value(state_vectors.squeeze(-1), coeff_at_t0)

    def _posterior_for_mcmc(self):
        #makes it look like pyro need it to look
        low = self.mcmc_starting_spatial_bounds[:, 0]
        high = self.mcmc_starting_spatial_bounds[:, 1]
        x = pyro.sample("x", pyro.distributions.Uniform(low, high).to_event(1))
        pyro.factor("logp", self._log_prob(x.unsqueeze(-1)))

    def _get_initial_positions(self):
        #output of shape (num_samples, spatial_dimensions)
        #uses mcmc sampler to sample given potential in uniform region specified
        if not self.potential.potential_model.fixed_starting:
            return MCMC(NUTS(self._posterior_for_mcmc), num_samples=self.mcmc_num_samples, warmup_steps=self.mcmc_warmup_ratio * self.mcmc_num_samples).get_samples()['x']
        else:
            return self.mcmc_sampler.get_samples()['x']

    def _get_initial_velocities(self):
        #Sample velocities from a Gaussian distribution with mean 0 and variance 1/(2*beta)
        var = 1 / (2 * self.beta)
        samples = torch.randn(self.mcmc_num_samples, device=self.device) * math.sqrt(var)
        return samples

    def _get_noise(self):
        std = torch.sqrt(torch.tensor(2 * self.gamma) / self.beta)
        samples = torch.randn(self.mcmc_num_samples, self.time_steps, device=self.device) * std * torch.sqrt(self.dt)
        return samples

    def simulate(self, initial_positions = None, initial_velocities = None, noise = None):
        #simulates the system for either given initial conditions or sampled initial conditions
        if initial_positions is None:
            initial_positions = self._get_initial_positions()
        if initial_velocities is None:
            initial_velocities = self._get_initial_velocities()
        if noise is None:
            noise = self._get_noise()
        coeff_grid = self.potential.potential_model.get_coeff_grid()
        return self.sim_engine.make_trajectories(self.potential, initial_positions, initial_velocities, self.time_steps, noise, coeff_grid, self.device)

    def _setup_optimizer(self, optimizer_class = torch.optim.Adam):
        self.optimizer = optimizer_class(self.potential.potential_model.trainable_params(), lr=self.learning_rate)

    def train(self):
        self._setup_optimizer()
        for epoch in range(self.epochs):
            pass