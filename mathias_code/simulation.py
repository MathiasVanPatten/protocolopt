#definition of the free paramters, defintion of the static setup
# from evolution import EulerMaruyama
import torch
import pyro
from pyro.infer.mcmc import MCMC, NUTS



class Simulation:
    def __init__(self, protocol, sim_engine, params) -> None:
        #need some kind of initial conditions
        #need some kind of parameters input
        #need some kind of starting conditions filter, let's just let it be x for now
        self.protocol = protocol
        self.sim_engine = sim_engine
        self.device = self.protocol.potential_parms.device
        self.spatial_dimensions = len(protocol.potential_params.shape) - 2 #minus 2 for time and params

        self.time_steps = params.get('time_steps', 1000)
        self.steps_per_spatial = params.get('steps_per_spatial', 100)

        self.mcmc_num_samples = params.get('mcmc_num_samples', 5000)
        self.mcmc_warmup_ratio = params.get('mcmc_warmup_ratio', 0.1)
        self.mcmc_starting_spatial_bounds = params.get('mcmc_starting_spatial_bounds', torch.tensor([[-5, 5]] * self.spatial_dimensions), device = self.device)
        pass
    
    def _log_prob(self, state_vectors):
        #exp(-beta * V), boltzman distribution assumed for posterior
        return - self.params['beta'] * self.protocol.potential_value(state_vectors)

    def _potential_model(self):
        #makes it look like pyro need it to look
        x = pyro.sample("x", pyro.distributions.Uniform(self.mcmc_starting_spatial_bounds[0], self.mcmc_starting_spatial_bounds[1]).to(self.device))
        pyro.factor("logp", self.log_prob(x.unsqueeze(-1)))

    def _get_initial_positions(self):
        return MCMC(NUTS(self._potential_model), num_samples=self.mcmc_num_samples, warmup_steps=self.mcmc_warmup_ratio * self.mcmc_num_samples).get_samples()['x']