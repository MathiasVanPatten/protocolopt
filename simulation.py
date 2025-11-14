#definition of the free paramters, defintion of the static setup
import torch
import pyro
from pyro.infer.mcmc import MCMC, NUTS
import math
from tqdm import tqdm
import itertools


class Simulation:
    def __init__(self, potential, sim_engine, loss, potential_model, params, callbacks = []) -> None:
        self.potential = potential
        self.sim_engine = sim_engine
        self.loss = loss
        self.potential_model = potential_model
        self.device = self.potential_model.device
        self.spatial_dimensions = params.get('spatial_dimensions', 1)
        self.callbacks = callbacks
        self.time_steps = params.get('time_steps', 1000)
        self.steps_per_spatial = params.get('steps_per_spatial', 100)


        self.mcmc_warmup_ratio = params.get('mcmc_warmup_ratio', 0.1)
        self.mcmc_starting_spatial_bounds = params.get('mcmc_starting_spatial_bounds', torch.tensor([[-5, 5]] * self.spatial_dimensions, device = self.device))
        self.mcmc_chains_per_well = params.get('mcmc_chains_per_well', 1)
        self.samples_per_well = params.get('samples_per_well', None)
        if self.samples_per_well is None:
            self.mcmc_num_samples = params.get('mcmc_num_samples', 5000)
        else:
            self.mcmc_num_samples = self.samples_per_well * self.mcmc_chains_per_well * 2**self.spatial_dimensions

        self.beta = params.get('beta', 1.0)
        self.gamma = self.sim_engine.gamma
        self.dt = self.sim_engine.dt
        #HACK WHEN NOISE IS GIVEN TO SIMULATE THE NOISE SIGMA WILL STILL BE SET AS IT WAS DURING TRAINING, POSSIBLY AN ISSUE
        #HACK MU IS FIXED AT 1 FOR NOW
        self.noise_sigma = torch.sqrt(torch.tensor(2 * self.gamma) / self.beta) #Einstein relation for overdamped systems TODO will need to change when underdamped is added
        self.epochs = params.get('epochs', 100)
        self.learning_rate = params.get('learning_rate', 0.001)
        self.grad_clip_max_norm = params.get('grad_clip_max_norm', 5.0)
        if self.potential_model.fixed_starting:
            self._run_multichain_mcmc()
    
    def _log_prob(self, state_vectors):
        #exp(-beta * U), boltzman distribution assumed for posterior
        coeff_at_t0 = self.potential_model.get_coeff_grid()[:, 0]
        return -self.beta * self.potential.potential_value(state_vectors, coeff_at_t0)

    def _posterior_for_mcmc(self, bounds_low, bounds_high):
        #makes it look like pyro need it to look
        x = pyro.sample("x", pyro.distributions.Uniform(bounds_low, bounds_high).to_event(1))
        pyro.factor("logp", self._log_prob(x.unsqueeze(-1)))

    def _partition_bounds_by_midpoints(self):
        """Partition the sampling bounds using midpoints to create per-well regions"""

        if not hasattr(self.loss, 'midpoints'):
            return [self.mcmc_starting_spatial_bounds]
        
        midpoints = self.loss.midpoints
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

    def _run_multichain_mcmc(self):
        """Run MCMC sampling, either per-well or global depending on parameters"""
        if self.samples_per_well is not None:
            # Per-well sampling mode
            well_bounds = self._partition_bounds_by_midpoints()
            all_samples = []
            
            for well_idx, bounds in enumerate(well_bounds):
                well_samples = []
                samples_per_chain = self.samples_per_well // self.mcmc_chains_per_well
                
                for chain_id in range(self.mcmc_chains_per_well):
                    sampler = MCMC(
                        NUTS(lambda: self._posterior_for_mcmc(bounds[:, 0], bounds[:, 1])),
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
                    NUTS(lambda: self._posterior_for_mcmc(bounds_low, bounds_high)),
                    num_samples=samples_per_chain,
                    warmup_steps=int(self.mcmc_warmup_ratio * samples_per_chain)
                )
                sampler.run()
                all_samples.append(sampler.get_samples()['x'])
            
            self.starting_pos = torch.cat(all_samples, dim=0)

    def _get_initial_velocities(self):
        #Sample velocities from a Gaussian distribution with mean 0 and variance 1/(2*beta)
        var = 1 / (2 * self.beta)
        samples = torch.randn(self.mcmc_num_samples, device=self.device) * math.sqrt(var)
        return samples

    def _get_noise(self):
        samples = torch.randn(self.mcmc_num_samples, self.time_steps, device=self.device) * self.noise_sigma * torch.sqrt(torch.tensor(self.dt))
        return samples

    def simulate(self, initial_positions = None, initial_velocities = None, noise = None, DEBUG_PRINT = False):
        #simulates the system for either given initial conditions or sampled initial conditions
        if initial_positions is None:
            if self.potential_model.fixed_starting:
                initial_positions = self.starting_pos
            else:
                initial_positions = self._run_multichain_mcmc()
            if len(initial_positions.shape) == 1:
                initial_positions = initial_positions.unsqueeze(1)
        if initial_velocities is None:
            initial_velocities = self._get_initial_velocities()
            if len(initial_velocities.shape) == 1:
                initial_velocities = initial_velocities.unsqueeze(1)
        if noise is None:
            noise = self._get_noise()
        
        coeff_grid = self.potential_model.get_coeff_grid()
        return self.sim_engine.make_trajectories(
            self.potential,
            initial_positions,
            initial_velocities,
            self.time_steps,
            noise,
            self.noise_sigma,
            coeff_grid,
            self.device,
            DEBUG_PRINT = DEBUG_PRINT
        ), coeff_grid

    def _setup_optimizer(self, optimizer_class = torch.optim.Adam):
        self.optimizer = optimizer_class(self.potential_model.trainable_params(), lr=self.learning_rate)

    def train(self):
        self._setup_optimizer()

        for callback in self.callbacks:
            callback.on_train_start(self)

        with tqdm(range(self.epochs), desc='Training') as pbar:
            for epoch in pbar:

                for callback in self.callbacks:
                    callback.on_epoch_start(self, epoch)

                self.optimizer.zero_grad()
                sim_dict, coeff_grid = self.simulate()
                dmw_dcoeff, loss_values = self.loss.compute_FRR_gradient(
                    sim_dict['potential'], sim_dict['trajectories'], sim_dict['malliavian_weight']
                )

                dmw_da = torch.autograd.grad(
                    outputs = coeff_grid,
                    inputs = self.potential_model.trainable_params(),
                    grad_outputs = dmw_dcoeff,
                    create_graph = False,
                    allow_unused = True
                )

                for param, grad in zip(self.potential_model.trainable_params(), dmw_da):
                    if grad is not None:
                        if param.grad is None:
                            param.grad = grad
                        else:
                            if epoch % 10 == 0 or epoch == 0:
                                print(f"Direct Grad: {param.grad}")
                                print(f'Malliavin Grad: {grad}')
                                print(f'Knot Locations are : {self.potential_model.trainable_params()[0]}')
                            param.grad += grad 

                torch.nn.utils.clip_grad_norm_(self.potential_model.trainable_params(), self.grad_clip_max_norm)
                self.optimizer.step()
                mean_loss = loss_values.mean().item()
                pbar.set_postfix({'loss': mean_loss})
                
                for callback in self.callbacks:
                    callback.on_epoch_end(self, sim_dict, loss_values, epoch)
        
        for callback in self.callbacks:
            callback.on_train_end(self, sim_dict, coeff_grid, epoch)
        
        print('Training complete')
        