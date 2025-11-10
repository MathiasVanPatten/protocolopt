#definition of the free paramters, defintion of the static setup
import torch
import pyro
from pyro.infer.mcmc import MCMC, NUTS
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.ioff()
from pathlib import Path
#TODO In theory now the ability to form a full trajectory tensor with all of the potential, protocol, and sim engine
# chain should work, in the next session we need to tie them together so they pass what's needed between
# and set up the actual epoch training loop in simulation which probably will be renamed to allow for 
# a full training loop

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

        self.mcmc_num_samples = params.get('mcmc_num_samples', 5000)
        self.mcmc_warmup_ratio = params.get('mcmc_warmup_ratio', 0.1)
        self.mcmc_starting_spatial_bounds = params.get('mcmc_starting_spatial_bounds', torch.tensor([[-5, 5]] * self.spatial_dimensions, device = self.device))
        
        self.beta = params.get('beta', 1.0)
        self.gamma = self.sim_engine.gamma
        self.dt = self.sim_engine.dt
        #HACK WHEN NOISE IS GIVEN TO SIMULATE THE NOISE SIGMA WILL STILL BE SET AS IT WAS DURING TRAINING, POSSIBLY AN ISSUE
        #HACK MU IS FIXED AT 1 FOR NOW
        self.noise_sigma = torch.sqrt(torch.tensor(2 * self.gamma) / self.beta) #Einstein relation for overdamped systems TODO will need to change when underdamped is added
        self.made_initial_plot = False #HACK
        self.epochs = params.get('epochs', 100)
        self.learning_rate = params.get('learning_rate', 0.001)
        if self.potential_model.fixed_starting:
            self._run_multichain_mcmc()
    
    def _log_prob(self, state_vectors):
        #exp(-beta * U), boltzman distribution assumed for posterior
        coeff_at_t0 = self.potential_model.get_coeff_grid()[:, 0]
        return -self.beta * self.potential.potential_value(state_vectors, coeff_at_t0)

    def _posterior_for_mcmc(self):
        #makes it look like pyro need it to look
        low = self.mcmc_starting_spatial_bounds[:, 0]
        high = self.mcmc_starting_spatial_bounds[:, 1]
        x = pyro.sample("x", pyro.distributions.Uniform(low, high).to_event(1))
        pyro.factor("logp", self._log_prob(x.unsqueeze(-1)))

    def _run_multichain_mcmc(self):
        all_samples = []
        num_chains = 4
        samples_per_chain = self.mcmc_num_samples // num_chains
        
        for chain_id in range(num_chains):
            sampler = MCMC(
                NUTS(self._posterior_for_mcmc),
                num_samples=samples_per_chain,
                warmup_steps=int(self.mcmc_warmup_ratio * samples_per_chain)
            )
            sampler.run()
            all_samples.append(sampler.get_samples()['x'])
        
        self.starting_pos = torch.cat(all_samples, dim=0)

    # def _get_initial_positions(self):
    #     #output of shape (num_samples, spatial_dimensions)
    #     #uses mcmc sampler to sample given potential in uniform region specified
    #     if not self.potential_model.fixed_starting:
    #         self._run_multichain_mcmc()
    #     else:
    #         return self.mcmc_sampler.get_samples()['x']

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
        if not self.made_initial_plot:
            self.made_initial_plot = True
            fig_save_path = Path('/mnt/b/wsl_code/[JIM]-2025_09-Optimal_Control/protocolopt/mathias_code/figs')
            # plt.figure()
            plt.scatter(initial_positions.squeeze().cpu().detach(), initial_velocities.squeeze().cpu().detach())
            plt.xlabel('Position')
            plt.ylabel('Velocity')
            plt.title('Initial Conditions Phase Space')
            plt.savefig(fig_save_path / f'initial_conditions_phase_space.png')

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

    def _plot_diagnostics(self, sim_dict, coeff_grid, epoch):
        """Plot trajectory positions and potential landscape for debugging"""
        fig_save_path = Path('/mnt/b/wsl_code/[JIM]-2025_09-Optimal_Control/protocolopt/mathias_code/figs')
        
        # Plot trajectory positions
        plt.figure()
        # Choose 100 random trajectory indices (or less if there are fewer than 100 trajectories)
        num_trajectories = sim_dict['trajectories'].shape[0]
        sample_size = min(100, num_trajectories)
        random_indices = torch.randperm(num_trajectories)[:sample_size]
        trajectory_positions = sim_dict['trajectories'][random_indices, :, :, 0].squeeze().cpu().detach()
        plt.plot(torch.linspace(0, self.time_steps, self.time_steps + 1), trajectory_positions.T)
        plt.xlabel('Time')
        plt.ylabel('Position')
        plt.title(f'Trajectory Positions Over Time (Epoch {epoch})')
        plt.savefig(fig_save_path / f'trajectory_positions_epoch_{epoch:04d}.png')
        plt.close()
        
        # Plot potential shape in space and time (3D contour)
        # Create spatial grid for visualization
        x_min, x_max = -2.5,2.5
        num_x_points = 100
        num_t_points = min(100, coeff_grid.shape[1])  # Subsample time for efficiency
        
        x_grid = torch.linspace(x_min, x_max, num_x_points, device=self.device)
        t_indices = torch.linspace(0, coeff_grid.shape[1] - 1, num_t_points, dtype=torch.long, device=self.device)
        
        # Compute potential values across space and time
        potential_values = torch.zeros(num_t_points, num_x_points, device=self.device)
        for i, t_idx in enumerate(t_indices):
            # Reshape x_grid to match expected input shape (num_samples, spatial_dims)
            x_grid_reshaped = x_grid.unsqueeze(1)
            coeff_at_t = coeff_grid[:, t_idx]
            potential_values[i, :] = self.potential.potential_value(x_grid_reshaped, coeff_at_t).squeeze()
        
        # Create meshgrid for plotting
        X, T = torch.meshgrid(x_grid.cpu(), t_indices.cpu().float(), indexing='xy')
        Z = potential_values.cpu().detach()
        
        # Create 3D surface plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X.numpy(), T.numpy(), Z.numpy(), cmap='viridis', alpha=0.8)
        ax.set_xlabel('Position')
        ax.set_ylabel('Time Step')
        ax.set_zlabel('Potential V(x, t)')
        ax.set_title(f'Potential Energy Landscape Evolution (Epoch {epoch})')
        plt.colorbar(surf, ax=ax, shrink=0.5)
        plt.savefig(fig_save_path / f'potential_landscape_3d_epoch_{epoch:04d}.png', dpi=150)
        plt.close()
        
        # Also create a 2D contour plot for easier visualization
        plt.figure(figsize=(10, 6))
        contour = plt.contourf(X.numpy(), T.numpy(), Z.numpy(), levels=20, cmap='viridis')
        plt.colorbar(contour, label='Potential V(x, t)')
        plt.xlabel('Position')
        plt.ylabel('Time Step')
        plt.title(f'Potential Energy Landscape Evolution (Epoch {epoch})')
        plt.savefig(fig_save_path / f'potential_landscape_2d_epoch_{epoch:04d}.png', dpi=150)
        plt.close()

    def train(self):
        self._setup_optimizer()
        with tqdm(range(self.epochs), desc='Training') as pbar:
            for epoch in pbar:
                for callback in self.callbacks:
                    callback.on_epoch_start(epoch)
                self.optimizer.zero_grad()
                sim_dict, coeff_grid = self.simulate()
                dmw_dcoeff, loss_values = self.loss.compute_FRR_gradient(
                    sim_dict['potential'], sim_dict['trajectories'], sim_dict['malliavian_weight']
                )

                dmw_da = torch.autograd.grad( #TODO might not be needed due to no detach on L32 potential
                    outputs = coeff_grid, #dummy to connect coeff grid to the upstream trainable params
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

                self.optimizer.step()
                mean_loss = loss_values.mean().item()
                pbar.set_postfix({'loss': mean_loss})
                
                # Plot diagnostics every 25 epochs
                if epoch % (self.epochs // 4) == 0 or epoch == self.epochs - 1:
                    self._plot_diagnostics(sim_dict, coeff_grid, epoch)
        
        print('Training complete')
        