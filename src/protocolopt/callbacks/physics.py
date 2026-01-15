import torch
import matplotlib.pyplot as plt
from pathlib import Path
from ..core.callback import BasePlottingCallback
import numpy as np
from matplotlib.colors import ListedColormap
from typing import Optional, Dict, Any, TYPE_CHECKING
from ..core.types import MicrostatePaths, ControlSignal, PotentialTensor
import math
from ..potentials.flat import FlatPotential

plt.ioff()

class FreeDiffusionCheckCallback(BasePlottingCallback):
    def __init__(self, save_dir: str = 'figs', num_vel_relaxation_times = 10, num_traj = 1000, mode = 'underdamped', mean_spatial = True):
        if num_vel_relaxation_times < 10:
             print('It is recommended to use at least 10 velocity relaxation times for FreeDiffusionCheckCallback. Consider increasing this value for more reliable results.')
        if mode not in ['underdamped', 'overdamped']:
            raise ValueError(f"Invalid mode: {mode} in FreeDiffusionCheckCallback, choose from 'underdamped' or 'overdamped'")
        self.num_vel_relaxation_times = num_vel_relaxation_times
        self.save_dir = save_dir
        self.num_traj = num_traj
        self.mean_spatial = mean_spatial
    
    def on_train_start(self, optimizer_object):
        self.device = optimizer_object.device
        self.sim = optimizer_object.simulator
        self.protocol = optimizer_object.protocol
        
        required_attrs = ['gamma', 'beta', 'mass', 'dt']
        if all(hasattr(self.sim, attr) for attr in required_attrs):
            self.beta = self.sim.beta
            self.gamma = self.sim.gamma
            self.mass = self.sim.mass
            self.dt = self.sim.dt
        else:
            raise ValueError(f"Simulator must have {', '.join(required_attrs)} attributes for the FreeDiffusionCheckCallback")
            
        if hasattr(optimizer_object.initial_condition_generator, 'spatial_dimensions'):
            self.num_dim = optimizer_object.initial_condition_generator.spatial_dimensions
        else:
            raise ValueError("Initial condition generator must have spatial_dimensions attribute for the FreeDiffusionCheckCallback")
        
        relax_time = self.mass / self.gamma
        self.time_steps = int(relax_time * self.num_vel_relaxation_times / self.dt)
        
        self.initial_positions, self.initial_velocities, self.noise = self._simple_initial_cond_generator(
            self.num_traj, self.num_dim, self.beta, self.mass, self.dt, self.time_steps, self.gamma
        )
        
        self.flat_potential = FlatPotential(compile_mode=self.sim.compile_mode)
        
        free_diffusion_paths = self._simulate_free_diffusion()[...,0] # just take position for now, there are tests with velocity could do another time
        if not self.mean_spatial and self.num_dim > 4:
            print('When mean_spatial is False, 2 plots per spatial dimension are made in FreeDiffusionCheckCallback, you have ', self.num_dim, ' spatial dimensions')
        if self.mean_spatial:
            free_diffusion_paths = free_diffusion_paths.mean(dim=0).mean(dim=0)
            ensemble_average_mean_displacement = free_diffusion_paths.mean(dim=0) # (time)
            ensemble_average_mean_squared_displacement = (free_diffusion_paths**2).mean(dim=0) # (time)
        else:
            ensemble_average_mean_displacement = free_diffusion_paths.mean(dim=0) # (spatial, time)
            ensemble_average_mean_squared_displacement = (free_diffusion_paths**2).mean(dim=0) # (spatial, time)

        self._plot_drift(ensemble_average_mean_displacement, optimizer_object)
        self._plot_diffusion(ensemble_average_mean_squared_displacement, optimizer_object)

        
    def _simulate_free_diffusion(self):
        dummy_protocol = torch.zeros((1, self.time_steps), device=self.device)
        
        microstate_paths, _, _ = self.sim.make_microstate_paths(
            potential=self.flat_potential,
            initial_pos=self.initial_positions,
            initial_vel=self.initial_velocities,
            time_steps=self.time_steps,
            noise=self.noise,
            protocol_tensor=dummy_protocol
        )
        return microstate_paths.detach().cpu()

    def _simple_initial_cond_generator(self, num_traj, num_dim, beta, mass, dt, time_steps, gamma):
        var = 1 / (beta * mass)
        initial_vel = torch.randn(num_traj, num_dim, device=self.device) * math.sqrt(var)
        initial_pos = torch.zeros(num_traj, num_dim, device=self.device)
        noise_sigma = math.sqrt(2 * gamma / beta)
        noise = torch.randn(num_traj, num_dim, time_steps, device=self.device) * noise_sigma * math.sqrt(dt)
        
        return initial_pos, initial_vel, noise

    def _plot_drift(self, mean_displacement, optimizer_object):
        # mean_displacement shape: (time) or (spatial, time)
        times = np.arange(self.time_steps) * self.dt
        velocity_relaxation_times = times / (self.mass / self.gamma)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if mean_displacement.ndim == 1:
            ax.plot(velocity_relaxation_times, mean_displacement, label='Mean Displacement')
        else:
            for i in range(mean_displacement.shape[0]):
                ax.plot(velocity_relaxation_times, mean_displacement[i], label=f'Dim {i}')
                
        ax.set_xlabel('Time (Relaxation Times)')
        ax.set_ylabel('Displacement')
        ax.set_title('Drift Check (Should be 0)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        self._log_or_save_figure(fig, 'free_diffusion_drift_check', 0, optimizer_object)
        plt.close(fig)

    def _plot_diffusion(self, mean_squared_displacement, optimizer_object):
        # mean_squared_displacement shape: (time) or (spatial, time)
        times = np.arange(self.time_steps) * self.dt
        
        # Theoretical MSD
        # D_eff depends on dimensions being summed over
        # if mean_spatial is True, we summed over all spatial dims, so we multiply by num_dim
        # if mean_spatial is False, we are looking at per-dimension, so effectively 1 dim
        
        effective_dim = self.num_dim if self.mean_spatial else 1
        
        D = 1 / (self.beta * self.gamma) # Diffusion coefficient kBT/gamma
        tau = self.mass / self.gamma # Relaxation time m/gamma
        
        # Langevin MSD formula: 2 * num_dims * D * (t - tau * (1 - exp(-t/tau)))
        theoretical_msd = 2 * effective_dim * D * (times - tau * (1 - np.exp(-times/tau)))
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        velocity_relaxation_times = times / tau
        
        if mean_squared_displacement.ndim == 1:
             ratio = mean_squared_displacement / (theoretical_msd + 1e-8) # avoid div by zero at t=0
             ax.plot(velocity_relaxation_times, ratio, label='MSD / Theory')
        else:
             for i in range(mean_squared_displacement.shape[0]):
                 ratio = mean_squared_displacement[i] / (theoretical_msd + 1e-8)
                 ax.plot(velocity_relaxation_times, ratio, label=f'Dim {i}')
        
        ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Theory')
        ax.set_xlabel('Time (Relaxation Times)')
        ax.set_ylabel('Ratio (Simulated / Theoretical MSD)')
        ax.set_title('Diffusion Check (Should be 1)')
        ax.legend()
        ax.set_ylim(0.8, 1.2) # Zoom in to see deviations near 1
        ax.grid(True, alpha=0.3)
        
        self._log_or_save_figure(fig, 'free_diffusion_diffusion_check', 0, optimizer_object)
        plt.close(fig)