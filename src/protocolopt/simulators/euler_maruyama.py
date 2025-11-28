#defintion of what a timestep looks like
#uses the Euler-Maruyama Method to numerically advance
#either over or underdamped systems

import torch
import sys
import os
from typing import Tuple, Any, TYPE_CHECKING

from ..utils import robust_compile, logger
from ..core.simulator import Simulator
from ..core.types import StateSpace, MicrostatePaths, PotentialTensor, MalliavinWeight, ControlSignal

if TYPE_CHECKING:
    from ..core.potential import Potential

class EulerMaruyama(Simulator):
    """Simulator implementation using the Euler-Maruyama method."""

    def __init__(self, mode: str, gamma: float, mass: float = 1.0, dt: float = None, compile_mode: bool = True) -> None:
        """Initializes the EulerMaruyama simulator.

        Args:
            mode: Simulation mode, either 'underdamped' or 'overdamped'.
            gamma: Friction coefficient.
            mass: Particle mass (default 1.0).
            dt: Time step size. If None, calculated as 1/time_steps.
            compile_mode: Whether to compile the step functions.

        Raises:
            ValueError: If mode is not 'underdamped' or 'overdamped'.
        """
        if mode not in ['underdamped', 'overdamped']:
            raise ValueError(f"Invalid mode: {mode}, choose from 'underdamped' or 'overdamped'")
        self.mode = mode
        self.dt = dt
        self.mass = mass
        self.gamma = gamma
        self.compile_mode = compile_mode

        self._compiled_underdamped_step = robust_compile(self._underdamped_step, compile_mode=self.compile_mode)
        self._compiled_overdamped_step = robust_compile(self._overdamped_step, compile_mode=self.compile_mode)
    
    @staticmethod
    def _overdamped_step(current_pos, dv_dx, noise, dt, gamma):
        return current_pos - (dv_dx * dt - noise) / gamma

    @staticmethod
    def _underdamped_step(current_pos, current_vel, dv_dx, noise, dt, gamma, mass):
        next_pos = current_pos + current_vel * dt
        next_vel = (current_vel - gamma * current_vel * dt - dv_dx * dt + noise) / mass
        return next_pos, next_vel

    def _compute_malliavian_weight(self, dv_dxda, noise, noise_sigma):
        # dv_dxda: (samples, spatial, coeffs, time)
        # noise: (samples, spatial, time)
        # output: (samples, coeffs, time)
        drift_grad = -dv_dxda
        noise_expanded = noise[:,:,None,:] # (samples, spatial, expanded to effect all coeffs equally, time)
        dot_product = (drift_grad * noise_expanded).sum(dim = 1) # (samples, coeffs, time)
        return dot_product / (noise_sigma ** 2)
        
    def make_microstate_paths(
        self,
        potential: "Potential",
        initial_pos: StateSpace,
        initial_vel: StateSpace,
        time_steps: int,
        noise: torch.Tensor,
        noise_sigma: float,
        protocol_tensor: ControlSignal,
        debug_print: bool = False
    ) -> Tuple[MicrostatePaths, PotentialTensor, MalliavinWeight]:
        """Generates microstate paths based on the system dynamics.

        Args:
            potential: The potential energy landscape object.
            initial_pos: Starting positions.
                         Shape: (Batch, Spatial_Dim)
            initial_vel: Starting velocities.
                         Shape: (Batch, Spatial_Dim)
            time_steps: Number of integration steps to perform.
            noise: Noise tensor (sampled or given).
                   Shape: (Batch, Spatial_Dim, Time_Steps)
            noise_sigma: Standard deviation of the noise.
            protocol_tensor: Time-dependent control signals for the potential.
                             Shape: (Control_Dim, Time_Steps)
            debug_print: If True, prints statistics about gradients during execution.

        Returns:
            A tuple containing:
            - **microstate_paths**: Full path of particles.
                                    Shape: (Batch, Spatial_Dim, Time_Steps+1, 2)
                                    Dimension 3 is (position, velocity).
            - **potential_val**: Potential energy at each step.
                                 Shape: (Batch, Time_Steps)
            - **malliavian_weight**: Computed path weights.
                                     Shape: (Batch, Control_Dim, Time_Steps)

        Raises:
            ValueError: If an invalid simulation mode is selected.
        """
        #potential is a Potential object
        #initial_phase of shape (num traj, spatial dimensions, 2 for position and velocity), 3 dimensions in total
        #output of shape (initial_phase.shape[0] num traj, initial_phase.shape[1] spatial dimensions, time_steps+1, 2 for position and velocity), 4 dimensions in total
        #and potential of shape (initial_phase.shape[0] num traj, initial_phase.shape[1] spatial dimensions, time_steps+1)
        #and potential of shape (initial_phase.shape[0] num traj, time_steps+1) potential is scalar
        if self.dt is None:
            dt = 1 / time_steps
        else:
            dt = self.dt
        
        # Use lists to build trajectories dynamically, preserving the computation graph
        traj_pos_list = [initial_pos]
        traj_vel_list = [initial_vel]
        potential_list = []
        dv_dxda_list = []

        for i in range(time_steps):
            if self.mode == 'underdamped':
                # dx = v * dt
                # dv = (-gamma * v * dt - dV/dx * dt + noise) / mass
                current_pos = traj_pos_list[-1]
                current_vel = traj_vel_list[-1]
                
                dv_dx = potential.dv_dx(current_pos, protocol_tensor, i)
                U = potential.get_potential_value(current_pos, protocol_tensor, i)
                dv_dxda = potential.dv_dxda(current_pos, protocol_tensor, i)

                # Compute next positions and velocities
                next_pos, next_vel = self._compiled_underdamped_step(
                                    current_pos, current_vel, dv_dx, noise[..., i], dt, self.gamma, self.mass
                                )
                traj_pos_list.append(next_pos)
                traj_vel_list.append(next_vel)
                potential_list.append(U.squeeze())
                dv_dxda_list.append(dv_dxda)
            elif self.mode == 'overdamped':
                # dv = 0
                # dx = - (dV/dx * dt + noise) / gamma
                
                current_pos = traj_pos_list[-1]

                dv_dx = potential.dv_dx(current_pos, protocol_tensor, i)
                U = potential.get_potential_value(current_pos, protocol_tensor, i)
                dv_dxda = potential.dv_dxda(current_pos, protocol_tensor, i)
            
                next_pos = self._compiled_overdamped_step(current_pos, dv_dx, noise[..., i], dt, self.gamma)
                traj_pos_list.append(next_pos)
                traj_vel_list.append(torch.zeros_like(next_pos))
                potential_list.append(U.squeeze())
                dv_dxda_list.append(dv_dxda)
            else:
                raise ValueError(f"Please choose a valid mode, got {self.mode}, choose from 'underdamped' or 'overdamped'")

        traj_pos = torch.stack(traj_pos_list, dim=-1)  # (num_traj, spatial_dims, time_steps+1)
        traj_vel = torch.stack(traj_vel_list, dim=-1)  # (num_traj, spatial_dims, time_steps+1)
        potential_tensor = torch.stack(potential_list, dim=-1)  # (num_traj, time_steps)
        dv_dxda_tensor = torch.stack(dv_dxda_list, dim=-1)  # (num_traj, coeff_count, time_steps)

        if debug_print:
            logger.info(f"Malliavin weight stats - mean: {self._compute_malliavian_weight(dv_dxda_tensor, noise, noise_sigma).mean().item()}, std: {self._compute_malliavian_weight(dv_dxda_tensor, noise, noise_sigma, dt).std().item()}")
            logger.info(f"dv_dxda_tensor stats - mean: {dv_dxda_tensor.mean().item()}, std: {dv_dxda_tensor.std().item()}, has nan: {torch.isnan(dv_dxda_tensor).any()}")

        microstate_paths = torch.cat([traj_pos.unsqueeze(-1), traj_vel.unsqueeze(-1)], dim=-1)
        malliavian_weight = self._compute_malliavian_weight(dv_dxda_tensor, noise, noise_sigma)

        return microstate_paths, potential_tensor, malliavian_weight

    def debug_gradients(self, dv_dx, U, traj_pos_slice, traj_vel_slice, potential_tensor, dv_dxda_tensor):
        pass
