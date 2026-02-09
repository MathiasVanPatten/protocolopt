import torch
import sys
import os
from typing import Tuple, Any, Dict, TYPE_CHECKING

from ..utils import robust_compile, logger
from ..core.simulator import Simulator
from ..core.types import StateSpace, MicrostatePaths, PotentialTensor, MalliavinWeight, ControlSignal

if TYPE_CHECKING:
    from ..core.potential import Potential

class GJF(Simulator):
    """Simulator using the 2013 Grønbech-Jensen and Farago Verlet Langevin Integrator. As defined in eqn 50 of https://doi.org/10.1007/s10955-025-03553-3 with help from 10.1080/00268976.2012.760055 since the review paper had a typo in the next_pos equation."""
    def __init__(self, mode: str, gamma: float, beta: float = 1.0, mass: float = 1.0, dt: float = None, compile_mode: bool = True):
        """Initializes the GJF simulator.

        Args:
            mode: Simulation mode, must be underdamped, arg left in for dropin compatibility 
            gamma: Friction coefficient.
            beta: 1/kT
            mass: Particle mass (default 1.0).
            dt: Time step size. If None, calculated as 1/time_steps.
            compile_mode: Whether to compile the step functions.

        Raises:
            ValueError: If mode is not 'underdamped'
        """
        if mode != "underdamped":
            raise ValueError("GJF only supports underdamped mode as it is a second order integrator.")
        
        self.mode = mode
        self.dt = dt
        self.mass = mass
        self.gamma = gamma
        self.beta = beta
        self.compile_mode = compile_mode
        self.noise_sigma = torch.sqrt(torch.tensor(2 * self.gamma / self.beta))
        self._compiled_next_pos = robust_compile(self._next_pos, compile_mode=self.compile_mode)
        self._compiled_next_vel = robust_compile(self._next_vel, compile_mode=self.compile_mode)
        self.hparams = {
            'mode': self.mode,
            'gamma': self.gamma,
            'beta': self.beta,
            'mass': self.mass,
            'noise_sigma': self.noise_sigma,
            'dt': self.dt,
            'compile_mode': self.compile_mode,
            'name': self.__class__.__name__
        }
        
    @staticmethod
    def _next_pos(pos, vel, force, noise, dt, mass, c1):
        return pos + c1 * (dt * vel + (dt**2 / (2 * mass)) * force + (dt / (2 * mass)) * noise) #testing if dt in num of last term is correct

    @staticmethod
    def _next_vel(vel, force, force_next, noise, dt, mass, c1, c2):
        return c2 * vel + (dt / (2 * mass)) * (c2 * force + force_next) + (c1 / mass) * noise

    def make_microstate_paths(
        self,
        potential: Any,
        initial_pos: torch.Tensor,
        initial_vel: torch.Tensor,
        time_steps: int,
        noise: torch.Tensor,
        protocol_tensor: torch.Tensor
    ) -> Tuple[MicrostatePaths, PotentialTensor, MalliavinWeight]:
        """Generates microstate paths based on the system dynamics using the GJF integrator.

        Args:
            potential: The potential energy landscape object.
            initial_pos: Starting positions. Shape: (Num_Traj, Spatial_Dim).
            initial_vel: Starting velocities. Shape: (Num_Traj, Spatial_Dim).
            time_steps: Number of integration steps to perform.
            noise: Noise tensor (sampled or given).
                   Shape: (Batch, Spatial_Dim, Time_Steps)
            protocol_tensor: Time-dependent control signals for the potential.
                             Shape: (Control_Dim, Time_Steps+1)

        Returns:
            A tuple containing:
            - **microstate_paths**: Full path of particles.
                                    Shape: (Batch, Spatial_Dim, Time_Steps+1, 2)
                                    Dimension 3 is (position, velocity).
            - **potential_val**: Potential energy at each step.
                                 Shape: (Batch, Time_Steps)
            - **malliavin_weight**: Computed path weights.
                                     Shape: (Batch, Control_Dim, Time_Steps)
            - **dw**: Change in potential energy at each step.
                      Shape: (Batch, Time_Steps)

        Raises:
            ValueError: If an invalid simulation mode is selected, only underdamped is supported.
        """
        if self.dt is None:
            dt = 1 / time_steps
        else:
            dt = self.dt
        c2 = (1 - 0.5 * self.gamma * dt / self.mass) / (1 + 0.5 * self.gamma * dt / self.mass) # Eqn 8 for mass div, 50c for the greater c2 form

        c1 = (1 + c2) / 2 #Eq. 9
        # Use lists to build trajectories dynamically, preserving the computation graph
        traj_pos_list = [initial_pos]
        traj_vel_list = [initial_vel]
        potential_list = []
        dv_dxda_list = []
        dw_list = []
        for i in range(time_steps):
                current_pos = traj_pos_list[-1]
                current_vel = traj_vel_list[-1]
                
                dv_dx = potential.dv_dx(current_pos, protocol_tensor, i)
                U = potential.get_potential_value(current_pos, protocol_tensor, i)
                dv_dxda = potential.dv_dxda(current_pos, protocol_tensor, i)

                next_pos = self._compiled_next_pos(current_pos, current_vel, -dv_dx, noise[..., i], dt, self.mass, c1)
                next_dv_dx = potential.dv_dx(next_pos, protocol_tensor, i+1) #dv_dx that will be in the future in the next pos. This is ambiguous in the review paper, but is clear in the original GJF paper 10.1080/00268976.2012.760055

                next_vel = self._compiled_next_vel(current_vel, -dv_dx, -next_dv_dx, noise[..., i], dt, self.mass, c1, c2)

                U_next = potential.get_potential_value(current_pos, protocol_tensor, i + 1)
                dw_list.append(U_next - U)

                traj_pos_list.append(next_pos)
                traj_vel_list.append(next_vel)
                potential_list.append(U.squeeze())
                dv_dxda_list.append(dv_dxda)


        traj_pos = torch.stack(traj_pos_list, dim=-1)  # (num_traj, spatial_dims, time_steps+1)
        traj_vel = torch.stack(traj_vel_list, dim=-1)  # (num_traj, spatial_dims, time_steps+1)
        potential_tensor = torch.stack(potential_list, dim=-1)  # (num_traj, time_steps)
        dv_dxda_tensor = torch.stack(dv_dxda_list, dim=-1)  # (num_traj, coeff_count, time_steps)
        dw_tensor = torch.stack(dw_list, dim=-1)  # (num_traj, time_steps)

        microstate_paths = torch.cat([traj_pos.unsqueeze(-1), traj_vel.unsqueeze(-1)], dim=-1)
        malliavin_weight = self._compute_malliavin_weight(dv_dxda_tensor, noise, self.noise_sigma)

        return microstate_paths, potential_tensor, malliavin_weight, dw_tensor


