from abc import ABC, abstractmethod
from typing import Tuple, Any
import torch

class Simulator(ABC):
    """Abstract base class for simulation engines."""

    @abstractmethod
    def make_trajectories(
        self,
        potential: Any,
        initial_pos: torch.Tensor,
        initial_vel: torch.Tensor,
        time_steps: int,
        noise: torch.Tensor,
        noise_sigma: float,
        protocol_tensor: torch.Tensor,
        debug_print: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generates trajectories based on the system dynamics.

        Args:
            potential: The potential energy landscape object.
            initial_pos: Starting positions. Shape: (Num_Traj, Spatial_Dim).
            initial_vel: Starting velocities. Shape: (Num_Traj, Spatial_Dim).
            time_steps: Number of integration steps to perform.
            noise: Brownian noise tensor. Shape: (Num_Traj, Spatial_Dim, Time_Steps).
            noise_sigma: Standard deviation of the noise.
            protocol_tensor: Time-dependent coefficients for the potential. Shape: (Control_Dim, Time_Steps).
            debug_print: If True, prints statistics about gradients during execution.

        Returns:
            A tuple containing:
            - **trajectories**: Full path of particles. Shape (Num_Traj, Spatial_Dim, Time_Steps+1, 2).
            - **potential_val**: Potential energy at each step. Shape (Num_Traj, Time_Steps).
            - **malliavian_weight**: Computed path weights. Shape (Num_Traj, Control_Dim, Time_Steps).
        """
        pass
