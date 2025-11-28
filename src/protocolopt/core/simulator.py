from abc import ABC, abstractmethod
from typing import Tuple, TYPE_CHECKING
import torch
from .types import Trajectories, PotentialTensor, MalliavinWeight, StateSpace, ControlSignal

if TYPE_CHECKING:
    from .potential import Potential

class Simulator(ABC):
    """Abstract base class for simulation engines."""

    @abstractmethod
    def make_trajectories(
        self,
        potential: "Potential",
        initial_pos: StateSpace,
        initial_vel: StateSpace,
        time_steps: int,
        noise: torch.Tensor,
        noise_sigma: float,
        protocol_tensor: ControlSignal,
        debug_print: bool = False
    ) -> Tuple[Trajectories, PotentialTensor, MalliavinWeight]:
        """Generates trajectories based on the system dynamics.

        Args:
            potential: The potential energy landscape object.
            initial_pos: Starting positions.
                         Shape: (Batch, Spatial_Dim)
            initial_vel: Starting velocities.
                         Shape: (Batch, Spatial_Dim)
            time_steps: Number of integration steps to perform.
            noise: Brownian noise tensor.
                   Shape: (Batch, Spatial_Dim, Time_Steps)
            noise_sigma: Standard deviation of the noise.
            protocol_tensor: Time-dependent coefficients for the potential.
                             Shape: (Control_Dim, Time_Steps)
            debug_print: If True, prints statistics about gradients during execution.

        Returns:
            A tuple containing:
            - **trajectories**: Full path of particles.
                                Shape: (Batch, Spatial_Dim, Time_Steps+1, 2)
            - **potential_val**: Potential energy at each step.
                                 Shape: (Batch, Time_Steps)
            - **malliavian_weight**: Computed path weights.
                                     Shape: (Batch, Control_Dim, Time_Steps)
        """
        pass
