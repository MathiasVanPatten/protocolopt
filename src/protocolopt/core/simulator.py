from abc import ABC, abstractmethod
from typing import Tuple, Any, TYPE_CHECKING
import torch
from .types import MicrostatePaths, PotentialTensor, MalliavinWeight, StateSpace, ControlSignal, WorkTensor

if TYPE_CHECKING:
    from .potential import Potential

class Simulator(ABC):
    """Abstract base class for simulation engines."""

    @abstractmethod
    def make_microstate_paths(
        self,
        potential: Any,
        initial_pos: torch.Tensor,
        initial_vel: torch.Tensor,
        time_steps: int,
        noise: torch.Tensor,
        protocol_tensor: torch.Tensor,
    ) -> Tuple[MicrostatePaths, PotentialTensor, MalliavinWeight, WorkTensor]:
        """Generates microstate paths based on the system dynamics.

        Args:
            potential: The potential energy landscape object.
            initial_pos: Starting positions. Shape: (Num_Traj, Spatial_Dim).
            initial_vel: Starting velocities. Shape: (Num_Traj, Spatial_Dim).
            time_steps: Number of integration steps to perform.
            noise: Noise tensor (sampled or given).
                   Shape: (Batch, Spatial_Dim, Time_Steps)
            protocol_tensor: Time-dependent control signals for the potential.
                             Shape: (Control_Dim, Time_Steps)

        Returns:
            A tuple containing:
            - **microstate_paths**: Full path of particles.
                                    Shape: (Batch, Spatial_Dim, Time_Steps+1, 2)
                                    Dimension 3 is (position, velocity).
            - **potential_val**: Potential energy at each step.
                                 Shape: (Batch, Time_Steps)
            - **malliavin_weight**: Computed path weights.
                                    Shape: (Batch, Control_Dim, Time_Steps)
            - **dw_tensor**: Change in potential energy at each step.
                      Shape: (Batch, Time_Steps)
        """
        pass
