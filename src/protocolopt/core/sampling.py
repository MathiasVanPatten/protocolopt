from abc import ABC, abstractmethod
import torch
from typing import Tuple, TYPE_CHECKING
from .types import StateSpace, Trajectories

if TYPE_CHECKING:
    from .potential import Potential
    from .protocol import Protocol
    from .loss import Loss

class InitialConditionGenerator(ABC):
    @abstractmethod
    def generate_initial_conditions(
        self,
        potential: "Potential",
        protocol: "Protocol",
        loss: "Loss"
    ) -> Tuple[StateSpace, StateSpace, torch.Tensor]:
        """Generates initial conditions for the simulation.

        Args:
            potential: The potential energy landscape object.
            protocol: The protocol object.
            loss: The loss object (often used to determine bounds/regions).

        Returns:
            Tuple containing:
            - **initial_pos**: Starting positions. Shape: (Batch, Spatial_Dim)
            - **initial_vel**: Starting velocities. Shape: (Batch, Spatial_Dim)
            - **noise**: Brownian noise. Shape: (Batch, Spatial_Dim, Time_Steps)
        """
        pass
