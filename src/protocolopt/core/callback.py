from typing import Dict, Any, TYPE_CHECKING
import torch

if TYPE_CHECKING:
    from .simulation import Simulation

class Callback:
    """Base class for callbacks."""

    def on_train_start(self, simulation_object: "Simulation") -> None:
        """Called at the beginning of training.

        Args:
            simulation_object: The main Simulation instance.
        """
        pass

    def on_epoch_start(self, simulation_object: "Simulation", epoch: int) -> None:
        """Called at the start of each epoch.

        Args:
            simulation_object: The main Simulation instance.
            epoch: Current epoch index.
        """
        pass

    def on_epoch_end(self, simulation_object: "Simulation", sim_dict: Dict[str, Any], loss_values: torch.Tensor, epoch: int) -> None:
        """Called at the end of each epoch.

        Args:
            simulation_object: The main Simulation instance.
            sim_dict: Dictionary containing simulation results (trajectories, etc.).
            loss_values: Tensor of loss values for the batch.
            epoch: Current epoch index.
        """
        pass

    def on_train_end(self, simulation_object: "Simulation", sim_dict: Dict[str, Any], protocol_tensor: torch.Tensor, epoch: int) -> None:
        """Called at the end of training.

        Args:
            simulation_object: The main Simulation instance.
            sim_dict: Dictionary containing simulation results.
            protocol_tensor: Final protocol tensor.
            epoch: Final epoch index.
        """
        pass
