from typing import Dict, Any, TYPE_CHECKING
import torch

if TYPE_CHECKING:
    from .protocol_optimizer import ProtocolOptimizer

class Callback:
    """Base class for callbacks."""

    def on_train_start(self, optimizer_object: "ProtocolOptimizer") -> None:
        """Called at the beginning of training.

        Args:
            optimizer_object: The main ProtocolOptimizer instance.
        """
        pass

    def on_epoch_start(self, optimizer_object: "ProtocolOptimizer", epoch: int) -> None:
        """Called at the start of each epoch.

        Args:
            optimizer_object: The main ProtocolOptimizer instance.
            epoch: Current epoch index.
        """
        pass

    def on_epoch_end(self, optimizer_object: "ProtocolOptimizer", sim_dict: Dict[str, Any], loss_values: torch.Tensor, epoch: int) -> None:
        """Called at the end of each epoch.

        Args:
            optimizer_object: The main ProtocolOptimizer instance.
            sim_dict: Dictionary containing simulation results (microstate_paths, etc.).
            loss_values: Tensor of loss values for the batch.
            epoch: Current epoch index.
        """
        pass

    def on_train_end(self, optimizer_object: "ProtocolOptimizer", sim_dict: Dict[str, Any], protocol_tensor: torch.Tensor, epoch: int) -> None:
        """Called at the end of training.

        Args:
            optimizer_object: The main ProtocolOptimizer instance.
            sim_dict: Dictionary containing simulation results.
            protocol_tensor: Final protocol tensor.
            epoch: Final epoch index.
        """
        pass
