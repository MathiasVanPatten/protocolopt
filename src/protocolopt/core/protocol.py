import torch
from abc import ABC, abstractmethod
from typing import List

class Protocol(torch.nn.Module, ABC):
    """Abstract base class for time-dependent protocols (coefficients)."""

    def __init__(self, time_steps: int, fixed_starting: bool) -> None:
        """Initializes the Protocol.

        Args:
            time_steps: The number of time steps in the protocol.
            fixed_starting: If True, the starting configuration is fixed and not trainable/random.
        """
        super().__init__()
        self.time_steps = time_steps
        self.fixed_starting = fixed_starting

    def save(self, path: str) -> None:
        """Saves the protocol parameters to a file.

        Args:
            path: The file path to save the parameters to.
        """
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        """Loads the protocol parameters from a file.

        Args:
            path: The file path to load the parameters from.
        """
        self.load_state_dict(torch.load(path))

    @abstractmethod
    def get_protocol_tensor(self) -> torch.Tensor:
        """Returns the full grid of coefficients over time.

        Returns:
            A tensor of coefficients. Shape: (Control_Dim, Time_Steps).
        """
        pass

    @abstractmethod
    def trainable_params(self) -> List[torch.nn.Parameter]:
        """Returns the list of trainable parameters for the optimizer.

        Returns:
            A list of torch.nn.Parameter objects.
        """
        pass
