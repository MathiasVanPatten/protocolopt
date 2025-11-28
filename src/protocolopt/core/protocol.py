import torch
from abc import ABC, abstractmethod
from typing import List

class Protocol(ABC):
    """Abstract base class for time-dependent protocols (coefficients)."""

    def __init__(self, time_steps: int, fixed_starting: bool) -> None:
        """Initializes the Protocol.

        Args:
            time_steps: The number of time steps in the protocol.
            fixed_starting: If True, the starting configuration is fixed and not trainable/random.
        """
        self.time_steps = time_steps
        self.fixed_starting = fixed_starting

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
