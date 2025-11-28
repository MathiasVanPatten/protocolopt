from ..core.protocol import Protocol
from ..core.types import ControlSignal
import torch
import torch.nn.functional as F
from typing import Optional, List

class LinearPiecewise(Protocol):
    """Protocol parameterized by linear interpolation between knots."""

    def __init__(self, control_dim: int, time_steps: int, knot_count: int, initial_coeff_guess: torch.Tensor, endpoints: Optional[torch.Tensor] = None) -> None:
        """Initializes the LinearPiecewise protocol.

        Args:
            control_dim: Number of coefficients to model.
            time_steps: Number of time steps for interpolation.
            knot_count: Total number of knots (including endpoints).
            initial_coeff_guess: Initial guess for the trainable knots.
                                 Shape: (Control_Dim, Knot_Count) or (Control_Dim, Knot_Count - 2)
            endpoints: Fixed values for the start and end knots.
                       Shape: (Control_Dim, 2).

        Raises:
            ValueError: If initial guess shape is inconsistent with knot count.
        """
        if endpoints is not None:
            fixed_starting = True
        else:
            fixed_starting = False
        super().__init__(time_steps, fixed_starting)
        if initial_coeff_guess.shape != (control_dim, knot_count if endpoints is None else knot_count - 2):
            raise ValueError(f"Initial coefficient guess must be of shape (control_dim, knot_count if endpoints is None else knot_count - 2), got {initial_coeff_guess.shape}")
        
        self.control_dim = control_dim
        self.knot_count = knot_count
        self.endpoints = endpoints
        if self.endpoints is not None:
            self._trainable_params = torch.nn.Parameter(initial_coeff_guess.clone())
        else:
            self._trainable_params = torch.nn.Parameter(initial_coeff_guess.clone())

        self.device = initial_coeff_guess.device

    def _get_knots(self) -> torch.Tensor:
        if self.endpoints is not None:
            return torch.cat([self.endpoints[..., 0:1], self._trainable_params, self.endpoints[..., -1:]], dim=-1)
        else:
            return self.trainable_params
        
    def get_protocol_tensor(self) -> ControlSignal:
        """Interpolates knots to get the full coefficient grid.

        Returns:
            Coefficient grid. Shape: (Control_Dim, Time_Steps).
        """
        knots = self._get_knots()
        protocol_tensor = F.interpolate(knots.unsqueeze(0), size=self.time_steps, mode='linear', align_corners=True)
        return protocol_tensor.squeeze(0)
    
    def trainable_params(self) -> List[torch.nn.Parameter]:
        """Returns the trainable knots."""
        return [self._trainable_params]
