from ..core.potential import Potential
from typing import Tuple, Optional
import torch

class QuarticPotential(Potential):
    """Potential of form V(x, t) = a(t)x^4 - b(t)x^2."""

    def potential_value(self, space_grid: torch.Tensor, protocol_tensor: torch.Tensor) -> torch.Tensor:
        """Computes the quartic potential value.

        Args:
            space_grid: Spatial coordinates. Shape: (Batch, Spatial_Dim) or (Spatial_Dim,).
            protocol_tensor: Coefficients [a, b]. Shape: (2,) or (2, Spatial_Dim) depending on implementation.
                Current implementation assumes protocol_tensor is (Control_Dim,) and handles 1D.

        Returns:
            Potential value.
        """
        return torch.sum(protocol_tensor[0] * space_grid**4 - protocol_tensor[1] * space_grid**2, dim=-1)

class QuarticPotentialWithLinearTerm(Potential):
    """Potential of form V(x, t) = a(t)x^4 - b(t)x^2 + c(t)x."""

    def potential_value(self, space_grid: torch.Tensor, protocol_tensor: torch.Tensor) -> torch.Tensor:
        """Computes the quartic potential with linear term.

        Args:
            space_grid: Spatial coordinates.
            protocol_tensor: Coefficients [a, b, c].

        Returns:
            Potential value.
        """
        return torch.sum(protocol_tensor[0] * space_grid**4 - protocol_tensor[1] * space_grid**2 + protocol_tensor[2] * space_grid, dim=-1)
