from ..core.potential import Potential
from typing import Tuple, Optional
import torch

class QuarticPotential(Potential):
    """Potential of form V(x, t) = a(t)x^4 - b(t)x^2."""

    def potential_value(self, space_grid: torch.Tensor, coeff_grid: torch.Tensor) -> torch.Tensor:
        """Computes the quartic potential value.

        Args:
            space_grid: Spatial coordinates. Shape: (Batch, Spatial_Dim) or (Spatial_Dim,).
            coeff_grid: Coefficients [a, b]. Shape: (2,) or (2, Spatial_Dim) depending on implementation.
                Current implementation assumes coeff_grid is (Num_Coeffs,) and handles 1D.

        Returns:
            Potential value.
        """
        return torch.sum(coeff_grid[0] * space_grid**4 - coeff_grid[1] * space_grid**2, dim=-1)

class QuarticPotentialWithLinearTerm(Potential):
    """Potential of form V(x, t) = a(t)x^4 - b(t)x^2 + c(t)x."""

    def potential_value(self, space_grid: torch.Tensor, coeff_grid: torch.Tensor) -> torch.Tensor:
        """Computes the quartic potential with linear term.

        Args:
            space_grid: Spatial coordinates.
            coeff_grid: Coefficients [a, b, c].

        Returns:
            Potential value.
        """
        return torch.sum(coeff_grid[0] * space_grid**4 - coeff_grid[1] * space_grid**2 + coeff_grid[2] * space_grid, dim=-1)
