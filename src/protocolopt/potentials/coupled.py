from ..core.potential import Potential
from ..core.types import StateSpace, ControlVector
from typing import Tuple, Optional, Dict, Any
import torch

class GeneralCoupledPotential(Potential):
    """A generalized coupled potential including quartic, quadratic, linear, and interaction terms.

    The potential is given by:
    V(x) = sum(a_i * x_i^4 - b_i * x_i^2 + c_i * x_i) + sum(mix_ij * x_i * x_j)
    """

    def __init__(self, spatial_dimensions: int, has_c: bool = True, has_mix: bool = True, compile_mode: bool = True) -> None:
        """Initializes the GeneralCoupledPotential.

        Args:
            spatial_dimensions: Number of spatial dimensions.
            has_c: Whether to include linear terms (c_i * x_i).
            has_mix: Whether to include interaction terms (mix_ij * x_i * x_j).
            compile_mode: Whether to compile gradient functions.
        """
        super().__init__(compile_mode)
        self.spatial_dim = spatial_dimensions
        self.has_c = has_c
        self.has_mix = has_mix
        if self.spatial_dim == 1:
            self.has_mix = False

        self.triu_indices = torch.triu_indices(row=spatial_dimensions, col=spatial_dimensions, offset=1)

        self.hparams = {
            'spatial_dim': self.spatial_dim,
            'has_c': self.has_c,
            'has_mix': self.has_mix,
            'compile_mode': self.compile_mode,
            'name': self.__class__.__name__
        }

    def potential_value(self, space_grid: StateSpace, protocol_tensor: ControlVector) -> torch.Tensor:
        """Computes the coupled potential value.

        Args:
            space_grid: Spatial coordinates.
                        Shape: (Batch, N) or (N,)
            protocol_tensor: Control vector. Layout: [N Quartics, N Quadratics, N Linears (optional), K Interactions (optional)].
                             Shape: (Total_Coeffs,)

        Returns:
            Potential value. Shape: (Batch,) or scalar.
        """
        # space_grid: (Batch, N) or (N,)
        # protocol_tensor: (Total_Coeffs)
        is_unbatched = space_grid.ndim == 1
        if is_unbatched:
            space_grid = space_grid.unsqueeze(0)
        # assume layout: [N Quartics, N Quadratics, N Linears, K Interactions]
        N = self.spatial_dim

        current_idx = 0

        # Quartic terms (a)
        a = protocol_tensor[current_idx : current_idx + N]
        current_idx += N

        # Quadratic terms (b)
        b = protocol_tensor[current_idx : current_idx + N]
        current_idx += N

        # Linear terms (c)
        if self.has_c:
            c = protocol_tensor[current_idx : current_idx + N]
            current_idx += N
        else:
            c = torch.zeros_like(a)

        # Interaction terms
        if self.has_mix:
            mix = protocol_tensor[current_idx : ]
        else:
            mix = None

        V_independent = torch.sum(a * space_grid**4 - b * space_grid**2 + c * space_grid, dim=-1)

        if N > 1 and self.has_mix:
            # grabs every unique pair for interaction
            vals_i = space_grid[:, self.triu_indices[0]]
            vals_j = space_grid[:, self.triu_indices[1]]

            V_interaction = torch.sum(mix * vals_i * vals_j, dim=-1)
        else:
            V_interaction = 0.0 #backwards compatibility
        result = V_independent + V_interaction

        if is_unbatched:
            return result.squeeze(0)
        return result
