from ..core.potential import Potential
from ..core.types import StateSpace, ControlVector
import torch

class FlatPotential(Potential):
    """Potential that is always zero (free diffusion)."""

    def __init__(self, compile_mode: bool = True):
        super().__init__(compile_mode)
        self.hparams = {
            'name': self.__class__.__name__,
            'compile_mode': self.compile_mode
        }

    def potential_value(self, space_grid: StateSpace, protocol_tensor: ControlVector) -> torch.Tensor:
        """Computes the flat potential value (always 0).

        Args:
            space_grid: Spatial coordinates.
                        Shape: (Batch, Spatial_Dim) or (Spatial_Dim,)
            protocol_tensor: Control vector (unused but kept for API consistency).
                             Shape: (Control_Dim,)

        Returns:
            Potential value (0).
        """
        target_shape = space_grid.shape[:-1]
        return torch.zeros(target_shape, dtype=space_grid.dtype, device=space_grid.device)
