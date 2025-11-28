from typing import Any, Tuple, Optional
import torch
import sys
from torch.func import vmap, grad, jacrev
from abc import ABC, abstractmethod
from ..utils import robust_compile

class Potential(ABC):
    """Abstract base class for potential energy landscapes."""

    def __init__(self, compile_mode: bool = True):
        """Initializes the Potential.

        Args:
            compile_mode: Whether to try compiling the gradient functions.
        """
        self.compile_mode = compile_mode

    @abstractmethod
    def potential_value(self, space_grid: torch.Tensor, protocol_tensor: torch.Tensor) -> torch.Tensor:
        """Computes the potential energy value.

        Args:
            space_grid: The spatial coordinates. Shape: (Batch, Spatial_Dim) or (Spatial_Dim,).
            protocol_tensor: The coefficients for the potential. Shape: (Control_Dim,).

        Returns:
            The potential energy at each point. Shape: (Batch,) or scalar.
        """
        pass

    def _get_kernels(self):
        if not hasattr(self, '_dv_dx_batched') or not hasattr(self, '_dv_dxda_batched'):
            single_sample_dv_dx = grad(self.potential_value, argnums=0)
            batched_dv_dx = vmap(single_sample_dv_dx, in_dims=(0, None))

            single_sample_dv_dxda = jacrev(single_sample_dv_dx, argnums=1)
            batched_dv_dxda = vmap(single_sample_dv_dxda, in_dims=(0, None))

            self._dv_dx_batched = robust_compile(batched_dv_dx, compile_mode=self.compile_mode)
            self._dv_dxda_batched = robust_compile(batched_dv_dxda, compile_mode=self.compile_mode)


        return self._dv_dx_batched, self._dv_dxda_batched

    def get_potential_value(self, space_grid: torch.Tensor, protocol_tensor: torch.Tensor, time_index: int) -> torch.Tensor:
        """Helper to get potential value at a specific time index."""
        return self.potential_value(space_grid, protocol_tensor[:, time_index])

    def dv_dx(self, space_grid: torch.Tensor, protocol_tensor: torch.Tensor, time_index: int) -> torch.Tensor:
        """Computes the gradient of the potential with respect to spatial coordinates."""
        coeff = protocol_tensor[:, time_index]
        batch_dvdx_func, _ = self._get_kernels()
        return batch_dvdx_func(space_grid, coeff)

    def dv_dxda(self, space_grid: torch.Tensor, protocol_tensor: torch.Tensor, time_index: int) -> torch.Tensor:
        """Computes the sensitivity of the gradient with respect to coefficients."""
        coeff = protocol_tensor[:, time_index]
        _, batch_dvdxda_func = self._get_kernels()
        return batch_dvdxda_func(space_grid, coeff)
