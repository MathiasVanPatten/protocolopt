#intended to hold the task definition, where do we start, where do we end up?
from typing import Any
import torch
import sys
from torch.func import vmap, grad, jacrev
from abc import ABC, abstractmethod
try:
    from protocolopt.utils import robust_compile
except ImportError:
    # Fallback for when running script directly vs as module
    from utils import robust_compile

class Potential(ABC):
    def __init__(self, compile_mode=True):
        self.compile_mode = compile_mode

    @abstractmethod
    def potential_value(self, space_grid, coeff_grid):
        #you should accept a tensor (spatial dimensions) with a coefficient grid (coefficient counts, spatial dimensions) and return the potential value at each point
        #do not detach from the graph, the protocol will automatically check
        #access potential parameters (control variables, spatial dimensions, time) with self.potential_params
        #NEVER detach self.potential_parms, use copy to not break the graph
        #control variables may be singular
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

    def get_potential_value(self, space_grid, coeff_grid, time_index):
        return self.potential_value(space_grid, coeff_grid[:, time_index])

    def dv_dx(self, space_grid, coeff_grid, time_index):
        coeff = coeff_grid[:, time_index]
        batch_dvdx_func, _ = self._get_kernels()
        return batch_dvdx_func(space_grid, coeff)

    def dv_dxda(self, space_grid, coeff_grid, time_index):
        # computes sensitivity of gradient w.r.t coeffs
        # used for malliavin weights
        coeff = coeff_grid[:, time_index]
        _, batch_dvdxda_func = self._get_kernels()
        return batch_dvdxda_func(space_grid, coeff)

class QuarticPotential(Potential):
    #the potential is V(x_i, t) = a_i(t) * x_i^4 - b_i(t) * x_i^2 for i in spatial dimensions
    #any number spatial dimensions is supported
    def potential_value(self, space_grid: torch.Tensor, coeff_grid: torch.Tensor) -> torch.Tensor:
        return torch.sum(coeff_grid[0] * space_grid**4 - coeff_grid[1] * space_grid**2, dim=-1)

class QuarticPotentialWithLinearTerm(Potential):
    #the potential is V(x_i, t) = a_i(t) * x_i^4 - b_i(t) * x_i^2 + c_i(t) * x_i for i in spatial dimensions
    #any number spatial dimensions is supported
    def potential_value(self, space_grid: torch.Tensor, coeff_grid: torch.Tensor) -> torch.Tensor:
        return torch.sum(coeff_grid[0] * space_grid**4 - coeff_grid[1] * space_grid**2 + coeff_grid[2] * space_grid, dim=-1)