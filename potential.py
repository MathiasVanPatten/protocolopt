#intended to hold the task definition, where do we start, where do we end up?
from typing import Any
import torch
#TODO finish type hinting
from abc import ABC, abstractmethod
class Potential(ABC):
    @abstractmethod
    def potential_value(self, space_grid, coeff_grid):
        #you should accept a tensor (spatial dimensions) with a coefficient grid (coefficient counts, spatial dimensions) and return the potential value at each point
        #do not detach from the graph, the protocol will automatically check
        #access potential parameters (control variables, spatial dimensions, time) with self.potential_params
        #NEVER detach self.potential_parms, use copy to not break the graph
        #control variables may be singular
        pass

    def get_potential_value(self, space_grid, coeff_grid, time_index):
        return self.potential_value(space_grid, coeff_grid[:, time_index])

    def dv_dx(self, space_grid, coeff_grid, time_index):
        coeff_grid_slice = coeff_grid[:, time_index]
        graphed_space_grid = space_grid.requires_grad_(True)
        V = self.potential_value(graphed_space_grid, coeff_grid_slice)
        dv_dx = torch.autograd.grad(
            outputs = V,
            inputs = graphed_space_grid,
            grad_outputs = torch.ones_like(V),
            create_graph = True
        )[0]
        return dv_dx

    def dv_dxda(self, space_grid, coeff_grid, time_index):
        coeff_for_grad = coeff_grid[:, time_index]
        graphed_space_grid = space_grid.requires_grad_(True)
        V = self.potential_value(graphed_space_grid, coeff_for_grad)

        dV_dx = torch.autograd.grad(
            outputs = V,
            inputs = graphed_space_grid,
            grad_outputs = torch.ones_like(V),
            create_graph = True
        )[0]
        
        dv_dxda = torch.autograd.grad(
            outputs = dV_dx,
            inputs = coeff_for_grad,
            grad_outputs=torch.ones_like(dV_dx) / len(dV_dx), #use mean to aggregate gradients since params effect all trajs
            create_graph = False
        )[0]
        
        return dv_dxda

class QuarticPotential(Potential):
    #the potential is V(x_i, t) = a_i(t) * x_i^4 - b_i(t) * x_i^2 for i in spatial dimensions
    #any number spatial dimensions is supported
    def potential_value(self, space_grid: torch.Tensor, coeff_grid: torch.Tensor) -> torch.Tensor:
        return coeff_grid[0] * space_grid**4 - coeff_grid[1] * space_grid**2