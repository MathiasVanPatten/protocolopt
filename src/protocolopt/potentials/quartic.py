from ..core.potential import Potential
import torch

class QuarticPotential(Potential):
    #the potential is V(x_i, t) = a_i(t) * x_i^4 - b_i(t) * x_i^2 for i in spatial dimensions
    #only works for 1 spatial dimension right now
    def potential_value(self, space_grid: torch.Tensor, coeff_grid: torch.Tensor) -> torch.Tensor:
        return torch.sum(coeff_grid[0] * space_grid**4 - coeff_grid[1] * space_grid**2, dim=-1)

class QuarticPotentialWithLinearTerm(Potential):
    #the potential is V(x_i, t) = a_i(t) * x_i^4 - b_i(t) * x_i^2 + c_i(t) * x_i for i in spatial dimensions
    #only works for 1 spatial dimension right now
    def potential_value(self, space_grid: torch.Tensor, coeff_grid: torch.Tensor) -> torch.Tensor:
        return torch.sum(coeff_grid[0] * space_grid**4 - coeff_grid[1] * space_grid**2 + coeff_grid[2] * space_grid, dim=-1)
