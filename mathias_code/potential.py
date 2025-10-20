#intended to hold the task definition, where do we start, where do we end up?
from abc import abstractmethod
from typing import Any
import torch
#TODO finish type hinting
class Potential:
    def __init__(self, potential_model) -> None:
        #the device of the initial potential parameters is taken as the device of the entire optimizer
        self._check_potential_function()
        self.potential_model = potential_model
        # self.potential_parms = torch.nn.Parameter(initial_potential_parms, requires_grad=True)
        # if len(self.potential_parms.shape) < 3:
        #     raise ValueError("Potential parameters must have at least 3 dimensions (control variables, spatial dimensions, time). Unsqueeze to add singleton control variable dimension if needed.")

    @abstractmethod
    def potential_value(self, space_grid, coeff_grid):
        #you should accept a tensor (spatial dimensions) with a coefficient grid (coefficient counts, spatial dimensions) and return the potential value at each point
        #do not detach from the graph, the protocol will automatically check
        #access potential parameters (control variables, spatial dimensions, time) with self.potential_params
        #NEVER detach self.potential_parms, use copy to not break the graph
        #control variables may be singular
        pass

    def refresh_coeff(self):
        self.coeff_grid = self.potential_model.get_coeff_grid()

    def get_potential_value(self, space_grid, time_index):
        return self.potential_value(space_grid, self.coeff_grid[:, time_index])

    def _check_potential_function(self):
        #TODO check the differentibility of the potential function with probes surrounding it
        pass


class QuarticPotential(Potential):
    #the potential is V(x_i, t) = a_i(t) * x_i^4 - b_i(t) * x_i^2 for i in spatial dimensions
    #any number spatial dimensions is supported
    def __init__(self, potential_model) -> None:
        super().__init__(potential_model)
        if potential_model.coefficient_count != 2:
            raise ValueError(f"QuarticPotential requires a potential model with 2 coefficients, got {potential_model.coefficient_count}")

    def potential_value(self, space_grid: torch.Tensor, coeff_grid: torch.Tensor) -> torch.Tensor:
        return coeff_grid[0] * space_grid**4 - coeff_grid[1] * space_grid**2