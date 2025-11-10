import torch
import torch.nn.functional as F 
from abc import ABC, abstractmethod
class PotentialModel(ABC):
    #potential models use any kind of differentiable math to go from a set
    #of trainable parameters to a full tensor of values to be used in the potential

    #fixed starting is a boolean, if True the starting potential is fixed and the mcmc sampler will only do warmup once 
    def __init__(self, time_steps, fixed_starting) -> None:
        self.time_steps = time_steps
        self.fixed_starting = fixed_starting
        
    @abstractmethod
    def get_coeff_grid(self):
        #return a tensor of shape (coefficients, spatial dimensions, time)
        pass

    @abstractmethod
    def trainable_params(self):
        # return the parameters you want attached to the optimizer as a list
        pass

class LinearPiecewise(PotentialModel):
    def __init__(self, coefficient_count, time_steps, knot_count, initial_coeff_guess, endpoints = None) -> None:
        if endpoints is not None:
            fixed_starting = True
        else:
            fixed_starting = False
        super().__init__(time_steps, fixed_starting)
        if initial_coeff_guess.shape != (coefficient_count, knot_count if endpoints is None else knot_count - 2):
            raise ValueError(f"Initial coefficient guess must be of shape (coefficient_count, knot_count if endpoints is None else knot_count - 2), got {initial_coeff_guess.shape}")
        
        self.coefficient_count = coefficient_count
        self.knot_count = knot_count
        self.endpoints = endpoints
        if self.endpoints is not None:
            self._trainable_params = torch.nn.Parameter(initial_coeff_guess.clone())
        else:
            self._trainable_params = torch.nn.Parameter(initial_coeff_guess.clone())

        self.device = initial_coeff_guess.device

    def _get_knots(self):
        if self.endpoints is not None:
            return torch.cat([self.endpoints[..., 0:1], self._trainable_params, self.endpoints[..., -1:]], dim=-1)
        else:
            return self.trainable_params
        
    def get_coeff_grid(self):
        #return a grid of shape (coefficient_count, time_steps)
        knots = self._get_knots()
        coeff_grid = F.interpolate(knots.unsqueeze(0), size=self.time_steps, mode='linear', align_corners=True)
        return coeff_grid.squeeze(0)
    
    def trainable_params(self):
        return [self._trainable_params]