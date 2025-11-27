from ..core.protocol import Protocol
import torch
import torch.nn.functional as F

class LinearPiecewise(Protocol):
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
