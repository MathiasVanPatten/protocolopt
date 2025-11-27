from ..core.potential import Potential
import torch

class GeneralCoupledPotential(Potential):
    def __init__(self, spatial_dimensions, has_c=True, has_mix=True, compile_mode=True):
        super().__init__(compile_mode)
        self.spatial_dim = spatial_dimensions
        self.has_c = has_c
        self.has_mix = has_mix
        if self.spatial_dim == 1:
            self.has_mix = False

        self.triu_indices = torch.triu_indices(row=spatial_dimensions, col=spatial_dimensions, offset=1)

    def potential_value(self, space_grid, coeff_grid):
        # space_grid: (Batch, N) or (N,)
        # coeff_grid: (Total_Coeffs)
        is_unbatched = space_grid.ndim == 1
        if is_unbatched:
            space_grid = space_grid.unsqueeze(0)
        # assume layout: [N Quartics, N Quadratics, N Linears, K Interactions]
        N = self.spatial_dim

        current_idx = 0

        # Quartic terms (a)
        a = coeff_grid[current_idx : current_idx + N]
        current_idx += N

        # Quadratic terms (b)
        b = coeff_grid[current_idx : current_idx + N]
        current_idx += N

        # Linear terms (c)
        if self.has_c:
            c = coeff_grid[current_idx : current_idx + N]
            current_idx += N
        else:
            c = torch.zeros_like(a)

        # Interaction terms
        if self.has_mix:
            mix = coeff_grid[current_idx : ]
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
