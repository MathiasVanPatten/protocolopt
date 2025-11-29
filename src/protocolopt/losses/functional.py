#for simple losses that are stateless
import torch
from ..core.types import PotentialTensor, MicrostatePaths, ControlSignal

def work_loss(potential_tensor):
    return (potential_tensor[...,1:] - potential_tensor[...,:-1]).sum(axis = -1)

def variance_loss(microstate_paths: MicrostatePaths, starting_bits_int: torch.Tensor, domain_size: int, phase_dimension: int = 0) -> torch.Tensor:
    """Computes the variance of microstate paths that started together.

    Args:
        microstate_paths: Microstate path data.
                           Shape: (Batch, Spatial_Dim, Time_Steps+1, 2)
                           Dimension 3 is (position, velocity).
        starting_bits_int: Starting bit states.
                           Shape: (Batch,)
        domain_size: Number of possible starting states (Decimal range of bitstring).
        phase_dimension: 0 for position, 1 for velocity.

    Returns:
        Variance loss. Shape: (Batch,)
    """
    #microstate_paths is of shape (num_samples, spatial_dimensions, time_steps+1, 2 (position and velocity))
    #starting_bits_int is of shape (num_samples,)
    #domain_size is the number of possible starting bits
    #phase_dimension is the dimension of the phase to compute the variance of, allowing for seperate position and velocity loss computations
    var_loss = torch.zeros(microstate_paths.shape[0], device=microstate_paths.device)
    for i in range(domain_size):
        mask = starting_bits_int == i
        if mask.any():
            var_loss[mask] = ((microstate_paths[mask, :, :, phase_dimension] -
                   microstate_paths[mask, :, :, phase_dimension].mean(axis=0, keepdim=True))**2
                  ).mean(dim=(1, 2)) #compute the variance using cohort mean over space and time then mean each trajectories var over space and time
    return var_loss

def temporal_smoothness_penalty(protocol_tensor, dt):
    dcoeff_dt = (protocol_tensor[:, 1:] - protocol_tensor[:, :-1]) / dt
    return (dcoeff_dt ** 2).mean()
