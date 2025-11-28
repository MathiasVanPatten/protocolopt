#for simple losses that are stateless
import torch
from ..core.types import PotentialTensor, Trajectories, ControlSignal

def work_loss(potential_tensor: PotentialTensor) -> torch.Tensor:
    """Computes work loss based on potential changes.

    Args:
        potential_tensor: Potential energy. Shape: (Batch, Time_Steps)

    Returns:
        Work loss. Shape: (Batch,)
    """
    return (potential_tensor[...,1:] - potential_tensor[...,:-1]).sum(axis = -1)

def variance_loss(trajectory_tensor: Trajectories, starting_bits_int: torch.Tensor, domain_size: int, phase_dimension: int = 0) -> torch.Tensor:
    """Computes the variance of trajectories that started together.

    Args:
        trajectory_tensor: Trajectory data.
                           Shape: (Batch, Spatial_Dim, Time_Steps+1, 2)
        starting_bits_int: Starting bit states.
                           Shape: (Batch,)
        domain_size: Number of possible starting states.
        phase_dimension: 0 for position, 1 for velocity.

    Returns:
        Variance loss. Shape: (Batch,)
    """
    #trajectory_tensor is of shape (num_samples, spatial_dimensions, time_steps+1, 2 (position and velocity))
    #starting_bits_int is of shape (num_samples,)
    #domain_size is the number of possible starting bits
    #phase_dimension is the dimension of the phase to compute the variance of, allowing for seperate position and velocity loss computations
    var_loss = torch.zeros(trajectory_tensor.shape[0], device=trajectory_tensor.device)
    for i in range(domain_size):
        mask = starting_bits_int == i
        if mask.any():
            var_loss[mask] = ((trajectory_tensor[mask, :, :, phase_dimension] - 
                   trajectory_tensor[mask, :, :, phase_dimension].mean(axis=0, keepdim=True))**2
                  ).mean(dim=(1, 2)) #compute the variance using cohort mean over space and time then mean each trajectories var over space and time
    return var_loss

def temporal_smoothness_penalty(protocol_tensor: ControlSignal, dt: float) -> torch.Tensor:
    """Computes temporal smoothness penalty for the protocol.

    Args:
        protocol_tensor: Control signals. Shape: (Control_Dim, Time_Steps)
        dt: Time step size.

    Returns:
        Smoothness penalty. Scalar.
    """
    dcoeff_dt = (protocol_tensor[:, 1:] - protocol_tensor[:, :-1]) / dt
    return (dcoeff_dt ** 2).mean()
