#for simple losses that are stateless
import torch

def work_loss(potential_tensor):
    return (potential_tensor[...,1:] - potential_tensor[...,:-1]).sum(axis = -1)

def variance_loss(trajectory_tensor, starting_bits_int, domain_size, phase_dimension = 0):
    #computes the variance of trajectories that started together
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

def temporal_smoothness_penalty(coeff_grid, dt):
    dcoeff_dt = (coeff_grid[:, 1:] - coeff_grid[:, :-1]) / dt
    return (dcoeff_dt ** 2).mean()
