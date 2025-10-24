#defintion of what a timestep looks like
#uses the Euler-Maruyama Method to numerically advance
#either over or underdamped systems

import torch

class EulerMaruyama:
    def __init__(self, mode, gamma, dt = None) -> None:
        #under or over
        if mode not in ['underdamped', 'overdamped']:
            raise ValueError(f"Invalid mode: {mode}, choose from 'underdamped' or 'overdamped'")
        self.mode = mode
        self.dt = dt
        self.gamma = gamma

    def make_trajectories(self, potential, initial_pos, initial_vel, time_steps, noise, coeff_grid, device):
        #potential is a Potential object
        #initial_phase of shape (num traj, spatial dimensions, 2 for position and velocity), 3 dimensions in total
        #output of shape (initial_phase.shape[0] num traj, initial_phase.shape[1] spatial dimensions, time_steps+1, 2 for position and velocity), 4 dimensions in total
        #and potential of shape (initial_phase.shape[0] num traj, initial_phase.shape[1] spatial dimensions, time_steps+1)
        if self.dt is None:
            dt = 1 / time_steps
        else:
            dt = self.dt
        
        traj_pos = torch.zeros(initial_pos.shape[0], initial_pos.shape[1], time_steps+1, 2, device=device)
        traj_vel = torch.zeros(initial_vel.shape[0], initial_vel.shape[1], time_steps+1, 2, device=device)
        potential_tensor = torch.zeros(initial_pos.shape[0], initial_pos.shape[1], time_steps+1, device=device)
        drift_values = torch.zeros(initial_pos.shape[0], initial_pos.shape[1], time_steps, device=device)
    
        traj_pos[..., 0] = initial_pos
        traj_vel[..., 0] = initial_vel
        for i in range(time_steps):
            if self.mode == 'underdamped':
                # dx = v * dt
                # dv = -gamma * v - dV/dx * dt + noise
                dv_dx = potential.dv_dx(traj_pos[..., i], coeff_grid, i)
                U = potential.get_potential_value(traj_pos[..., i], coeff_grid, i)

                
                traj_pos[..., i + 1] = traj_pos[..., i] + traj_vel[..., i] * dt
                traj_vel[..., i + 1] = traj_vel[..., i] - self.gamma * traj_vel[..., i] - dv_dx * dt + noise[..., i]
                potential_tensor[..., i] = U
                drift_values[..., i] = dv_dx
            else:
                raise NotImplementedError(f"Mode {self.mode} not implemented")

        potential_tensor[..., time_steps] = potential.get_potential_value(traj_pos[..., time_steps], time_steps)
        return {
            'trajectories': torch.cat([traj_pos, traj_vel], dim=-1),
            'potential': potential_tensor,
            'drift': drift_values,
            'noise': noise
        }