#defintion of what a timestep looks like
#uses the Euler-Maruyama Method to numerically advance
#either over or underdamped systems

import torch

class EulerMaruyama:
    #TODO vectorize to any dimensions
    def __init__(self, mode, gamma, dt = None) -> None:
        #under or over
        if mode not in ['underdamped', 'overdamped']:
            raise ValueError(f"Invalid mode: {mode}, choose from 'underdamped' or 'overdamped'")
        self.mode = mode
        self.dt = dt
        self.gamma = gamma

    def make_trajectories(self, potential, initial_phase, time_steps, noise, device):
        #potential is a Potential object
        #initial_phase of shape (num traj, spatial dimensions, 2 for position and velocity)
        #output of shape (initial_phase.shape[0] num traj, initial_phase.shape[1] spatial dimensions, time_steps+1, 2 for position and velocity)
        if self.dt is None: 
            dt = 1 / time_steps #TODO check and make sure this works as expected
        else:
            dt = self.dt

        traj_pos = torch.zeros(initial_phase.shape[0], initial_phase.shape[1], time_steps+1, 2, device=device)
        traj_vel = torch.zeros(initial_phase.shape[0], initial_phase.shape[1], time_steps+1, 2, device=device)
        traj_pos[..., 0] = initial_phase[..., 0]
        traj_vel[..., 0] = initial_phase[..., 1]
        potential.refresh_coeff()
        for i in range(time_steps-1):
            if self.mode == 'underdamped':
                # dx = v * dt
                # dv = -gamma * v + U * dt + noise
                traj_pos[..., i + 1] = traj_pos[..., i] + traj_vel[..., i] * dt
                traj_vel[..., i + 1] = traj_vel[..., i] - self.gamma * traj_vel[..., i] + potential.get_potential_value(traj_pos[..., i], i) * dt + noise[..., i]
            else:
                raise NotImplementedError(f"Mode {self.mode} not implemented")
        return torch.cat([traj_pos, traj_vel], dim=-1)