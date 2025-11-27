#defintion of what a timestep looks like
#uses the Euler-Maruyama Method to numerically advance
#either over or underdamped systems

import torch
import sys
import os

from utils import robust_compile

class EulerMaruyama:
    def __init__(self, mode, gamma, mass = 1, dt = None, compile_mode=True) -> None:
        if mode not in ['underdamped', 'overdamped']:
            raise ValueError(f"Invalid mode: {mode}, choose from 'underdamped' or 'overdamped'")
        self.mode = mode
        self.dt = dt
        self.mass = mass
        self.gamma = gamma
        self.compile_mode = compile_mode

        self._compiled_underdamped_step = robust_compile(self._underdamped_step, compile_mode=self.compile_mode)
        self._compiled_overdamped_step = robust_compile(self._overdamped_step, compile_mode=self.compile_mode)
    
    @staticmethod
    def _overdamped_step(current_pos, dv_dx, noise, dt, gamma):
        return current_pos - (dv_dx * dt - noise) / gamma

    @staticmethod
    def _underdamped_step(current_pos, current_vel, dv_dx, noise, dt, gamma, mass):
        next_pos = current_pos + current_vel * dt
        next_vel = (current_vel - gamma * current_vel * dt - dv_dx * dt + noise) / mass
        return next_pos, next_vel

    def _compute_malliavian_weight(self, dv_dxda, noise, noise_sigma):
        # dv_dxda: (samples, spatial, coeffs, time)
        # noise: (samples, spatial, time)
        # output: (samples, coeffs, time)
        drift_grad = -dv_dxda
        noise_expanded = noise[:,:,None,:] # (samples, spatial, expanded to effect all coeffs equally, time)
        dot_product = (drift_grad * noise_expanded).sum(dim = 1) # (samples, coeffs, time)
        return dot_product / (noise_sigma ** 2)
        
    def make_trajectories(self, potential, initial_pos, initial_vel, time_steps, noise, noise_sigma, coeff_grid, DEBUG_PRINT = False):
        #potential is a Potential object
        #initial_phase of shape (num traj, spatial dimensions, 2 for position and velocity), 3 dimensions in total
        #output of shape (initial_phase.shape[0] num traj, initial_phase.shape[1] spatial dimensions, time_steps+1, 2 for position and velocity), 4 dimensions in total
        #and potential of shape (initial_phase.shape[0] num traj, initial_phase.shape[1] spatial dimensions, time_steps+1)
        #and potential of shape (initial_phase.shape[0] num traj, time_steps+1) potential is scalar
        if self.dt is None:
            dt = 1 / time_steps
        else:
            dt = self.dt
        
        # Use lists to build trajectories dynamically, preserving the computation graph
        traj_pos_list = [initial_pos]
        traj_vel_list = [initial_vel]
        potential_list = []
        dv_dxda_list = []

        for i in range(time_steps):
            if self.mode == 'underdamped':
                # dx = v * dt
                # dv = (-gamma * v * dt - dV/dx * dt + noise) / mass
                current_pos = traj_pos_list[-1]
                current_vel = traj_vel_list[-1]
                
                dv_dx = potential.dv_dx(current_pos, coeff_grid, i)
                U = potential.get_potential_value(current_pos, coeff_grid, i)
                dv_dxda = potential.dv_dxda(current_pos, coeff_grid, i)

                # Compute next positions and velocities
                next_pos, next_vel = self._compiled_underdamped_step(
                                    current_pos, current_vel, dv_dx, noise[..., i], dt, self.gamma, self.mass
                                )
                traj_pos_list.append(next_pos)
                traj_vel_list.append(next_vel)
                potential_list.append(U.squeeze())
                dv_dxda_list.append(dv_dxda)
            elif self.mode == 'overdamped':
                # dv = 0
                # dx = - (dV/dx * dt + noise) / gamma
                
                current_pos = traj_pos_list[-1]

                dv_dx = potential.dv_dx(current_pos, coeff_grid, i)
                U = potential.get_potential_value(current_pos, coeff_grid, i)
                dv_dxda = potential.dv_dxda(current_pos, coeff_grid, i)
            
                next_pos = self._compiled_overdamped_step(current_pos, dv_dx, noise[..., i], dt, self.gamma)
                traj_pos_list.append(next_pos)
                traj_vel_list.append(torch.zeros_like(next_pos))
                potential_list.append(U.squeeze())
                dv_dxda_list.append(dv_dxda)
            else:
                raise ValueError(f"Please choose a valid mode, got {self.mode}, choose from 'underdamped' or 'overdamped'")

        traj_pos = torch.stack(traj_pos_list, dim=-1)  # (num_traj, spatial_dims, time_steps+1)
        traj_vel = torch.stack(traj_vel_list, dim=-1)  # (num_traj, spatial_dims, time_steps+1)
        potential_tensor = torch.stack(potential_list, dim=-1)  # (num_traj, time_steps)
        dv_dxda_tensor = torch.stack(dv_dxda_list, dim=-1)  # (num_traj, coeff_count, time_steps)

        if DEBUG_PRINT:
            print(f"Malliavin weight stats - mean: {self._compute_malliavian_weight(dv_dxda_tensor, noise, noise_sigma).mean().item()}, std: {self._compute_malliavian_weight(dv_dxda_tensor, noise, noise_sigma, dt).std().item()}")
            print(f"dv_dxda_tensor stats - mean: {dv_dxda_tensor.mean().item()}, std: {dv_dxda_tensor.std().item()}, has nan: {torch.isnan(dv_dxda_tensor).any()}")
        output_dict = {
            'trajectories': torch.cat([traj_pos.unsqueeze(-1), traj_vel.unsqueeze(-1)], dim=-1),
            'potential': potential_tensor,
            'malliavian_weight': self._compute_malliavian_weight(dv_dxda_tensor, noise, noise_sigma)
        }
        return output_dict

    def debug_gradients(self, dv_dx, U, traj_pos_slice, traj_vel_slice, potential_tensor, dv_dxda_tensor):
        pass