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

    def _compute_malliavian_weight(self, dv_dxda, noise, noise_sigma, dt):
        return (-dv_dxda * noise[:,None,:] / (noise_sigma ** 2 * dt))
        
    def make_trajectories(self, potential, initial_pos, initial_vel, time_steps, noise, noise_sigma, coeff_grid, device, DEBUG_PRINT = False):
        #potential is a Potential object
        #initial_phase of shape (num traj, spatial dimensions, 2 for position and velocity), 3 dimensions in total
        #output of shape (initial_phase.shape[0] num traj, initial_phase.shape[1] spatial dimensions, time_steps+1, 2 for position and velocity), 4 dimensions in total
        #and potential of shape (initial_phase.shape[0] num traj, initial_phase.shape[1] spatial dimensions, time_steps+1)
        #and potential of shape (initial_phase.shape[0] num traj, time_steps+1) potential is scalar TODO mightve messed this up by making is like a vector potential accidently
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
                # dv = -gamma * v - dV/dx * dt + noise
                current_pos = traj_pos_list[-1]
                current_vel = traj_vel_list[-1]
                
                dv_dx = potential.dv_dx(current_pos, coeff_grid, i)
                U = potential.get_potential_value(current_pos, coeff_grid, i)
                dv_dxda = potential.dv_dxda(current_pos, coeff_grid, i)

                # Compute next positions and velocities
                next_pos = current_pos + current_vel * dt
                next_vel = current_vel - self.gamma * current_vel * dt - dv_dx * dt + noise[..., i].unsqueeze(-1)
                
                traj_pos_list.append(next_pos)
                traj_vel_list.append(next_vel)
                potential_list.append(U.squeeze())
                dv_dxda_list.append(dv_dxda)
            else:
                raise NotImplementedError(f"Mode {self.mode} not implemented")

        # Stack lists into tensors
        traj_pos = torch.stack(traj_pos_list, dim=-1)  # (num_traj, spatial_dims, time_steps+1)
        traj_vel = torch.stack(traj_vel_list, dim=-1)  # (num_traj, spatial_dims, time_steps+1)
        potential_tensor = torch.stack(potential_list, dim=-1)  # (num_traj, time_steps)
        dv_dxda_tensor = torch.stack(dv_dxda_list, dim=-1)  # (num_traj, coeff_count, time_steps)

        if DEBUG_PRINT:
            print(f"Malliavin weight stats - mean: {self._compute_malliavian_weight(dv_dxda_tensor, noise, noise_sigma, dt).mean().item()}, std: {self._compute_malliavian_weight(dv_dxda_tensor, noise, noise_sigma, dt).std().item()}")
            print(f"dv_dxda_tensor stats - mean: {dv_dxda_tensor.mean().item()}, std: {dv_dxda_tensor.std().item()}, has nan: {torch.isnan(dv_dxda_tensor).any()}")
        # bun = torch.autograd.grad(
        #     outputs = traj_pos,
        #     inputs = potential_tensor,
        #     grad_outputs = torch.ones_like(traj_pos),
        #     create_graph = True
        # )[0]
        return {
            'trajectories': torch.cat([traj_pos.unsqueeze(-1), traj_vel.unsqueeze(-1)], dim=-1),
            'potential': potential_tensor,
            'malliavian_weight': self._compute_malliavian_weight(dv_dxda_tensor, noise, noise_sigma, dt)
        }
    def debug_gradients(self, dv_dx, U, traj_pos_slice, traj_vel_slice, potential_tensor, dv_dxda_tensor):
        pass