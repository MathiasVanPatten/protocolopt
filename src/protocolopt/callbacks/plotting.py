import torch
import matplotlib.pyplot as plt
from pathlib import Path
from ..core.callback import BasePlottingCallback
import numpy as np
from matplotlib.colors import ListedColormap
from typing import Optional, Dict, Any, TYPE_CHECKING
from ..core.types import MicrostatePaths, ControlSignal, PotentialTensor

if TYPE_CHECKING:
    from ..core.protocol_optimizer import ProtocolOptimizer

plt.ioff()


class TrajectoryPlotCallback(BasePlottingCallback):
    """Callback for plotting microstate paths over time"""
    
    def __init__(self, save_dir='figs', plot_frequency=None, num_trajectories=100):
        """
        Args:
            save_dir: Directory to save plots (relative to working directory or absolute path)
            plot_frequency: How often to plot (e.g., every N epochs). If None, plots at 0, 25%, 50%, 75%, 100%
            num_trajectories: Number of random paths to plot (default: 100)
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.plot_frequency = plot_frequency
        self.num_trajectories = num_trajectories
        self.total_epochs = None
    
    def _plot(self, optimizer_object: "ProtocolOptimizer", sim_dict: Dict[str, Any], epoch: int) -> None:
        time_steps = optimizer_object.protocol.time_steps
        microstate_paths: MicrostatePaths = sim_dict['microstate_paths']
        spatial_dimensions = optimizer_object.init_cond_generator.spatial_dimensions
        
        # Choose random path indices
        num_paths = microstate_paths.shape[0]
        sample_size = min(self.num_trajectories, num_paths)
        random_indices = torch.randperm(num_paths)[:sample_size]
        
        # Create separate plot for each spatial dimension
        for dim_idx in range(spatial_dimensions):
            # Extract position data for this dimension
            path_positions = microstate_paths[random_indices, dim_idx, :, 0].cpu()
            
            fig = plt.figure()
            plt.plot(torch.linspace(0, time_steps, time_steps + 1).cpu(), path_positions.T)
            plt.xlabel('Time')
            plt.ylabel(f'Position (Dimension {dim_idx})')
            plt.title(f'Microstate Positions Over Time - Dim {dim_idx} (Epoch {epoch})')
            self._log_or_save_figure(fig, f'trajectory_dim', epoch, optimizer_object, context={'dim': dim_idx})
            plt.close()

class ConfusionMatrixCallback(BasePlottingCallback):
    """Callback for plotting confusion matrix of binary state transitions"""
    
    def __init__(self, save_dir='figs', plot_frequency=None):
        """
        Args:
            save_dir: Directory to save plots
            plot_frequency: How often to plot (e.g., every N epochs). If None, plots at 0, 25%, 50%, 75%, 100%
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.plot_frequency = plot_frequency
        self.total_epochs = None
    
    def _plot(self, optimizer_object: "ProtocolOptimizer", sim_dict: Dict[str, Any], epoch: int) -> None:
        # Get the loss object
        loss_object = optimizer_object.loss
        
        # Check if loss supports binary trajectory computation
        if not hasattr(loss_object, 'compute_binary_trajectory_info'):
            return  # Only works with LogicGateEndpointLossBase-derived losses
        
        # Compute binary trajectory information
        binary_trajectory_dict = loss_object.compute_binary_trajectory_info(sim_dict['microstate_paths'])
        starting_bits_int = binary_trajectory_dict['starting_bits_int'].cpu().numpy()
        ending_bits_int = binary_trajectory_dict['ending_bits_int'].cpu().numpy()
        
        # Get the domain size (number of possible states)
        if not hasattr(loss_object, 'domain'):
            return
        
        domain = loss_object.domain
        validity = loss_object.validity.detach().cpu().numpy()
        
        # Create confusion matrix (counts)
        confusion_matrix = torch.zeros(domain, domain, dtype=torch.float32)
        for start, end in zip(starting_bits_int, ending_bits_int):
            confusion_matrix[start, end] += 1
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        valid_color = np.array([144, 238, 144]) / 255  # Light green
        invalid_color = np.array([255, 182, 193]) / 255  # Light pink
        
        background = np.zeros((domain, domain, 3))
        
        for i in range(domain):
            for j in range(domain):
                is_valid = validity[i, j]
                
                if is_valid:
                    background[i, j] = valid_color
                else:
                    background[i, j] = invalid_color
        
        # Display the background
        ax.imshow(background, aspect='auto')
        
        # Add text annotations with counts
        for i in range(domain):
            for j in range(domain):
                count = int(confusion_matrix[i, j].item())
                is_valid = validity[i, j]
                
                # Use dark text for visibility on light backgrounds
                text_color = 'black'
                
                text = f'{count}'
                
                ax.text(j, i, text, ha='center', va='center', 
                       color=text_color, fontsize=11, weight='bold')
        
        # Format binary labels
        bit_width = len(bin(domain - 1)) - 2  # Number of bits needed
        labels = [format(i, f'0{bit_width}b') for i in range(domain)]
        
        ax.set_xticks(range(domain))
        ax.set_yticks(range(domain))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_xlabel('Output Bit State')
        ax.set_ylabel('Input Bit State')
        ax.set_title(f'Binary State Transition Confusion Matrix (Epoch {epoch})\nGreen=Valid, Red=Invalid')
        
        plt.tight_layout()
        self._log_or_save_figure(fig, 'confusion_matrix', epoch, optimizer_object, dpi=150)
        plt.close()

class PotentialLandscapePlotCallback(BasePlottingCallback):
    
    def __init__(self, save_dir='figs', plot_frequency=None, spatial_resolution=100, trajectories_per_bit=10):
        self.save_dir = Path(save_dir)
        self.plot_frequency = plot_frequency
        self.spatial_resolution = spatial_resolution
        self.trajectories_per_bit = trajectories_per_bit
        self.total_epochs = None
    
    def _plot(self, optimizer_object: "ProtocolOptimizer", sim_dict: Dict[str, Any], epoch: int) -> None:
        protocol_tensor = sim_dict.get('protocol_tensor', None)
        if protocol_tensor is None:
            return
        
        microstate_paths: MicrostatePaths = sim_dict['microstate_paths']
        spatial_dimensions = optimizer_object.init_cond_generator.spatial_dimensions
        time_steps = optimizer_object.protocol.time_steps
        spatial_bounds = getattr(optimizer_object.init_cond_generator, 'mcmc_starting_spatial_bounds', None)
        if spatial_bounds is None:
            min_vals = microstate_paths[..., 0].min(dim=0)[0].min(dim=1)[0]
            max_vals = microstate_paths[..., 0].max(dim=0)[0].max(dim=1)[0]
            
            centers = (max_vals + min_vals) / 2
            spans = (max_vals - min_vals)
            

            spans = torch.maximum(spans, torch.tensor(1.0, device=optimizer_object.device))

            x_min_calc = centers - spans
            x_max_calc = centers + spans
            
            spatial_bounds = torch.stack([x_min_calc, x_max_calc], dim=1)

        bit_locations = optimizer_object.loss.bit_locations
        potential_obj = optimizer_object.potential
        
        for dim_idx in range(spatial_dimensions):
            x_min, x_max = spatial_bounds[dim_idx, 0].item(), spatial_bounds[dim_idx, 1].item()
            x_grid = torch.linspace(x_min, x_max, self.spatial_resolution, device=optimizer_object.device)
            t_indices = torch.arange(time_steps, device=optimizer_object.device)
            
            query_points = torch.zeros(self.spatial_resolution, spatial_dimensions, device=optimizer_object.device)
            query_points[:, dim_idx] = x_grid

            potential_values = torch.zeros(self.spatial_resolution, time_steps)
            for t_idx in range(time_steps):
                coeff_slice = protocol_tensor[:, t_idx]
                potential_values[:, t_idx] = potential_obj.potential_value(query_points, coeff_slice).cpu()
            
            potential_values = potential_values.sign() * ((potential_values.abs() + 1).log10())
            path_positions = microstate_paths[:, dim_idx, :, 0]
            
            traj_min = path_positions.min().item()
            traj_max = path_positions.max().item()
            spatial_buffer = (traj_max - traj_min) * 0.2
            
            mask = (x_grid.cpu() >= (traj_min - spatial_buffer)) & (x_grid.cpu() <= (traj_max + spatial_buffer))
            relevant_potential = potential_values[mask, :]
            
            v_min = torch.quantile(relevant_potential, 0.05).item()
            v_max = torch.quantile(relevant_potential, 0.95).item()
            start_positions = path_positions[:, 0]
            bit_locs_dim = bit_locations[:, dim_idx]
            distances = torch.abs(start_positions[:, None] - bit_locs_dim[None, :])
            bit_assignments = distances.argmin(dim=1)
            
            sampled_indices = []
            num_bits = bit_locs_dim.shape[0]
            for bit_idx in range(num_bits):
                mask = (bit_assignments == bit_idx)
                indices_in_class = torch.where(mask)[0]
                if len(indices_in_class) > 0:
                    num_to_sample = min(self.trajectories_per_bit, len(indices_in_class))
                    sampled = indices_in_class[torch.randperm(len(indices_in_class))[:num_to_sample]]
                    sampled_indices.append(sampled)
            if len(sampled_indices) > 0:
                sampled_indices = torch.cat(sampled_indices)
            else:
                sampled_indices = torch.tensor([], dtype=torch.long)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            im = ax.imshow(potential_values.T, aspect='auto', origin='lower', 
                          extent=[x_min, x_max, 0, time_steps], cmap='viridis', 
                          vmin=v_min, vmax=v_max)
            ax.set_xlabel('Position')
            ax.set_ylabel('Time')
            ax.set_title(f'Potential Landscape - Dim {dim_idx} (Epoch {epoch})')
            plt.colorbar(im, ax=ax, label='Signed log10(Potential Value)')
            
            
            high_contrast_colors = ['cyan', 'magenta', 'lime', 'orange', 'white']
            
            for idx in sampled_indices:
                bit_class = bit_assignments[idx].item()
                traj = path_positions[idx].cpu().numpy()
                time_array = np.linspace(0, time_steps, time_steps + 1)
                
                color_idx = bit_class % len(high_contrast_colors)
                color = high_contrast_colors[color_idx]
                
                ax.plot(traj, time_array, color=color, alpha=0.8, linewidth=1.5)
            
            self._log_or_save_figure(fig, f'potential_landscape', epoch, optimizer_object, context={'dim': dim_idx})
            plt.close(fig)

class ProtocolPlotCallback(BasePlottingCallback):
    
    def __init__(self, save_dir='figs', plot_frequency=None):
        self.save_dir = Path(save_dir)
        self.plot_frequency = plot_frequency
        self.total_epochs = None
    
    def _plot(self, optimizer_object: "ProtocolOptimizer", sim_dict: Dict[str, Any], epoch: int) -> None:
        protocol_tensor = sim_dict.get('protocol_tensor', None)
        if protocol_tensor is None:
            return
        
        protocol_tensor_cpu = protocol_tensor.cpu().numpy()
        control_dim, time_steps = protocol_tensor_cpu.shape
        time_array = np.arange(time_steps)
        
        if control_dim <= 4:
            ncols = control_dim
            nrows = 1
        else:
            ncols = 2
            nrows = (control_dim + 1) // 2
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows), squeeze=False)
        axes = axes.flatten()
        
        for coeff_idx in range(control_dim):
            ax = axes[coeff_idx]
            ax.plot(time_array, protocol_tensor_cpu[coeff_idx, :])
            ax.set_xlabel('Time')
            ax.set_ylabel(f'Coeff {coeff_idx}')
            ax.set_title(f'Coefficient {coeff_idx}')
            ax.grid(True, alpha=0.3)
        
        for idx in range(control_dim, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        self._log_or_save_figure(fig, 'protocol_control_time_evolution', epoch, optimizer_object)
        plt.close(fig)

class InfluenceHeatmapCallback(BasePlottingCallback):
    """
    Visualizes the Information Ripple: How much does Bit i depend on Bit i-1 over time?
    """
    def __init__(self, save_dir: str = 'figs', plot_frequency: Optional[int] = None):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.plot_frequency = plot_frequency
        self.total_epochs: Optional[int] = None

    def _plot(self, optimizer_object: "ProtocolOptimizer", sim_dict: Dict[str, Any], epoch: int) -> None:
        microstate_paths: MicrostatePaths = sim_dict['microstate_paths']
        batch_size, dims, time_steps, _ = microstate_paths.shape
        
        if dims < 2: return 

        midpoints = optimizer_object.loss.midpoints # Shape: (Spatial_Dims,)
            
        start_positions = sim_dict['microstate_paths'][:, :, 0, 0] # (Batch, Dims)
        
        binary_starts = (start_positions > midpoints).float() 

        logic_map = torch.zeros(2 * (dims - 1), time_steps, device=microstate_paths.device)
        y_labels = []

        for i in range(1, dims):
            prev_bit = i - 1
            
            mask_1 = (binary_starts[:, prev_bit] == 1.0)
            mask_0 = (binary_starts[:, prev_bit] == 0.0)
            
            if mask_0.sum() > 0:
                mean_path_0 = microstate_paths[mask_0, i, :, 0].mean(dim=0)
                logic_map[2*(i-1)] = mean_path_0
            y_labels.append(f'Bit {i} | Prev=0')
            
            if mask_1.sum() > 0:
                mean_path_1 = microstate_paths[mask_1, i, :, 0].mean(dim=0)
                logic_map[2*(i-1) + 1] = mean_path_1
            y_labels.append(f'Bit {i} | Prev=1')

        logic_map = logic_map.cpu().numpy()
        
        limit = np.abs(logic_map).max()
        
        fig, ax = plt.subplots(figsize=(12, 0.8 * len(y_labels) + 2))
        im = ax.imshow(logic_map, aspect='auto', origin='lower', 
                       cmap='seismic', vmin=-limit, vmax=limit)
        
        plt.colorbar(im, ax=ax, label='Mean Position (Blue=0, Red=1)')
        
        for y in range(0, len(y_labels), 2):
            ax.axhline(y - 0.5, color='black', linewidth=2)
            
        ax.set_yticks(range(len(y_labels)))
        ax.set_yticklabels(y_labels)
        ax.set_xlabel('Time Steps')
        ax.set_title(f'Logic Flow Analysis (Epoch {epoch})')
        
        plt.tight_layout()
        self._log_or_save_figure(fig, 'influence_ripple', epoch, optimizer_object)
        plt.close(fig)