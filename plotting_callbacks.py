import torch
import matplotlib.pyplot as plt
from pathlib import Path
from callbacks import Callback
from PIL import Image
import io
import numpy as np
from matplotlib.colors import ListedColormap

try:
    from aim import Image as AimImage
    AIM_AVAILABLE = True
except ImportError:
    AIM_AVAILABLE = False

plt.ioff()


class TrajectoryPlotCallback(Callback):
    """Callback for plotting trajectory positions over time"""
    
    def __init__(self, save_dir='figs', plot_frequency=None, num_trajectories=100):
        """
        Args:
            save_dir: Directory to save plots (relative to working directory or absolute path)
            plot_frequency: How often to plot (e.g., every N epochs). If None, plots at 0, 25%, 50%, 75%, 100%
            num_trajectories: Number of random trajectories to plot (default: 100)
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.plot_frequency = plot_frequency
        self.num_trajectories = num_trajectories
        self.total_epochs = None
    
    def _log_or_save_figure(self, fig, name, epoch, simulation_object, dpi=150):
        """Log figure to Aim if available, otherwise save to disk"""
        aim_callback = next((cb for cb in simulation_object.callbacks 
                            if type(cb).__name__ == 'AimCallback'), None)
        
        if AIM_AVAILABLE and aim_callback and hasattr(aim_callback, 'run') and aim_callback.run:
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=dpi)
            buf.seek(0)
            pil_image = Image.open(buf)
            
            aim_callback.run.track(AimImage(pil_image), name=name, epoch=epoch)
            buf.close()
        else:
            filepath = self.save_dir / f'{name}_epoch_{epoch:04d}.png'
            fig.savefig(filepath, dpi=dpi)
    
    def on_train_start(self, simulation_object):
        self.total_epochs = simulation_object.epochs
    
    def on_epoch_end(self, simulation_object, sim_dict, loss_values, epoch):
        should_plot = False
        if self.plot_frequency is not None:
            should_plot = (epoch % self.plot_frequency == 0) or (epoch == self.total_epochs - 1)
        else:
            if self.total_epochs is not None:
                should_plot = (epoch % (self.total_epochs // 4) == 0) or (epoch == self.total_epochs - 1)
        
        if not should_plot:
            return
        
        time_steps = simulation_object.time_steps
        trajectories = sim_dict['trajectories']
        spatial_dimensions = simulation_object.spatial_dimensions
        
        # Choose random trajectory indices
        num_trajectories = trajectories.shape[0]
        sample_size = min(self.num_trajectories, num_trajectories)
        random_indices = torch.randperm(num_trajectories)[:sample_size]
        
        # Create separate plot for each spatial dimension
        for dim_idx in range(spatial_dimensions):
            # Extract position data for this dimension
            trajectory_positions = trajectories[random_indices, dim_idx, :, 0].cpu()
            
            fig = plt.figure()
            plt.plot(torch.linspace(0, time_steps, time_steps + 1).cpu(), trajectory_positions.T)
            plt.xlabel('Time')
            plt.ylabel(f'Position (Dimension {dim_idx})')
            plt.title(f'Trajectory Positions Over Time - Dim {dim_idx} (Epoch {epoch})')
            self._log_or_save_figure(fig, f'trajectory_dim_{dim_idx}', epoch, simulation_object)
            plt.close()

class ConfusionMatrixCallback(Callback):
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
    
    def _log_or_save_figure(self, fig, name, epoch, simulation_object, dpi=150):
        """Log figure to Aim if available, otherwise save to disk"""
        aim_callback = next((cb for cb in simulation_object.callbacks 
                            if type(cb).__name__ == 'AimCallback'), None)
        
        if AIM_AVAILABLE and aim_callback and hasattr(aim_callback, 'run') and aim_callback.run:
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=dpi)
            buf.seek(0)
            pil_image = Image.open(buf)
            
            aim_callback.run.track(AimImage(pil_image), name=name, epoch=epoch)
            buf.close()
        else:
            filepath = self.save_dir / f'{name}_epoch_{epoch:04d}.png'
            fig.savefig(filepath, dpi=dpi)
    
    def on_train_start(self, simulation_object):
        self.total_epochs = simulation_object.epochs
    
    def on_epoch_end(self, simulation_object, sim_dict, loss_values, epoch):
        # Determine if we should plot this epoch
        should_plot = False
        if self.plot_frequency is not None:
            should_plot = (epoch % self.plot_frequency == 0) or (epoch == self.total_epochs - 1)
        else:
            if self.total_epochs is not None:
                should_plot = (epoch % (self.total_epochs // 4) == 0) or (epoch == self.total_epochs - 1)
        
        if not should_plot:
            return
        
        # Get the loss object
        loss_object = simulation_object.loss
        
        # Check if loss supports binary trajectory computation
        if not hasattr(loss_object, 'compute_binary_trajectory_info'):
            return  # Only works with EndpointLossBase-derived losses
        
        # Compute binary trajectory information
        binary_trajectory_dict = loss_object.compute_binary_trajectory_info(sim_dict['trajectories'])
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
        self._log_or_save_figure(fig, 'confusion_matrix', epoch, simulation_object, dpi=150)
        plt.close()

class PotentialLandscapePlotCallback(Callback):
    
    def __init__(self, save_dir='figs', plot_frequency=None, spatial_resolution=100, trajectories_per_bit=10):
        self.save_dir = Path(save_dir)
        self.plot_frequency = plot_frequency
        self.spatial_resolution = spatial_resolution
        self.trajectories_per_bit = trajectories_per_bit
        self.total_epochs = None
    
    def _log_or_save_figure(self, fig, name, epoch, simulation_object, dpi=150):
        aim_callback = next((cb for cb in simulation_object.callbacks 
                            if type(cb).__name__ == 'AimCallback'), None)
        
        if AIM_AVAILABLE and aim_callback and hasattr(aim_callback, 'run') and aim_callback.run:
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=dpi)
            buf.seek(0)
            pil_image = Image.open(buf)
            
            aim_callback.run.track(AimImage(pil_image), name=name, epoch=epoch)
            buf.close()
        else:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            filepath = self.save_dir / f'{name}_epoch_{epoch:04d}.png'
            fig.savefig(filepath, dpi=dpi)
    
    def on_train_start(self, simulation_object):
        self.total_epochs = simulation_object.epochs
    
    def on_epoch_end(self, simulation_object, sim_dict, loss_values, epoch):
        should_plot = False
        if self.plot_frequency is not None:
            should_plot = (epoch % self.plot_frequency == 0) or (epoch == self.total_epochs - 1)
        else:
            if self.total_epochs is not None:
                should_plot = (epoch % (self.total_epochs // 4) == 0) or (epoch == self.total_epochs - 1)
        
        if not should_plot:
            return
        
        coeff_grid = sim_dict.get('coeff_grid', None)
        if coeff_grid is None:
            return
        
        trajectories = sim_dict['trajectories']
        spatial_dimensions = simulation_object.spatial_dimensions
        time_steps = simulation_object.time_steps
        spatial_bounds = simulation_object.mcmc_starting_spatial_bounds
        bit_locations = simulation_object.loss.bit_locations
        potential_obj = simulation_object.potential
        
        for dim_idx in range(spatial_dimensions):
            x_min, x_max = spatial_bounds[dim_idx, 0].item(), spatial_bounds[dim_idx, 1].item()
            x_grid = torch.linspace(x_min, x_max, self.spatial_resolution, device=simulation_object.device)
            t_indices = torch.arange(time_steps, device=simulation_object.device)
            
            potential_values = torch.zeros(self.spatial_resolution, time_steps)
            for t_idx in range(time_steps):
                coeff_slice = coeff_grid[:, t_idx]
                potential_values[:, t_idx] = potential_obj.potential_value(x_grid, coeff_slice).cpu()
            potential_values = potential_values.sign() * ((potential_values.abs() + 1).log10())
            traj_positions = trajectories[:, dim_idx, :, 0]
            
            traj_min = traj_positions.min().item()
            traj_max = traj_positions.max().item()
            spatial_buffer = (traj_max - traj_min) * 0.2
            
            mask = (x_grid.cpu() >= (traj_min - spatial_buffer)) & (x_grid.cpu() <= (traj_max + spatial_buffer))
            relevant_potential = potential_values[mask, :]
            
            v_min = torch.quantile(relevant_potential, 0.05).item()
            v_max = torch.quantile(relevant_potential, 0.95).item()
            start_positions = traj_positions[:, 0]
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
                          extent=[x_min, x_max, 0, time_steps], cmap='plasma', 
                          vmin=v_min, vmax=v_max)
            ax.set_xlabel('Position')
            ax.set_ylabel('Time')
            ax.set_title(f'Potential Landscape - Dim {dim_idx} (Epoch {epoch})')
            plt.colorbar(im, ax=ax, label='Potential Value')
            
            colors = plt.cm.tab10(np.arange(num_bits) % 10)
            for idx in sampled_indices:
                bit_class = bit_assignments[idx].item()
                traj = traj_positions[idx].cpu().numpy()
                time_array = np.linspace(0, time_steps, time_steps + 1)
                ax.plot(traj, time_array, color=colors[bit_class], alpha=0.4, linewidth=0.8)
            
            self._log_or_save_figure(fig, f'potential_landscape_dim_{dim_idx}', epoch, simulation_object)
            plt.close(fig)

class CoefficientPlotCallback(Callback):
    
    def __init__(self, save_dir='figs', plot_frequency=None):
        self.save_dir = Path(save_dir)
        self.plot_frequency = plot_frequency
        self.total_epochs = None
    
    def _log_or_save_figure(self, fig, name, epoch, simulation_object, dpi=150):
        aim_callback = next((cb for cb in simulation_object.callbacks 
                            if type(cb).__name__ == 'AimCallback'), None)
        
        if AIM_AVAILABLE and aim_callback and hasattr(aim_callback, 'run') and aim_callback.run:
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=dpi)
            buf.seek(0)
            pil_image = Image.open(buf)
            
            aim_callback.run.track(AimImage(pil_image), name=name, epoch=epoch)
            buf.close()
        else:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            filepath = self.save_dir / f'{name}_epoch_{epoch:04d}.png'
            fig.savefig(filepath, dpi=dpi)
    
    def on_train_start(self, simulation_object):
        self.total_epochs = simulation_object.epochs
    
    def on_epoch_end(self, simulation_object, sim_dict, loss_values, epoch):
        should_plot = False
        if self.plot_frequency is not None:
            should_plot = (epoch % self.plot_frequency == 0) or (epoch == self.total_epochs - 1)
        else:
            if self.total_epochs is not None:
                should_plot = (epoch % (self.total_epochs // 4) == 0) or (epoch == self.total_epochs - 1)
        
        if not should_plot:
            return
        
        coeff_grid = sim_dict.get('coeff_grid', None)
        if coeff_grid is None:
            return
        
        coeff_grid_cpu = coeff_grid.cpu().numpy()
        coefficient_count, time_steps = coeff_grid_cpu.shape
        time_array = np.arange(time_steps)
        
        if coefficient_count <= 4:
            ncols = coefficient_count
            nrows = 1
        else:
            ncols = 2
            nrows = (coefficient_count + 1) // 2
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows), squeeze=False)
        axes = axes.flatten()
        
        for coeff_idx in range(coefficient_count):
            ax = axes[coeff_idx]
            ax.plot(time_array, coeff_grid_cpu[coeff_idx, :])
            ax.set_xlabel('Time')
            ax.set_ylabel(f'Coeff {coeff_idx}')
            ax.set_title(f'Coefficient {coeff_idx}')
            ax.grid(True, alpha=0.3)
        
        for idx in range(coefficient_count, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        self._log_or_save_figure(fig, 'coefficient_evolution', epoch, simulation_object)
        plt.close(fig)

