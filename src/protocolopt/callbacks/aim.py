try:
    from aim import Run
    AIM_AVAILABLE = True
except ImportError:
    AIM_AVAILABLE = False
    print("Warning: Aim not installed. Install with 'pip install aim' to use AimCallback")

from ..core.callback import Callback
from typing import Optional, Dict, Any, TYPE_CHECKING
import torch

if TYPE_CHECKING:
    from ..core.protocol_optimizer import ProtocolOptimizer

class AimCallback(Callback):
    """Callback for experiment tracking with Aim"""
    
    def __init__(self, experiment_name: Optional[str] = None, repo_path: Optional[str] = None, log_system_params: bool = True,
                 capture_terminal_logs: bool = False, run_hash: Optional[str] = None):
        """
        Args:
            experiment_name: Name for the experiment
            repo_path: Path to Aim repository (defaults to current directory)
            log_system_params: Whether to log system parameters (GPU, CPU, etc.)
            capture_terminal_logs: Whether to capture terminal output
            run_hash: Optional hash to continue a previous run
        """
        if not AIM_AVAILABLE:
            raise ImportError("Aim is not installed. Install with 'pip install aim'")
        
        self.experiment_name = experiment_name
        self.repo_path = repo_path
        self.log_system_params = log_system_params
        self.capture_terminal_logs = capture_terminal_logs
        self.run_hash = run_hash
        self.run = None
    
    def on_train_start(self, optimizer_object: "ProtocolOptimizer") -> None:
        """Initialize Aim Run and log hyperparameters"""
        # Initialize Aim run
        self.run = Run(
            repo=self.repo_path,
            experiment=self.experiment_name,
            log_system_params=self.log_system_params,
            capture_terminal_logs=self.capture_terminal_logs,
            run_hash=self.run_hash
        )
        
        # Log hyperparameters
        self.run['hparams'] = {
            'learning_rate': optimizer_object.learning_rate,
            'epochs': optimizer_object.epochs,
            'time_steps': optimizer_object.time_steps,
            'spatial_dimensions': optimizer_object.spatial_dimensions,
            'beta': optimizer_object.beta,
            'gamma': optimizer_object.gamma,
            'dt': optimizer_object.dt,
            'noise_sigma': optimizer_object.noise_sigma.item() if torch.is_tensor(optimizer_object.noise_sigma) else optimizer_object.noise_sigma,
        }
        
        # Add MCMC parameters from generator if available
        if hasattr(optimizer_object, 'init_cond_generator'):
            ic_gen = optimizer_object.init_cond_generator
            self.run['hparams']['ic_gen_type'] = type(ic_gen).__name__

            if hasattr(ic_gen, 'mcmc_warmup_ratio'):
                self.run['hparams']['mcmc_warmup_ratio'] = ic_gen.mcmc_warmup_ratio
                
            if hasattr(ic_gen, 'samples_per_well') and ic_gen.samples_per_well is not None:
                self.run['hparams']['samples_per_well'] = ic_gen.samples_per_well
                if hasattr(ic_gen, 'mcmc_chains_per_well'):
                    self.run['hparams']['mcmc_chains_per_well'] = ic_gen.mcmc_chains_per_well
                self.run['hparams']['sampling_mode'] = 'per_well'
            elif hasattr(ic_gen, 'mcmc_num_samples'):
                self.run['hparams']['mcmc_num_samples'] = ic_gen.mcmc_num_samples
                self.run['hparams']['sampling_mode'] = 'global'
            elif hasattr(ic_gen, 'num_samples'):
                 self.run['hparams']['num_samples'] = ic_gen.num_samples

            # ConditionalFlow specific
            if hasattr(ic_gen, 'flow_epochs'):
                self.run['hparams']['flow_epochs'] = ic_gen.flow_epochs
            if hasattr(ic_gen, 'flow_batch_size'):
                self.run['hparams']['flow_batch_size'] = ic_gen.flow_batch_size
            if hasattr(ic_gen, 'flow_training_samples_per_well'):
                self.run['hparams']['flow_training_samples_per_well'] = ic_gen.flow_training_samples_per_well
                
            # Laplace specific
            if hasattr(ic_gen, 'centers'):
                 self.run['hparams']['num_centers'] = ic_gen.centers.shape[0] if torch.is_tensor(ic_gen.centers) else len(ic_gen.centers)
        
        # Log loss function parameters if available
        if hasattr(optimizer_object.loss, 'endpoint_weight'):
            self.run['hparams']['endpoint_weight'] = optimizer_object.loss.endpoint_weight
        if hasattr(optimizer_object.loss, 'work_weight'):
            self.run['hparams']['work_weight'] = optimizer_object.loss.work_weight
        if hasattr(optimizer_object.loss, 'var_weight'):
            self.run['hparams']['var_weight'] = optimizer_object.loss.var_weight
        if hasattr(optimizer_object.loss, 'smoothness_weight'):
            self.run['hparams']['smoothness_weight'] = optimizer_object.loss.smoothness_weight
        
        # Log potential model info
        self.run['hparams']['protocol_type'] = type(optimizer_object.protocol).__name__
        self.run['hparams']['potential_type'] = type(optimizer_object.potential).__name__
        self.run['hparams']['simulator_type'] = type(optimizer_object.simulator).__name__
    
    def on_epoch_end(self, optimizer_object: "ProtocolOptimizer", sim_dict: Dict[str, Any], loss_values: torch.Tensor, epoch: int) -> None:
        """Log metrics for each epoch"""
        if self.run is None:
            return
        
        # Log loss metrics
        mean_loss = loss_values.mean().item()
        std_loss = loss_values.std().item()
        min_loss = loss_values.min().item()
        max_loss = loss_values.max().item()
        
        self.run.track(mean_loss, name='loss/mean', epoch=epoch)
        self.run.track(std_loss, name='loss/std', epoch=epoch)
        self.run.track(min_loss, name='loss/min', epoch=epoch)
        self.run.track(max_loss, name='loss/max', epoch=epoch)
        
        # Compute and track additional metrics
        loss_object = optimizer_object.loss
        
        microstate_paths_detached = sim_dict['microstate_paths']
        potential_detached = sim_dict['potential']
        
        # 1. Binary error rate (% of paths ending in invalid states)
        if hasattr(loss_object, 'compute_binary_error_rate'):
            binary_error_rate = loss_object.compute_binary_error_rate(microstate_paths_detached)
            self.run.track(binary_error_rate, name='metrics/binary_error_rate', epoch=epoch)
        
        # 2. Individual loss components
        if hasattr(loss_object, 'compute_loss_components'):
            loss_components = loss_object.compute_loss_components(
                potential_detached, microstate_paths_detached,
                sim_dict['protocol_tensor'], optimizer_object.dt
            )
            
            # Track each component
            endpoint_mean = loss_components['endpoint_loss'].mean().item()
            work_mean = loss_components['work_loss'].mean().item()
            variance_mean = loss_components['variance_loss'].mean().item()
            smoothness_mean = loss_components['smoothness_loss'].mean().item()
            
            self.run.track(endpoint_mean, name='loss_components/endpoint', epoch=epoch)
            self.run.track(work_mean, name='loss_components/work', epoch=epoch)
            self.run.track(variance_mean, name='loss_components/variance', epoch=epoch)
            self.run.track(smoothness_mean, name='loss_components/smoothness', epoch=epoch)
        
        # 3. Average work (mean delta V across paths)
        # Work is computed as sum of (V[t+1] - V[t]) over time for each path
        work_per_path = (potential_detached[..., 1:] - potential_detached[..., :-1]).sum(dim=-1)
        average_work = work_per_path.mean().item()
        self.run.track(average_work, name='metrics/average_work', epoch=epoch)
    
    def on_train_end(self, optimizer_object: "ProtocolOptimizer", sim_dict: Dict[str, Any], protocol_tensor: torch.Tensor, epoch: int) -> None:
        """Finalize and close Aim run"""
        if self.run is None:
            return
        
        # Mark training as completed
        self.run['status'] = 'completed'
        self.run['final_epoch'] = epoch
        
        # Close the run
        self.run.close()
        
        print(f"Aim tracking completed. Run hash: {self.run.hash}")
