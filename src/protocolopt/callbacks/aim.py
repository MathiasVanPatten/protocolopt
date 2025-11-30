try:
    from aim import Run, Image as AimImage
    AIM_AVAILABLE = True
except ImportError:
    AIM_AVAILABLE = False
    print("Warning: Aim not installed. Install with 'pip install aim' to use AimCallback")

from ..core.callback import Callback
import torch
from typing import Dict, Any, TYPE_CHECKING
import io

if TYPE_CHECKING:
    from ..core.protocol_optimizer import ProtocolOptimizer
    import matplotlib.pyplot as plt

class AimCallback(Callback):
    """Callback for experiment tracking with Aim"""
    
    def __init__(self, experiment_name=None, repo_path=None, log_system_params=True, 
                 capture_terminal_logs=False, run_hash=None):
        """
        Args:
            experiment_name: Name for the experiment
            repo_path: Path to Aim repository (defaults to current directory)
            log_system_params: Whether to log system parameters (Installed packages, git info, env vars, NOT GPU/CPU info which is always logged)
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

    def _sanitize_value(self, value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.item() if value.numel() == 1 else value.tolist()
        elif isinstance(value, dict):
            return {k: self._sanitize_value(v) for k, v in value.items()}
        elif isinstance(value, (list, tuple)):
            return [self._sanitize_value(v) for v in value]
        return value

    def on_train_start(self, optimizer: "ProtocolOptimizer") -> None:
        self.run = Run(
            repo=self.repo_path,
            experiment=self.experiment_name,
            run_hash=self.run_hash,
            log_system_params=self.log_system_params,
            capture_terminal_logs=self.capture_terminal_logs,
        )

        config = {}

        components = {
            'optimizer': optimizer,
            'potential': optimizer.potential,
            'simulator': optimizer.simulator,
            'loss': optimizer.loss,
            'protocol': optimizer.protocol,
            'initial_condition_generator': optimizer.init_cond_generator,
        }

        for name, comp in components.items():
            if hasattr(comp, 'hparams'):
                config[name] = self._sanitize_value(comp.hparams)
            elif hasattr(comp, '__dict__'):
                config[name] = {k: self._sanitize_value(v) for k, v in comp.__dict__.items() 
                              if isinstance(v, (int, float, str, bool, torch.Tensor))}
        
        def get_class_name(obj):
            if isinstance(obj, type):
                return obj.__name__
            return obj.__class__.__name__
            
        config['optimizer'] = config['optimizer'] | {
            'optimizer': get_class_name(optimizer.optimizer_class),
            'optimizer_kwargs': self._sanitize_value(optimizer.optimizer_kwargs),
            'scheduler': get_class_name(optimizer.scheduler_class) if optimizer.scheduler_class else None,
            'scheduler_kwargs': self._sanitize_value(optimizer.scheduler_kwargs),
            'scheduler_restart_decay': optimizer.scheduler_restart_decay
        }
        self.run['hparams'] = config

    def on_epoch_end(self, optimizer: "ProtocolOptimizer", sim_dict: Dict[str, Any], 
                     loss_values: torch.Tensor, epoch: int) -> None:
        
        self.run.track(loss_values.mean().item(), name='loss', epoch=epoch, context={'agg': 'mean'})
        self.run.track(loss_values.std().item(), name='loss', epoch=epoch, context={'agg': 'std'})
        self.run.track(loss_values.min().item(), name='loss', epoch=epoch, context={'agg': 'min'})
        self.run.track(loss_values.max().item(), name='loss', epoch=epoch, context={'agg': 'max'})

        if hasattr(optimizer.loss, 'log_components'):
            metrics = optimizer.loss.log_components(
                sim_dict['potential'], 
                sim_dict['microstate_paths'],
                sim_dict['protocol_tensor'], 
                optimizer.simulator.dt
            )
            
            for metric_name, value in metrics.items():
                val = value.mean().item() if isinstance(value, torch.Tensor) else value
                
                ctx = {'group': 'loss_components'} if 'loss' in metric_name else {'group': 'metrics'}
                
                self.run.track(val, name=metric_name, epoch=epoch, context=ctx)

    def track_figure(self, figure: Any, name: str, epoch: int, context: Dict[str, Any] = {}, dpi: int = 150) -> None:
        """
        Track a matplotlib figure as an Aim Image.
        
        Args:
            figure: Matplotlib figure to track
            name: Name of the image
            epoch: Current epoch
            context: Optional context dictionary
            dpi: DPI for the saved image
        """
        if not self.run:
            return

        aim_image = AimImage(figure)
        self.run.track(aim_image, name=name, epoch=epoch, context=context)

    def on_train_end(self, *args, **kwargs) -> None:
        if self.run:
            self.run.close()
