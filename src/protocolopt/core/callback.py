from typing import Dict, Any, TYPE_CHECKING, Optional
import torch
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from pathlib import Path

if TYPE_CHECKING:
    from .protocol_optimizer import ProtocolOptimizer

class Callback:
    """Base class for callbacks."""

    def on_train_start(self, optimizer_object: "ProtocolOptimizer") -> None:
        """Called at the beginning of training.

        Args:
            optimizer_object: The main ProtocolOptimizer instance.
        """
        pass

    def on_epoch_start(self, optimizer_object: "ProtocolOptimizer", epoch: int) -> None:
        """Called at the start of each epoch.

        Args:
            optimizer_object: The main ProtocolOptimizer instance.
            epoch: Current epoch index.
        """
        pass

    def on_epoch_end(self, optimizer_object: "ProtocolOptimizer", sim_dict: Dict[str, Any], loss_values: torch.Tensor, epoch: int) -> None:
        """Called at the end of each epoch.

        Args:
            optimizer_object: The main ProtocolOptimizer instance.
            sim_dict: Dictionary containing simulation results (microstate_paths, etc.).
            loss_values: Tensor of loss values for the batch.
            epoch: Current epoch index.
        """
        pass

    def on_train_end(self, optimizer_object: "ProtocolOptimizer", sim_dict: Dict[str, Any], protocol_tensor: torch.Tensor, epoch: int) -> None:
        """Called at the end of training.

        Args:
            optimizer_object: The main ProtocolOptimizer instance.
            sim_dict: Dictionary containing simulation results.
            protocol_tensor: Final protocol tensor.
            epoch: Final epoch index.
        """
        pass

class BasePlottingCallback(Callback, ABC):
    """Abstract base callback for plotting that handles schedule management and saving/logging figures.
    
    Subclasses must implement the `_plot` method.
    
    Attributes:
        save_dir (Path): Directory where plots will be saved if Aim is not available
        plot_frequency (Optional[int]): Frequency of plotting in epochs.
        total_epochs (Optional[int]): Total number of training epochs.
    """
    
    def on_train_start(self, optimizer_object: "ProtocolOptimizer") -> None:
        """Captures total epochs at the start of training.
        
        Args:
            optimizer_object: The main ProtocolOptimizer instance.
        """
        self.total_epochs = optimizer_object.epochs

    def should_plot(self, epoch: int) -> bool:
        """Determines if plotting should occur for the given epoch.
        
        Checks against `plot_frequency` if set, otherwise defaults to plotting
        at 0, 25%, 50%, 75%, and 100% of total epochs.
        
        Args:
            epoch: Current epoch index.
            
        Returns:
            bool: True if plotting should occur, False otherwise.
        """
        if getattr(self, 'plot_frequency', None) is not None:
            return (epoch % self.plot_frequency == 0)
        
        if getattr(self, 'total_epochs', None) is not None:
            div = max(1, self.total_epochs // 4)
            return (epoch % div == 0)
        
        return False

    def on_epoch_end(self, optimizer_object: "ProtocolOptimizer", sim_dict: Dict[str, Any], loss_values: torch.Tensor, epoch: int) -> None:
        """Triggers plotting at the end of an epoch if scheduled.
        
        Args:
            optimizer_object: The main ProtocolOptimizer instance.
            sim_dict: Simulation results.
            loss_values: Loss values.
            epoch: Current epoch index.
        """
        if self.should_plot(epoch):
            self._plot(optimizer_object, sim_dict, epoch)
            
    def on_train_end(self, optimizer_object: "ProtocolOptimizer", sim_dict: Dict[str, Any], protocol_tensor: torch.Tensor, epoch: int) -> None:
        """Ensures a final plot is generated at the end of training.
        
        Args:
            optimizer_object: The main ProtocolOptimizer instance.
            sim_dict: Simulation results.
            protocol_tensor: Final protocol tensor.
            epoch: Final epoch index.
        """
        if not self.should_plot(epoch):
             self._plot(optimizer_object, sim_dict, epoch)
             
    @abstractmethod
    def _plot(self, optimizer_object: "ProtocolOptimizer", sim_dict: Dict[str, Any], epoch: int) -> None:
        """Abstract method to implement specific plotting logic.
        
        Args:
            optimizer_object: The main ProtocolOptimizer instance.
            sim_dict: Simulation results containing microstate paths, protocol tensor, etc.
            epoch: Current epoch index.
            
        """
    
    def _log_or_save_figure(self, fig: plt.Figure, name: str, epoch: int, optimizer_object: "ProtocolOptimizer", context: Dict[str, Any] = {}, dpi: int = 150) -> None:
        """Helper to save figure to disk or log to tracking service (e.g., Aim).
        
        Args:
            fig: The matplotlib figure to save.
            name: Base name for the file/log.
            epoch: Current epoch.
            optimizer_object: Optimizer instance (to check for AimCallback).
            context: Additional context for logging.
            dpi: DPI for saving the figure.
            
        Raises:
            AttributeError: If `save_dir` is not defined in the subclass and no AimCallback is found.
        """
        aim_callback = next((cb for cb in optimizer_object.callbacks
                            if type(cb).__name__ == 'AimCallback'), None)
        
        if aim_callback and hasattr(aim_callback, 'track_figure'):
            aim_callback.track_figure(fig, name, epoch, context=context, dpi=dpi)
        else:
            # Assumes self.save_dir is set by subclass
            if not hasattr(self, 'save_dir'):
                 raise AttributeError(f"{self.__class__.__name__} must define save_dir to use _log_or_save_figure locally")
            
            filepath = self.save_dir / f'{name}_epoch_{epoch:04d}.png'
            fig.savefig(filepath, dpi=dpi)
