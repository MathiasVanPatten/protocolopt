#definition of the free paramters, defintion of the static setup
import torch
from tqdm import tqdm
from typing import Dict, Any, List, Optional, Union
from ..utils import logger
from .potential import Potential
from .simulator import Simulator
from .loss import Loss
from .protocol import Protocol
from .sampling import InitialConditionGenerator
from .callback import Callback

class ProtocolOptimizer:
    """The main orchestrator for running simulations and training protocols."""

    def __init__(
        self,
        potential: Potential,
        simulator: Simulator,
        loss: Loss,
        protocol: Protocol,
        initial_condition_generator: InitialConditionGenerator,
        callbacks: List[Callback] = [],
        epochs: int = 100,
        learning_rate: float = 0.001,
        grad_clip_max_norm: float = 1.0,
        optimizer_class: Any = torch.optim.Adam,
        optimizer_kwargs: Dict[str, Any] = {},
        scheduler_class: Any = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
        scheduler_kwargs: Dict[str, Any] = {},
        scheduler_restart_decay: float = 1.0
    ) -> None:
        """Initializes the ProtocolOptimizer.

        Args:
            potential: The potential energy landscape.
            simulator: The simulation engine (e.g., EulerMaruyama).
            loss: The loss function to minimize.
            protocol: The time-dependent protocol model.
            initial_condition_generator: Generator for starting states.
            callbacks: List of callbacks to run during training.
            epochs: Number of training epochs.
            learning_rate: Learning rate for the optimizer.
            grad_clip_max_norm: Maximum norm for gradient clipping.
            optimizer_class: The optimizer class to use.
            optimizer_kwargs: Additional arguments for the optimizer.
            scheduler_class: The scheduler class to use.
            scheduler_kwargs: Additional arguments for the scheduler. If CosineAnnealingWarmRestarts 
                is used and T_0 is not specified, it defaults to epochs.
            scheduler_restart_decay: Decay factor for peak LR on restarts (only for 
                CosineAnnealingWarmRestarts). Defaults to 1.0 (no decay). LR multiplied by this 
                factor on each restart.
        """
        self.potential = potential
        self.simulator = simulator
        self.loss = loss
        self.protocol = protocol
        self.init_cond_generator = initial_condition_generator
        self.callbacks = callbacks
        self.device = getattr(self.protocol, 'device', torch.device('cpu'))
        
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.grad_clip_max_norm = grad_clip_max_norm
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs
        self.scheduler_class = scheduler_class
        self.scheduler_kwargs = scheduler_kwargs
        self.scheduler_restart_decay = scheduler_restart_decay
        self.optimizer = None
        self.scheduler = None

    def simulate(
        self,
        manual_initial_pos: Optional[torch.Tensor] = None,
        manual_initial_vel: Optional[torch.Tensor] = None,
        manual_noise: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Runs a single simulation batch.

        Args:
            manual_initial_pos: Optional override for initial positions.
            manual_initial_vel: Optional override for initial velocities.
            manual_noise: Optional override for noise.

        Returns:
            A tuple of (microstate_paths, potential_val, malliavin_weight, dw_tensor, protocol_tensor).
        """
        initial_pos, initial_vel, noise = self.init_cond_generator.generate_initial_conditions(self.potential, self.protocol, self.loss)

        if manual_initial_pos is not None:
            initial_pos = manual_initial_pos
        if manual_initial_vel is not None:
            initial_vel = manual_initial_vel
        if manual_noise is not None:
            noise = manual_noise

        protocol_tensor = self.protocol.get_protocol_tensor()

        microstate_paths, potential_val, malliavin_weight, dw_tensor = self.simulator.make_microstate_paths(
            self.potential,
            initial_pos,
            initial_vel,
            self.protocol.time_steps,
            noise,
            protocol_tensor
        )

        return microstate_paths, potential_val, malliavin_weight, dw_tensor, protocol_tensor

    def _setup_optimizer(self):
        self.optimizer = self.optimizer_class(self.protocol.trainable_params(), lr=self.learning_rate, **self.optimizer_kwargs)
        
        if self.scheduler_class is not None:
            if self.scheduler_class == torch.optim.lr_scheduler.CosineAnnealingWarmRestarts:
                if 'T_0' not in self.scheduler_kwargs:
                    self.scheduler_kwargs['T_0'] = self.epochs
            
            self.scheduler = self.scheduler_class(self.optimizer, **self.scheduler_kwargs)
        else:
            self.scheduler = None

    def train(self) -> None:
        """Runs the training loop.

        Mutates the `protocol` object in-place.
        """
        self._setup_optimizer()

        for callback in self.callbacks:
            callback.on_train_start(self)

        with tqdm(range(self.epochs), desc='Training') as pbar:
            for epoch in pbar:

                for callback in self.callbacks:
                    callback.on_epoch_start(self, epoch)

                self.optimizer.zero_grad()

                microstate_paths, potential_val, malliavin_weight, dw_tensor, protocol_tensor = self.simulate()

                dt = getattr(self.simulator, 'dt', None)
                if dt is None:
                    dt = 1.0 / self.protocol.time_steps

                total_loss, per_traj_loss = self.loss.compute_FRR_gradient( self.potential,
                    potential_val, microstate_paths, malliavin_weight, dw_tensor,
                    protocol_tensor, dt
                )

                total_loss.backward()

                if self.grad_clip_max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.protocol.trainable_params(), self.grad_clip_max_norm)

                self.optimizer.step()
                
                if self.scheduler is not None:
                    self.scheduler.step()
                    
                    if (self.scheduler_restart_decay < 1.0 and 
                        isinstance(self.scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts)):
                        if self.scheduler.T_cur == 0:
                            for param_group in self.optimizer.param_groups:
                                param_group['lr'] *= self.scheduler_restart_decay
                            self.scheduler.base_lrs = [pg['lr'] for pg in self.optimizer.param_groups]

                pbar.set_postfix({'loss': total_loss.item()})

                sim_dict_detached = {
                    'microstate_paths': microstate_paths.detach(),
                    'potential': potential_val.detach(),
                    'malliavin_weight': malliavin_weight.detach(),
                    'dw_tensor': dw_tensor.detach(),
                    'protocol_tensor': protocol_tensor.detach()
                }

                for callback in self.callbacks:
                    callback.on_epoch_end(self, sim_dict_detached, per_traj_loss, epoch)

        for callback in self.callbacks:
            callback.on_train_end(self, sim_dict_detached, protocol_tensor, epoch)

        logger.info('Training complete')
