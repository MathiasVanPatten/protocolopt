#definition of the free paramters, defintion of the static setup
import torch
from tqdm import tqdm
from typing import Dict, Any, List, Optional, Union
from .config import SimulationConfig
from ..utils import logger
from .potential import Potential
from .simulator import Simulator
from .loss import Loss
from .protocol import Protocol
from .sampling import InitialConditionGenerator
from .callback import Callback

class Simulation:
    """The main orchestrator for running simulations and training protocols."""

    def __init__(
        self,
        potential: Potential,
        simulator: Simulator,
        loss: Loss,
        protocol: Protocol,
        initial_condition_generator: InitialConditionGenerator,
        params: Union[SimulationConfig, Dict[str, Any]],
        callbacks: List[Callback] = []
    ) -> None:
        """Initializes the Simulation.

        Args:
            potential: The potential energy landscape.
            simulator: The simulation engine (e.g., EulerMaruyama).
            loss: The loss function to minimize.
            protocol: The time-dependent protocol model.
            initial_condition_generator: Generator for starting states.
            params: Configuration parameters. Can be a dict or SimulationConfig object.
            callbacks: List of callbacks to run during training.
        """
        self.potential = potential
        self.simulator = simulator
        self.loss = loss
        self.protocol = protocol
        self.init_cond_generator = initial_condition_generator
        self.device = getattr(self.protocol, 'device', torch.device('cpu'))
        self.callbacks = callbacks

        if isinstance(params, dict):
            self.config = SimulationConfig.from_dict(params)
        else:
            self.config = params

        self.spatial_dimensions = self.config.spatial_dimensions
        self.time_steps = self.config.time_steps
        self.epochs = self.config.epochs
        self.learning_rate = self.config.learning_rate
        self.beta = self.config.beta
        self.grad_clip_max_norm = self.config.grad_clip_max_norm

        # Simulator params override config if present in simulator
        self.gamma = getattr(self.simulator, 'gamma', self.config.gamma)
        self.dt = getattr(self.simulator, 'dt', self.config.dt)
        if self.dt is None:
             self.dt = 1.0 / self.time_steps

        self.noise_sigma = torch.sqrt(torch.tensor(2 * self.gamma / self.beta))

        if self.protocol.fixed_starting:
            self.init_cond_generator.generate_initial_conditions(self.potential, self.protocol, self.loss)

    def simulate(
        self,
        manual_inital_pos: Optional[torch.Tensor] = None,
        manual_initial_vel: Optional[torch.Tensor] = None,
        manual_noise: Optional[torch.Tensor] = None,
        debug_print: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Runs a single simulation batch.

        Args:
            manual_inital_pos: Optional override for initial positions.
            manual_initial_vel: Optional override for initial velocities.
            manual_noise: Optional override for noise.
            debug_print: Whether to print debug info from simulator.

        Returns:
            A tuple of (trajectories, potential_val, malliavian_weight, coeff_grid).
        """
        initial_pos, initial_vel, noise = self.init_cond_generator.generate_initial_conditions(self.potential, self.protocol, self.loss)

        if manual_inital_pos is not None:
            initial_pos = manual_inital_pos
        if manual_initial_vel is not None:
            initial_vel = manual_initial_vel
        if manual_noise is not None:
            noise = manual_noise

        coeff_grid = self.protocol.get_coeff_grid()

        trajectories, potential_val, malliavian_weight = self.simulator.make_trajectories(
            self.potential,
            initial_pos,
            initial_vel,
            self.time_steps,
            noise,
            self.noise_sigma,
            coeff_grid,
            debug_print = debug_print
        )

        return trajectories, potential_val, malliavian_weight, coeff_grid

    def _setup_optimizer(self, optimizer_class = torch.optim.Adam):
        self.optimizer = optimizer_class(self.protocol.trainable_params(), lr=self.learning_rate)

    def train(self) -> None:
        """Runs the training loop."""
        self._setup_optimizer()

        for callback in self.callbacks:
            callback.on_train_start(self)

        with tqdm(range(self.epochs), desc='Training') as pbar:
            for epoch in pbar:

                for callback in self.callbacks:
                    callback.on_epoch_start(self, epoch)

                self.optimizer.zero_grad()

                trajectories, potential_val, malliavian_weight, coeff_grid = self.simulate()

                total_loss, per_traj_loss = self.loss.compute_FRR_gradient( self.potential,
                    potential_val, trajectories, malliavian_weight,
                    coeff_grid, self.dt
                )

                total_loss.backward()

                if self.grad_clip_max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.protocol.trainable_params(), self.grad_clip_max_norm)

                self.optimizer.step()

                pbar.set_postfix({'loss': total_loss.item()})

                sim_dict_detached = {
                    'trajectories': trajectories.detach(),
                    'potential': potential_val.detach(),
                    'malliavian_weight': malliavian_weight.detach(),
                    'coeff_grid': coeff_grid.detach()
                }

                for callback in self.callbacks:
                    callback.on_epoch_end(self, sim_dict_detached, per_traj_loss, epoch)

        # Reconstruct sim_dict for train_end callback
        sim_dict = {
            'trajectories': trajectories,
            'potential': potential_val,
            'malliavian_weight': malliavian_weight
        }
        for callback in self.callbacks:
            callback.on_train_end(self, sim_dict, coeff_grid, epoch)

        logger.info('Training complete')
