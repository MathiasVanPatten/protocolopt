#definition of the free paramters, defintion of the static setup
import torch
from tqdm import tqdm


class Simulation:
    def __init__(self, potential, simulator, loss, protocol, initial_condition_generator, params, callbacks = []) -> None:
        self.potential = potential
        self.simulator = simulator
        self.loss = loss
        self.protocol = protocol
        self.init_cond_generator = initial_condition_generator
        self.device = self.protocol.device
        self.spatial_dimensions = params.get('spatial_dimensions', 1)
        self.callbacks = callbacks
        self.time_steps = params.get('time_steps', 1000)

        self.beta = params.get('beta', 1.0)
        self.gamma = self.simulator.gamma
        self.dt = self.simulator.dt
        #NOTE WHEN NOISE IS GIVEN TO SIMULATE THE NOISE SIGMA WILL STILL BE SET AS IT WAS DURING TRAINING, POSSIBLY AN ISSUE

        self.noise_sigma = torch.sqrt(torch.tensor(2 * self.gamma / self.beta))
        self.epochs = params.get('epochs', 100)
        self.learning_rate = params.get('learning_rate', 0.001)
        self.grad_clip_max_norm = params.get('grad_clip_max_norm', 1.0)
        if self.protocol.fixed_starting:
            self.init_cond_generator.generate_initial_conditions(self.potential, self.protocol, self.loss)

    def simulate(self, manual_inital_pos = None, manual_initial_vel = None, manual_noise = None, DEBUG_PRINT = False):
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
            DEBUG_PRINT = DEBUG_PRINT
        )

        return trajectories, potential_val, malliavian_weight, coeff_grid

    def _setup_optimizer(self, optimizer_class = torch.optim.Adam):
        self.optimizer = optimizer_class(self.protocol.trainable_params(), lr=self.learning_rate)

    def train(self):
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
        
        print('Training complete')
