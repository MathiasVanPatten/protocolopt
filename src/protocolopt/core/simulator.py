from abc import ABC, abstractmethod

class Simulator(ABC):
    @abstractmethod
    def make_trajectories(self, potential, initial_pos, initial_vel, time_steps, noise, noise_sigma, coeff_grid, DEBUG_PRINT = False):
        # Must return (trajectories, potential, malliavian_weight)
        pass
