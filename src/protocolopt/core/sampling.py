from abc import ABC, abstractmethod
import torch

class InitialConditionGenerator(ABC):
    @abstractmethod
    def generate_initial_conditions(self, potential, potential_model, loss):
        #you should return initial positions, initial velocities, and noise
        pass
