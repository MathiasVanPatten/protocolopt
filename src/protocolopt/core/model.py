import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod

class PotentialModel(ABC):
    #potential models use any kind of differentiable math to go from a set
    #of trainable parameters to a full tensor of values to be used in the potential

    #fixed starting is a boolean, if True the starting potential is fixed and the mcmc sampler will only do warmup once
    def __init__(self, time_steps, fixed_starting) -> None:
        self.time_steps = time_steps
        self.fixed_starting = fixed_starting

    @abstractmethod
    def get_coeff_grid(self):
        #return a tensor of shape (coefficients, spatial dimensions, time)
        pass

    @abstractmethod
    def trainable_params(self):
        # return the parameters you want attached to the optimizer as a list
        pass
