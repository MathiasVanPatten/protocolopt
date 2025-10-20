from simulation import Simulation
from potential import QuarticPotential
import torch


if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

if __name__ == '__main__':
    #order potential model
    #potential
    #sim engine
    #simulation
    #then train
    experiment_potential = QuarticPotential(torch.tensor([[5, -10], [5, -10]], device=device))
    