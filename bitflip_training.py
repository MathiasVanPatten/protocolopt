import torch
import math
from potential import QuarticPotential
from potential_model import LinearPiecewise
from sim_engine import EulerMaruyama
from loss_classes import StandardLoss
from simulation import Simulation
from plotting_callbacks import TrajectoryPlotCallback, ConfusionMatrixCallback, PotentialLandscapePlotCallback, CoefficientPlotCallback
try:
    from aim_callback import AimCallback
    AIM_AVAILABLE = True
except ImportError:
    AIM_AVAILABLE = False
    print("Aim not available - skipping experiment tracking")

# Device configuration
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

print(f"Using device: {device}")

# Simulation parameters
time_steps = 100
dt = 1/time_steps
gamma = 0.1
beta = 1.0
mass = 1.0
# Protocol parameters
num_coefficients = 16
a_endpoints = [5.0, 5.0]
b_endpoints = [20.0, 20.0]

# Training parameters
samples_per_well = 2000 
training_iterations = 400
learning_rate = 0.025
alpha = 2.0  # endpoint_weight
alpha_1 = 0.1  # var_weight
alpha_2 = 0.1  # work_weight
alpha_3 = 5e-3  # smoothness_weight

# Additional parameters
spatial_dimensions = 1
mcmc_warmup_ratio = 0.1
mcmc_starting_spatial_bounds = torch.tensor([[-5.0, 5.0]], device=device)

# Calculate centers
centers = math.sqrt(b_endpoints[0] / (2 * a_endpoints[0]))

# Create endpoints tensor
endpoints = torch.tensor([
    [a_endpoints[0], a_endpoints[1]], 
    [b_endpoints[0], b_endpoints[1]]
], device=device)

# Create initial coefficient guess
initial_coeff_guess = torch.randn((2, num_coefficients), device=device)

# Instantiate PotentialModel (LinearPiecewise)
potential_model = LinearPiecewise(
    coefficient_count=2,
    time_steps=time_steps,
    knot_count=num_coefficients+2,
    initial_coeff_guess=initial_coeff_guess,
    endpoints=endpoints
)

# Instantiate Potential (QuarticPotential)
potential = QuarticPotential()

# Instantiate SimEngine (EulerMaruyama)
sim_engine = EulerMaruyama(
    mode='underdamped',
    gamma=gamma,
    mass=mass,
    dt=dt
)

# Create loss function (StandardLoss)
midpoints = torch.tensor([0.0], device=device)
bit_locations = torch.tensor([[-centers], [centers]], device=device)
truth_table = {0: ['1'], 1: ['0']}

loss = StandardLoss(
    midpoints=midpoints,
    truth_table=truth_table,
    bit_locations=bit_locations,
    endpoint_weight=alpha,
    work_weight=alpha_2,
    var_weight=alpha_1,
    smoothness_weight=alpha_3,
    exponent=2
)

# Build params dictionary for Simulation
params = {
    'spatial_dimensions': spatial_dimensions,
    'time_steps': time_steps,
    'samples_per_well': samples_per_well,
    'mcmc_warmup_ratio': mcmc_warmup_ratio,
    'mcmc_starting_spatial_bounds': mcmc_starting_spatial_bounds,
    'mcmc_chains_per_well': 1,
    'beta': beta,
    'epochs': training_iterations,
    'learning_rate': learning_rate
}

# Create callbacks
callbacks = []

# Add plotting callbacks
trajectory_callback = TrajectoryPlotCallback(
    save_dir='figs',
    plot_frequency=None,
    num_trajectories=100
)
callbacks.append(trajectory_callback)

confusion_matrix_callback = ConfusionMatrixCallback(
    save_dir='figs',
    plot_frequency=None
)
callbacks.append(confusion_matrix_callback)

# Add Aim callback if available
if AIM_AVAILABLE:
    aim_callback = AimCallback(
        experiment_name='bitflip_training',
        log_system_params=False
    )
    callbacks.append(aim_callback)

potential_landscape_callback = PotentialLandscapePlotCallback(
    save_dir='figs',
    plot_frequency=None
)
callbacks.append(potential_landscape_callback)

coefficient_callback = CoefficientPlotCallback(
    save_dir='figs',
    plot_frequency=None
)
callbacks.append(coefficient_callback)
# Instantiate Simulation
simulation = Simulation(
    potential=potential,
    sim_engine=sim_engine,
    loss=loss,
    potential_model=potential_model,
    params=params,
    callbacks=callbacks
)

if __name__ == '__main__':
    print("Starting bitflip training...")
    simulation.train()
    print("Training complete!")
