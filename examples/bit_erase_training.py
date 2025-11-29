import torch
import math
from protocolopt.potentials import GeneralCoupledPotential
from protocolopt.protocols import LinearPiecewise
from protocolopt.simulators import EulerMaruyama
from protocolopt.losses import StandardLogicGateLoss
from protocolopt import ProtocolOptimizer
from protocolopt.sampling import LaplaceApproximation
from protocolopt.callbacks import TrajectoryPlotCallback, ConfusionMatrixCallback, PotentialLandscapePlotCallback, ProtocolPlotCallback
try:
    from protocolopt.callbacks import AimCallback
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

# ProtocolOptimizer parameters
# time_steps = 100
time_steps = 1000 #overdamped
dt = 1/time_steps
gamma = 1.0
beta = 1.0

# Protocol parameters
num_coefficients = 16
a_endpoints = [10.0, 10.0]
b_endpoints = [20.0, 20.0]
c_endpoints = [0.0, 0.0]
# Training parameters
samples_per_well = 3000
training_iterations = 250
learning_rate = 0.25
alpha = 2.0  # endpoint_weight
alpha_1 = 0.1  # var_weight
alpha_2 = 0.1  # work_weight
alpha_3 = 5e-4  # smoothness_weight

# Additional parameters
spatial_dimensions = 1
mcmc_warmup_ratio = 0.1
mcmc_starting_spatial_bounds = torch.tensor([[-5.0, 5.0]], device=device)

# Calculate centers
centers = math.sqrt(b_endpoints[0] / (2 * a_endpoints[0]))

# Create endpoints tensor
endpoints = torch.tensor([
    [a_endpoints[0], a_endpoints[1]], 
    [b_endpoints[0], b_endpoints[1]],
    [c_endpoints[0], c_endpoints[1]]
], device=device)


guess_list = []
for i in range(endpoints.shape[0]):
    start_val = endpoints[i, 0]
    end_val = endpoints[i, 1]
    
    if i == 0: #a
        target_mid = 1.0
    elif i == 1: #b
        target_mid = 2.0
    else: #c
        target_mid = 0.0
        
    total_points = num_coefficients + 2
    mid_idx = total_points // 2
    
    part1 = torch.linspace(start_val, target_mid, mid_idx + 1, device=device)
    part2 = torch.linspace(target_mid, end_val, total_points - mid_idx, device=device)

    full_path = torch.cat([part1[:-1], part2])
    
    guess_list.append(full_path[1:-1])

initial_coeff_guess = torch.stack(guess_list)

initial_coeff_guess += 0.01 * torch.randn_like(initial_coeff_guess)

# Instantiate Protocol (LinearPiecewise)
protocol = LinearPiecewise(
    control_dim=endpoints.shape[0],
    time_steps=time_steps,
    knot_count=num_coefficients+2,
    initial_coeff_guess=initial_coeff_guess,
    endpoints=endpoints
)

# Instantiate Potential (GeneralCoupledPotential)
potential = GeneralCoupledPotential(spatial_dimensions=spatial_dimensions, has_c=True, compile_mode=True)

# Instantiate Simulator (EulerMaruyama)
simulator = EulerMaruyama(
    mode='overdamped',
    gamma=gamma,
    mass=1.0,
    dt=dt,
    compile_mode=True
)

# Create loss function (StandardLogicGateLoss)
midpoints = torch.tensor([0.0], device=device)
bit_locations = torch.tensor([[-centers], [centers]], device=device)
truth_table = {0: ['0'], 1: ['0']}

loss = StandardLogicGateLoss(
    midpoints=midpoints,
    truth_table=truth_table,
    bit_locations=bit_locations,
    endpoint_weight=alpha,
    work_weight=alpha_2,
    var_weight=alpha_1,
    smoothness_weight=alpha_3,
    exponent=2
)

# Build params dictionary for ProtocolOptimizer
params = {
    'spatial_dimensions': spatial_dimensions,
    'time_steps': time_steps,
    'samples_per_well': samples_per_well,
    'mcmc_warmup_ratio': mcmc_warmup_ratio,
    'mcmc_starting_spatial_bounds': mcmc_starting_spatial_bounds,
    'mcmc_chains_per_well': 1,
    'beta': beta,
    'epochs': training_iterations,
    'learning_rate': learning_rate,
    'dt': dt,
    'gamma': gamma,
    'mass': 1.0 # default mass
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
        experiment_name='bit_erase_training',
        log_system_params=False
    )
    callbacks.append(aim_callback)

potential_landscape_callback = PotentialLandscapePlotCallback(
    save_dir='figs',
    plot_frequency=None
)
callbacks.append(potential_landscape_callback)

coefficient_callback = ProtocolPlotCallback(
    save_dir='figs',
    plot_frequency=None
)
callbacks.append(coefficient_callback)

# init_cond_generator = McmcNuts(params, device)
init_cond_generator = LaplaceApproximation(
    params=params,
    centers=bit_locations,
    device=device
)

# Instantiate ProtocolOptimizer
simulation = ProtocolOptimizer(
    potential=potential,
    simulator=simulator,
    loss=loss,
    protocol=protocol,
    initial_condition_generator=init_cond_generator,
    params=params,
    callbacks=callbacks
)

if __name__ == '__main__':
    print("Starting bitflip training...")
    simulation.train()
    print("Training complete!")
