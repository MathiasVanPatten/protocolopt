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
time_steps = 100
dt = 1/time_steps
gamma = 0.1
beta = 1.0
mass = 1.0

# Protocol parameters
spatial_dimensions = 2
num_interactions = (spatial_dimensions * (spatial_dimensions - 1)) // 2
control_dim = 3 * spatial_dimensions + num_interactions
num_coefficients_time = 16

# Define Endpoints
endpoints_list = []

# 1. Quartic (a): High walls at start/end
for _ in range(spatial_dimensions): endpoints_list.append([5.0, 5.0]) 

# 2. Quadratic (b): Standard wells
for _ in range(spatial_dimensions): endpoints_list.append([10.0, 10.0])

# 3. Linear (c): Neutral start/end
for _ in range(spatial_dimensions): endpoints_list.append([0.0, 0.0])

# 4. Interactions: Neutral start/end (Let it learn the twist!)
for _ in range(num_interactions): endpoints_list.append([0.0, 0.0])

endpoints = torch.tensor(endpoints_list, device=device)

# Training parameters
samples_per_well = 2000 
training_iterations = 400
learning_rate = 0.25
alpha = 2.0  # endpoint_weight
alpha_1 = 0.1  # var_weight
alpha_2 = 0.1  # work_weight
alpha_3 = 5e-4  # smoothness_weight

guess_list = []
target_mids = []
for _ in range(spatial_dimensions): target_mids.append(1.0)
for _ in range(spatial_dimensions): target_mids.append(2.0)
for _ in range(spatial_dimensions): target_mids.append(0.0)
for _ in range(num_interactions): target_mids.append(0.0)

for i in range(endpoints.shape[0]):
    start_val = endpoints[i, 0]
    end_val = endpoints[i, 1]
    target_mid = target_mids[i]
    
    total_points = num_coefficients_time + 2
    mid_idx = total_points // 2
    
    part1 = torch.linspace(start_val, target_mid, mid_idx + 1, device=device)
    part2 = torch.linspace(target_mid, end_val, total_points - mid_idx, device=device)

    full_path = torch.cat([part1[:-1], part2])
    
    guess_list.append(full_path[1:-1])

initial_coeff_guess = torch.stack(guess_list)
initial_coeff_guess += 0.01 * torch.randn_like(initial_coeff_guess)

# Instantiate Protocol (LinearPiecewise)
protocol = LinearPiecewise(
    control_dim=control_dim,
    time_steps=time_steps,
    knot_count=num_coefficients_time+2,
    initial_coeff_guess=initial_coeff_guess,
    endpoints=endpoints
)

# Instantiate Potential (GeneralCoupledPotential)
potential = GeneralCoupledPotential(spatial_dimensions=spatial_dimensions, compile_mode=True)

# Instantiate Simulator (EulerMaruyama)
simulator = EulerMaruyama(
    mode='underdamped',
    gamma=gamma,
    mass=mass,
    dt=dt,
    compile_mode=True
)

# Create loss function (StandardLogicGateLoss)
midpoints = torch.tensor([0.0, 0.0], device=device)

# bit_locations ordered 0 to 3 for integers 00, 01, 10, 11
# -1 is 0, 1 is 1
bit_locations = torch.tensor([
    [-1.0, -1.0], # 00 (0)
    [-1.0,  1.0], # 01 (1)
    [ 1.0, -1.0], # 10 (2)
    [ 1.0,  1.0]  # 11 (3)
], device=device)

# Strict NAND logic
truth_table = {
    0: {
        0: ['11'],
        1: ['11']
    },
    1: {
        0: ['11'],
        1: ['00', '10', '01']
    }
}

# Weight the starting bit 11 (int 3) 3 times higher than others to prevent everything just going to 11
starting_bit_weights = torch.ones(4, device=device)
starting_bit_weights[3] = 3.0

loss = StandardLogicGateLoss(
    midpoints=midpoints,
    truth_table=truth_table,
    bit_locations=bit_locations,
    endpoint_weight=alpha,
    work_weight=alpha_2,
    var_weight=alpha_1,
    smoothness_weight=alpha_3,
    exponent=2,
    starting_bit_weights=starting_bit_weights
)

# Build params dictionary for ProtocolOptimizer
params = {
    'spatial_dimensions': spatial_dimensions,
    'time_steps': time_steps,
    'samples_per_well': samples_per_well,
    'beta': beta,
    'epochs': training_iterations,
    'learning_rate': learning_rate,
    'dt': dt,
    'gamma': gamma,
    'mass': mass,
    'num_samples': 5000
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

# IMPORTANT: AimCallback must be last
# Add Aim callback if available
if AIM_AVAILABLE:
    aim_callback = AimCallback(
        experiment_name='nand_training',
        log_system_params=False
    )
    callbacks.append(aim_callback)

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
    epochs=training_iterations,
    learning_rate=learning_rate,
    callbacks=callbacks
)

if __name__ == '__main__':
    print("Starting NAND training...")
    simulation.train()
    print("Training complete!")
