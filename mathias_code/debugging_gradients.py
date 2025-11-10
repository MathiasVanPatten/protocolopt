"""
test_autograd.py - Minimal script to understand gradient flow through your architecture

Run this to see:
1. How gradients flow from trainable_params -> coeff_grid -> potential -> loss
2. What happens with .detach() at different points
3. Manual gradient computation vs autograd
"""

import torch
import torch.nn.functional as F
from potential import QuarticPotential
from potential_model import LinearPiecewise

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")

# ============================================================================
# SETUP: Same as your actual training
# ============================================================================
time_steps = 50
num_coefficients = 5  # a and b quartic coefficients
spatial_dimensions = 1
knot_count = 3  # Internal knots between endpoints

# Create potential model with endpoints
a_endpoints = torch.tensor([[5.0, 5.0]], device=device)  
b_endpoints = torch.tensor([[10.0, 10.0]], device=device)
endpoints = torch.cat([a_endpoints, b_endpoints], dim=0)  # Shape: (2, 2)

initial_guess = torch.randn(2, knot_count, device=device) * 0.5 + 7.0

protocol_model = LinearPiecewise(
    coefficient_count=2,  # a and b
    time_steps=time_steps,
    knot_count=knot_count+2,
    initial_coeff_guess=initial_guess,
    endpoints=endpoints
)


potential = QuarticPotential()

# ============================================================================
# TEST 1: Check gradient flow with simple loss
# ============================================================================
print("="*70)
print("TEST 1: Basic gradient flow check")
print("="*70)

trainable_params = protocol_model.trainable_params()[0]
print(f"Trainable params shape: {trainable_params.shape}")
print(f"Trainable params requires_grad: {trainable_params.requires_grad}")
print(f"Trainable params is_leaf: {trainable_params.is_leaf}\n")

# Get coefficient grid
coeff_grid = protocol_model.get_coeff_grid()
print(f"Coeff grid shape: {coeff_grid.shape}")
print(f"Coeff grid requires_grad: {coeff_grid.requires_grad}")
print(f"Coeff grid is_leaf: {coeff_grid.is_leaf}")
print(f"Coeff grid grad_fn: {coeff_grid.grad_fn}\n")

# Sample time-varying positions (simulating evolving trajectories)
num_samples = 100
# Initial positions
initial_positions = torch.randn(num_samples, spatial_dimensions, device=device) * 2.0 + 5.0

# Create positions that evolve over time: shape (num_samples, spatial_dimensions, time_steps)
positions_over_time = torch.zeros(num_samples, spatial_dimensions, time_steps, device=device)
# Add a deterministic drift so positions change over time (no noise)
time_drift = torch.linspace(0, 3, time_steps, device=device).view(1, 1, time_steps)
positions_over_time = initial_positions.unsqueeze(-1) + time_drift
positions_over_time = positions_over_time.detach()  # Detached like in your sim_engine

print(f"Positions over time shape: {positions_over_time.shape}")
print(f"Positions requires_grad: {positions_over_time.requires_grad}")
print(f"Position range: t=0: [{positions_over_time[..., 0].min().item():.2f}, {positions_over_time[..., 0].max().item():.2f}], "
      f"t={time_steps-1}: [{positions_over_time[..., -1].min().item():.2f}, {positions_over_time[..., -1].max().item():.2f}]\n")

# Compute potential across ALL timesteps using time-varying positions
V_all = torch.zeros(num_samples, spatial_dimensions, time_steps, device=device)
for t in range(time_steps):
    V_all[..., t] = potential.get_potential_value(positions_over_time[..., t], coeff_grid, t)

print(f"Potential values shape (trajectories, spatial_dims, time): {V_all.shape}")
print(f"Potential requires_grad: {V_all.requires_grad}")
print(f"Potential grad_fn: {V_all.grad_fn}\n")

# Simple loss: sum of all potential values across all timesteps
loss = V_all.sum()
print(f"Loss: {loss.item():.4f}")
print(f"Loss requires_grad: {loss.requires_grad}")
print(f"Loss grad_fn: {loss.grad_fn}\n")

# ============================================================================
# Check dV/dcoeff
# ============================================================================
print("="*70)
print("dV/dcoeff (gradient of potential w.r.t. coeff_grid)")
print("="*70)
dV_dcoeff = torch.autograd.grad(
    outputs=loss,
    inputs=coeff_grid,
    create_graph=True,
    retain_graph=True
)[0]
print(f"dV/dcoeff shape: {dV_dcoeff.shape}")
print(f"dV/dcoeff:\n{dV_dcoeff}")

# Show which timesteps have non-zero gradients
nonzero_timesteps = (dV_dcoeff.abs() > 1e-8).any(dim=0).nonzero(as_tuple=True)[0]
print(f"\nTimesteps with non-zero gradients: {nonzero_timesteps.tolist()}")
print(f"Count: {len(nonzero_timesteps)}/{time_steps}\n")

# ============================================================================
# Check dcoeff/dtrainable
# ============================================================================
print("="*70)
print("dcoeff/dtrainable (gradient of coeff_grid w.r.t. trainable_params)")
print("="*70)
dcoeff_dtrainable = torch.autograd.grad(
    outputs=coeff_grid,
    inputs=trainable_params,
    grad_outputs=torch.ones_like(coeff_grid),
    create_graph=True,
    retain_graph=True
)[0]
print(f"dcoeff/dtrainable shape: {dcoeff_dtrainable.shape}")
print(f"dcoeff/dtrainable:\n{dcoeff_dtrainable}\n")

# ============================================================================
# Check dV/dtrainable
# ============================================================================
print("="*70)
print("dV/dtrainable (gradient of potential w.r.t. trainable_params)")
print("="*70)
dV_dtrainable = torch.autograd.grad(
    outputs=loss,
    inputs=trainable_params,
    create_graph=False,
    retain_graph=False
)[0]
print(f"dV/dtrainable shape: {dV_dtrainable.shape}")
print(f"dV/dtrainable:\n{dV_dtrainable}")

# Summary statistics
nonzero_count = (dV_dtrainable.abs() > 1e-8).sum().item()
total_count = dV_dtrainable.numel()
print(f"\nNon-zero gradient elements: {nonzero_count}/{total_count}")
print(f"Max gradient magnitude: {dV_dtrainable.abs().max().item():.6e}")
print(f"Mean gradient magnitude: {dV_dtrainable.abs().mean().item():.6e}\n")

# ============================================================================
# TEST 2: Check dv_dxda computation (for Malliavin weights)
# ============================================================================
print("\n" + "="*70)
print("TEST 2: dv_dxda gradient tracking")
print("="*70)

# Reset gradients
trainable_params.grad = None

# Use positions at a middle timestep for testing
test_positions = positions_over_time[..., 25].contiguous()

# Compute dv_dxda (as used in sim_engine)
dv_dxda = potential.dv_dxda(test_positions, coeff_grid, time_index=25)
print(f"dv_dxda shape: {dv_dxda.shape}")
print(f"dv_dxda requires_grad: {dv_dxda.requires_grad}")
print(f"dv_dxda grad_fn: {dv_dxda.grad_fn}\n")

# Can we backprop through dv_dxda to trainable_params?
test_loss = dv_dxda.sum()
test_loss.backward()

print(f"After dv_dxda.sum().backward():")
print(f"  trainable_params.grad is None: {trainable_params.grad is None}")
if trainable_params.grad is not None:
    print(f"  Non-zero gradient elements: {(trainable_params.grad.abs() > 1e-8).sum().item()}/{trainable_params.grad.numel()}")
    print(f"  Max gradient magnitude: {trainable_params.grad.abs().max().item():.6e}")
else:
    print("  ERROR: dv_dxda is detached from trainable_params!")

# ============================================================================
# TEST 3: Work loss gradient (multi-timestep)
# ============================================================================
print("\n" + "="*70)
print("TEST 3: Work loss over time")
print("="*70)

trainable_params.grad = None

# Compute potential at multiple timesteps using time-varying positions
potential_over_time = torch.zeros(num_samples, spatial_dimensions, time_steps + 1, device=device)
for t in range(time_steps + 1):
    t_idx = min(t, time_steps - 1)
    pos_idx = min(t, time_steps - 1)
    potential_over_time[..., t] = potential.get_potential_value(positions_over_time[..., pos_idx], coeff_grid, t_idx)

# Work loss: sum of differences
work = (potential_over_time[..., 1:] - potential_over_time[..., :-1]).sum()
print(f"Total work: {work.item():.4f}")
print(f"Work requires_grad: {work.requires_grad}\n")

work.backward()

print(f"After work.backward():")
print(f"  trainable_params.grad is None: {trainable_params.grad is None}")
if trainable_params.grad is not None:
    print(f"  Non-zero gradient elements: {(trainable_params.grad.abs() > 1e-8).sum().item()}/{trainable_params.grad.numel()}")
    print(f"  Max gradient magnitude: {trainable_params.grad.abs().max().item():.6e}")
    print(f"  Gradient values:\n{trainable_params.grad.squeeze()}")
else:
    print("  ERROR: No gradient!")

# ============================================================================
# TEST 4: Manual gradient check
# ============================================================================
print("\n" + "="*70)
print("TEST 4: Manual gradient via torch.autograd.grad")
print("="*70)

trainable_params.grad = None

# Recompute work loss with time-varying positions
coeff_grid = protocol_model.get_coeff_grid()
potential_over_time = torch.zeros(num_samples, spatial_dimensions, time_steps + 1, device=device)
for t in range(time_steps + 1):
    t_idx = min(t, time_steps - 1)
    pos_idx = min(t, time_steps - 1)
    potential_over_time[..., t] = potential.get_potential_value(positions_over_time[..., pos_idx], coeff_grid, t_idx)

work = (potential_over_time[..., 1:] - potential_over_time[..., :-1]).sum()

# Manual gradient computation
manual_grad = torch.autograd.grad(
    outputs=work,
    inputs=trainable_params,
    create_graph=False,
    retain_graph=False
)

print(f"Manual gradient via torch.autograd.grad:")
if manual_grad[0] is not None:
    print(f"  Gradient shape: {manual_grad[0].shape}")
    print(f"  Non-zero elements: {(manual_grad[0].abs() > 1e-8).sum().item()}/{manual_grad[0].numel()}")
    print(f"  Max magnitude: {manual_grad[0].abs().max().item():.6e}")
    print(f"  Gradient values:\n{manual_grad[0].squeeze()}")
else:
    print("  ERROR: Manual gradient is None!")

# ============================================================================
# TEST 5: Effect of endpoint variation
# ============================================================================
print("\n" + "="*70)
print("TEST 5: How much do coefficients vary over time?")
print("="*70)

coeff_grid = protocol_model.get_coeff_grid()
a_coeff = coeff_grid[0, 0, :]  # First coefficient over time
b_coeff = coeff_grid[1, 0, :]  # Second coefficient over time

print(f"Coefficient 'a' variation:")
print(f"  Start: {a_coeff[0].item():.4f}")
print(f"  End: {a_coeff[-1].item():.4f}")
print(f"  Range: {(a_coeff.max() - a_coeff.min()).item():.4f}")
print(f"  Std: {a_coeff.std().item():.4f}\n")

print(f"Coefficient 'b' variation:")
print(f"  Start: {b_coeff[0].item():.4f}")
print(f"  End: {b_coeff[-1].item():.4f}")
print(f"  Range: {(b_coeff.max() - b_coeff.min()).item():.4f}")
print(f"  Std: {b_coeff.std().item():.4f}\n")

# Check how much potential changes over time at a fixed position
test_pos = torch.tensor([[5.0]], device=device)
V_over_time = torch.zeros(time_steps + 1, device=device)
for t in range(time_steps + 1):
    t_idx = min(t, time_steps - 1)
    V_over_time[t] = potential.get_potential_value(test_pos, coeff_grid, t_idx).item()

print(f"Potential at position 5.0 over time:")
print(f"  Start: {V_over_time[0].item():.4f}")
print(f"  End: {V_over_time[-1].item():.4f}")
print(f"  Total change: {(V_over_time[-1] - V_over_time[0]).item():.4f}")
print(f"  Max step change: {(V_over_time[1:] - V_over_time[:-1]).abs().max().item():.6f}")

print("\n" + "="*70)
print("DIAGNOSTICS COMPLETE")
print("="*70)
