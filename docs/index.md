# ProtocolOpt

ProtocolOpt is a library designed for optimizing time-dependent protocols in physical systems, particularly focusing on non-equilibrium statistical physics and thermodynamic computing applications.

## Overview

The core of the library is the `ProtocolOptimizer`, which orchestrates the training of protocols to minimize a defined loss function (such as work or entropy production) over a simulation.

## Key Components

- **Potentials**: Define the energy landscape of the system.
- **Simulators**: Propagate the system dynamics (e.g., Euler-Maruyama for Langevin dynamics).
- **Protocols**: Define how control parameters change over time.
- **Losses**: Define the objective function for optimization.
- **Sampling**: Generate initial conditions for simulations.

Optionally, you may add callbacks to the `ProtocolOptimizer` to perform additional actions during the training process.

## Getting Started

See the [API Reference](reference/core.md) for detailed documentation of the classes and functions.

