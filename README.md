# ProtocolOpt

[![Development Status](https://img.shields.io/badge/status-alpha-orange)](https://github.com/mathiasvanpatten/protocolopt)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

**ProtocolOpt** provides a modular framework for optimizing time-dependent control protocols in stochastic systems using **differentiable programming**.

### What is this?
Optimizing control in noisy, thermal environments (like superconducting circuits) is notoriously difficult because the noise is non-differentiable. ProtocolOpt solves this by providing:

1.  [cite_start]A **`ProtocolOptimizer`** loop that handles the training orchestration[cite: 50].
2.  **Differentiable Simulators** for SDEs (Stochastic Differential Equations).
3.  **Malliavin Weight Estimators** that re-express sensitivity as a correlation function, allowing for stable gradient descent on thermodynamic quantities.

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/MathiasVanPatten/protocolopt.git
cd protocolopt
```

2. Install the package in development mode (make sure your virtual environment is activated):
```bash
pip install -e .
```

3. Run the examples:
```bash
python examples/bitflip_training.py
```

4. Start the aim server to view the results in a browser:
```bash
aim up
```

## Project Overview

This library provides a modular framework for setting up and running optimization experiments. To use the `ProtocolOptimizer`, you need to define and compose the following core components:

1.  **Potential**: Defines the energy landscape of the system.
2.  **Simulator**: Handles the dynamics of the system (e.g., Euler-Maruyama for Langevin dynamics).
3.  **Loss**: Specifies the objective function to be minimized during training.
4.  **Protocol**: Represents the time-dependent control parameters that are being optimized.
5.  **InitialConditionGenerator**: Generates the starting states for the simulation.

Optionally, you may add callbacks to the `ProtocolOptimizer` to perform additional actions during the training process.

## Documentation & Resources

- **[Documentation](https://mathiasvanpatten.github.io/protocolopt/)**: For detailed API reference and guides, please refer to the documentation in the `docs/` directory.
- **[Roadmap](https://mathiasvanpatten.github.io/protocolopt/roadmap/)**: Check out the development roadmap for upcoming features and milestones.
- **[Discussions](https://github.com/MathiasVanPatten/protocolopt/discussions)**: Have questions or ideas? Join the discussion.
- **[Bug Reports](https://github.com/MathiasVanPatten/protocolopt/issues)**: Found a bug? Open an issue.
