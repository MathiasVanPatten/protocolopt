# ProtocolOpt

ProtocolOpt is a library designed for optimizing time-dependent protocols in physical systems. The primary goal of this project is to provide a flexible and efficient `ProtocolOptimizer` that can train control protocols to minimize specific loss functions (such as thermodynamic work or entropy production) in stochastic simulations.

## Project Overview

This library provides a modular framework for setting up and running optimization experiments. To use the `ProtocolOptimizer`, you need to define and compose the following core components:

1.  **Potential**: Defines the energy landscape of the system.
2.  **Simulator**: Handles the dynamics of the system (e.g., Euler-Maruyama for Langevin dynamics).
3.  **Loss**: Specifies the objective function to be minimized during training.
4.  **Protocol**: Represents the time-dependent control parameters that are being optimized.
5.  **InitialConditionGenerator**: Generates the starting states for the simulation.

Optionally, you can add callbacks to the `ProtocolOptimizer` to perform additional actions during the training process.

## Documentation & Resources

- **[Documentation](docs/index.md)**: For detailed API reference and guides, please refer to the documentation in the `docs/` directory.
- **[Roadmap](docs/roadmap.md)**: Check out the development roadmap for upcoming features and milestones.
- **[Discussions](https://github.com/MathiasVanPatten/protocolopt/discussions)**: Have questions or ideas? Join the discussion.
- **[Bug Reports](https://github.com/MathiasVanPatten/protocolopt/issues)**: Found a bug or have a feature request? Open an issue.
