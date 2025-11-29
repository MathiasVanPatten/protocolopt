import torch
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any
from torch.func import vmap
from .types import PotentialTensor, MicrostatePaths, ControlSignal, MalliavinWeight

if TYPE_CHECKING:
    from .potential import Potential

class TruthTableError(Exception):
    def __init__(self, message, path=''):
        self.path = path
        super().__init__(f"{message} at branch '{path}'")

class Loss(ABC):
    """Abstract base class for loss functions."""

    @abstractmethod
    def loss(self, potential_tensor: PotentialTensor, microstate_paths: MicrostatePaths, protocol_tensor: ControlSignal, dt: float) -> torch.Tensor:
        """Computes the loss for a batch of microstate paths.

        Args:
            potential_tensor: Potential values along paths.
                              Shape: (Batch, Time_Steps) or (Batch, Time_Steps+1)
            microstate_paths: Microstate path data.
                              Shape: (Batch, Spatial_Dim, Time_Steps+1, 2)
                              Dimension 3 is (position, velocity).
            protocol_tensor: Control signals from the protocol.
                             Shape: (Control_Dim, Time_Steps)
            dt: Time step size.

        Returns:
            The loss value for each path. Shape: (Batch,).
        """
        pass

    def _compute_direct_grad(self, loss_values):
        loss_values.mean(axis = -1).backward()
        pass

    def _compute_malliavin_grad(self, loss_values, malliavian_weights):
        #malliavian_weights are (num_samples, coeff_count, time_steps)
        return (loss_values[:,None, None] * malliavian_weights).mean(axis = 0)

    def compute_FRR_gradient(
        self,
        potential_obj: "Potential",
        potential_tensor: PotentialTensor,
        microstate_paths: MicrostatePaths,
        malliavian_weights: MalliavinWeight,
        protocol_tensor: ControlSignal,
        dt: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes the gradient using the Fluctuation Response Relation (FRR).

        Args:
            potential_obj: The potential object.
            potential_tensor: Recorded potential values.
            microstate_paths: Recorded microstate paths.
            malliavian_weights: Computed Malliavin weights for gradient estimation.
            protocol_tensor: The current control signals from the protocol.
            dt: Time step.

        Returns:
            A tuple (total_loss, per_path_loss).
        """
        pos_tensor_detached = microstate_paths[..., 0].detach()
        # recompute to freeze the secondary reliance on the protocol through the trajectories
        # we want dLoss/da where loss is given a trajectory and the mall weights carry the probability of the trajectory
        def potential_at_t(pos_t, coeff_t):
            return potential_obj.potential_value(pos_t, coeff_t)

        # in dims: (2,1) -> for arg 0 pos iterate over dim 2 time, for arg 1 coeff iterate over dim 1 which is time
        # outdims: 1, put time at the end
        batched_time_potential = vmap(potential_at_t, in_dims=(2, 1), out_dims=1)

        clean_potential_tensor = batched_time_potential(pos_tensor_detached[...,:-1], protocol_tensor)

        loss_values_direct = self.loss(
            clean_potential_tensor,
            microstate_paths.detach(), # detach to avoid double counting through the stochastic correction loss
            protocol_tensor,
            dt
        )

        direct_grad_scalar = loss_values_direct.mean()

        # direct loss for backwards, detach to avoid double counting through the stochastic correction loss
        with torch.no_grad():
            loss_values_for_scoring = self.loss(
                potential_tensor,
                microstate_paths,
                protocol_tensor,
                dt
            )

        # Make sure the eventual backwards goes back through to the control signals and trainable params only
        # Sum over (Control_Dim, Time)
        frr_term = (malliavian_weights.detach() * protocol_tensor).sum(dim=(-2, -1))
        surrogate_grad_scalar = (loss_values_for_scoring * frr_term).mean()

        return direct_grad_scalar + surrogate_grad_scalar, loss_values_for_scoring
