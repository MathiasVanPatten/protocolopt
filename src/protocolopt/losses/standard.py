from ..core.loss import Loss, TruthTableError
from ..core.types import PotentialTensor, MicrostatePaths, ControlSignal, WorkTensor
from .functional import variance_loss, work_loss, temporal_smoothness_penalty
from typing import Dict, Optional, Union, List, Any
import torch

class LogicGateEndpointLossBase(Loss):
    """Base class for losses that depend on the final state of the trajectory relative to target bits."""

    def __init__(self, midpoints: torch.Tensor, truth_table: Dict[int, Union[List[str], Dict]], bit_locations: torch.Tensor, exponent: int = 2, starting_bit_weights: Optional[torch.Tensor] = None):
        """Initializes the LogicGateEndpointLossBase.

        Args:
            midpoints: Midpoints defining bit boundaries in each spatial dimension. Currently only
                supports binary mapping. Should be a vector of midpoints for each spatial dimension.
            truth_table: Dictionary defining valid transitions. It should be nested with one input bit
                at a time as the keys with the innermost level showing the acceptable outputs (as a list
                of bit strings) for that bit configuration. You must fully define the input/output space
                in a 1-Many or 1-1 enforced configuration.
            bit_locations: Target locations for each bit configuration. It is assumed that index 0 is
                the location of 0b0, index 1 is the location of 0b1, etc. To the left of the midpoint
                is 0 and the right is 1. Exactly on the midpoint maps to 0.
                Shape: (Domain_Size, Spatial_Dim)
                Domain_Size is 2^Spatial_Dim.
            exponent: Exponent for the distance metric (p-norm).
            starting_bit_weights: Optional tensor of weights for each starting state (int representation).
                Shape should match domain size.

        Examples:
            NAND with 2 bits with the rightmost bit being used as the output:

            {
                0: {
                    0: ['01', '11'],
                    1: ['01', '11']
                },
                1: {
                    0: ['01', '11'],
                    1: ['00', '10']
                }
            }

            NAND where only 11 is used as 1:

            {
                0: {
                    0: ['11'],
                    1: ['11']
                },
                1: {
                    0: ['11'],
                    1: ['00', '10', '01']
                }
            }
        """
        
        self.midpoints = midpoints
        self.bit_locations = bit_locations
        self.exponent = exponent
        self.starting_bit_weights = starting_bit_weights
        self.truth_table = truth_table
        self._validate_input_sequence_in_truth_table(truth_table)
        self.domain = 2**self._get_depth_of_truth_table(truth_table)
        self.flattened_truth_table = {k : None for k in range(self.domain)}
        self._flatten_truth_table(truth_table)
        self._gen_validity_mapping()
        self.hparams = {
            'midpoints': self.midpoints.tolist() if isinstance(self.midpoints, torch.Tensor) else self.midpoints,
            'truth_table': self.truth_table,
            'bit_locations_shape': list(self.bit_locations.shape),
            'exponent': self.exponent,
            'starting_bit_weights': self.starting_bit_weights.tolist() if self.starting_bit_weights is not None else None,
            'domain': self.domain,
            'name': self.__class__.__name__
        }

    def _get_depth_of_truth_table(self, truth_table_dict):
        if not isinstance(truth_table_dict[0], dict):
            return 1
        else:
            return 1 + self._get_depth_of_truth_table(truth_table_dict[0])

    def _compute_starting_bits_int(self, microstate_paths: MicrostatePaths) -> torch.Tensor:
        starting_bits = microstate_paths[:,:,0,0] > self.midpoints[None,:]
        starting_bits_int = torch.sum(starting_bits.int() * torch.tensor(list(reversed([2**x for x in range(self.midpoints.shape[0])])), device = microstate_paths.device)[None, :], axis = -1)
        return starting_bits_int
    
    def _compute_ending_bits_int(self, microstate_paths: MicrostatePaths) -> torch.Tensor:
        ending_bits = microstate_paths[:,:,-1,0] > self.midpoints[None,:]
        ending_bits_int = torch.sum(ending_bits.int() * torch.tensor(list(reversed([2**x for x in range(self.midpoints.shape[0])])), device = microstate_paths.device)[None, :], axis = -1)
        return ending_bits_int

    def _endpoint_loss(self, microstate_paths: MicrostatePaths) -> torch.Tensor:
        starting_bits_int = self._compute_starting_bits_int(microstate_paths)
        ending_positions = microstate_paths[:,:,-1,0]
        distances = torch.norm(ending_positions[:,None,:] - self.bit_locations[None,:,:], dim = -1, p = self.exponent)
        valid_mask = self.validity[starting_bits_int, :]
        masked_distances = torch.where(valid_mask, distances, torch.inf)
        
        # Get minimum distance to a valid target state
        min_distances = masked_distances.min(axis = -1).values
        
        # Apply starting bit weights if provided
        if self.starting_bit_weights is not None:
            weights = self.starting_bit_weights[starting_bits_int]
            return min_distances * weights
            
        return min_distances


    def _gen_validity_mapping(self):
        self.validity = torch.zeros(self.domain, self.domain, dtype = torch.bool, device = self.bit_locations.device)
        for inputstate, outputstate in self.flattened_truth_table.items():
            self.validity[inputstate, outputstate] = True
    
    def compute_binary_trajectory_info(self, microstate_paths: MicrostatePaths) -> Dict[str, torch.Tensor]:
        """
        Compute starting and ending binary states for microstate paths.
        
        Args:
            microstate_paths: Microstate paths tensor.
                              Shape: (Batch, Spatial_Dim, Time_Steps+1, 2)
        
        Returns:
            Dictionary with 'starting_bits_int' and 'ending_bits_int' tensors
        """
        return {
            'starting_bits_int': self._compute_starting_bits_int(microstate_paths),
            'ending_bits_int': self._compute_ending_bits_int(microstate_paths)
        }
    
    def compute_binary_error_rate(self, microstate_paths: MicrostatePaths) -> float:
        """
        Compute the percentage of microstate paths ending in invalid states.
        
        Args:
            microstate_paths: Microstate paths tensor (should be detached)
                              Shape: (Batch, Spatial_Dim, Time_Steps+1, 2)
            
        Returns:
            Float representing the error rate (0.0 to 1.0)
        """
        starting_bits_int = self._compute_starting_bits_int(microstate_paths)
        ending_bits_int = self._compute_ending_bits_int(microstate_paths)
        valid_transitions = self.validity[starting_bits_int, ending_bits_int]
        error_rate = (~valid_transitions).float().mean()
        return error_rate.item()

    def _validate_input_sequence_in_truth_table(self, truth_table_dict, current_bit_sequence = ''):
        for bit in [0, 1]:
            if bit not in truth_table_dict:
                raise TruthTableError(f"Missing key {bit}", current_bit_sequence)
            if isinstance(truth_table_dict[bit], dict):
                try:
                    self._validate_input_sequence_in_truth_table(
                        truth_table_dict[bit],
                        current_bit_sequence = current_bit_sequence + str(bit)
                    )
                except TruthTableError as e:
                    raise
                except Exception as e:
                    raise TruthTableError(f"Unexpected error ({type(e).__name__}: {e})", current_bit_sequence + str(bit)) from e

    def _flatten_truth_table(self, truth_table_dict, current_bit_sequence = ''):
        for bit in [0, 1]:
            if isinstance(truth_table_dict[bit], dict):
                self._flatten_truth_table(truth_table_dict[bit], current_bit_sequence + str(bit))
            else:
                list_to_add = [int(i, base = 2) for i in truth_table_dict[bit]]
                if list_to_add == []:
                    raise TruthTableError(f"Missing output for {bit}", current_bit_sequence + str(bit))
                self.flattened_truth_table[int(current_bit_sequence + str(bit), base = 2)] = list_to_add

class StandardLogicGateLoss(LogicGateEndpointLossBase):
    """Standard loss function combining logic gate endpoint error, work, variance, and smoothness."""

    def __init__(self, midpoints, truth_table, bit_locations, endpoint_weight = 1, work_weight = 1, var_weight = 1, smoothness_weight = 1, exponent = 2, starting_bit_weights=None):
        """Initializes StandardLogicGateLoss.

        Args:
            midpoints: Bit boundaries.
            truth_table: Valid transitions.
            bit_locations: Target locations.
            endpoint_weight: Weight for endpoint loss.
            work_weight: Weight for work loss.
            var_weight: Weight for variance loss.
            smoothness_weight: Weight for smoothness penalty.
            exponent: Exponent for distance.
            starting_bit_weights: Weights for starting bits.
        """
        super().__init__(midpoints, truth_table, bit_locations, exponent, starting_bit_weights)
        self.endpoint_weight = endpoint_weight
        self.work_weight = work_weight
        self.var_weight = var_weight
        self.smoothness_weight = smoothness_weight

        self.hparams.update({
            'endpoint_weight': self.endpoint_weight,
            'work_weight': self.work_weight,
            'var_weight': self.var_weight,
            'smoothness_weight': self.smoothness_weight
        })

    def loss(self, potential_tensor: PotentialTensor, microstate_paths: MicrostatePaths, dw_tensor: WorkTensor, protocol_tensor: ControlSignal, dt: float) -> torch.Tensor:
        """Computes the combined loss.

        Args:
            potential_tensor: Potential energy values. Shape: (Batch, Time_Steps)
            microstate_paths: Microstate paths data. Shape: (Batch, Spatial_Dim, Time_Steps+1, 2)
            dw_tensor: Change in potential energy at each step. Shape: (Batch, Time_Steps)
            protocol_tensor: Control signals. Shape: (Control_Dim, Time_Steps)
            dt: Time step size.

        Returns:
            Loss value. Shape: (Batch,)
        """
        starting_bits_int = self._compute_starting_bits_int(microstate_paths)
        endpoint_loss = self._endpoint_loss(microstate_paths)
        work_loss_value = work_loss(dw_tensor)
        var_loss_value = variance_loss(microstate_paths, starting_bits_int, self.domain)
        smoothness_loss_value = temporal_smoothness_penalty(protocol_tensor, dt)
        return (
            self.endpoint_weight * endpoint_loss
            + self.work_weight * work_loss_value
            + self.var_weight * var_loss_value
            + self.smoothness_weight * smoothness_loss_value
        )
    
    def log_components(self, potential_tensor: PotentialTensor, microstate_paths: MicrostatePaths, dw_tensor: WorkTensor, protocol_tensor: ControlSignal, dt: float) -> Dict[str, torch.Tensor]:
        """
        Compute individual loss components for logging/analysis.
        
        Args:
            potential_tensor: Detached potential tensor
            microstate_paths: Detached microstate paths tensor
            dw_tensor: Detached work tensor
            protocol_tensor: Protocol tensor
            dt: Time step
            
        Returns:
            Dictionary with loss component values (already detached from input)
        """
        starting_bits_int = self._compute_starting_bits_int(microstate_paths)
        endpoint_loss_val = self._endpoint_loss(microstate_paths)
        work_loss_val = work_loss(dw_tensor)
        var_loss_val = variance_loss(microstate_paths, starting_bits_int, self.domain)
        smoothness_loss_val = temporal_smoothness_penalty(protocol_tensor, dt)
        
        return {
            'endpoint_loss': endpoint_loss_val,
            'work_loss': work_loss_val,
            'variance_loss': var_loss_val,
            'smoothness_loss': smoothness_loss_val
        }
