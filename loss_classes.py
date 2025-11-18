#no not the comic

import torch
from abc import ABC, abstractmethod
from loss_methods import variance_loss, work_loss, temporal_smoothness_penalty

class TruthTableError(Exception):
    def __init__(self, message, path=''):
        self.path = path
        super().__init__(f"{message} at branch '{path}'")

class Loss(ABC):
    @abstractmethod
    def loss(self, potential_tensor, trajectory_tensor, coeff_grid, dt):
        #potential_tensor is a tensor of shape (num_samples, spatial_dimensions, time_steps+1)
        #trajectory_tensor is a tensor of shape (num_samples, spatial_dimensions, time_steps+1, 2 (position and velocity))
        #you are expected to compute the loss for each sample in the batch such that the graph is not broken, return only the value of the loss for each sample
        pass

    def _compute_direct_grad(self, loss_values):
        loss_values.mean(axis = -1).backward()
        pass

    def _compute_malliavin_grad(self, loss_values, malliavian_weights):
        #malliavian_weights are (num_samples, coeff_count, time_steps)
        return (loss_values[:,None, None] * malliavian_weights).mean(axis = 0)

    def compute_FRR_gradient(self, potential_tensor, trajectory_tensor, malliavian_weights, coeff_grid, dt, return_loss_values = True):
        #loss is a tensor of shape (num_samples,)
        #malliavian_weights is a tensor of shape (num_samples, time_steps)

        loss_values = self.loss(potential_tensor, trajectory_tensor, coeff_grid, dt)
        self._compute_direct_grad(loss_values)
        malliavin_grad = self._compute_malliavin_grad(loss_values, malliavian_weights)
        if return_loss_values:
            return malliavin_grad, loss_values
        else:
            return malliavin_grad


class EndpointLossBase(Loss):
    #all losses that need to hold state to evaluate the ending positions should inherit from this class
    def __init__(self, midpoints, truth_table, bit_locations, exponent = 2): 
        #midpoints only supports binary mapping currently, it should be a vector of midpoints for each spatial dimension
        #bit_locations is the ideal location of each bit for each spatial dimension, it's assumed that index 0 is the location of 0b0, index 1 is the location of 0b1, etc.
        #to the left of which is 0 and the right of which is 1. Exactly the mapping midpoint will be a 0
        #truth_table is a dict with the general form of the key being the input and the value being the expected output, it should be nested
        #with one input bit at a time as the keys with the innermost level showing the acceptable outputs for that bit configuration.
        #you must have a list of values, even if you only have one element
        #you must fully define the input/output space in a 1-Many or 1-1 enforced configuration
        #Example for NAND with 2 bits with the rightmost bit being used as the output:
        '''
            {
                0:{
                    0: ['01', '11'],
                    1: ['01', '11']
                    }
                1:{
                    0: ['01', '11'],
                    1: ['00', '10']
                }
            }
        '''
        #example for NAND where only 11 is used as 1
        '''
            {
                0:{
                    0: ['11'],
                    1: ['11']
                    }
                1:{
                    0: ['11'],
                    1: ['00', '10', '01']
                }
            }
        '''
        
        self.midpoints = midpoints
        self.bit_locations = bit_locations
        self.exponent = exponent
        self._validate_input_sequence_in_truth_table(truth_table)
        self.domain = 2**self._get_depth_of_truth_table(truth_table)
        self.flattened_truth_table = {k : None for k in range(self.domain)}
        self._flatten_truth_table(truth_table)
        print(self.flattened_truth_table)
        # self._validate_flattened_truth_table()
        self._gen_validity_mapping()

    def _get_depth_of_truth_table(self, truth_table_dict):
        if not isinstance(truth_table_dict[0], dict):
            return 1
        else:
            return 1 + self._get_depth_of_truth_table(truth_table_dict[0])

    def _compute_starting_bits_int(self, trajectory_tensor):
        starting_bits = trajectory_tensor[:,:,0,0] > self.midpoints[None,:]
        starting_bits_int = torch.sum(starting_bits.int() * torch.tensor(list(reversed([2**x for x in range(self.midpoints.shape[0])])), device = trajectory_tensor.device)[None, :], axis = -1)
        return starting_bits_int
    
    def _compute_ending_bits_int(self, trajectory_tensor):
        ending_bits = trajectory_tensor[:,:,-1,0] > self.midpoints[None,:]
        ending_bits_int = torch.sum(ending_bits.int() * torch.tensor(list(reversed([2**x for x in range(self.midpoints.shape[0])])), device = trajectory_tensor.device)[None, :], axis = -1)
        return ending_bits_int

    def _endpoint_loss(self, trajectory_tensor):
        starting_bits_int = self._compute_starting_bits_int(trajectory_tensor)
        ending_positions = trajectory_tensor[:,:,-1,0]
        distances = torch.norm(ending_positions[:,None,:] - self.bit_locations[None,:,:], dim = -1, p = self.exponent)
        valid_mask = self.validity[starting_bits_int, :]
        masked_distances = torch.where(valid_mask, distances, torch.inf)
        endpoint_loss = masked_distances.min(axis = -1).values
        return endpoint_loss

    # def loss(self, potential_tensor, trajectory_tensor):
    #     return self._endpoint_loss(trajectory_tensor)

    def _gen_validity_mapping(self):
        self.validity = torch.zeros(self.domain, self.domain, dtype = torch.bool, device = self.bit_locations.device)
        for inputstate, outputstate in self.flattened_truth_table.items():
            self.validity[inputstate, outputstate] = True
    
    def compute_binary_trajectory_info(self, trajectory_tensor):
        """
        Compute starting and ending binary states for trajectories.
        
        Args:
            trajectory_tensor: Trajectory tensor of shape (num_samples, spatial_dimensions, time_steps+1, 2)
        
        Returns:
            Dictionary with 'starting_bits_int' and 'ending_bits_int' tensors
        """
        return {
            'starting_bits_int': self._compute_starting_bits_int(trajectory_tensor),
            'ending_bits_int': self._compute_ending_bits_int(trajectory_tensor)
        }
    
    def compute_binary_error_rate(self, trajectory_tensor):
        """
        Compute the percentage of trajectories ending in invalid states.
        
        Args:
            trajectory_tensor: Trajectory tensor (should be detached)
            
        Returns:
            Float representing the error rate (0.0 to 1.0)
        """
        starting_bits_int = self._compute_starting_bits_int(trajectory_tensor)
        ending_bits_int = self._compute_ending_bits_int(trajectory_tensor)
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

    # def _validate_flattened_truth_table(self):
    #     output_values = []
    #     for k, v in self.flattened_truth_table.items():
    #         output_values.extend(v)
    #     missing = set(range(self.domain)) - set(output_values)
    #     if missing:
    #         missing_bin = [bin(x)[2:].zfill(len(self.midpoints)) for x in sorted(missing)]
    #         raise ValueError(f"Missing output values: {missing_bin} (as binary), found at least one instance of {set(output_values)}")


class StandardLoss(EndpointLossBase):
    def __init__(self, midpoints, truth_table, bit_locations, endpoint_weight = 1, work_weight = 1, var_weight = 1, smoothness_weight = 1, exponent = 2):
        super().__init__(midpoints, truth_table, bit_locations, exponent)
        self.endpoint_weight = endpoint_weight
        self.work_weight = work_weight
        self.var_weight = var_weight
        self.smoothness_weight = smoothness_weight

    def loss(self, potential_tensor, trajectory_tensor, coeff_grid, dt):
        # trajectory_tensor = trajectory_tensor.detach().requires_grad_(True)    
        starting_bits_int = self._compute_starting_bits_int(trajectory_tensor)
        endpoint_loss = self._endpoint_loss(trajectory_tensor)
        work_loss_value = work_loss(potential_tensor)
        var_loss_value = variance_loss(trajectory_tensor, starting_bits_int, self.domain)
        smoothness_loss_value = temporal_smoothness_penalty(coeff_grid, dt)
        # bun = torch.autograd.grad(
        #     outputs = endpoint_loss.sum(),
        #     inputs = trajectory_tensor,
        #     create_graph = True
        # )[0]
        return (
            self.endpoint_weight * endpoint_loss
            + self.work_weight * work_loss_value
            + self.var_weight * var_loss_value
            + self.smoothness_weight * smoothness_loss_value
        )
    
    def compute_loss_components(self, potential_tensor, trajectory_tensor, coeff_grid, dt):
        """
        Compute individual loss components for logging/analysis.
        
        Args:
            potential_tensor: Detached potential tensor
            trajectory_tensor: Detached trajectory tensor
            
        Returns:
            Dictionary with loss component values (already detached from input)
        """
        starting_bits_int = self._compute_starting_bits_int(trajectory_tensor)
        endpoint_loss_val = self._endpoint_loss(trajectory_tensor)
        work_loss_val = work_loss(potential_tensor)
        var_loss_val = variance_loss(trajectory_tensor, starting_bits_int, self.domain)
        smoothness_loss_val = temporal_smoothness_penalty(coeff_grid, dt)
        
        return {
            'endpoint_loss': endpoint_loss_val,
            'work_loss': work_loss_val,
            'variance_loss': var_loss_val,
            'smoothness_loss': smoothness_loss_val
        }

if __name__ == '__main__':
    good_truth_table = {
        0: {
            0: ['01', '11'],
            1: ['01', '11']
        },
        1: {
            0: ['01', '11'],
            1: ['00', '10']
        }
    }

    bad_truth_table = {
        0: {
            0: ['01'],
            1: ['01']
        },
        1: {
            0: ['01'],
            1: ['00']
        }
    }

    test = StandardLoss(
        midpoints=torch.tensor([0, 0]),
        bit_locations=torch.tensor([
            [-0.5, -0.5],
            [-0.5,  0.5],
            [ 0.5, -0.5],
            [ 0.5,  0.5]
        ]),
        truth_table=good_truth_table
    )

    test.loss(
        torch.randn(10, 10),
        torch.randn(10, 2, 10, 2)
    )


    # test = EndpointLossBase(midpoints = [0.5], truth_table = bad_truth_table)




