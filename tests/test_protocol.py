import torch
import unittest
from protocolopt.protocols.piecewise import LinearPiecewise

class TestLinearPiecewise(unittest.TestCase):
    def test_get_protocol_tensor_shape(self):
        # Setup
        control_dim = 3
        time_steps = 100
        knot_count = 5
        initial_coeff_guess = torch.randn(control_dim, knot_count - 2)

        protocol = LinearPiecewise(
            control_dim=control_dim,
            time_steps=time_steps,
            knot_count=knot_count,
            initial_coeff_guess=initial_coeff_guess,
            endpoints=torch.zeros(control_dim, 2)
        )

        # Action
        protocol_tensor = protocol.get_protocol_tensor()

        # Assert
        self.assertEqual(protocol_tensor.shape, (control_dim, time_steps))

if __name__ == '__main__':
    unittest.main()
