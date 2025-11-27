import torch
import unittest
from protocolopt.protocols.piecewise import LinearPiecewise

class TestLinearPiecewise(unittest.TestCase):
    def test_get_coeff_grid_shape(self):
        # Setup
        coefficient_count = 3
        time_steps = 100
        knot_count = 5
        initial_coeff_guess = torch.randn(coefficient_count, knot_count - 2)

        protocol = LinearPiecewise(
            coefficient_count=coefficient_count,
            time_steps=time_steps,
            knot_count=knot_count,
            initial_coeff_guess=initial_coeff_guess,
            endpoints=torch.zeros(coefficient_count, 2)
        )

        # Action
        coeff_grid = protocol.get_coeff_grid()

        # Assert
        self.assertEqual(coeff_grid.shape, (coefficient_count, time_steps))

if __name__ == '__main__':
    unittest.main()
