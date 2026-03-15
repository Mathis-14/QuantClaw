"""
Unit tests for vol_surface.svi module using unittest (no numpy dependency).

Tests:
- Input validation for SVI.calibrate.
- Parameter bounds and constraints.
"""

import unittest
from unittest.mock import patch
from vol_surface.svi import SVI


class TestSVI(unittest.TestCase):
    """Test suite for SVI model (logic-only tests)."""

    def test_calibrate_invalid_inputs(self) -> None:
        """Test SVI.calibrate with invalid inputs."""
        with self.assertRaises(ValueError):
            SVI.calibrate([-1.0, 0.0], [0.25], 1.0)  # Length mismatch
        
        with self.assertRaises(ValueError):
            SVI.calibrate([-1.0, 0.0], [0.25, 0.30], -1.0)  # Invalid T
        
        with self.assertRaises(ValueError):
            SVI.calibrate([1.0, 0.0], [0.25, 0.30], 1.0)  # Unsorted k

    @patch('vol_surface.svi.SVI.svi_implied_vol')
    @patch('vol_surface.svi.minimize')
    def test_calibrate_success(self, mock_minimize, mock_svi_implied_vol) -> None:
        """Test SVI.calibrate success path (mocked)."""
        # Mock minimize to return success
        mock_minimize.return_value.success = True
        mock_minimize.return_value.x = [0.04, 0.1, -0.5, 0.0, 0.1]
        mock_minimize.return_value.fun = 0.01
        
        # Mock svi_implied_vol to return dummy values
        mock_svi_implied_vol.return_value = [0.25, 0.30, 0.35]
        
        params = SVI.calibrate([-1.0, 0.0, 1.0], [0.25, 0.30, 0.35], 1.0)
        self.assertEqual(len(params), 5)
        self.assertGreaterEqual(params[0], 0)  # a >= 0
        self.assertGreaterEqual(params[1], 0)  # b >= 0
        self.assertGreaterEqual(params[2], -1)  # rho >= -1
        self.assertLessEqual(params[2], 1)     # rho <= 1
        self.assertGreater(params[4], 0)       # epsilon > 0

    @patch('vol_surface.svi.minimize')
    def test_calibrate_failure(self, mock_minimize) -> None:
        """Test SVI.calibrate failure path (mocked)."""
        mock_minimize.return_value.success = False
        mock_minimize.return_value.message = "Optimization failed"
        
        with self.assertRaises(ValueError):
            SVI.calibrate([-1.0, 0.0, 1.0], [0.25, 0.30, 0.35], 1.0)


if __name__ == "__main__":
    unittest.main()