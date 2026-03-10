"""Tests for SVI model parametric constraints and calibration."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from vol_surface.data.schema import SVIParams, VolSlice
from vol_surface.models.svi import (
    svi_initial_guess,
    svi_parameter_bounds,
    svi_total_variance,
    svi_implied_vol,
    vector_to_params,
    params_to_vector,
)
from vol_surface.calibration.optimizer import calibrate_svi_slice


class TestSVIParametricConstraints:
    """Verify SVI parameter validation."""

    def test_valid_params(self, svi_params_valid: SVIParams):
        assert svi_params_valid.b >= 0
        assert abs(svi_params_valid.rho) < 1
        assert svi_params_valid.sigma > 0

    def test_no_arb_lower_bound_satisfied(self, svi_params_valid: SVIParams):
        assert svi_params_valid.no_arb_lower_bound() >= 0

    def test_negative_b_rejected(self):
        with pytest.raises(ValidationError):
            SVIParams(a=0.04, b=-0.1, rho=-0.3, m=0.0, sigma=0.1)

    def test_rho_out_of_bounds(self):
        with pytest.raises(ValidationError):
            SVIParams(a=0.04, b=0.2, rho=1.0, m=0.0, sigma=0.1)
        with pytest.raises(ValidationError):
            SVIParams(a=0.04, b=0.2, rho=-1.0, m=0.0, sigma=0.1)

    def test_sigma_zero_rejected(self):
        with pytest.raises(ValidationError):
            SVIParams(a=0.04, b=0.2, rho=-0.3, m=0.0, sigma=0.0)

    def test_sigma_negative_rejected(self):
        with pytest.raises(ValidationError):
            SVIParams(a=0.04, b=0.2, rho=-0.3, m=0.0, sigma=-0.1)

    def test_no_arb_violation_detectable(self):
        """Params that violate a + b*sigma*sqrt(1-rho^2) >= 0."""
        p = SVIParams(a=-0.5, b=0.01, rho=0.0, m=0.0, sigma=0.01)
        assert p.no_arb_lower_bound() < 0


class TestSVIFunction:
    """Test the SVI function itself."""

    def test_atm_value(self, svi_params_valid: SVIParams):
        p = svi_params_valid
        w_atm = svi_total_variance(np.array([0.0]), p.a, p.b, p.rho, p.m, p.sigma)
        assert w_atm[0] > 0

    def test_positive_variance(self, svi_params_valid: SVIParams):
        p = svi_params_valid
        k = np.linspace(-0.5, 0.5, 100)
        w = svi_total_variance(k, p.a, p.b, p.rho, p.m, p.sigma)
        assert np.all(w > 0), "Total variance must be positive everywhere"

    def test_wings_increase(self, svi_params_valid: SVIParams):
        """SVI should increase for large |k|."""
        p = svi_params_valid
        k_far = np.array([-2.0, 0.0, 2.0])
        w = svi_total_variance(k_far, p.a, p.b, p.rho, p.m, p.sigma)
        assert w[0] > w[1] or w[2] > w[1]

    def test_roundtrip_vector(self, svi_params_valid: SVIParams):
        v = params_to_vector(svi_params_valid)
        recovered = vector_to_params(v)
        assert abs(recovered.a - svi_params_valid.a) < 1e-12


class TestSVICalibration:
    """Test SVI calibration on synthetic data."""

    def test_exact_recovery(self, synthetic_svi_slice: VolSlice, svi_params_valid: SVIParams):
        """Calibrate on noiseless synthetic data and check curve fit quality.

        SVI is NOT globally identifiable: multiple (a, b, rho, m, sigma) tuples
        can produce the same total-variance curve.  The meaningful assertion is
        that the fitted model reproduces the market curve to near-machine precision,
        not that the optimizer lands on the exact generating parameters.
        """
        from vol_surface.calibration.diagnostics import svi_slice_rmse

        svi_p, opt = calibrate_svi_slice(synthetic_svi_slice)
        assert svi_p is not None
        assert opt.success
        # Fitted curve must reproduce market total variance to < 1e-6
        assert svi_slice_rmse(synthetic_svi_slice, svi_p) < 1e-6
        # Arbitrage constraint: a + b*sigma >= 0
        assert svi_p.no_arb_lower_bound() >= -1e-8

    def test_noisy_recovery(self, synthetic_svi_slice: VolSlice, svi_params_valid: SVIParams):
        """Add noise and verify calibration fits the noisy slice well."""
        from vol_surface.calibration.diagnostics import svi_slice_rmse

        rng = np.random.default_rng(42)
        noise = rng.normal(0, 0.001, len(synthetic_svi_slice.total_variance))
        noisy_w = [max(w + n, 1e-6) for w, n in zip(synthetic_svi_slice.total_variance, noise)]
        noisy_iv = [np.sqrt(w / synthetic_svi_slice.T) for w in noisy_w]

        noisy_slice = synthetic_svi_slice.model_copy(
            update={"total_variance": noisy_w, "implied_vols": noisy_iv}
        )
        svi_p, opt = calibrate_svi_slice(noisy_slice)
        assert svi_p is not None
        # Fit quality should be at most ~2x the noise level
        assert svi_slice_rmse(noisy_slice, svi_p) < 0.002
