"""Tests for SSVI model no-arbitrage conditions and calibration."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from vol_surface.data.schema import SSVIParams, VolSlice
from vol_surface.models.ssvi import (
    check_ssvi_no_arb,
    phi_func,
    ssvi_total_variance,
    ssvi_implied_vol,
)
from vol_surface.calibration.optimizer import calibrate_ssvi_surface


class TestSSVINoArbitrage:
    """Verify SSVI no-arbitrage conditions."""

    def test_valid_params(self, ssvi_params_valid: SSVIParams):
        assert check_ssvi_no_arb(ssvi_params_valid.rho, ssvi_params_valid.eta, ssvi_params_valid.gamma)

    def test_eta_too_large(self):
        """eta*(1+|rho|) > 4 should fail."""
        with pytest.raises(ValidationError):
            SSVIParams(rho=-0.5, eta=3.0, gamma=0.5)  # 3*(1+0.5)=4.5 > 4

    def test_gamma_zero_rejected(self):
        with pytest.raises(ValidationError):
            SSVIParams(rho=-0.3, eta=1.0, gamma=0.0)

    def test_gamma_over_one_rejected(self):
        with pytest.raises(ValidationError):
            SSVIParams(rho=-0.3, eta=1.0, gamma=1.01)

    def test_boundary_case(self):
        """eta*(1+|rho|) = 4 exactly should pass."""
        p = SSVIParams(rho=0.0, eta=4.0, gamma=0.5)
        assert check_ssvi_no_arb(p.rho, p.eta, p.gamma)

    def test_check_function_rejects_bad(self):
        assert not check_ssvi_no_arb(rho=-0.5, eta=3.0, gamma=0.5)
        assert not check_ssvi_no_arb(rho=0.0, eta=1.0, gamma=0.0)
        assert not check_ssvi_no_arb(rho=0.0, eta=1.0, gamma=1.5)
        assert not check_ssvi_no_arb(rho=0.0, eta=-1.0, gamma=0.5)


class TestSSVIFunction:
    """Test SSVI function behavior."""

    def test_positive_variance(self, ssvi_params_valid: SSVIParams):
        p = ssvi_params_valid
        k = np.linspace(-0.4, 0.4, 100)
        theta = 0.04
        w = ssvi_total_variance(k, theta, p.rho, p.eta, p.gamma)
        assert np.all(w > 0)

    def test_atm_equals_theta(self, ssvi_params_valid: SSVIParams):
        """At k=0, w(0) = theta if rho=0. With rho != 0, w(0) ~ theta."""
        p = ssvi_params_valid
        theta = 0.04
        w_atm = ssvi_total_variance(np.array([0.0]), theta, p.rho, p.eta, p.gamma)
        # w(0) = (theta/2)*(1 + sqrt(rho^2 + 1 - rho^2)) = (theta/2)*(1+1) = theta
        assert abs(w_atm[0] - theta) < 1e-10

    def test_phi_decreasing_in_theta(self, ssvi_params_valid: SSVIParams):
        """phi should generally decrease as theta increases (for gamma > 0)."""
        p = ssvi_params_valid
        phi_small = phi_func(0.01, p.eta, p.gamma)
        phi_large = phi_func(0.5, p.eta, p.gamma)
        assert phi_small > phi_large


class TestSSVICalibration:
    """Test SSVI calibration on synthetic data."""

    def test_exact_recovery(
        self,
        synthetic_ssvi_slices: list[VolSlice],
        ssvi_params_valid: SSVIParams,
    ):
        """Calibrate on noiseless SSVI data and check param recovery."""
        thetas = [0.02, 0.04, 0.08, 0.12]
        ssvi_p, opt = calibrate_ssvi_surface(synthetic_ssvi_slices, thetas)
        assert ssvi_p is not None
        assert opt.success

        assert abs(ssvi_p.rho - ssvi_params_valid.rho) < 0.05
        assert abs(ssvi_p.eta - ssvi_params_valid.eta) < 0.1
        assert abs(ssvi_p.gamma - ssvi_params_valid.gamma) < 0.1

    def test_noisy_recovery(
        self,
        synthetic_ssvi_slices: list[VolSlice],
        ssvi_params_valid: SSVIParams,
    ):
        """Add noise and verify calibration still approximately recovers."""
        thetas = [0.02, 0.04, 0.08, 0.12]
        rng = np.random.default_rng(123)

        noisy_slices = []
        for s in synthetic_ssvi_slices:
            noise = rng.normal(0, 0.0005, len(s.total_variance))
            noisy_w = [max(w + n, 1e-6) for w, n in zip(s.total_variance, noise)]
            noisy_iv = [np.sqrt(w / s.T) for w in noisy_w]
            noisy_slices.append(s.model_copy(update={"total_variance": noisy_w, "implied_vols": noisy_iv}))

        ssvi_p, opt = calibrate_ssvi_surface(noisy_slices, thetas)
        assert ssvi_p is not None
        assert abs(ssvi_p.rho - ssvi_params_valid.rho) < 0.2
        assert abs(ssvi_p.eta - ssvi_params_valid.eta) < 0.5
