"""Round-trip test: generate surface -> add noise -> calibrate -> check recovery."""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from vol_surface.calibration.optimizer import calibrate_ssvi_surface, calibrate_svi_slice
from vol_surface.data.schema import SVIParams, SSVIParams, VolSlice
from vol_surface.models.svi import svi_total_variance
from vol_surface.models.ssvi import ssvi_total_variance


def _make_svi_slice(
    params: SVIParams, T: float, n_strikes: int = 25, noise_std: float = 0.0, seed: int = 0,
) -> VolSlice:
    k = np.linspace(-0.3, 0.3, n_strikes)
    w = svi_total_variance(k, params.a, params.b, params.rho, params.m, params.sigma)
    if noise_std > 0:
        rng = np.random.default_rng(seed)
        w = w + rng.normal(0, noise_std, len(w))
        w = np.maximum(w, 1e-6)
    iv = np.sqrt(w / T)
    return VolSlice(
        expiry=date(2025, 6, 20),
        T=T,
        forward=4500.0,
        strikes=(4500.0 * np.exp(k)).tolist(),
        log_moneyness=k.tolist(),
        total_variance=w.tolist(),
        implied_vols=iv.tolist(),
        weights=[1.0 / n_strikes] * n_strikes,
    )


class TestSVIRoundTrip:
    """Generate -> noise -> calibrate -> recover for SVI."""

    @pytest.mark.parametrize("noise_std", [0.0, 0.0005, 0.001])
    def test_param_recovery(self, noise_std: float):
        true_params = SVIParams(a=0.04, b=0.15, rho=-0.25, m=0.01, sigma=0.15)
        vol_slice = _make_svi_slice(true_params, T=0.25, noise_std=noise_std)

        recovered, opt = calibrate_svi_slice(vol_slice)
        assert recovered is not None

        tol = 0.02 + noise_std * 100  # tolerance scales with noise
        assert abs(recovered.a - true_params.a) < tol
        assert abs(recovered.b - true_params.b) < tol
        assert abs(recovered.rho - true_params.rho) < tol * 3
        assert abs(recovered.m - true_params.m) < tol * 2
        assert abs(recovered.sigma - true_params.sigma) < tol * 2


class TestSSVIRoundTrip:
    """Generate -> noise -> calibrate -> recover for SSVI."""

    def test_param_recovery_noiseless(self):
        true_ssvi = SSVIParams(rho=-0.3, eta=1.5, gamma=0.5)
        thetas = [0.02, 0.04, 0.08, 0.15]
        Ts = [0.1, 0.25, 0.5, 1.0]

        slices = []
        for theta, T in zip(thetas, Ts):
            k = np.linspace(-0.3, 0.3, 30)
            w = ssvi_total_variance(k, theta, true_ssvi.rho, true_ssvi.eta, true_ssvi.gamma)
            iv = np.sqrt(np.maximum(w, 1e-10) / T)
            slices.append(
                VolSlice(
                    expiry=date(2025, 6, 20),
                    T=T,
                    forward=4500.0,
                    strikes=(4500.0 * np.exp(k)).tolist(),
                    log_moneyness=k.tolist(),
                    total_variance=w.tolist(),
                    implied_vols=iv.tolist(),
                    weights=[1.0 / len(k)] * len(k),
                )
            )

        recovered, opt = calibrate_ssvi_surface(slices, thetas)
        assert recovered is not None
        assert abs(recovered.rho - true_ssvi.rho) < 0.05
        assert abs(recovered.eta - true_ssvi.eta) < 0.1
        assert abs(recovered.gamma - true_ssvi.gamma) < 0.1

    def test_param_recovery_noisy(self):
        true_ssvi = SSVIParams(rho=-0.25, eta=1.2, gamma=0.6)
        thetas = [0.02, 0.05, 0.10, 0.18]
        Ts = [0.1, 0.25, 0.5, 1.0]
        rng = np.random.default_rng(99)

        slices = []
        for theta, T in zip(thetas, Ts):
            k = np.linspace(-0.3, 0.3, 30)
            w = ssvi_total_variance(k, theta, true_ssvi.rho, true_ssvi.eta, true_ssvi.gamma)
            w = w + rng.normal(0, 0.0005, len(w))
            w = np.maximum(w, 1e-6)
            iv = np.sqrt(w / T)
            slices.append(
                VolSlice(
                    expiry=date(2025, 6, 20),
                    T=T,
                    forward=4500.0,
                    strikes=(4500.0 * np.exp(k)).tolist(),
                    log_moneyness=k.tolist(),
                    total_variance=w.tolist(),
                    implied_vols=iv.tolist(),
                    weights=[1.0 / len(k)] * len(k),
                )
            )

        recovered, opt = calibrate_ssvi_surface(slices, thetas)
        assert recovered is not None
        assert abs(recovered.rho - true_ssvi.rho) < 0.3
        assert abs(recovered.eta - true_ssvi.eta) < 0.5
        assert abs(recovered.gamma - true_ssvi.gamma) < 0.3
