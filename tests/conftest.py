"""Shared fixtures for tests."""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from vol_surface.data.schema import SVIParams, SSVIParams, VolSlice
from vol_surface.models.svi import svi_total_variance
from vol_surface.models.ssvi import ssvi_total_variance


@pytest.fixture
def svi_params_valid() -> SVIParams:
    return SVIParams(a=0.04, b=0.2, rho=-0.3, m=0.0, sigma=0.1)


@pytest.fixture
def ssvi_params_valid() -> SSVIParams:
    return SSVIParams(rho=-0.3, eta=1.5, gamma=0.5)


@pytest.fixture
def synthetic_svi_slice(svi_params_valid: SVIParams) -> VolSlice:
    """A synthetic vol slice generated from known SVI params."""
    p = svi_params_valid
    T = 0.25
    k = np.linspace(-0.3, 0.3, 30)
    w = svi_total_variance(k, p.a, p.b, p.rho, p.m, p.sigma)
    iv = np.sqrt(w / T)
    return VolSlice(
        expiry=date(2025, 6, 20),
        T=T,
        forward=4500.0,
        strikes=(4500.0 * np.exp(k)).tolist(),
        log_moneyness=k.tolist(),
        total_variance=w.tolist(),
        implied_vols=iv.tolist(),
        weights=[1.0 / len(k)] * len(k),
    )


@pytest.fixture
def synthetic_ssvi_slices(ssvi_params_valid: SSVIParams) -> list[VolSlice]:
    """Multiple synthetic slices from known SSVI params."""
    p = ssvi_params_valid
    thetas = [0.02, 0.04, 0.08, 0.12]
    Ts = [0.1, 0.25, 0.5, 1.0]
    slices = []
    for theta, T in zip(thetas, Ts):
        k = np.linspace(-0.3, 0.3, 25)
        w = ssvi_total_variance(k, theta, p.rho, p.eta, p.gamma)
        iv = np.sqrt(np.maximum(w, 1e-10) / T)
        slices.append(
            VolSlice(
                expiry=date(2025, 3, 1),
                T=T,
                forward=4500.0,
                strikes=(4500.0 * np.exp(k)).tolist(),
                log_moneyness=k.tolist(),
                total_variance=w.tolist(),
                implied_vols=iv.tolist(),
                weights=[1.0 / len(k)] * len(k),
            )
        )
    return slices
