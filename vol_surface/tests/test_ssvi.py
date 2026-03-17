"""Tests for SSVI calibration and constraints."""

from __future__ import annotations

import logging
from datetime import date, timedelta

import numpy as np
import pytest

from vol_surface.data.schema import VolSlice
from vol_surface.calibration.verification import (
    recalibrate_ssvi_with_constraints,
    validate_ssvi_calibration,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture
def synthetic_btc_slices() -> list[VolSlice]:
    """Generate synthetic BTC volatility slices for testing."""
    np.random.seed(42)
    slices = []
    expiries = [7, 30, 90]
    spot = 74110.48
    today = date.today()
    
    for T in expiries:
        strikes = np.linspace(0.7 * spot, 1.3 * spot, 15)
        log_moneyness = np.log(strikes / spot)
        total_variance = 0.02 * T + 0.01 * (log_moneyness**2)
        implied_vols = np.sqrt(total_variance / (T / 365.25))
        weights = np.ones_like(implied_vols)
        
        slices.append(
            VolSlice(
                expiry=today + timedelta(days=T),
                T=T / 365.25,
                forward=spot,
                strikes=strikes.tolist(),
                log_moneyness=log_moneyness.tolist(),
                total_variance=total_variance.tolist(),
                implied_vols=implied_vols.tolist(),
                weights=weights.tolist(),
            )
        )
    return slices


@pytest.fixture
def synthetic_eth_slices() -> list[VolSlice]:
    """Generate synthetic ETH volatility slices for testing."""
    np.random.seed(42)
    slices = []
    expiries = [7, 30, 90]
    spot = 3500.0
    today = date.today()
    
    for T in expiries:
        strikes = np.linspace(0.7 * spot, 1.3 * spot, 15)
        log_moneyness = np.log(strikes / spot)
        total_variance = 0.03 * T + 0.015 * (log_moneyness**2)
        implied_vols = np.sqrt(total_variance / (T / 365.25))
        weights = np.ones_like(implied_vols)
        
        slices.append(
            VolSlice(
                expiry=today + timedelta(days=T),
                T=T / 365.25,
                forward=spot,
                strikes=strikes.tolist(),
                log_moneyness=log_moneyness.tolist(),
                total_variance=total_variance.tolist(),
                implied_vols=implied_vols.tolist(),
                weights=weights.tolist(),
            )
        )
    return slices


def test_ssvi_recalibration(synthetic_btc_slices: list[VolSlice]) -> None:
    """Test SSVI recalibration with constraints."""
    atm_total_vars = [0.02 * T for T in [7, 30, 90]]
    params, result = recalibrate_ssvi_with_constraints(synthetic_btc_slices, atm_total_vars)
    assert result["success"] is True
    assert params is not None
    assert params.rho <= 0
    assert params.eta > 0
    assert 0 < params.gamma <= 1


def test_ssvi_validation(synthetic_btc_slices: list[VolSlice]) -> None:
    """Test SSVI validation: RMSE and arbitrage checks."""
    atm_total_vars = [0.02 * T for T in [7, 30, 90]]
    params, _ = recalibrate_ssvi_with_constraints(synthetic_btc_slices, atm_total_vars)
    validation = validate_ssvi_calibration(synthetic_btc_slices, params, atm_total_vars)
    assert validation["rmse"] < 0.02
    assert validation["no_arb"] is True