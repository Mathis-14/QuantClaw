"""
Tests for SVI and SSVI calibration.

Run with:
    python -m pytest tests/test_svi.py -v
"""

import numpy as np
import pandas as pd
import pytest
from src.svi import SVIParams, SSVIParams, calibrate_svi, calibrate_ssvi, raw_svi, ssvi


@pytest.fixture
def sample_df():
    """Fixture for a sample volatility surface DataFrame."""
    data = {
        "expiry_date": [
            pd.Timestamp("2026-06-26"), pd.Timestamp("2026-06-26"), 
            pd.Timestamp("2026-09-25"), pd.Timestamp("2026-09-25"),
        ],
        "log_moneyness": [-0.1, 0.0, 0.1, -0.1],
        "total_variance": [0.04, 0.045, 0.05, 0.055],
        "weight": [1.0, 1.0, 1.0, 1.0],
    }
    return pd.DataFrame(data)


def test_raw_svi():
    """Test raw SVI formula."""
    k = np.array([-0.1, 0.0, 0.1])
    w = raw_svi(k, a=0.01, b=0.1, rho=-0.5, m=0.0, sigma=0.1)
    assert len(w) == 3
    assert all(w >= 0)


def test_ssvi():
    """Test SSVI formula."""
    theta = 0.04
    k = np.array([-0.1, 0.0, 0.1])
    w = ssvi(theta, k, rho=-0.5, eta=0.1, gamma=0.5)
    assert len(w) == 3
    assert all(w >= 0)


def test_calibrate_svi(sample_df):
    """Test SVI calibration per expiry slice."""
    svi_params = calibrate_svi(sample_df)
    assert len(svi_params) == 2  # Two expiries
    assert all(isinstance(p, SVIParams) for p in svi_params)
    assert all(p.rmse is not None for p in svi_params)
    assert all(p.r_squared is not None for p in svi_params)


def test_calibrate_ssvi(sample_df):
    """Test SSVI calibration for the entire surface."""
    ssvi_params = calibrate_ssvi(sample_df)
    assert isinstance(ssvi_params, SSVIParams)
    assert ssvi_params.rmse is not None
    assert ssvi_params.r_squared is not None


def test_svi_rmse(sample_df):
    """Test SVI calibration RMSE is reasonable."""
    svi_params = calibrate_svi(sample_df)
    assert all(p.rmse < 0.1 for p in svi_params)  # RMSE should be small


def test_ssvi_rmse(sample_df):
    """Test SSVI calibration RMSE is reasonable."""
    ssvi_params = calibrate_ssvi(sample_df)
    assert ssvi_params.rmse < 0.1  # RMSE should be small