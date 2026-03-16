"""
Tests for Deribit data pipeline.

Run with:
    python -m pytest tests/test_pipeline.py -v
"""

import pandas as pd
import pytest
import pandera as pa
from src.pipeline import (
    RawOptionSchema,
    CleanOptionSchema,
    filter_instruments,
    clean_and_enrich,
    day_count_act_365,
    compute_forward_price,
)


@pytest.fixture
def raw_df():
    """Fixture for raw options DataFrame."""
    data = {
        "instrument_name": ["BTC-26JUN26-50000-C", "BTC-26JUN26-50000-P"],
        "strike": [50000.0, 50000.0],
        "expiry_date": ["2026-06-26T08:00:00.000Z", "2026-06-26T08:00:00.000Z"],
        "option_type": ["call", "put"],
        "bid": [1000.0, 500.0],
        "ask": [1010.0, 510.0],
        "underlying_price": [52000.0, 52000.0],
        "implied_volatility": [0.5, 0.55],
        "funding_rate": [0.0001, 0.0001],
        "timestamp": ["2026-03-16T02:30:00.000Z", "2026-03-16T02:30:00.000Z"],
    }
    return pd.DataFrame(data)


def test_raw_schema_validation(raw_df):
    """Test RawOptionSchema validation."""
    RawOptionSchema.validate(raw_df, lazy=True)


def test_raw_schema_validation_fails(raw_df):
    """Test RawOptionSchema validation fails with invalid data."""
    raw_df["bid"] = -100.0  # Invalid bid
    with pytest.raises(pa.errors.SchemaErrors):
        RawOptionSchema.validate(raw_df, lazy=True)


def test_filter_instruments(raw_df):
    """Test filter_instruments removes invalid instruments."""
    # Add invalid instrument
    raw_df.loc[2] = {
        "instrument_name": "BTC-26JUN26-60000-C",
        "strike": 60000.0,
        "expiry_date": "2026-06-26T08:00:00.000Z",
        "option_type": "call",
        "bid": 0.0,  # Invalid bid
        "ask": 10.0,
        "underlying_price": 52000.0,
        "implied_volatility": 0.5,
        "funding_rate": 0.0001,
        "timestamp": "2026-03-16T02:30:00.000Z",
    }
    
    filtered_df = filter_instruments(raw_df)
    assert len(filtered_df) == 2  # Only valid instruments
    assert all(filtered_df["bid"] > 0)


def test_clean_and_enrich(raw_df):
    """Test clean_and_enrich computes correct moneyness and forward price."""
    clean_df = clean_and_enrich(raw_df)
    
    # Check moneyness
    expected_moneyness = 50000.0 / 52000.0
    assert clean_df["moneyness"].iloc[0] == pytest.approx(expected_moneyness)
    
    # Check forward price
    days_to_expiry = day_count_act_365(
        pd.Timestamp("2026-06-26T08:00:00.000Z"), pd.Timestamp("2026-03-16T02:30:00.000Z")
    )
    expected_forward = compute_forward_price(52000.0, 0.0001, days_to_expiry)
    assert clean_df["forward_price"].iloc[0] == pytest.approx(expected_forward)


def test_day_count_act_365():
    """Test ACT/365 day count."""
    expiry = pd.Timestamp("2026-06-26T08:00:00.000Z")
    timestamp = pd.Timestamp("2026-03-16T02:30:00.000Z")
    days = day_count_act_365(expiry, timestamp)
    assert days == pytest.approx(102.229, rel=1e-3)


def test_compute_forward_price():
    """Test forward price computation."""
    forward = compute_forward_price(52000.0, 0.0001, 102.229)
    assert forward == pytest.approx(52000.0 * (1 + 0.0001 * 102.229 / 365.0), rel=1e-6)