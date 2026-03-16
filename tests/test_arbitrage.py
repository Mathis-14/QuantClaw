"""
Tests for arbitrage checks.

Run with:
    python -m pytest tests/test_arbitrage.py -v
"""

import pandas as pd
import pytest
from src.arbitrage import (
    ArbitrageViolationWarning,
    butterfly_check,
    calendar_spread_check,
    compute_density,
    check_arbitrage,
)


@pytest.fixture
def sample_df():
    """Fixture for a sample volatility surface DataFrame."""
    data = {
        "expiry_date": [
            pd.Timestamp("2026-06-26"), pd.Timestamp("2026-06-26"), 
            pd.Timestamp("2026-09-25"), pd.Timestamp("2026-09-25"),
        ],
        "strike": [50000.0, 52000.0, 50000.0, 52000.0],
        "total_variance": [0.05, 0.045, 0.04, 0.035],  # Calendar arbitrage: 0.04 < 0.05
        "forward_price": [52000.0, 52000.0, 52000.0, 52000.0],
    }
    return pd.DataFrame(data)


def test_calendar_spread_check(sample_df):
    """Test calendar spread check detects violations."""
    violations = calendar_spread_check(sample_df)
    assert len(violations) == 2
    assert all(v.violation_type == "calendar" for v in violations)
    assert {v.strike for v in violations} == {50000.0, 52000.0}
    assert all(v.severity > 0 for v in violations)


def test_butterfly_check(sample_df):
    """Test butterfly check detects violations."""
    # Add density column (manually computed for test)
    sample_df["density"] = [1.0, 0.9, 1.0, -0.1]  # Butterfly arbitrage: -0.1 < 0
    violations = butterfly_check(sample_df)
    assert len(violations) == 1
    assert violations[0].violation_type == "butterfly"
    assert violations[0].strike == 52000.0
    assert violations[0].severity > 0


def test_compute_density(sample_df):
    """Test density computation."""
    df_with_density = compute_density(sample_df)
    assert "density" in df_with_density.columns
    assert not df_with_density["density"].isna().any()


def test_check_arbitrage(sample_df):
    """Test full arbitrage check."""
    print("Test DataFrame columns:", sample_df.columns)
    violations = check_arbitrage(sample_df)
    assert len(violations) == 1  # Only calendar violation (butterfly requires density)


def test_no_arbitrage():
    """Test no arbitrage case."""
    data = {
        "expiry_date": [
            pd.Timestamp("2026-06-26"), pd.Timestamp("2026-06-26"), 
            pd.Timestamp("2026-09-25"), pd.Timestamp("2026-09-25"),
        ],
        "strike": [50000.0, 52000.0, 50000.0, 52000.0],
        "total_variance": [0.04, 0.045, 0.05, 0.055],  # No calendar arbitrage
        "forward_price": [52000.0, 52000.0, 52000.0, 52000.0],
    }
    df = pd.DataFrame(data)
    df = compute_density(df)
    
    violations = check_arbitrage(df)
    assert len(violations) == 0