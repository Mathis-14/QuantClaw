"""Data loading and validation for ETH options."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_eth_options_data(file_path: Path) -> pd.DataFrame:
    """Load ETH options data from CSV and validate.

    Args:
        file_path: Path to the CSV file.

    Returns:
        Validated DataFrame with ETH options data.
    """
    df = pd.read_csv(file_path)
    
    # Validate required columns
    required_columns = {
        'strike', 'expiry_date', 'option_type', 'best_bid_price',
        'best_ask_price', 'underlying_price', 'mark_iv'
    }
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        raise ValueError(f"Missing required columns: {missing}")
    
    # Filter invalid rows
    df = df[df['mark_iv'].notna()]
    df = df[(df['best_bid_price'] > 0) & (df['best_ask_price'] > 0)]
    
    logger.info(f"Loaded {len(df)} valid ETH options from {file_path}")
    return df


def generate_synthetic_eth_data(
    output_path: Path,
    spot_price: float = 3500.0,
    strikes: Optional[np.ndarray] = None,
    expiries: Optional[list] = None,
    num_strikes: int = 20,
    num_expiries: int = 4,
) -> None:
    """Generate synthetic ETH options data for testing.

    Args:
        output_path: Path to save the synthetic data.
        spot_price: Underlying spot price.
        strikes: Array of strike prices. If None, generates linearly spaced strikes.
        expiries: List of expiry dates. If None, generates monthly expiries.
        num_strikes: Number of strikes to generate if `strikes` is None.
        num_expiries: Number of expiries to generate if `expiries` is None.
    """
    np.random.seed(42)
    
    if strikes is None:
        strikes = np.linspace(2500, 4500, num_strikes)
    if expiries is None:
        expiries = pd.date_range(start='2026-03-27', periods=num_expiries, freq='ME').tolist()
    
    option_types = ['C', 'P']
    data = []
    
    for expiry in expiries:
        for strike in strikes:
            for option_type in option_types:
                # Simulate realistic implied volatility (smile)
                if option_type == 'C':
                    iv = 0.6 + 0.2 * np.exp(-0.5 * ((strike - spot_price) / 500) ** 2)
                else:
                    iv = 0.6 + 0.3 * np.exp(-0.5 * ((strike - spot_price) / 500) ** 2)
                
                # Simulate bid/ask prices
                bid = max(0.01, np.random.normal(0.1, 0.02))
                ask = bid + np.random.normal(0.01, 0.005)
                
                data.append({
                    'instrument_name': f'ETH-{expiry.strftime("%d%b%y")}-{int(strike)}-{option_type}',
                    'strike': strike,
                    'expiry_date': expiry.strftime('%Y-%m-%d %H:%M:%S%z'),
                    'option_type': option_type,
                    'best_bid_price': bid,
                    'best_ask_price': ask,
                    'underlying_price': spot_price,
                    'mark_iv': iv,
                    'timestamp': '2026-03-16 14:01:44.365850+00:00',
                })
    
    # Convert to DataFrame and save
    df = pd.DataFrame(data)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Generated synthetic ETH data saved to {output_path}")