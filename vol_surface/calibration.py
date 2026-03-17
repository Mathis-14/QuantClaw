"""Calibration utilities for volatility models."""

from typing import Optional
import numpy as np
import pandas as pd
from vol_surface.fetcher import OptionChain, OptionQuote


def load_and_preprocess_data(
    file_path: str,
    underlying_price: Optional[float] = None,
) -> OptionChain:
    """Load and preprocess options data from a CSV file.

    Args:
        file_path: Path to the CSV file.
        underlying_price: Manually specified underlying price. If None, estimated.

    Returns:
        OptionChain: Preprocessed option chain.
    """
    df = pd.read_csv(file_path)
    df["expiry_date"] = pd.to_datetime(df["expiry_date"], unit="ms")
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Estimate underlying price if not provided
    if underlying_price is None:
        underlying_price = df["strike"].median()  # Placeholder: refine later

    # Create OptionQuote objects
    quotes = []
    for _, row in df.iterrows():
        quotes.append(
            OptionQuote(
                strike=float(row["strike"]),
                expiry=row["expiry_date"].date(),
                bid=float(row["bid"]),
                ask=float(row["ask"]),
                mid=(float(row["bid"]) + float(row["ask"])) / 2,
                implied_vol=float(row["implied_volatility"]) if pd.notna(row["implied_volatility"]) else None,
                open_interest=0,
                volume=0,
                option_type="call" if "C" in row["instrument_name"] else "put",
            )
        )

    # Separate calls and puts
    calls = [q for q in quotes if q.option_type == "call"]
    puts = [q for q in quotes if q.option_type == "put"]

    return OptionChain(
        ticker="BTC" if "BTC" in file_path else "ETH",
        spot=underlying_price,
        timestamp=pd.Timestamp.now(),
        quotes=quotes,
        calls=calls,
        puts=puts,
    )