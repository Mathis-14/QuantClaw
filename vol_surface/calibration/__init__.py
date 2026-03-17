"""Calibration utilities for volatility models."""

from typing import Optional
import numpy as np
import pandas as pd
from vol_surface.data.schema import OptionChain, OptionQuote


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
        implied_vol = float(row["implied_volatility"]) if pd.notna(row["implied_volatility"]) else None
        # If implied_vol is missing, use a placeholder (e.g., 0.5)
        if implied_vol is None:
            implied_vol = 0.5  # Placeholder: refine later
        quotes.append(
            OptionQuote(
                strike=float(row["strike"]),
                expiry=row["expiry_date"].date(),
                bid=float(row["bid"]),
                ask=float(row["ask"]),
                mid=(float(row["bid"]) + float(row["ask"])) / 2,
                implied_vol=implied_vol,
                open_interest=0,
                volume=0,
                option_type="call" if "C" in row["instrument_name"] else "put",
            )
        )

    # Separate calls and puts
    calls = [q for q in quotes if q.option_type == "call"]
    puts = [q for q in quotes if q.option_type == "put"]

    # Convert to DataFrames for compatibility
    calls_df = pd.DataFrame([
        {
            "strike": q.strike,
            "expiry_date": q.expiry,
            "bid": q.bid,
            "ask": q.ask,
            "mid": q.mid,
            "impliedVolatility": q.implied_vol,
            "option_type": q.option_type,
        }
        for q in calls
    ])
    puts_df = pd.DataFrame([
        {
            "strike": q.strike,
            "expiry_date": q.expiry,
            "bid": q.bid,
            "ask": q.ask,
            "mid": q.mid,
            "impliedVolatility": q.implied_vol,
            "option_type": q.option_type,
        }
        for q in puts
    ])

    # Convert expiry_date to datetime for filtering
    if not calls_df.empty:
        calls_df["expiry_date"] = pd.to_datetime(calls_df["expiry_date"])
    if not puts_df.empty:
        puts_df["expiry_date"] = pd.to_datetime(puts_df["expiry_date"])

    print(f"Calls DataFrame: {calls_df.head()}")
    print(f"Puts DataFrame: {puts_df.head()}")

    chain = OptionChain(
        ticker="BTC" if "BTC" in file_path else "ETH",
        spot=underlying_price,
        timestamp=pd.Timestamp.now(),
        quotes=quotes,
        calls=calls,
        puts=puts,
        calls_df=calls_df,
        puts_df=puts_df,
    )
    return chain