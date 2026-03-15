"""
Fetch option chains from market data sources.

This module provides:
- Fetching option chains (Calls/Puts/Expiries) for a given underlying.
- Support for multiple data sources (e.g., Yahoo Finance, local files).
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class OptionChain:
    """Dataclass for option chain data."""
    strikes: np.ndarray
    maturities: np.ndarray
    calls: pd.DataFrame  # Columns: [strike, maturity, price, iv]
    puts: pd.DataFrame   # Columns: [strike, maturity, price, iv]


class OptionChainFetcher:
    """Fetch option chains from market data sources."""

    def __init__(self, source: str = "yfinance"):
        """
        Initialize the fetcher.

        Args:
            source (str): Data source (e.g., "yfinance", "local"). Defaults to "yfinance".
        """
        self.source = source

    def fetch(
        self,
        ticker: str,
        maturities: Optional[List[float]] = None,
    ) -> OptionChain:
        """
        Fetch option chains for a given ticker.

        Args:
            ticker (str): Underlying asset ticker (e.g., "AAPL").
            maturities (List[float], optional): List of maturities (in years).
                If None, fetches all available maturities.

        Returns:
            OptionChain: Fetched option chain data.
        """
        if self.source == "yfinance":
            return self._fetch_yfinance(ticker, maturities)
        elif self.source == "local":
            return self._fetch_local(ticker, maturities)
        else:
            raise ValueError(f"Unknown data source: {self.source}")

    def _fetch_yfinance(
        self,
        ticker: str,
        maturities: Optional[List[float]],
    ) -> OptionChain:
        """
        Fetch option chains from Yahoo Finance.

        Args:
            ticker (str): Underlying asset ticker.
            maturities (List[float], optional): List of maturities (in years).

        Returns:
            OptionChain: Fetched option chain data.
        """
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError("yfinance is required. Install with `pip install yfinance`.")

        underlying = yf.Ticker(ticker)
        expirations = underlying.options  # List of expiration dates
        
        calls_data = []
        puts_data = []
        
        for exp in expirations:
            opt = underlying.option_chain(exp)
            maturity = (pd.to_datetime(exp) - pd.Timestamp.now()).days / 365.0
            
            if maturities is not None and maturity not in maturities:
                continue
            
            # Calls
            calls = opt.calls
            calls["maturity"] = maturity
            calls_data.append(calls[["strike", "maturity", "lastPrice", "impliedVolatility"]])
            
            # Puts
            puts = opt.puts
            puts["maturity"] = maturity
            puts_data.append(puts[["strike", "maturity", "lastPrice", "impliedVolatility"]])
        
        calls_df = pd.concat(calls_data, ignore_index=True)
        puts_df = pd.concat(puts_data, ignore_index=True)
        
        strikes = np.unique(calls_df["strike"].values)
        maturities = np.unique(calls_df["maturity"].values)
        
        return OptionChain(
            strikes=strikes,
            maturities=maturities,
            calls=calls_df,
            puts=puts_df,
        )

    def _fetch_local(
        self,
        ticker: str,
        maturities: Optional[List[float]],
    ) -> OptionChain:
        """
        Fetch option chains from local files (for testing).

        Args:
            ticker (str): Underlying asset ticker.
            maturities (List[float], optional): List of maturities (in years).

        Returns:
            OptionChain: Fetched option chain data.
        """
        # Mock data for testing
        strikes = np.linspace(80, 120, 5)
        maturities = np.array([0.1, 0.25, 0.5]) if maturities is None else np.array(maturities)
        
        calls_data = []
        puts_data = []
        
        for maturity in maturities:
            for strike in strikes:
                calls_data.append({
                    "strike": strike,
                    "maturity": maturity,
                    "lastPrice": max(0.0, 100 - strike) * np.exp(-0.05 * maturity),
                    "impliedVolatility": 0.2 + 0.1 * (strike - 100) / 100,
                })
                puts_data.append({
                    "strike": strike,
                    "maturity": maturity,
                    "lastPrice": max(0.0, strike - 100) * np.exp(-0.05 * maturity),
                    "impliedVolatility": 0.2 + 0.1 * (100 - strike) / 100,
                })
        
        calls_df = pd.DataFrame(calls_data)
        puts_df = pd.DataFrame(puts_data)
        
        return OptionChain(
            strikes=strikes,
            maturities=maturities,
            calls=calls_df,
            puts=puts_df,
        )