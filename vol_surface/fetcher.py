"""
Fetch option chains from Yahoo Finance and cache in SQLite.

This module provides:
- Fetching option chains (Calls/Puts/Expiries) for a given underlying.
- Caching fetched data in SQLite to avoid redundant API calls.
"""

import sqlite3
import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import yfinance as yf
from dataclasses import dataclass


@dataclass
class OptionChain:
    """Dataclass for option chain data."""
    strikes: np.ndarray
    maturities: np.ndarray
    calls: pd.DataFrame  # Columns: [strike, maturity, price, iv]
    puts: pd.DataFrame   # Columns: [strike, maturity, price, iv]


class OptionChainFetcher:
    """Fetch option chains from Yahoo Finance and cache in SQLite."""

    def __init__(self, cache_dir: str = "data"):
        """
        Initialize the fetcher.

        Args:
            cache_dir (str): Directory to store the SQLite database. Defaults to "data".
        """
        self.cache_dir = cache_dir
        self.db_path = os.path.join(cache_dir, "option_chains.db")
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the SQLite database."""
        os.makedirs(self.cache_dir, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS option_chains (
            ticker TEXT NOT NULL,
            expiration TEXT NOT NULL,
            strike REAL NOT NULL,
            option_type TEXT NOT NULL,  -- 'call' or 'put'
            last_price REAL,
            implied_volatility REAL,
            PRIMARY KEY (ticker, expiration, strike, option_type)
        );
        """)
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS forward_prices (
            ticker TEXT PRIMARY KEY,
            forward_price REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        """)
        
        conn.commit()
        conn.close()

    def _cache_chain(
        self,
        ticker: str,
        expiration: str,
        option_type: str,
        data: pd.DataFrame,
    ) -> None:
        """Cache option chain data in SQLite."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Clear old data for this ticker/expiration/option_type
        cursor.execute(
            "DELETE FROM option_chains WHERE ticker = ? AND expiration = ? AND option_type = ?",
            (ticker, expiration, option_type),
        )
        
        # Insert new data
        records = data[["strike", "lastPrice", "impliedVolatility"]].to_records(index=False)
        cursor.executemany(
            "INSERT INTO option_chains (ticker, expiration, strike, option_type, last_price, implied_volatility) VALUES (?, ?, ?, ?, ?, ?)",
            [(ticker, expiration, strike, option_type, price, iv) for strike, price, iv in records],
        )
        
        conn.commit()
        conn.close()

    def _fetch_cached_chain(
        self,
        ticker: str,
        expiration: str,
        option_type: str,
    ) -> Optional[pd.DataFrame]:
        """Fetch cached option chain data from SQLite."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT strike, last_price, implied_volatility FROM option_chains WHERE ticker = ? AND expiration = ? AND option_type = ?",
            (ticker, expiration, option_type),
        )
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return None
        
        return pd.DataFrame(rows, columns=["strike", "lastPrice", "impliedVolatility"])

    def _cache_forward_price(
        self,
        ticker: str,
        forward_price: float,
    ) -> None:
        """Cache forward price in SQLite."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM forward_prices WHERE ticker = ?", (ticker,))
        cursor.execute(
            "INSERT INTO forward_prices (ticker, forward_price) VALUES (?, ?)",
            (ticker, forward_price),
        )
        
        conn.commit()
        conn.close()

    def _fetch_cached_forward_price(
        self,
        ticker: str,
    ) -> Optional[float]:
        """Fetch cached forward price from SQLite."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT forward_price FROM forward_prices WHERE ticker = ?", (ticker,))
        row = cursor.fetchone()
        conn.close()
        
        return row[0] if row else None

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
        underlying = yf.Ticker(ticker)
        expirations = underlying.options  # List of expiration dates
        
        calls_data = []
        puts_data = []
        
        for exp in expirations:
            maturity = (pd.to_datetime(exp) - pd.Timestamp.now()).days / 365.0
            
            if maturities is not None and maturity not in maturities:
                continue
            
            # Check cache
            cached_calls = self._fetch_cached_chain(ticker, exp, "call")
            cached_puts = self._fetch_cached_chain(ticker, exp, "put")
            
            if cached_calls is not None and cached_puts is not None:
                calls = cached_calls
                puts = cached_puts
            else:
                opt = underlying.option_chain(exp)
                calls = opt.calls
                puts = opt.puts
                
                # Cache data
                self._cache_chain(ticker, exp, "call", calls)
                self._cache_chain(ticker, exp, "put", puts)
            
            calls["maturity"] = maturity
            puts["maturity"] = maturity
            
            calls_data.append(calls[["strike", "maturity", "lastPrice", "impliedVolatility"]])
            puts_data.append(puts[["strike", "maturity", "lastPrice", "impliedVolatility"]])
        
        calls_df = pd.concat(calls_data, ignore_index=True)
        puts_df = pd.concat(puts_data, ignore_index=True)
        
        strikes = np.unique(calls_df["strike"].values)
        maturities = np.unique(calls_df["maturity"].values)
        
        # Cache forward price
        forward = self._fetch_cached_forward_price(ticker)
        if forward is None:
            forward = self._estimate_forward(calls_df, puts_df)
            self._cache_forward_price(ticker, forward)
        
        return OptionChain(
            strikes=strikes,
            maturities=maturities,
            calls=calls_df,
            puts=puts_df,
        )

    def _estimate_forward(
        self,
        calls: pd.DataFrame,
        puts: pd.DataFrame,
    ) -> float:
        """Estimate the forward price from calls and puts."""
        # Use put-call parity: F = C - P + K
        merged = pd.merge(
            calls,
            puts,
            on=["strike", "maturity"],
            suffixes=["_call", "_put"],
        )
        merged["diff"] = (
            merged["lastPrice_call"] - merged["lastPrice_put"] + merged["strike"]
        )
        return float(merged["diff"].mean())