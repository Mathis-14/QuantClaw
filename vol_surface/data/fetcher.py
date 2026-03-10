"""Abstract data fetcher and yfinance adapter."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime

import pandas as pd
import yfinance as yf

from vol_surface.data.schema import OptionChain, OptionQuote

logger = logging.getLogger(__name__)


class DataFetcher(ABC):
    """Abstract base class for option-chain data sources."""

    @abstractmethod
    def fetch(self, ticker: str) -> OptionChain:
        """Fetch the full option chain for *ticker*."""
        ...

    @abstractmethod
    def available_expiries(self, ticker: str) -> list[str]:
        """Return available expiry dates."""
        ...


class YFinanceFetcher(DataFetcher):
    """Fetch option chains via yfinance."""

    def available_expiries(self, ticker: str) -> list[str]:
        t = yf.Ticker(ticker)
        return list(t.options)

    def fetch(self, ticker: str) -> OptionChain:
        t = yf.Ticker(ticker)
        spot = t.info.get("regularMarketPrice") or t.info.get("previousClose")
        if spot is None:
            raise RuntimeError(f"Cannot determine spot price for {ticker}")

        quotes: list[OptionQuote] = []
        for expiry_str in t.options:
            chain = t.option_chain(expiry_str)
            expiry_date = pd.Timestamp(expiry_str).date()
            for side, df in [("call", chain.calls), ("put", chain.puts)]:
                quotes.extend(self._parse_frame(df, expiry_date, side))

        logger.info("Fetched %d quotes for %s (spot=%.2f)", len(quotes), ticker, spot)
        return OptionChain(
            ticker=ticker,
            spot=float(spot),
            timestamp=datetime.utcnow(),
            quotes=quotes,
        )

    @staticmethod
    def _parse_frame(
        df: pd.DataFrame, expiry_date, option_type: str
    ) -> list[OptionQuote]:
        out: list[OptionQuote] = []
        for _, row in df.iterrows():
            bid = _safe_float(row.get("bid", 0))
            ask = _safe_float(row.get("ask", 0))
            if bid <= 0 or ask <= 0 or bid > ask:
                continue
            mid = (bid + ask) / 2.0
            iv = row.get("impliedVolatility")
            iv = float(iv) if pd.notna(iv) and iv > 0 else None
            out.append(
                OptionQuote(
                    strike=float(row["strike"]),
                    expiry=expiry_date,
                    bid=bid,
                    ask=ask,
                    mid=mid,
                    implied_vol=iv,
                    open_interest=_safe_int(row.get("openInterest", 0)),
                    volume=_safe_int(row.get("volume", 0)),
                    option_type=option_type,
                )
            )
        return out


def _safe_float(val, default: float = 0.0) -> float:
    try:
        v = float(val)
        return v if not pd.isna(v) else default
    except (TypeError, ValueError):
        return default


def _safe_int(val, default: int = 0) -> int:
    try:
        v = float(val)
        return int(v) if not pd.isna(v) else default
    except (TypeError, ValueError):
        return default
