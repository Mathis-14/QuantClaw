"""Abstract data fetcher and yfinance adapter."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime

import pandas as pd
import yfinance as yf

from schema import OptionChain, OptionQuote
from cleaner import _safe_float, _safe_int

logger = logging.getLogger(__name__)


TICKER_MAP: dict[str, str] = {
    "SPX": "^SPX",  # index spot (no options on yfinance)
    "SPY": "SPY",   # ETF proxy with liquid options
    "NDX": "^NDX",
    "QQQ": "QQQ",
}

OPTIONS_PROXY_MAP: dict[str, str] = {
    "SPX": "SPY",
    "NDX": "QQQ",
}


def resolve_tickers(logical_ticker: str) -> tuple[str, str]:
    """Return (spot_ticker, options_ticker) for a logical index/ETF name.

    Examples
    --------
    - \"SPX\" -> (\"^SPX\", \"SPY\")
    - \"SPY\" -> (\"SPY\", \"SPY\")
    - unknown -> (ticker, ticker)
    """
    spot_ticker = TICKER_MAP.get(logical_ticker, logical_ticker)
    options_ticker = OPTIONS_PROXY_MAP.get(logical_ticker, spot_ticker)
    return spot_ticker, options_ticker


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
    """Fetch option chains via yfinance.

    For index tickers like \"SPX\" and \"NDX\", spot is taken from the index
    (^SPX, ^NDX) while options are taken from a liquid ETF proxy (SPY, QQQ).
    """

    def available_expiries(self, ticker: str) -> list[str]:
        _, options_ticker = resolve_tickers(ticker)
        t_opt = yf.Ticker(options_ticker)
        return list(t_opt.options)

    def fetch(self, ticker: str) -> OptionChain:
        spot_ticker, options_ticker = resolve_tickers(ticker)

        # Spot from index (or ETF) using fast_info as primary source.
        t_spot = yf.Ticker(spot_ticker)
        spot: float | None
        try:
            fi = t_spot.fast_info
            spot = fi.last_price or fi.previous_close
        except Exception:  # pragma: no cover - defensive against yfinance changes
            spot = None

        if spot is None:
            info = t_spot.info or {}
            spot = info.get("regularMarketPrice") or info.get("previousClose")

        if spot is None:
            raise RuntimeError(f"Cannot determine spot price for {ticker} via {spot_ticker}")

        # Options from ETF proxy (or same ticker).
        t_opt = yf.Ticker(options_ticker)

        quotes: list[OptionQuote] = []
        for expiry_str in t_opt.options:
            chain = t_opt.option_chain(expiry_str)
            expiry_date = pd.Timestamp(expiry_str).date()
            for side, df in [("call", chain.calls), ("put", chain.puts)]:
                quotes.extend(self._parse_frame(df, expiry_date, side))

        logger.info(
            "Fetched %d quotes for %s using options=%s (spot=%.2f from %s)",
            len(quotes),
            ticker,
            options_ticker,
            spot,
            spot_ticker,
        )
        return OptionChain(
            ticker=ticker,  # logical ticker (e.g. \"SPX\")
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



