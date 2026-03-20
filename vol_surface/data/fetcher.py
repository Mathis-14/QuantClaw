"""Abstract data fetcher and yfinance adapter."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf

from vol_surface.data.schema import OptionChain, OptionQuote

logger = logging.getLogger(__name__)


TICKER_MAP: dict[str, str] = {
    "SPX": "^SPX",
    "SPY": "SPY",
    "NDX": "^NDX",
    "QQQ": "QQQ",
}

OPTIONS_PROXY_MAP: dict[str, str] = {
    "SPX": "SPY",
    "NDX": "QQQ",
}


def resolve_tickers(logical_ticker: str) -> tuple[str, str]:
    """Return (spot_ticker, options_ticker) for a logical index/ETF name.

    Examples: "SPX" -> ("^SPX", "SPY"),  "SPY" -> ("SPY", "SPY").
    """
    key = logical_ticker.upper()
    # Accept aliases such as "spx" and "^SPX".
    canonical = key[1:] if key.startswith("^") else key
    spot_ticker = TICKER_MAP.get(canonical, logical_ticker)
    options_ticker = OPTIONS_PROXY_MAP.get(canonical, spot_ticker)
    return spot_ticker, options_ticker


class DataFetcher(ABC):
    """Abstract base class for option-chain data sources."""

    @abstractmethod
    def fetch(self, ticker: str) -> OptionChain: ...

    @abstractmethod
    def available_expiries(self, ticker: str) -> list[str]: ...


class YFinanceFetcher(DataFetcher):
    """Fetch option chains via yfinance.

    For index tickers (SPX, NDX), spot comes from the index while options
    come from the liquid ETF proxy (SPY, QQQ).
    """

    def available_expiries(self, ticker: str) -> list[str]:
        _, options_ticker = resolve_tickers(ticker)
        return list(yf.Ticker(options_ticker).options)

    def fetch(self, ticker: str) -> OptionChain:
        spot_ticker, options_ticker = resolve_tickers(ticker)

        t_spot = yf.Ticker(spot_ticker)
        spot = _get_spot_price(t_spot)
        if spot is None:
            raise RuntimeError(
                f"Cannot determine spot price for {ticker} via {spot_ticker}"
            )

        t_opt = yf.Ticker(options_ticker)
        quotes: list[OptionQuote] = []
        for expiry_str in t_opt.options:
            chain = t_opt.option_chain(expiry_str)
            expiry_date = pd.Timestamp(expiry_str).date()
            for side, df in [("call", chain.calls), ("put", chain.puts)]:
                quotes.extend(_parse_frame(df, expiry_date, side))

        logger.info(
            "Fetched %d quotes for %s (options=%s, spot=%.2f from %s)",
            len(quotes), ticker, options_ticker, spot, spot_ticker,
        )
        return OptionChain(
            ticker=ticker,
            spot=float(spot),
            timestamp=datetime.now(timezone.utc),
            quotes=quotes,
        )


# ── Private helpers ─────────────────────────────────────────────────────────


def _get_spot_price(t: yf.Ticker) -> float | None:
    """Extract spot price from a yfinance Ticker, trying fast_info first."""
    try:
        fi = t.fast_info
        price = fi.last_price or fi.previous_close
        if price:
            return float(price)
    except Exception:
        pass
    info = t.info or {}
    return info.get("regularMarketPrice") or info.get("previousClose")


def _parse_frame(
    df: pd.DataFrame, expiry_date, option_type: str,
) -> list[OptionQuote]:
    """Convert a yfinance options DataFrame into OptionQuote objects."""
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
                bid=bid, ask=ask, mid=mid,
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
