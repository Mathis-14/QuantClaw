"""Option chain data schema."""

from dataclasses import dataclass
from datetime import date
from typing import List


@dataclass
class OptionQuote:
    """Single option quote."""
    strike: float
    expiry: date
    bid: float
    ask: float
    mid: float
    implied_vol: float | None
    open_interest: int
    volume: int
    option_type: str  # "call" or "put"


@dataclass
class OptionChain:
    """Full option chain for a ticker."""
    ticker: str
    spot: float
    timestamp: date
    quotes: List[OptionQuote]