"""Pydantic v2 schemas for option chain data and calibration results."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator, model_validator


class OptionQuote(BaseModel):
    """Single option quote."""

    strike: float = Field(gt=0)
    expiry: date
    bid: float = Field(ge=0)
    ask: float = Field(ge=0)
    mid: float = Field(ge=0)
    implied_vol: float | None = Field(ge=0)
    open_interest: int = Field(ge=0)
    volume: int = Field(ge=0)
    option_type: str


class OptionChain(BaseModel):
    """Raw option chain for a single underlying."""

    model_config = {"arbitrary_types_allowed": True}

    ticker: str
    spot: float = Field(gt=0)
    timestamp: datetime
    quotes: list[OptionQuote]
    calls: list[OptionQuote] | pd.DataFrame = Field(default_factory=list)
    puts: list[OptionQuote] | pd.DataFrame = Field(default_factory=list)
    calls_df: pd.DataFrame | None = None
    puts_df: pd.DataFrame | None = None

    @property
    def strikes(self) -> np.ndarray:
        """Unique strikes across all options."""
        return np.unique([q.strike for q in self.quotes])

    @property
    def maturities(self) -> np.ndarray:
        """Unique maturities across all options (as datetime.date)."""
        return np.unique([q.expiry for q in self.quotes])

    @property
    def time_to_maturities(self) -> np.ndarray:
        """Time to maturities in years."""
        today = pd.Timestamp.now().date()
        return [(expiry - today).days / 365.25 for expiry in self.maturities]


class SVIParams(BaseModel):
    """SVI parameter set."""
    a: float = Field(gt=0)
    b: float = Field(gt=0)
    rho: float = Field(le=0, ge=-1)
    m: float = Field(ge=0)
    sigma: float = Field(gt=0)


class SSVIParams(BaseModel):
    """SSVI parameter set."""
    theta: float = Field(gt=0)
    rho: float = Field(le=0, ge=-1)
    eta: float = Field(gt=0)
    gamma: float = Field(le=1, gt=0)


OptionChain.model_rebuild()