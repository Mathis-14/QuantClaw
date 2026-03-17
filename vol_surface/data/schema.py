"""Pydantic v2 schemas for option chain data and calibration results."""

from __future__ import annotations

from datetime import date, datetime
from typing import List

import numpy as np
import pandas as pd
from datetime import date, datetime
from pydantic import BaseModel, Field
from typing import List
from numpy.typing import NDArray
import pandas as pd  # type: ignore


class VolSlice(BaseModel):
    """Volatility slice for a single maturity."""
    expiry: date
    T: float = Field(gt=0)
    forward: float = Field(gt=0)
    strikes: List[float]
    log_moneyness: List[float]
    total_variance: List[float]
    implied_vols: List[float]
    weights: List[float]
    
    def as_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return arrays for calibration."""
        return (
            np.array(self.log_moneyness),
            np.array(self.total_variance),
            np.array(self.implied_vols),
            np.array(self.weights),
        )


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


class VolSlice(BaseModel):
    """Single expiry slice for calibration."""
    T: float = Field(gt=0)
    forward: float = Field(gt=0)
    strikes: List[float]
    log_moneyness: List[float]
    total_variance: List[float]
    implied_vols: List[float]
    weights: List[float]


class OptionChain(BaseModel):
    """Container for a chain of options (calls/puts) with metadata."""
    calls: pd.DataFrame  # Standardized on DataFrame for compatibility
    puts: pd.DataFrame
    spot: float
    forward: float
    maturities: list[date]
    time_to_maturities: NDArray[np.float64]  # Array of time-to-maturity in years
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


class SliceResult(BaseModel):
    """Result of a single expiry slice calibration."""
    expiry: date | str  # Accept both for flexibility
    params: "SVIParams | SSVIParams"  # Forward reference
    rmse: float
    arbitrage_violations: List[str] = Field(default_factory=list)
    T: float | None = None  # Time to expiry (years)
    status: str | None = None  # e.g., "success", "failed"
    message: str | None = None  # Error message (if any)


class VolSurface(BaseModel):
    """Container for a full volatility surface."""
    slices: List[SliceResult]
    spot: float
    forward: float
    timestamp: datetime | None = None
    ticker: str | None = None
    spot_source: str | None = None
    options_source: str | None = None
    maturities: list[date] | None = None
    ssvi_params: SSVIParams | None = None
    surface_rmse: float | None = None
    arbitrage_violations: List[str] = Field(default_factory=list)


class SVIParams(BaseModel):
    """SVI parameter set."""
    a: float = Field(gt=0)
    b: float = Field(gt=0)
    rho: float = Field(ge=-1, le=1)
    m: float = Field(default=0.0)
    sigma: float = Field(gt=0)
    no_arb_lower_bound: float | None = None  # For arbitrage constraint checks
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