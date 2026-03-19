"""Pydantic v2 schemas for option-chain data and calibration results."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

import numpy as np
from pydantic import BaseModel, Field, model_validator


# ── Market data ─────────────────────────────────────────────────────────────


class OptionQuote(BaseModel):
    """Single option contract quote."""

    strike: float = Field(gt=0)
    expiry: date
    bid: float = Field(ge=0)
    ask: float = Field(ge=0)
    mid: float = Field(ge=0)
    implied_vol: float | None = None
    open_interest: int = Field(ge=0, default=0)
    volume: int = Field(ge=0, default=0)
    option_type: str = Field(pattern=r"^(call|put)$")

    @model_validator(mode="after")
    def _bid_leq_ask(self) -> OptionQuote:
        if self.bid > self.ask:
            raise ValueError(f"bid ({self.bid}) > ask ({self.ask})")
        return self


class OptionChain(BaseModel):
    """Raw option chain for a single underlying."""

    ticker: str
    spot: float = Field(gt=0)
    timestamp: datetime
    quotes: list[OptionQuote]


class VolSlice(BaseModel):
    """Cleaned implied-vol slice for a single maturity."""

    expiry: date
    T: float = Field(gt=0, description="Time to expiry in years")
    forward: float = Field(gt=0)
    strikes: list[float]
    log_moneyness: list[float]
    total_variance: list[float]
    implied_vols: list[float]
    weights: list[float]

    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode="after")
    def _consistent_lengths(self) -> VolSlice:
        n = len(self.strikes)
        for name in ("log_moneyness", "total_variance", "implied_vols", "weights"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"{name} length mismatch: expected {n}")
        return self

    def as_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (k, w, iv, weights) as numpy arrays."""
        return (
            np.array(self.log_moneyness),
            np.array(self.total_variance),
            np.array(self.implied_vols),
            np.array(self.weights),
        )


# ── Model parameters ────────────────────────────────────────────────────────


class SVIParams(BaseModel):
    """Raw SVI parameters for a single slice."""

    a: float
    b: float = Field(ge=0)
    rho: float = Field(gt=-1, lt=1)
    m: float
    sigma: float = Field(gt=0)

    @property
    def no_arb_lower_bound(self) -> float:
        """a + b * sigma * sqrt(1 - rho^2) >= 0 is required for no-arbitrage."""
        return self.a + self.b * self.sigma * np.sqrt(1 - self.rho**2)


class SSVIParams(BaseModel):
    """SSVI surface parameters (Gatheral-Jacquier 2014)."""

    rho: float = Field(gt=-1, lt=1)
    eta: float = Field(gt=0)
    gamma: float = Field(gt=0, le=1)

    @model_validator(mode="after")
    def _no_arb_condition(self) -> SSVIParams:
        lhs = self.eta * (1 + abs(self.rho))
        if lhs > 4:
            raise ValueError(f"No-arb violated: eta*(1+|rho|) = {lhs:.4f} > 4")
        return self


# ── Calibration output ──────────────────────────────────────────────────────


class SliceResult(BaseModel):
    """Calibration result for a single maturity slice."""

    expiry: str
    T: float
    svi_params: SVIParams | None = None
    slice_rmse: float | None = None
    status: str = "ok"
    message: str = ""


class ArbitrageViolation(BaseModel):
    """A detected arbitrage violation."""

    type: str = Field(pattern=r"^(butterfly|calendar)$")
    maturity: str
    strike: float
    severity: float = 0.0


class VolSurface(BaseModel):
    """Full calibration output."""

    timestamp: str
    ticker: str
    spot: float
    spot_source: str | None = None
    options_source: str | None = None
    maturities: list[SliceResult]
    ssvi_params: SSVIParams | None = None
    surface_rmse: float | None = None
    arbitrage_violations: list[ArbitrageViolation] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
