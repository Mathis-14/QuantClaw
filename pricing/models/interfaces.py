"""
Interfaces for pricing models and volatility surfaces.

This module defines abstract base classes (ABCs) for:
- VolatilityModel: Interface for volatility models (e.g., Black-Scholes, SABR).
- PricingEngine: Interface for pricing engines (e.g., Monte Carlo, PDE).
"""

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable
import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class VolatilityModel(Protocol):
    """Interface for volatility models."""

    @abstractmethod
    def volatility(self, strike: float, maturity: float) -> float:
        """
        Compute the implied volatility for a given strike and maturity.

        Args:
            strike (float): Strike price of the option.
            maturity (float): Time to maturity (in years).

        Returns:
            float: Implied volatility.
        """
        ...


class PricingEngine(ABC):
    """Abstract base class for pricing engines."""

    @abstractmethod
    def price(
        self,
        spot: float,
        strike: float,
        maturity: float,
        vol_model: VolatilityModel,
        **kwargs,
    ) -> float:
        """
        Compute the price of an option using the specified volatility model.

        Args:
            spot (float): Current spot price of the underlying.
            strike (float): Strike price of the option.
            maturity (float): Time to maturity (in years).
            vol_model (VolatilityModel): Volatility model to use.
            **kwargs: Additional engine-specific parameters.

        Returns:
            float: Option price.
        """
        ...


class PathDependentOption(ABC):
    """Abstract base class for path-dependent options."""

    @abstractmethod
    def payoff(self, paths: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute the payoff for a set of simulated paths.

        Args:
            paths (NDArray[np.float64]): Simulated price paths of the underlying.
                Shape: (num_paths, num_time_steps).

        Returns:
            NDArray[np.float64]: Payoff for each path. Shape: (num_paths,).
        """
        ...