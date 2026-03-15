"""
Barrier option pricing models.

This module implements:
- Up-and-Out Call/Put
- Down-and-In Call/Put

Supports discrete monitoring and rebates.
"""

from typing import Literal, Optional
import numpy as np
from numpy.typing import NDArray
from pricing.models.interfaces import PathDependentOption, VolatilityModel


class BarrierOption(PathDependentOption):
    """Base class for barrier options."""

    def __init__(
        self,
        barrier_type: Literal["up-and-out", "down-and-in", "up-and-in", "down-and-out"],
        option_type: Literal["call", "put"],
        barrier: float,
        strike: float,
        rebate: Optional[float] = None,
        monitoring: Literal["continuous", "discrete"] = "discrete",
    ):
        """
        Initialize a barrier option.

        Args:
            barrier_type (str): Type of barrier (e.g., "up-and-out").
            option_type (str): "call" or "put".
            barrier (float): Barrier level.
            strike (float): Strike price.
            rebate (float, optional): Rebate paid if barrier is hit. Defaults to None.
            monitoring (str): "continuous" or "discrete". Defaults to "discrete".
        """
        self.barrier_type = barrier_type
        self.option_type = option_type
        self.barrier = barrier
        self.strike = strike
        self.rebate = rebate
        self.monitoring = monitoring

    def payoff(self, paths: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute the payoff for a set of simulated paths.

        Args:
            paths (NDArray[np.float64]): Simulated price paths of the underlying.
                Shape: (num_paths, num_time_steps).

        Returns:
            NDArray[np.float64]: Payoff for each path. Shape: (num_paths,).
        """
        final_prices = paths[:, -1]
        
        # Compute vanilla payoff
        if self.option_type == "call":
            vanilla_payoff = np.maximum(final_prices - self.strike, 0.0)
        else:
            vanilla_payoff = np.maximum(self.strike - final_prices, 0.0)
        
        # Check barrier condition
        if self.barrier_type == "up-and-out":
            hit_barrier = np.any(paths >= self.barrier, axis=1)
            payoff = np.where(hit_barrier, self.rebate or 0.0, vanilla_payoff)
        elif self.barrier_type == "down-and-in":
            hit_barrier = np.any(paths <= self.barrier, axis=1)
            payoff = np.where(hit_barrier, vanilla_payoff, self.rebate or 0.0)
        elif self.barrier_type == "up-and-in":
            hit_barrier = np.any(paths >= self.barrier, axis=1)
            payoff = np.where(hit_barrier, vanilla_payoff, self.rebate or 0.0)
        elif self.barrier_type == "down-and-out":
            hit_barrier = np.any(paths <= self.barrier, axis=1)
            payoff = np.where(hit_barrier, self.rebate or 0.0, vanilla_payoff)
        else:
            raise ValueError(f"Unknown barrier type: {self.barrier_type}")
        
        return payoff