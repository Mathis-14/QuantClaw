"""
Autocallable option pricing models.

This module implements:
- Athena: Memory coupon + capital protection.
- Phoenix: Memory coupon + conditional capital protection.
"""

from typing import Literal, Optional
import numpy as np
from numpy.typing import NDArray
from pricing.models.interfaces import PathDependentOption


class AutocallableOption(PathDependentOption):
    """Base class for autocallable options."""

    def __init__(
        self,
        autocall_type: Literal["athena", "phoenix"],
        strike: float,
        coupon: float,
        barrier: float,
        observation_dates: NDArray[np.float64],
        memory: bool = True,
    ):
        """
        Initialize an autocallable option.

        Args:
            autocall_type (str): "athena" or "phoenix".
            strike (float): Strike price.
            coupon (float): Coupon rate (e.g., 0.1 for 10%).
            barrier (float): Barrier level for autocall/protection.
            observation_dates (NDArray[np.float64]): Array of observation dates (in years).
            memory (bool): Whether to pay missed coupons at autocall. Defaults to True.
        """
        self.autocall_type = autocall_type
        self.strike = strike
        self.coupon = coupon
        self.barrier = barrier
        self.observation_dates = observation_dates
        self.memory = memory

    def payoff(self, paths: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute the payoff for a set of simulated paths.

        Args:
            paths (NDArray[np.float64]): Simulated price paths of the underlying.
                Shape: (num_paths, num_time_steps).

        Returns:
            NDArray[np.float64]: Payoff for each path. Shape: (num_paths,).
        """
        num_paths, num_steps = paths.shape
        payoff = np.zeros(num_paths)
        
        # Find the indices of observation dates in the time grid
        time_grid = np.linspace(0, self.observation_dates[-1], num_steps)
        obs_indices = np.searchsorted(time_grid, self.observation_dates)
        
        # Track missed coupons for memory autocallables
        missed_coupons = np.zeros(num_paths)
        
        for i, obs_idx in enumerate(obs_indices):
            current_prices = paths[:, obs_idx]
            autocall_condition = current_prices >= self.strike
            
            if self.memory:
                # Pay missed coupons + current coupon if autocall
                payoff += np.where(
                    autocall_condition,
                    (1.0 + missed_coupons + self.coupon) * (i + 1) / len(self.observation_dates),
                    0.0,
                )
                # Update missed coupons for paths not autocalled
                missed_coupons += np.where(~autocall_condition, self.coupon, 0.0)
            else:
                # Pay only current coupon if autocall
                payoff += np.where(
                    autocall_condition,
                    (1.0 + self.coupon) * (i + 1) / len(self.observation_dates),
                    0.0,
                )
            
            # Early exit for autocalled paths
            paths = paths[~autocall_condition]
            if len(paths) == 0:
                break
        
        # Final payoff for non-autocalled paths
        if len(paths) > 0:
            final_prices = paths[:, -1]
            if self.autocall_type == "athena":
                # Capital protection: pay back 100% of notional
                payoff[~autocall_condition] = 1.0
            elif self.autocall_type == "phoenix":
                # Conditional capital protection: pay back min(100%, S_T / strike)
                payoff[~autocall_condition] = np.minimum(1.0, final_prices / self.strike)
            else:
                raise ValueError(f"Unknown autocall type: {self.autocall_type}")
        
        return payoff