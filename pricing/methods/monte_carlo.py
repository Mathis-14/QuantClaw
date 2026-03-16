"""
Monte Carlo pricing engine for path-dependent options.

This module implements:
- Geometric Brownian Motion (GBM) path generation.
- Pricing for path-dependent options (e.g., barrier, autocallable).
"""

from typing import Optional
import numpy as np
from numpy.typing import NDArray
from pricing.models.interfaces import PricingEngine, VolatilityModel, PathDependentOption


class MonteCarloPricer(PricingEngine):
    """Monte Carlo pricing engine for path-dependent options."""

    def __init__(
        self,
        num_paths: int = 100_000,
        num_steps: int = 252,
        seed: Optional[int] = None,
    ):
        """
        Initialize the Monte Carlo pricer.

        Args:
            num_paths (int): Number of Monte Carlo paths. Defaults to 100,000.
            num_steps (int): Number of time steps per path. Defaults to 252.
            seed (int, optional): Random seed for reproducibility. Defaults to None.
        """
        self.num_paths = num_paths
        self.num_steps = num_steps
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

    def price(
        self,
        spot: float,
        strike: float,
        maturity: float,
        vol_model: VolatilityModel,
        option: PathDependentOption,
        risk_free_rate: float = 0.0,
    ) -> float:
        """
        Compute the price of a path-dependent option using Monte Carlo simulation.

        Args:
            spot (float): Current spot price of the underlying.
            strike (float): Strike price of the option.
            maturity (float): Time to maturity (in years).
            vol_model (VolatilityModel): Volatility model to use.
            option (PathDependentOption): Path-dependent option to price.
            risk_free_rate (float): Risk-free interest rate. Defaults to 0.0.

        Returns:
            float: Option price.
        """
        # Generate paths using Geometric Brownian Motion
        paths = self._generate_paths(spot, maturity, vol_model)
        
        # Compute payoff for each path
        payoffs = option.payoff(paths)
        
        # Discount and average
        discount_factor = np.exp(-risk_free_rate * maturity)
        price = discount_factor * np.mean(payoffs)
        
        return price

    def _generate_paths(
        self,
        spot: float,
        maturity: float,
        vol_model: VolatilityModel,
    ) -> NDArray[np.float64]:
        """
        Generate Monte Carlo paths using Geometric Brownian Motion.

        Args:
            spot (float): Current spot price.
            maturity (float): Time to maturity (in years).
            vol_model (VolatilityModel): Volatility model to use.

        Returns:
            NDArray[np.float64]: Simulated price paths. Shape: (num_paths, num_steps).
        """
        dt = maturity / self.num_steps
        paths = np.zeros((self.num_paths, self.num_steps + 1))
        paths[:, 0] = spot
        
        for t in range(1, self.num_steps + 1):
            current_spot = paths[:, t - 1]
            current_time = (t - 1) * dt
            
            # Compute volatility for each path (stochastic volatility models would vary here)
            vol = vol_model.volatility(strike=current_spot, maturity=maturity - current_time)
            
            # Generate random shocks
            shocks = np.random.standard_normal(self.num_paths)
            
            # GBM update
            paths[:, t] = current_spot * np.exp(
                (risk_free_rate - 0.5 * vol ** 2) * dt + vol * np.sqrt(dt) * shocks
            )
        
        return paths[:, 1:]  # Exclude initial spot