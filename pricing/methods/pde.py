"""
PDE pricing engine for barrier options using finite difference methods.

This module implements:
- Crank-Nicolson scheme for the Black-Scholes PDE.
- Barrier option pricing with discrete monitoring.
"""

from typing import Optional
import numpy as np
from numpy.typing import NDArray
from pricing.models.interfaces import PricingEngine, VolatilityModel


class PDEPricer(PricingEngine):
    """PDE pricing engine for barrier options."""

    def __init__(
        self,
        num_spatial_steps: int = 100,
        num_time_steps: int = 1000,
    ):
        """
        Initialize the PDE pricer.

        Args:
            num_spatial_steps (int): Number of spatial steps. Defaults to 100.
            num_time_steps (int): Number of time steps. Defaults to 1000.
        """
        self.num_spatial_steps = num_spatial_steps
        self.num_time_steps = num_time_steps

    def price(
        self,
        spot: float,
        strike: float,
        maturity: float,
        vol_model: VolatilityModel,
        barrier: float,
        barrier_type: str,
        option_type: str,
        risk_free_rate: float = 0.0,
        dividend_yield: float = 0.0,
    ) -> float:
        """
        Compute the price of a barrier option using finite difference methods.

        Args:
            spot (float): Current spot price.
            strike (float): Strike price.
            maturity (float): Time to maturity (in years).
            vol_model (VolatilityModel): Volatility model to use.
            barrier (float): Barrier level.
            barrier_type (str): Type of barrier (e.g., "up-and-out").
            option_type (str): "call" or "put".
            risk_free_rate (float): Risk-free interest rate. Defaults to 0.0.
            dividend_yield (float): Dividend yield. Defaults to 0.0.

        Returns:
            float: Option price.
        """
        # Spatial grid
        s_max = max(spot * 2, barrier * 1.5)
        s_min = max(0.01, barrier * 0.5)
        s_grid = np.linspace(s_min, s_max, self.num_spatial_steps)
        ds = s_grid[1] - s_grid[0]
        
        # Time grid
        dt = maturity / self.num_time_steps
        
        # Initialize option value grid
        if option_type == "call":
            option_values = np.maximum(s_grid - strike, 0.0)
        else:
            option_values = np.maximum(strike - s_grid, 0.0)
        
        # Boundary conditions
        if barrier_type == "up-and-out":
            option_values[s_grid >= barrier] = 0.0
        elif barrier_type == "down-and-out":
            option_values[s_grid <= barrier] = 0.0
        
        # Crank-Nicolson scheme
        alpha = 0.5 * dt * (
            (risk_free_rate - dividend_yield) * s_grid / ds ** 2
            + 0.5 * vol_model.volatility(strike, maturity) ** 2 * s_grid ** 2 / ds ** 2
        )
        beta = 0.5 * dt * (
            - (risk_free_rate - dividend_yield) * s_grid / ds
            - 0.5 * vol_model.volatility(strike, maturity) ** 2 * s_grid ** 2 / ds ** 2
        )
        
        # Tridiagonal matrix
        A = np.diag(1 + 2 * alpha) + np.diag(-alpha[1:], k=1) + np.diag(-alpha[:-1], k=-1)
        B = np.diag(1 - 2 * alpha) + np.diag(alpha[1:], k=1) + np.diag(alpha[:-1], k=-1)
        
        # Apply boundary conditions
        if barrier_type in ["up-and-out", "up-and-in"]:
            A[-1, :] = 0
            A[-1, -1] = 1
            B[-1, :] = 0
        if barrier_type in ["down-and-out", "down-and-in"]:
            A[0, :] = 0
            A[0, 0] = 1
            B[0, :] = 0
        
        # Time-stepping
        for _ in range(self.num_time_steps):
            option_values = np.linalg.solve(A, B @ option_values)
            
            # Apply barrier conditions
            if barrier_type == "up-and-out":
                option_values[s_grid >= barrier] = 0.0
            elif barrier_type == "down-and-out":
                option_values[s_grid <= barrier] = 0.0
        
        # Interpolate to spot price
        price = np.interp(spot, s_grid, option_values)
        return price