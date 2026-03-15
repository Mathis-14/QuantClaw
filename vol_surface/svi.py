"""
SVI (Stochastic Volatility Inspired) calibration with no-arbitrage constraints.

This module implements:
- 5-parameter SVI model for implied volatility.
- No-butterfly-arbitrage and no-calendar-spread-arbitrage checks.
- Calibration using scipy.optimize.minimize with debugging.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from typing import Tuple, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SVI:
    """SVI volatility model with no-arbitrage constraints."""

    @staticmethod
    def svi_total_variance(
        k: NDArray[np.float64],
        a: float,
        b: float,
        rho: float,
        m: float,
        epsilon: float,
    ) -> NDArray[np.float64]:
        """
        Compute total variance using SVI parameterization.

        Args:
            k (NDArray[np.float64]): Log-moneyness (log(K/F)).
            a (float): SVI parameter.
            b (float): SVI parameter.
            rho (float): SVI parameter (correlation).
            m (float): SVI parameter (shift).
            epsilon (float): SVI parameter (smoothing).

        Returns:
            NDArray[np.float64]: Total variance (sigma^2 * T).
        """
        return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + epsilon ** 2))

    @staticmethod
    def svi_implied_vol(
        k: NDArray[np.float64],
        a: float,
        b: float,
        rho: float,
        m: float,
        epsilon: float,
        T: float,
    ) -> NDArray[np.float64]:
        """
        Compute implied volatility from SVI total variance.

        Args:
            k (NDArray[np.float64]): Log-moneyness (log(K/F)).
            a (float): SVI parameter.
            b (float): SVI parameter.
            rho (float): SVI parameter (correlation).
            m (float): SVI parameter (shift).
            epsilon (float): SVI parameter (smoothing).
            T (float): Time to maturity (in years).

        Returns:
            NDArray[np.float64]: Implied volatility.
        """
        if T < 0.01:
            T = 0.01  # Avoid division by zero
        total_variance = SVI.svi_total_variance(k, a, b, rho, m, epsilon)
        return np.sqrt(total_variance / T)

    @staticmethod
    def _compute_derivatives(
        k: NDArray[np.float64],
        b: float,
        rho: float,
        m: float,
        epsilon: float,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Compute first and second derivatives of total variance w.r.t. log-moneyness.

        Args:
            k (NDArray[np.float64]): Log-moneyness (log(K/F)).
            b (float): SVI parameter.
            rho (float): SVI parameter (correlation).
            m (float): SVI parameter (shift).
            epsilon (float): SVI parameter (smoothing).

        Returns:
            Tuple[NDArray[np.float64], NDArray[np.float64]]: First and second derivatives.
        """
        denominator = np.sqrt((k - m) ** 2 + epsilon ** 2)
        dw_dk = b * (rho + (k - m) / denominator)
        d2w_dk2 = b * epsilon ** 2 / (denominator ** 3)
        return dw_dk, d2w_dk2

    @staticmethod
    def butterfly_arbitrage_condition(
        k: NDArray[np.float64],
        a: float,
        b: float,
        rho: float,
        m: float,
        epsilon: float,
    ) -> NDArray[np.float64]:
        """
        Check no-butterfly-arbitrage condition: g(k) >= 0 for all k.

        Args:
            k (NDArray[np.float64]): Log-moneyness (log(K/F)).
            a (float): SVI parameter.
            b (float): SVI parameter.
            rho (float): SVI parameter (correlation).
            m (float): SVI parameter (shift).
            epsilon (float): SVI parameter (smoothing).

        Returns:
            NDArray[np.float64]: g(k) values. Must be >= 0 for no arbitrage.
        """
        w = SVI.svi_total_variance(k, a, b, rho, m, epsilon)
        dw_dk, d2w_dk2 = SVI._compute_derivatives(k, b, rho, m, epsilon)
        
        g_k = (
            (1 - (k / w) * dw_dk) ** 2
            - 0.25 * (-1 / w + 0.25) * (dw_dk ** 2)
            + d2w_dk2
        )
        return g_k

    @staticmethod
    def calibrate(
        k: NDArray[np.float64],
        market_vols: NDArray[np.float64],
        T: float,
        initial_params: Optional[Tuple[float, float, float, float, float]] = None,
        enforce_arbitrage: bool = False,
        max_iter: int = 1000,
        ftol: float = 1e-6,
    ) -> Tuple[float, float, float, float, float]:
        """
        Calibrate SVI parameters to market implied volatilities.

        Args:
            k (NDArray[np.float64]): Log-moneyness (log(K/F)). Must be sorted.
            market_vols (NDArray[np.float64]): Market implied volatilities.
            T (float): Time to maturity (in years). Must be > 0.
            initial_params (Tuple[float, float, float, float, float], optional):
                Initial guess for (a, b, rho, m, epsilon). Defaults to None.
            enforce_arbitrage (bool): Whether to enforce no-arbitrage constraints.
                Defaults to False for debugging.
            max_iter (int): Maximum iterations for optimization. Defaults to 1000.
            ftol (float): Tolerance for optimization convergence. Defaults to 1e-6.

        Returns:
            Tuple[float, float, float, float, float]: Calibrated SVI parameters.

        Raises:
            ValueError: If inputs are invalid or calibration fails.
        """
        # Input validation
        if len(k) != len(market_vols):
            raise ValueError("k and market_vols must have the same length.")
        if len(k) < 5:
            raise ValueError("At least 5 data points required for calibration.")
        if T <= 0.0:
            raise ValueError("Time to maturity (T) must be > 0.")
        if not np.all(np.diff(k) >= 0):
            raise ValueError("Log-moneyness (k) must be sorted in ascending order.")
        
        logger.info("Starting SVI calibration for T=%.2fY with %d data points.", T, len(k))
        logger.debug("Input log-moneyness (k): %s", k)
        logger.debug("Input market implied vols: %s", market_vols)
        
        # Clip implied vols to avoid extreme values
        market_vols = np.clip(market_vols, 0.01, 2.0)
        
        # Initial guess based on market data
        if initial_params is None:
            atm_vol = np.interp(0, k, market_vols)  # ATM implied vol
            initial_params = (
                atm_vol ** 2 * T,  # a: ATM total variance
                0.1,              # b
                -0.5,             # rho
                0.0,              # m
                0.1,              # epsilon
            )
        logger.info("Initial params: %s", initial_params)
        
        def objective(params: NDArray[np.float64]) -> float:
            """Objective function: RMSE between model and market implied vols."""
            a, b, rho, m, epsilon = params
            model_vols = SVI.svi_implied_vol(k, a, b, rho, m, epsilon, T)
            rmse = np.sqrt(np.mean((model_vols - market_vols) ** 2))
            logger.debug("Params: %s, RMSE: %.6f", params, rmse)
            return rmse
        
        # Constraints: no butterfly arbitrage (g(k) >= 0)
        constraints = []
        if enforce_arbitrage:
            def constraint(params: NDArray[np.float64]) -> float:
                a, b, rho, m, epsilon = params
                g_k = SVI.butterfly_arbitrage_condition(k, a, b, rho, m, epsilon)
                min_g_k = np.min(g_k)
                logger.debug("Min g(k): %.6f", min_g_k)
                return min_g_k  # Must be >= 0
            constraints.append({"type": "ineq", "fun": constraint})
        
        # Parameter bounds
        bounds = [
            (0, None),      # a
            (0, None),      # b
            (-0.999, 0.999), # rho
            (None, None),   # m
            (1e-6, None),   # epsilon
        ]
        
        result = minimize(
            objective,
            initial_params,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": max_iter, "ftol": ftol},
        )
        
        if not result.success:
            logger.error("SVI calibration failed: %s", result.message)
            raise ValueError(f"SVI calibration failed: {result.message}")
        
        logger.info("Calibration successful. Final params: %s, RMSE: %.6f", result.x, result.fun)
        return tuple(result.x)  # type: ignore