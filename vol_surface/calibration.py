"""
Calibration routines for volatility surfaces (SABR, SVI).
"""

from typing import Tuple
import numpy as np
from scipy.optimize import minimize
from numpy.typing import NDArray


class SABR:
    """SABR volatility model calibration."""

    @staticmethod
    def sabr_vol(
        strike: float,
        forward: float,
        maturity: float,
        alpha: float,
        beta: float,
        rho: float,
        nu: float,
    ) -> float:
        """
        Compute SABR implied volatility.

        Args:
            strike (float): Strike price.
            forward (float): Forward price.
            maturity (float): Time to maturity (in years).
            alpha (float): Alpha parameter.
            beta (float): Beta parameter.
            rho (float): Rho parameter.
            nu (float): Nu parameter.

        Returns:
            float: Implied volatility.
        """
        if abs(strike - forward) < 1e-8:
            # ATM formula
            return alpha / (forward ** (1 - beta)) * (
                1
                + (
                    ((1 - beta) ** 2 * alpha ** 2) / (24 * forward ** (2 - 2 * beta))
                    + (rho * beta * nu * alpha) / (4 * forward ** (1 - beta))
                    + (nu ** 2 * (2 - 3 * rho ** 2)) / 24
                )
                * maturity
            )
        else:
            # Non-ATM formula
            z = (nu / alpha) * (forward * strike) ** ((1 - beta) / 2) * np.log(forward / strike)
            x_z = np.log((np.sqrt(1 - 2 * rho * z + z ** 2) + z - rho) / (1 - rho))
            return (
                alpha
                / ((forward * strike) ** ((1 - beta) / 2) * (1 + ((1 - beta) ** 2 / 24) * np.log(forward / strike) ** 2))
                * (z / x_z)
                * (
                    1
                    + (
                        ((1 - beta) ** 2 * alpha ** 2) / (24 * (forward * strike) ** (1 - beta))
                        + (rho * beta * nu * alpha) / (4 * (forward * strike) ** ((1 - beta) / 2))
                        + (nu ** 2 * (2 - 3 * rho ** 2)) / 24
                    )
                    * maturity
                )
            )

    @staticmethod
    def calibrate(
        strikes: NDArray[np.float64],
        maturities: NDArray[np.float64],
        market_vols: NDArray[np.float64],
        forward: float,
        beta: float = 0.5,
    ) -> Tuple[float, float, float]:
        """
        Calibrate SABR parameters to market volatilities.

        Args:
            strikes (NDArray[np.float64]): Array of strike prices.
            maturities (NDArray[np.float64]): Array of maturities (in years).
            market_vols (NDArray[np.float64]): Array of market implied volatilities.
            forward (float): Forward price.
            beta (float): Beta parameter. Defaults to 0.5.

        Returns:
            Tuple[float, float, float]: Calibrated (alpha, rho, nu).
        """
        def objective(params: NDArray[np.float64]) -> float:
            alpha, rho, nu = params
            model_vols = np.array([
                SABR.sabr_vol(strike, forward, maturity, alpha, beta, rho, nu)
                for strike, maturity in zip(strikes, maturities)
            ])
            return np.sum((model_vols - market_vols) ** 2)
        
        # Initial guess
        initial_params = np.array([0.2, 0.0, 0.2])
        
        # Bounds
        bounds = [(1e-6, None), (-0.999, 0.999), (1e-6, None)]
        
        # Optimize
        result = minimize(
            objective,
            initial_params,
            bounds=bounds,
            method="L-BFGS-B",
        )
        
        if not result.success:
            raise ValueError("SABR calibration failed.")
        
        return tuple(result.x)  # type: ignore