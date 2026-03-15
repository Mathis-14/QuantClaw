"""
Volatility surface construction and interpolation.
"""

from typing import Optional
import numpy as np
import pandas as pd
from scipy.interpolate import RectBivariateSpline
from vol_surface.fetcher import OptionChain
from vol_surface.calibration import SABR


class VolatilitySurface:
    """Volatility surface construction and interpolation."""

    def __init__(
        self,
        option_chain: OptionChain,
        forward: Optional[float] = None,
    ):
        """
        Initialize the volatility surface.

        Args:
            option_chain (OptionChain): Option chain data.
            forward (float, optional): Forward price. If None, estimated from calls/puts.
        """
        self.option_chain = option_chain
        self.forward = forward or self._estimate_forward()
        self.surface = self._build_surface()

    def _estimate_forward(self) -> float:
        """
        Estimate the forward price from calls and puts.

        Returns:
            float: Estimated forward price.
        """
        # Use put-call parity: F = C - P + K
        calls = self.option_chain.calls
        puts = self.option_chain.puts
        
        # Find the strike with the smallest difference between call and put
        merged = pd.merge(
            calls,
            puts,
            on=["strike", "maturity"],
            suffixes=["_call", "_put"],
        )
        merged["diff"] = (
            merged["lastPrice_call"] - merged["lastPrice_put"] + merged["strike"]
        )
        
        # Average the differences to estimate forward
        return float(merged["diff"].mean())

    def _build_surface(self) -> RectBivariateSpline:
        """
        Build the volatility surface using SABR calibration.

        Returns:
            RectBivariateSpline: Interpolated volatility surface.
        """
        strikes = self.option_chain.strikes
        maturities = self.option_chain.maturities
        
        # Calibrate SABR for each maturity
        sabr_params = {}
        for maturity in maturities:
            calls = self.option_chain.calls[self.option_chain.calls["maturity"] == maturity]
            puts = self.option_chain.puts[self.option_chain.puts["maturity"] == maturity]
            
            # Use calls for calibration (puts are similar)
            market_vols = calls["impliedVolatility"].values
            alpha, rho, nu = SABR.calibrate(
                strikes=calls["strike"].values,
                maturities=np.full_like(calls["strike"].values, maturity),
                market_vols=market_vols,
                forward=self.forward,
            )
            sabr_params[maturity] = (alpha, rho, nu)
        
        # Build grid for interpolation
        vol_grid = np.zeros((len(strikes), len(maturities)))
        for i, strike in enumerate(strikes):
            for j, maturity in enumerate(maturities):
                alpha, rho, nu = sabr_params[maturity]
                vol_grid[i, j] = SABR.sabr_vol(
                    strike=strike,
                    forward=self.forward,
                    maturity=maturity,
                    alpha=alpha,
                    beta=0.5,
                    rho=rho,
                    nu=nu,
                )
        
        # Interpolate
        return RectBivariateSpline(strikes, maturities, vol_grid)

    def volatility(
        self,
        strike: float,
        maturity: float,
    ) -> float:
        """
        Get the implied volatility for a given strike and maturity.

        Args:
            strike (float): Strike price.
            maturity (float): Time to maturity (in years).

        Returns:
            float: Implied volatility.
        """
        return float(self.surface(strike, maturity))