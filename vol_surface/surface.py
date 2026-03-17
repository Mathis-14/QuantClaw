"""
Volatility surface construction and interpolation.
"""

from typing import Optional, Literal
import numpy as np
import pandas as pd
from scipy.interpolate import RectBivariateSpline
from vol_surface.fetcher import OptionChain
from vol_surface.models.svi import svi_implied_vol
from vol_surface.models.ssvi import ssvi_implied_vol, calibrate_ssvi


class VolatilitySurface:
    """Volatility surface for a single underlying."""

    def __init__(
        self,
        option_chain: OptionChain,
        forward: Optional[float] = None,
        model: Literal["SVI", "SSVI"] = "SSVI",
    ):
        """
        Initialize the volatility surface.

        Args:
            option_chain (OptionChain): Option chain data.
            forward (float, optional): Forward price. If None, estimated from calls/puts.
            model (str): Model to use for surface construction (SVI, SSVI).
        """
        self.option_chain = option_chain
        self.forward = forward or self._estimate_forward()
        self.model = model
        self.surface = self._build_surface()

    def _estimate_forward(self) -> float:
        """Estimate forward price from put-call parity."""
        calls = self.option_chain.calls_df
        puts = self.option_chain.puts_df

        # If puts are missing, use spot as forward
        if puts.empty:
            return self.option_chain.spot

        # Merge calls and puts on strike
        merged = pd.merge(
            calls, puts, on=["strike", "expiry_date"], suffixes=("_call", "_put")
        )

        # Use mid prices
        merged["diff"] = (
            merged["mid_call"] - merged["mid_put"] + merged["strike"]
        )

        # Average the differences to estimate forward
        return float(merged["diff"].mean())

    def _build_surface(self) -> RectBivariateSpline:
        """
        Build the volatility surface using the specified model (SVI, SSVI).

        Returns:
            RectBivariateSpline: Interpolated volatility surface.
        """
        strikes = self.option_chain.strikes
        maturities = self.option_chain.maturities

        if self.model == "SVI":
            return self._build_svi_surface(strikes, maturities)
        elif self.model == "SSVI":
            return self._build_ssvi_surface(strikes, maturities)
        else:
            raise ValueError(f"Unsupported model: {self.model}")

    def _build_svi_surface(self, strikes: np.ndarray, maturities: np.ndarray) -> RectBivariateSpline:
        """Build the volatility surface using SVI calibration."""
        svi_params = {}
        for maturity in maturities:
            calls = self.option_chain.calls_df[self.option_chain.calls_df["expiry_date"] == maturity]
# puts = self.option_chain.puts_df[self.option_chain.puts_df["expiry_date"] == maturity]  # Unused (puts not needed for calibration)

            # Use calls for calibration (puts are similar)
# market_vols = calls["impliedVolatility"].values  # Unused (implied vols not needed here)
            k = np.log(calls["strike"].values / self.forward)
# w = market_vols**2 * maturity  # Unused (total variance not needed here)

            # Calibrate SVI (placeholder: use existing SVI calibration logic)
            # For now, use dummy parameters; replace with actual calibration
            svi_params[maturity] = (0.1, 0.5, -0.3, 0.0, 0.2)

        # Build grid for interpolation
        vol_grid = np.zeros((len(strikes), len(maturities)))
        for i, strike in enumerate(strikes):
            for j, maturity in enumerate(maturities):
                a, b, rho, m, sigma = svi_params[maturity]
                k = np.log(strike / self.forward)
                vol_grid[i, j] = svi_implied_vol(k, maturity, a, b, rho, m, sigma)

        # Interpolate
        return RectBivariateSpline(strikes, self.option_chain.time_to_maturities, vol_grid)

    def _build_ssvi_surface(self, strikes: np.ndarray, maturities: np.ndarray) -> RectBivariateSpline:
        """Build the volatility surface using SSVI calibration."""
        ssvi_params = {}
        for i, maturity in enumerate(self.option_chain.time_to_maturities):
            calls = self.option_chain.calls_df[self.option_chain.calls_df["expiry_date"] == pd.to_datetime(self.option_chain.maturities[i])]
# puts = self.option_chain.puts_df[self.option_chain.puts_df["expiry_date"] == pd.to_datetime(self.option_chain.maturities[i])]  # Unused if not self.option_chain.puts_df.empty else None

            # Use calls for calibration (puts are similar)
            market_vols = calls["impliedVolatility"].values
            k = np.log(calls["strike"].values / self.forward)

            # Estimate theta (ATM total variance)
            atm_vol = market_vols[np.argmin(np.abs(k))]
            theta = atm_vol**2 * maturity

            # Calibrate SSVI
            rho, eta, gamma = calibrate_ssvi(k, maturity, market_vols, theta)
            ssvi_params[maturity] = (theta, rho, eta, gamma)

        # Build grid for interpolation
        vol_grid = np.zeros((len(strikes), len(self.option_chain.time_to_maturities)))
        for i, strike in enumerate(strikes):
            for j, maturity in enumerate(self.option_chain.time_to_maturities):
                theta, rho, eta, gamma = ssvi_params[maturity]
                k = np.log(strike / self.forward)
                vol_grid[i, j] = ssvi_implied_vol(k, maturity, theta, rho, eta, gamma)

        # Interpolate
        return RectBivariateSpline(strikes, self.option_chain.time_to_maturities, vol_grid)

    def volatility(self, strike: float, maturity: float) -> float:
        """Get implied volatility for a given strike and maturity."""
        return float(self.surface(strike, maturity, grid=False))