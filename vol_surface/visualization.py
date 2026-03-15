"""
Diagnostic plots for volatility surfaces and smiles.
"""

from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from vol_surface.surface import VolatilitySurface
import os


class VolatilityVisualizer:
    """Generate diagnostic plots for volatility surfaces."""

    def __init__(self, output_dir: str = "exports/plots"):
        """
        Initialize the visualizer.

        Args:
            output_dir (str): Directory to save plots. Defaults to "exports/plots".
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_smile(
        self,
        surface: VolatilitySurface,
        maturity: float,
        ticker: str = "UNDL",
    ) -> str:
        """
        Plot the volatility smile for a given maturity.

        Args:
            surface (VolatilitySurface): Volatility surface.
            maturity (float): Time to maturity (in years).
            ticker (str): Underlying ticker. Defaults to "UNDL".

        Returns:
            str: Path to the saved plot.
        """
        strikes = surface.option_chain.strikes
        vols = [surface.volatility(strike, maturity) for strike in strikes]
        
        plt.figure(figsize=(10, 6))
        plt.plot(strikes, vols, label=f"Maturity = {maturity:.2f}Y", marker="o")
        plt.title(f"Volatility Smile for {ticker} (Maturity = {maturity:.2f}Y)")
        plt.xlabel("Strike")
        plt.ylabel("Implied Volatility")
        plt.grid(True)
        plt.legend()
        
        path = os.path.join(self.output_dir, f"smile_{ticker}_{maturity:.2f}Y.png")
        plt.savefig(path)
        plt.close()
        
        return path

    def plot_surface(
        self,
        surface: VolatilitySurface,
        ticker: str = "UNDL",
    ) -> str:
        """
        Plot the volatility surface as a heatmap.

        Args:
            surface (VolatilitySurface): Volatility surface.
            ticker (str): Underlying ticker. Defaults to "UNDL".

        Returns:
            str: Path to the saved plot.
        """
        strikes = surface.option_chain.strikes
        maturities = surface.option_chain.maturities
        
        # Create grid
        strike_grid, maturity_grid = np.meshgrid(strikes, maturities)
        vol_grid = np.array([
            [surface.volatility(strike, maturity) for strike in strikes]
            for maturity in maturities
        ])
        
        plt.figure(figsize=(10, 6))
        plt.contourf(strike_grid, maturity_grid, vol_grid, levels=20, cmap="viridis")
        plt.colorbar(label="Implied Volatility")
        plt.title(f"Volatility Surface for {ticker}")
        plt.xlabel("Strike")
        plt.ylabel("Maturity (Y)")
        plt.grid(True)
        
        path = os.path.join(self.output_dir, f"surface_{ticker}.png")
        plt.savefig(path)
        plt.close()
        
        return path