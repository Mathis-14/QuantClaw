"""
Generate diagnostic plots for the volatility surface and return as base64.
"""

import base64
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from vol_surface.fetcher import OptionChainFetcher
from vol_surface.surface import VolatilitySurface
from vol_surface.visualization import VolatilityVisualizer


def plot_to_base64() -> dict:
    """Generate plots and return as base64-encoded strings."""
    # Fetch option chain (using local mock data)
    fetcher = OptionChainFetcher(source="local")
    option_chain = fetcher.fetch("AAPL")
    
    # Build volatility surface
    surface = VolatilitySurface(option_chain)
    
    # Generate plots in memory
    plt.switch_backend('Agg')  # Non-interactive backend
    
    # Plot 1: Volatility Smile
    strikes = surface.option_chain.strikes
    maturity = 0.25
    vols = [surface.volatility(strike, maturity) for strike in strikes]
    
    plt.figure(figsize=(10, 6))
    plt.plot(strikes, vols, label=f"Maturity = {maturity:.2f}Y", marker="o")
    plt.title(f"Volatility Smile for AAPL (Maturity = {maturity:.2f}Y)")
    plt.xlabel("Strike")
    plt.ylabel("Implied Volatility")
    plt.grid(True)
    plt.legend()
    
    # Save to buffer
    buffer = base64.b64encode(plt_to_buffer()).decode('utf-8')
    plt.close()
    smile_plot = buffer
    
    # Plot 2: Volatility Surface
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
    plt.title("Volatility Surface for AAPL")
    plt.xlabel("Strike")
    plt.ylabel("Maturity (Y)")
    plt.grid(True)
    
    # Save to buffer
    buffer = base64.b64encode(plt_to_buffer()).decode('utf-8')
    plt.close()
    surface_plot = buffer
    
    return {
        "smile_plot": smile_plot,
        "surface_plot": surface_plot,
    }


def plt_to_buffer() -> bytes:
    """Convert current matplotlib plot to bytes."""
    buffer = plt.ioff()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    return buffer.getvalue()


if __name__ == "__main__":
    plots = plot_to_base64()
    print("Plots generated successfully.")
    # Write to files for debugging
    os.makedirs("exports/plots", exist_ok=True)
    with open("exports/plots/smile_plot_base64.txt", "w") as f:
        f.write(plots["smile_plot"])
    with open("exports/plots/surface_plot_base64.txt", "w") as f:
        f.write(plots["surface_plot"])