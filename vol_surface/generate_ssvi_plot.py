"""Generate and plot SSVI volatility smiles for BTC and ETH."""

import os
import sys

# Set Matplotlib config directory
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

# Add project root to Python path
sys.path.append("/home/openclaw_daemon/.openclaw/workspace-engineer/projects/QuantClaw")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from vol_surface.surface import VolatilitySurface
from vol_surface.calibration import load_and_preprocess_data


def generate_ssvi_plot(
    btc_file: str,
    eth_file: str,
    output_dir: str = "plots",
) -> None:
    """Generate and plot SSVI volatility smiles for BTC and ETH.

    Args:
        btc_file: Path to BTC options CSV.
        eth_file: Path to ETH options CSV.
        output_dir: Directory to save plots.
    """
    # Load and preprocess data
    btc_chain = load_and_preprocess_data(btc_file)
    eth_chain = load_and_preprocess_data(eth_file)

    # Calibrate SSVI
    btc_surface = VolatilitySurface(btc_chain, model="SSVI")
    eth_surface = VolatilitySurface(eth_chain, model="SSVI")

    # Generate volatility smiles
    def plot_volatility_smiles(surface, ticker: str):
        strikes = np.linspace(
            min(surface.option_chain.strikes) * 0.8,
            max(surface.option_chain.strikes) * 1.2,
            100,
        )
        maturities = [0.25, 0.5, 1.0]  # Example maturities
        plt.figure(figsize=(10, 6))
        for T in maturities:
            vols = [surface.volatility(strike, T) for strike in strikes]
            plt.plot(strikes, vols, label=f"T={T}Y")
        plt.title(f"SSVI Volatility Smile for {ticker}")
        plt.xlabel("Strike")
        plt.ylabel("Implied Volatility")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{output_dir}/{ticker}_ssvi_smiles.png")
        plt.close()

    plot_volatility_smiles(btc_surface, "BTC")
    plot_volatility_smiles(eth_surface, "ETH")


if __name__ == "__main__":
    generate_ssvi_plot(
        btc_file="/home/openclaw_daemon/.openclaw/workspace-engineer/projects/QuantClaw/data/raw/btc_options_20260316.csv",
        eth_file="/home/openclaw_daemon/.openclaw/workspace-engineer/projects/QuantClaw/data/raw/eth_options_20260316.csv",
        output_dir="/home/openclaw_daemon/.openclaw/workspace-engineer/projects/QuantClaw/plots",
    )