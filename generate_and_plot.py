"""
Fetch real data, calibrate SABR, and generate plots in-memory.
"""

import os
import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from io import BytesIO


def fetch_option_chain(ticker: str, expiration: str) -> pd.DataFrame:
    """Fetch option chain for a given ticker and expiration."""
    try:
        opt = yf.Ticker(ticker).option_chain(expiration)
        calls = opt.calls
        calls["option_type"] = "call"
        puts = opt.puts
        puts["option_type"] = "put"
        chain = pd.concat([calls, puts])
        chain["expiration"] = expiration
        return chain
    except Exception as e:
        print(f"Failed to fetch {ticker} {expiration}: {e}")
        return pd.DataFrame()


def calibrate_sabr(
    strikes: np.ndarray,
    vols: np.ndarray,
    forward: float,
    maturity: float,
) -> tuple:
    """Calibrate SABR parameters."""
    from scipy.optimize import minimize
    
    def sabr_vol(strike, alpha, beta, rho, nu):
        if abs(strike - forward) < 1e-8:
            return alpha / (forward ** (1 - beta)) * (
                1 + ((1 - beta) ** 2 * alpha ** 2) / (24 * forward ** (2 - 2 * beta))
                + (rho * beta * nu * alpha) / (4 * forward ** (1 - beta))
                + (nu ** 2 * (2 - 3 * rho ** 2)) / 24
            ) * maturity
        else:
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
    
    def objective(params):
        alpha, beta, rho, nu = params
        model_vols = np.array([sabr_vol(strike, alpha, beta, rho, nu) for strike in strikes])
        return np.sum((model_vols - vols) ** 2)
    
    result = minimize(
        objective,
        x0=[0.2, 0.5, 0.0, 0.2],
        bounds=[(1e-6, None), (0, 1), (-0.999, 0.999), (1e-6, None)],
    )
    return result.x


def plot_smile(ticker: str, maturity: float, strikes: np.ndarray, vols: np.ndarray) -> str:
    """Generate volatility smile plot and return as base64."""
    plt.figure(figsize=(10, 6))
    plt.plot(strikes, vols, label=f"Maturity = {maturity:.2f}Y", marker="o")
    plt.title(f"Volatility Smile for {ticker} (Maturity = {maturity:.2f}Y)")
    plt.xlabel("Strike")
    plt.ylabel("Implied Volatility")
    plt.grid(True)
    plt.legend()
    
    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    plt.close()
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def plot_surface(ticker: str, strikes: np.ndarray, maturities: np.ndarray, vol_grid: np.ndarray) -> str:
    """Generate volatility surface plot and return as base64."""
    plt.figure(figsize=(10, 6))
    plt.contourf(strikes, maturities, vol_grid, levels=20, cmap="viridis")
    plt.colorbar(label="Implied Volatility")
    plt.title(f"Volatility Surface for {ticker}")
    plt.xlabel("Strike")
    plt.ylabel("Maturity (Y)")
    plt.grid(True)
    
    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    plt.close()
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def main():
    tickers = ["AAPL", "SPY"]
    os.makedirs("exports/plots", exist_ok=True)
    
    for ticker in tickers:
        print(f"Processing {ticker}...")
        try:
            # Fetch option chain
            expirations = yf.Ticker(ticker).options[:2]  # Use first 2 expirations
            all_chains = []
            for exp in expirations:
                chain = fetch_option_chain(ticker, exp)
                if not chain.empty:
                    all_chains.append(chain)
            
            if not all_chains:
                print(f"No data for {ticker}")
                continue
            
            chain = pd.concat(all_chains)
            maturity = (pd.to_datetime(chain["expiration"].iloc[0]) - pd.Timestamp.now()).days / 365.0
            
            # Calibrate SABR
            strikes = chain["strike"].unique()
            vols = chain.groupby("strike")["impliedVolatility"].mean().values
            forward = chain["lastPrice"].mean()  # Simplified forward estimate
            
            alpha, beta, rho, nu = calibrate_sabr(strikes, vols, forward, maturity)
            print(f"SABR params for {ticker}: alpha={alpha:.4f}, beta={beta:.4f}, rho={rho:.4f}, nu={nu:.4f}")
            
            # Generate plots
            smile_plot = plot_smile(ticker, maturity, strikes, vols)
            with open(f"exports/plots/{ticker}_smile.png", "wb") as f:
                f.write(base64.b64decode(smile_plot))
            
            # Mock surface plot (real implementation would use multiple maturities)
            maturities = np.array([maturity])
            vol_grid = np.array([vols])
            surface_plot = plot_surface(ticker, strikes, maturities, vol_grid)
            with open(f"exports/plots/{ticker}_surface.png", "wb") as f:
                f.write(base64.b64decode(surface_plot))
            
            print(f"Generated plots for {ticker}")
            
        except Exception as e:
            print(f"Failed to process {ticker}: {e}")


if __name__ == "__main__":
    main()