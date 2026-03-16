"""
Calibrate SVI and generate volatility surface plots from fetched data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize
from pathlib import Path

# Load data
eth_df = pd.read_csv("data/raw/eth_options_20260316_113618.csv")

def raw_svi(k, a, b, rho, m, sigma):
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))

def ssvi(theta, k, rho, eta, gamma):
    return theta / 2 * (1 + rho * eta * k + np.sqrt((eta * k + rho) ** 2 + (1 - rho ** 2)))

def calibrate_svi(df: pd.DataFrame, output_dir: Path = Path("plots")):
    """Calibrate SVI per expiry and generate plots."""
    output_dir.mkdir(exist_ok=True)
    svi_params = []
    
    for expiry, group in df.groupby("expiry_date"):
        group = group.dropna(subset=["implied_volatility"])
        if len(group) < 3:
            continue
        
        # Prepare data
        k = np.log(group["strike"] / group["underlying_price"]).values
        w = (group["implied_volatility"] ** 2) * (pd.Timestamp(expiry) - pd.Timestamp.now()).days / 365.0
        weights = 1.0 / ((group["ask"] - group["bid"]) / group["bid"] * 100.0)
        
        # Calibrate SVI
        def svi_objective(params):
            a, b, rho, m, sigma = params
            w_pred = raw_svi(k, a, b, rho, m, sigma)
            return np.sqrt(np.mean(weights * (w - w_pred) ** 2))
        
        p0 = [0.01, 0.1, -0.5, 0.0, 0.1]
        bounds = [(-np.inf, np.inf), (0, np.inf), (-1, 1), (-np.inf, np.inf), (1e-6, np.inf)]
        result = minimize(svi_objective, p0, bounds=bounds, method="L-BFGS-B")
        a, b, rho, m, sigma = result.x
        
        # Save params
        svi_params.append({
            "expiry": expiry,
            "a": a, "b": b, "rho": rho, "m": m, "sigma": sigma,
            "rmse": result.fun
        })
        
        # Plot SVI fit
        k_range = np.linspace(min(k) - 0.1, max(k) + 0.1, 100)
        w_pred = raw_svi(k_range, a, b, rho, m, sigma)
        
        plt.figure(figsize=(10, 6))
        plt.scatter(k, w, label="Market", color="blue")
        plt.plot(k_range, w_pred, label="SVI Fit", color="red")
        plt.title(f"SVI Fit for {pd.Timestamp(expiry).strftime('%Y-%m-%d')}\nRMSE={result.fun:.6f}")
        plt.xlabel("Log-Moneyness (log(K/F))")
        plt.ylabel("Total Variance")
        plt.legend()
        plt.grid(True)
        plt.savefig(output_dir / f"svi_fit_{pd.Timestamp(expiry).strftime('%Y%m%d')}.png")
        plt.close()
    
    return svi_params

def calibrate_ssvi(df: pd.DataFrame, output_dir: Path = Path("plots")):
    """Calibrate SSVI and generate surface plot."""
    output_dir.mkdir(exist_ok=True)
    
    # Prepare data
    df = df.dropna(subset=["implied_volatility", "underlying_price"])
    df = df[df["underlying_price"] > 0]
    if len(df) < 3:
        return {"rho": np.nan, "eta": np.nan, "gamma": np.nan, "rmse": np.nan}
    
    theta = (df["implied_volatility"] ** 2).mean()  # ATM total variance
    k = np.log(df["strike"] / df["underlying_price"]).values
    w = (df["implied_volatility"] ** 2) * (pd.to_datetime(df["expiry_date"]) - pd.Timestamp.now()).dt.days / 365.0
    weights = 1.0 / ((df["ask"] - df["bid"]) / df["bid"] * 100.0)
    
    # Calibrate SSVI
    def ssvi_objective(params):
        rho, eta, gamma = params
        w_pred = ssvi(theta, k, rho, eta, gamma)
        return np.sqrt(np.mean(weights * (w - w_pred) ** 2))
    
    p0 = [-0.5, 0.1, 0.5]
    bounds = [(-1, 1), (0, np.inf), (0, 1)]
    result = minimize(ssvi_objective, p0, bounds=bounds, method="L-BFGS-B")
    rho, eta, gamma = result.x
    
    # Generate surface
    k_range = np.linspace(min(k) - 0.1, max(k) + 0.1, 50)
    theta_range = np.linspace(0.01, 0.1, 50)
    K, Theta = np.meshgrid(k_range, theta_range)
    W = ssvi(Theta, K, rho, eta, gamma)
    
    # Plot
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(K, Theta, W, cmap="viridis", alpha=0.8)
    ax.scatter(k, theta, w, color="red", label="Market Data")
    ax.set_title("SSVI Volatility Surface")
    ax.set_xlabel("Log-Moneyness (log(K/F))")
    ax.set_ylabel("ATM Total Variance (theta)")
    ax.set_zlabel("Total Variance")
    ax.legend()
    plt.savefig(output_dir / "ssvi_surface.png")
    plt.close()
    
    return {"rho": rho, "eta": eta, "gamma": gamma, "rmse": result.fun}

# Calibrate and plot
eth_svi_params = calibrate_svi(eth_df)
eth_ssvi_params = calibrate_ssvi(eth_df)

print("SVI and SSVI calibration complete. Plots saved to plots/")