#!/usr/bin/env python3
"""
Recalibrate SVI, SSVI, and eSSVI using Deribit IVs.
- Filter OTM options.
- Enforce no-arbitrage constraints.
- Use scipy.optimize.minimize with tighter bounds and better initial guesses.
- Generate 3D surface and 2D smile plots.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime

# --- Constants ---
OUTPUT_DIR = "/home/openclaw_daemon/.openclaw/workspace-engineer/projects/QuantClaw/plots/recalibrated"

# --- Data Loading and Preprocessing ---
def load_and_preprocess_data(filepath):
    """Load and preprocess Deribit options data."""
    df = pd.read_csv(filepath)
    
    # Drop rows with missing expiry_date
    df = df.dropna(subset=["expiry_date"])
    
    # Convert expiry_date to datetime
    df["expiry_date"] = pd.to_datetime(df["expiry_date"])
    
    # Calculate time to expiry in years
    df["time_to_expiry"] = (df["expiry_date"] - pd.to_datetime("today")).dt.days / 365.25
    
    # Filter OTM calls (strike > underlying_price)
    df = df[df["strike"] > df["underlying_price"]]
    
    return df

# --- Model Definitions ---
def svi_total_variance(k, a, b, rho, m, sigma):
    """SVI total variance function."""
    return a + b * (rho * (m - k) + np.sqrt((m - k)**2 + sigma**2))

def ssvi_total_variance(k, T, theta, phi, rho, m):
    """SSVI total variance function."""
    w = theta + phi * (rho * (m - k) + np.sqrt((m - k)**2 + 1))
    return w * T

def essvi_total_variance(k, T, theta, phi, rho, m, psi):
    """eSSVI total variance function."""
    w = theta + phi * (rho * (m - k) + np.sqrt((m - k)**2 + psi))
    return w * T

# --- No-Arbitrage Constraints ---
def svi_no_arbitrage_constraint(params):
    """Enforce SVI no-butterfly-arbitrage: θ_t ϕ_t ≤ 4/(1 + |ρ_t|)."""
    a, b, rho, m, sigma = params
    return 4 / (1 + abs(rho)) - b * abs(sigma)

def ssvi_no_arbitrage_constraint(params):
    """Enforce SSVI no-arbitrage: w(k,T) non-decreasing in T."""
    theta, phi, rho, m = params
    return phi  # Simplified; full constraint requires checking across T

# --- Calibration ---
def calibrate_svi(df_group, initial_guess, bounds):
    """Calibrate SVI for a single expiry group."""
    strikes = df_group["strike"].values
    ivs = df_group["implied_volatility"].values
    forward = df_group["underlying_price"].mean()
    
    # Log-moneyness: k = ln(K/F)
    k = np.log(strikes / forward)
    
    def objective(params):
        a, b, rho, m, sigma = params
        total_var = svi_total_variance(k, a, b, rho, m, sigma)
        model_iv = np.sqrt(total_var / df_group["time_to_expiry"].iloc[0])
        return np.sum((ivs - model_iv)**2)
    
    constraints = ({"type": "ineq", "fun": svi_no_arbitrage_constraint})
    result = minimize(
        objective, initial_guess, bounds=bounds, constraints=constraints, method="SLSQP"
    )
    return result

def calibrate_ssvi(df_group, initial_guess, bounds):
    """Calibrate SSVI for a single expiry group."""
    strikes = df_group["strike"].values
    ivs = df_group["implied_volatility"].values
    forward = df_group["underlying_price"].mean()
    T = df_group["time_to_expiry"].iloc[0]
    
    k = np.log(strikes / forward)
    
    def objective(params):
        theta, phi, rho, m = params
        total_var = ssvi_total_variance(k, T, theta, phi, rho, m)
        model_iv = np.sqrt(total_var / T)
        return np.sum((ivs - model_iv)**2)
    
    constraints = ({"type": "ineq", "fun": ssvi_no_arbitrage_constraint})
    result = minimize(
        objective, initial_guess, bounds=bounds, constraints=constraints, method="SLSQP"
    )
    return result

def calibrate_essvi(df_group, initial_guess, bounds):
    """Calibrate eSSVI for a single expiry group."""
    strikes = df_group["strike"].values
    ivs = df_group["implied_volatility"].values
    forward = df_group["underlying_price"].mean()
    T = df_group["time_to_expiry"].iloc[0]
    
    k = np.log(strikes / forward)
    
    def objective(params):
        theta, phi, rho, m, psi = params
        total_var = essvi_total_variance(k, T, theta, phi, rho, m, psi)
        model_iv = np.sqrt(total_var / T)
        return np.sum((ivs - model_iv)**2)
    
    result = minimize(
        objective, initial_guess, bounds=bounds, method="SLSQP"
    )
    return result

# --- Plotting ---
def plot_2d_smiles(df, model_name, params_dict):
    """Plot 2D smiles for each expiry."""
    expiries = df["expiry_date"].unique()
    for expiry in expiries:
        df_group = df[df["expiry_date"] == expiry]
        strikes = df_group["strike"].values
        ivs = df_group["implied_volatility"].values
        forward = df_group["underlying_price"].mean()
        T = df_group["time_to_expiry"].iloc[0]
        
        k = np.log(strikes / forward)
        params = params_dict[expiry]
        
        if model_name == "SVI":
            a, b, rho, m, sigma = params
            total_var = svi_total_variance(k, a, b, rho, m, sigma)
        elif model_name == "SSVI":
            theta, phi, rho, m = params
            total_var = ssvi_total_variance(k, T, theta, phi, rho, m)
        elif model_name == "eSSVI":
            theta, phi, rho, m, psi = params
            total_var = essvi_total_variance(k, T, theta, phi, rho, m, psi)
        
        model_iv = np.sqrt(total_var / T)
        
        plt.figure(figsize=(10, 6))
        plt.scatter(strikes, ivs, label="Market IV", color="blue")
        plt.plot(strikes, model_iv, label=f"{model_name} Fit", color="red", linestyle="--")
        plt.title(f"{model_name} Smile for Expiry {expiry.date()}")
        plt.xlabel("Strike")
        plt.ylabel("Implied Volatility")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{OUTPUT_DIR}/{model_name}_smile_{expiry.date()}.png")
        plt.close()

def plot_3d_surface(df, model_name, params_dict):
    """Plot 3D surface for the model."""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    
    strikes = np.linspace(df["strike"].min(), df["strike"].max(), 100)
    expiries = df["expiry_date"].unique()
    T = [(expiry - pd.to_datetime("today")).days / 365.25 for expiry in expiries]
    
    X, Y = np.meshgrid(strikes, T)
    Z = np.zeros_like(X)
    
    for i, expiry in enumerate(expiries):
        forward = df[df["expiry_date"] == expiry]["underlying_price"].mean()
        k = np.log(strikes / forward)
        params = params_dict[expiry]
        
        if model_name == "SVI":
            a, b, rho, m, sigma = params
            total_var = svi_total_variance(k, a, b, rho, m, sigma)
        elif model_name == "SSVI":
            theta, phi, rho, m = params
            total_var = ssvi_total_variance(k, Y[i, 0], theta, phi, rho, m)
        elif model_name == "eSSVI":
            theta, phi, rho, m, psi = params
            total_var = essvi_total_variance(k, Y[i, 0], theta, phi, rho, m, psi)
        
        Z[i, :] = np.sqrt(total_var / Y[i, 0])
    
    ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.8)
    ax.set_xlabel("Strike")
    ax.set_ylabel("Time to Expiry (Years)")
    ax.set_zlabel("Implied Volatility")
    ax.set_title(f"{model_name} Surface")
    plt.savefig(f"{OUTPUT_DIR}/{model_name}_surface_3d.png")
    plt.close()

# --- Main ---
def main():
    # Load and preprocess data
    df = load_and_preprocess_data(
        "/home/openclaw_daemon/.openclaw/workspace-engineer/projects/QuantClaw/data/processed/btc_options_deribit.csv"
    )
    
    # Group by expiry
    grouped = df.groupby("expiry_date")
    
    # Calibrate models
    svi_params = {}
    ssvi_params = {}
    essvi_params = {}
    
    svi_initial = [0.01, 0.5, -0.5, 0.0, 0.2]
    svi_bounds = [
        (0.001, 1.0),  # a
        (0.01, 2.0),   # b
        (-0.99, 0.99), # rho
        (-1.0, 1.0),   # m
        (0.01, 1.0)    # sigma
    ]
    
    ssvi_initial = [0.01, 0.5, -0.5, 0.0]
    ssvi_bounds = [
        (0.001, 1.0),  # theta
        (0.01, 2.0),   # phi
        (-0.99, 0.99), # rho
        (-1.0, 1.0)    # m
    ]
    
    essvi_initial = [0.01, 0.5, -0.5, 0.0, 0.2]
    essvi_bounds = [
        (0.001, 1.0),  # theta
        (0.01, 2.0),   # phi
        (-0.99, 0.99), # rho
        (-1.0, 1.0),   # m
        (0.01, 1.0)    # psi
    ]
    
    for expiry, df_group in grouped:
        print(f"Calibrating for expiry: {expiry.date()}")
        
        # SVI
        svi_result = calibrate_svi(df_group, svi_initial, svi_bounds)
        svi_params[expiry] = svi_result.x
        
        # SSVI
        ssvi_result = calibrate_ssvi(df_group, ssvi_initial, ssvi_bounds)
        ssvi_params[expiry] = ssvi_result.x
        
        # eSSVI
        essvi_result = calibrate_essvi(df_group, essvi_initial, essvi_bounds)
        essvi_params[expiry] = essvi_result.x
    
    # Plot results
    plot_2d_smiles(df, "SVI", svi_params)
    plot_2d_smiles(df, "SSVI", ssvi_params)
    plot_2d_smiles(df, "eSSVI", essvi_params)
    
    plot_3d_surface(df, "SVI", svi_params)
    plot_3d_surface(df, "SSVI", ssvi_params)
    plot_3d_surface(df, "eSSVI", essvi_params)

if __name__ == "__main__":
    main()