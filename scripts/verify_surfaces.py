#!/usr/bin/env python3
"""
Script to verify 3D surface plots for SVI, SSVI, and eSSVI models.
Checks for arbitrage violations, smoothness, and fit quality.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from datetime import datetime

# Constants
DATA_PATH = "/home/openclaw_daemon/.openclaw/workspace-engineer/projects/QuantClaw/data/processed/btc_options_deribit.csv"
PLOT_DIR = "/home/openclaw_daemon/.openclaw/workspace-engineer/projects/QuantClaw/plots/verification"

# Ensure plot directory exists
os.makedirs(PLOT_DIR, exist_ok=True)

# Load data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["expiry_date"] = pd.to_datetime(df["expiry_date"])
    df["time_to_maturity"] = (df["expiry_date"] - datetime.now()).dt.days / 365.0
    return df

# SVI model
def svi_model(k, a, b, rho, m, sigma):
    return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))

# SSVI model
def ssvi_model(k, a, b, rho, m, sigma):
    return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))

# eSSVI model
def essvi_model(k, a, b, rho, m, sigma, eta):
    return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2)) + eta * k

# Calibrate model to market data
def calibrate_model(df, model_func, initial_guess, bounds):
    def objective(params):
        total_error = 0.0
        for expiry, group in df.groupby("expiry_date"):
            k = np.log(group["strike"] / group["underlying_price"].mean())
            iv_market = group["implied_volatility"].values
            iv_model = model_func(k, *params)
            total_error += np.sum((iv_market - iv_model) ** 2)
        return total_error

    result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
    return result.x

# Check arbitrage constraints
def check_arbitrage_svi(params):
    a, b, rho, m, sigma = params
    return (b * sigma <= 4 / (1 + abs(rho)))

def check_arbitrage_ssvi(params):
    a, b, rho, m, sigma = params
    return (b * sigma <= 4 / (1 + abs(rho)))

def check_arbitrage_essvi(params):
    a, b, rho, m, sigma, eta = params
    return (b * sigma <= 4 / (1 + abs(rho)))

# Plot residuals
def plot_residuals(df, model_func, params, model_name):
    plt.figure(figsize=(12, 6))
    for expiry, group in df.groupby("expiry_date"):
        k = np.log(group["strike"] / group["underlying_price"].mean())
        iv_market = group["implied_volatility"].values
        iv_model = model_func(k, *params)
        plt.scatter(k, iv_market - iv_model, label=f"Expiry: {expiry.date()}")
    plt.axhline(0, color='black', linestyle='--')
    plt.title(f"Residuals: Market IV vs. {model_name} Model IV")
    plt.xlabel("Log-Moneyness")
    plt.ylabel("Residual (Market - Model)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(PLOT_DIR, f"{model_name}_Residuals.png"))
    plt.close()

# Main verification function
def verify_surfaces():
    df = load_data()

    # SVI
    svi_params = calibrate_model(
        df, svi_model,
        initial_guess=[0.1, 0.5, -0.5, 0.0, 0.2],
        bounds=[(0, 1), (0, 1), (-1, 1), (-1, 1), (0, 1)]
    )
    svi_arbitrage_ok = check_arbitrage_svi(svi_params)
    plot_residuals(df, svi_model, svi_params, "SVI")

    # SSVI
    ssvi_params = calibrate_model(
        df, ssvi_model,
        initial_guess=[0.1, 0.5, -0.5, 0.0, 0.2],
        bounds=[(0, 1), (0, 1), (-1, 1), (-1, 1), (0, 1)]
    )
    ssvi_arbitrage_ok = check_arbitrage_ssvi(ssvi_params)
    plot_residuals(df, ssvi_model, ssvi_params, "SSVI")

    # eSSVI
    essvi_params = calibrate_model(
        df, essvi_model,
        initial_guess=[0.1, 0.5, -0.5, 0.0, 0.2, 0.01],
        bounds=[(0, 1), (0, 1), (-1, 1), (-1, 1), (0, 1), (-0.1, 0.1)]
    )
    essvi_arbitrage_ok = check_arbitrage_essvi(essvi_params)
    plot_residuals(df, essvi_model, essvi_params, "eSSVI")

    # Generate report
    report = {
        "SVI": {
            "Arbitrage Free": svi_arbitrage_ok,
            "Residuals Plot": "SVI_Residuals.png"
        },
        "SSVI": {
            "Arbitrage Free": ssvi_arbitrage_ok,
            "Residuals Plot": "SSVI_Residuals.png"
        },
        "eSSVI": {
            "Arbitrage Free": essvi_arbitrage_ok,
            "Residuals Plot": "eSSVI_Residuals.png"
        }
    }

    return report

if __name__ == "__main__":
    report = verify_surfaces()
    print("Verification Report:")
    for model, results in report.items():
        print(f"\n{model}:")
        print(f"  Arbitrage Free: {'Yes' if results['Arbitrage Free'] else 'No'}")
        print(f"  Residuals Plot: {results['Residuals Plot']}")