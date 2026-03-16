"""
Standalone script to generate SSVI surface plot from demo data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize

# Demo data
data = {
    "instrument_name": ["BTC-26JUN26-50000-C", "BTC-26JUN26-52000-C", "BTC-26JUN26-50000-P", "BTC-26JUN26-52000-P"],
    "strike": [50000, 52000, 50000, 52000],
    "expiry_date": [pd.Timestamp("2026-06-26"), pd.Timestamp("2026-06-26"), pd.Timestamp("2026-06-26"), pd.Timestamp("2026-06-26")],
    "option_type": ["call", "call", "put", "put"],
    "bid": [1000, 800, 500, 300],
    "ask": [1010, 810, 510, 310],
    "underlying_price": [52000, 52000, 52000, 52000],
    "implied_volatility": [0.5, 0.55, 0.6, 0.65],
    "funding_rate": [0.0001, 0.0001, 0.0001, 0.0001],
    "timestamp": [pd.Timestamp("2026-03-16"), pd.Timestamp("2026-03-16"), pd.Timestamp("2026-03-16"), pd.Timestamp("2026-03-16")],
}

df = pd.DataFrame(data)
df["log_moneyness"] = np.log(df["strike"] / df["underlying_price"])
df["total_variance"] = (df["implied_volatility"] ** 2) * 102 / 365.0  # 102 days to expiry
df["weight"] = 1.0 / ((df["ask"] - df["bid"]) / df["bid"] * 100.0)

# SSVI formula
def ssvi(theta, k, rho, eta, gamma):
    return theta / 2 * (1 + rho * eta * k + np.sqrt((eta * k + rho) ** 2 + (1 - rho ** 2)))

# Objective function
def ssvi_objective(params, theta, k, w, weights):
    rho, eta, gamma = params
    w_pred = ssvi(theta, k, rho, eta, gamma)
    return np.sqrt(np.mean(weights * (w - w_pred) ** 2))

# Calibrate SSVI
theta = df["total_variance"].mean()  # ATM total variance
k = df["log_moneyness"].values
w = df["total_variance"].values
weights = df["weight"].values
p0 = [-0.5, 0.1, 0.5]
bounds = [(-1, 1), (0, np.inf), (0, 1)]
result = minimize(ssvi_objective, p0, args=(theta, k, w, weights), bounds=bounds, method="L-BFGS-B")
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
plt.savefig("plots/ssvi_surface_demo.png")
plt.close()

print("SSVI surface plot saved to plots/ssvi_surface_demo.png")