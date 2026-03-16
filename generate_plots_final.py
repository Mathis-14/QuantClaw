import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import io
import base64

# Mock data for volatility smile
strikes = np.linspace(80, 120, 5)
vols = 0.2 + 0.1 * (strikes - 100) / 100

# Plot 1: Volatility Smile
plt.figure(figsize=(10, 6))
plt.plot(strikes, vols, label="Maturity = 0.25Y", marker="o")
plt.title("Volatility Smile for AAPL (Maturity = 0.25Y)")
plt.xlabel("Strike")
plt.ylabel("Implied Volatility")
plt.grid(True)
plt.legend()

# Save to base64
buffer = io.BytesIO()
plt.savefig(buffer, format='png')
buffer.seek(0)
smile_plot = base64.b64encode(buffer.getvalue()).decode('utf-8')
plt.close()

# Mock data for volatility surface
strike_grid, maturity_grid = np.meshgrid(strikes, [0.1, 0.25, 0.5])
vol_grid = np.array([
    [0.25, 0.22, 0.20],
    [0.24, 0.21, 0.19],
    [0.23, 0.20, 0.18],
    [0.22, 0.19, 0.17],
    [0.21, 0.18, 0.16],
])

# Plot 2: Volatility Surface
plt.figure(figsize=(10, 6))
plt.contourf(strike_grid, maturity_grid, vol_grid, levels=20, cmap="viridis")
plt.colorbar(label="Implied Volatility")
plt.title("Volatility Surface for AAPL")
plt.xlabel("Strike")
plt.ylabel("Maturity (Y)")
plt.grid(True)

# Save to base64
buffer = io.BytesIO()
plt.savefig(buffer, format='png')
buffer.seek(0)
surface_plot = base64.b64encode(buffer.getvalue()).decode('utf-8')
plt.close()

# Write to files
with open('/tmp/smile_plot_base64.txt', 'w') as f:
    f.write(smile_plot)
with open('/tmp/surface_plot_base64.txt', 'w') as f:
    f.write(surface_plot)

print("Plots generated successfully.")
