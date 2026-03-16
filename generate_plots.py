"""
Generate diagnostic plots for the volatility surface.
"""

from vol_surface.fetcher import OptionChainFetcher
from vol_surface.surface import VolatilitySurface
from vol_surface.visualization import VolatilityVisualizer


def main():
    # Fetch option chain (using local mock data)
    fetcher = OptionChainFetcher(source="local")
    option_chain = fetcher.fetch("AAPL")
    
    # Build volatility surface
    surface = VolatilitySurface(option_chain)
    
    # Generate plots
    visualizer = VolatilityVisualizer()
    smile_path = visualizer.plot_smile(surface, maturity=0.25, ticker="AAPL")
    surface_path = visualizer.plot_surface(surface, ticker="AAPL")
    
    print(f"Smile plot saved to: {smile_path}")
    print(f"Surface plot saved to: {surface_path}")


if __name__ == "__main__":
    main()