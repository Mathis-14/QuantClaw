"""
QuantClaw Pricing Engine.

This module provides:
- Path-dependent option pricing (barrier, autocallable).
- Monte Carlo and PDE-based methods.
- Interface-based design for volatility models.
"""

from pricing.models.barrier import BarrierOption
from pricing.models.autocallable import AutocallableOption
from pricing.models.interfaces import VolatilityModel, PricingEngine
from pricing.methods.monte_carlo import MonteCarloPricer
from pricing.methods.pde import PDEPricer
from pricing.utils import black_scholes

__all__ = [
    "BarrierOption",
    "AutocallableOption",
    "VolatilityModel",
    "PricingEngine",
    "MonteCarloPricer",
    "PDEPricer",
    "black_scholes",
]