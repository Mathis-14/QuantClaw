"""SSVI surface model (Gatheral-Jacquier 2014).

Parametrization:
    w(k, t) = (theta_t / 2) * (1 + rho*phi*k + sqrt((phi*k + rho)^2 + (1 - rho^2)))

where:
    phi(theta) = eta / (theta^gamma * (1 + theta)^(1 - gamma))

No-arbitrage conditions:  0 < gamma <= 1,  eta > 0,  eta*(1+|rho|) <= 4.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from vol_surface.data.schema import SSVIParams


def phi_func(theta: float, eta: float, gamma: float) -> float:
    """SSVI mixing function."""
    return eta / (theta**gamma * (1 + theta) ** (1 - gamma))


def ssvi_total_variance(
    k: NDArray[np.float64],
    theta: float,
    rho: float,
    eta: float,
    gamma: float,
) -> NDArray[np.float64]:
    """Evaluate SSVI total variance for a single maturity."""
    p = phi_func(theta, eta, gamma)
    pk = p * k
    return (theta / 2) * (1 + rho * pk + np.sqrt((pk + rho) ** 2 + 1 - rho**2))


def ssvi_implied_vol(
    k: NDArray[np.float64],
    T: float,
    theta: float,
    rho: float,
    eta: float,
    gamma: float,
) -> NDArray[np.float64]:
    """Implied vol from SSVI."""
    w = np.maximum(ssvi_total_variance(k, theta, rho, eta, gamma), 1e-10)
    return np.sqrt(w / T)


def check_ssvi_no_arb(rho: float, eta: float, gamma: float) -> bool:
    """Return True if SSVI no-arbitrage conditions are satisfied."""
    return 0 < gamma <= 1 and eta > 0 and eta * (1 + abs(rho)) <= 4


# ── Initial guess & bounds ──────────────────────────────────────────────────


def ssvi_initial_guess() -> NDArray[np.float64]:
    """Heuristic initial guess for [rho, eta, gamma]."""
    return np.array([-0.3, 1.0, 0.5])


def ssvi_parameter_bounds() -> tuple[list[float], list[float]]:
    """(lower, upper) bounds for [rho, eta, gamma].

    Equity-specific:
    - rho in (-0.95, 0.0)  — equity skew is always negative.
    - eta in (0.05, 1.5)   — avoids flat-surface and explosions.
    - gamma in (0.1, 0.9)  — keeps power-law decay well-conditioned.

    Strictly inside the mathematical no-arb region eta*(1+|rho|) <= 4.
    """
    lower = [-0.95, 0.05, 0.10]
    upper = [0.00,  1.50, 0.90]
    return lower, upper
