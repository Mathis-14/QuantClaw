"""SSVI surface model (Gatheral-Jacquier 2014).

Parametrization:
    w(k, t) = (theta_t / 2) * (1 + rho*phi*k + sqrt((phi*k + rho)^2 + (1 - rho^2)))

where:
    phi(theta) = eta / (theta^gamma * (1 + theta)^(1 - gamma))
    theta_t = ATM total variance at maturity t

No-arbitrage conditions:
    0 < gamma <= 1
    eta > 0
    eta * (1 + |rho|) <= 4
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
    w = ssvi_total_variance(k, theta, rho, eta, gamma)
    w = np.maximum(w, 1e-10)
    return np.sqrt(w / T)


def ssvi_from_params(
    params: SSVIParams,
) -> dict[str, float]:
    return dict(rho=params.rho, eta=params.eta, gamma=params.gamma)


def ssvi_initial_guess() -> NDArray[np.float64]:
    """Heuristic initial guess for [rho, eta, gamma]."""
    return np.array([-0.3, 1.0, 0.5])


def ssvi_parameter_bounds() -> tuple[list[float], list[float]]:
    """Return (lower, upper) bounds for [rho, eta, gamma]."""
    lower = [-0.999, 1e-4, 1e-4]
    upper = [0.999, 4.0, 1.0]
    return lower, upper


def check_ssvi_no_arb(rho: float, eta: float, gamma: float) -> bool:
    """Return True if SSVI no-arbitrage conditions are satisfied."""
    if gamma <= 0 or gamma > 1:
        return False
    if eta <= 0:
        return False
    if eta * (1 + abs(rho)) > 4:
        return False
    return True
