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
from scipy.optimize import minimize

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
    lower = [-0.95, 0.05, 0.10]
    upper = [0.00,  1.50, 0.90]
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


def calibrate_ssvi(
    k: NDArray[np.float64],
    T: float,
    market_vols: NDArray[np.float64],
    theta: float,
    initial_guess: NDArray[np.float64] | None = None,
) -> tuple[float, float, float]:
    """Calibrate SSVI parameters [rho, eta, gamma] to market implied volatilities.

    Args:
        k: Log-moneyness (k = log(K/F)).
        T: Time to maturity (in years).
        market_vols: Market implied volatilities.
        theta: ATM total variance (theta_t).
        initial_guess: Initial guess for [rho, eta, gamma].

    Returns:
        Calibrated [rho, eta, gamma].
    """
    if initial_guess is None:
        initial_guess = ssvi_initial_guess()

    def objective(x: NDArray[np.float64]) -> float:
        rho, eta, gamma = x
        if not check_ssvi_no_arb(rho, eta, gamma):
            return 1e10  # Penalize invalid parameters
        model_vols = ssvi_implied_vol(k, T, theta, rho, eta, gamma)
        return float(np.sum((model_vols - market_vols) ** 2))

    lower, upper = ssvi_parameter_bounds()
    bounds = list(zip(lower, upper))
    result = minimize(
        objective,
        initial_guess,
        bounds=bounds,
        method="L-BFGS-B",
    )
    if not result.success:
        raise RuntimeError(f"SSVI calibration failed: {result.message}")
    return tuple(result.x)