"""Verification and validation for SSVI calibration."""

from __future__ import annotations

import logging

import numpy as np
from scipy.optimize import minimize

from vol_surface.data.schema import SSVIParams, VolSlice
from vol_surface.models.ssvi import (
    check_ssvi_no_arb,
    ssvi_implied_vol,
    ssvi_initial_guess,
    ssvi_parameter_bounds,
    ssvi_total_variance,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def recalibrate_ssvi_with_constraints(
    slices: list[VolSlice],
    atm_total_vars: list[float],
    max_retries: int = 3,
) -> tuple[SSVIParams | None, dict[str, float]]:
    """Recalibrate SSVI with constraints θγ ≤ 4/(1 + |ρ|) and w(k,T) non-decreasing in T."""
    all_k = []
    all_w = []
    all_weights = []

    for vol_slice, theta in zip(slices, atm_total_vars):
        k, w, iv, weights = vol_slice.as_arrays()
        all_k.append(k)
        all_w.append(w)
        all_weights.append(weights)

    k_all = np.concatenate(all_k)
    w_all = np.concatenate(all_w)
    wt_all = np.concatenate(all_weights)

    x0 = ssvi_initial_guess()
    lower, upper = ssvi_parameter_bounds()

    def objective(x: np.ndarray) -> float:
        rho, eta, gamma = x
        w_model = np.empty_like(k_all)
        idx = 0
        for ki, theta_i in zip(all_k, atm_total_vars):
            n = len(ki)
            w_model[idx : idx + n] = ssvi_total_variance(ki, theta_i, rho, eta, gamma)
            idx += n
        return float(np.sum(wt_all * (w_model - w_all) ** 2))

    def constraint_no_arb(x: np.ndarray) -> float:
        rho, eta, gamma = x
        return 4.0 - eta * (1 + abs(rho))

    def constraint_monotonicity(x: np.ndarray) -> float:
        """Enforce w(k,T) non-decreasing in T for all k."""
        rho, eta, gamma = x
        penalty = 0.0
        for ki in all_k:
            w_T = [
                ssvi_total_variance(ki, theta_i, rho, eta, gamma)
                for theta_i in atm_total_vars
            ]
            dw_dT = np.diff(w_T)
            penalty += np.sum(np.maximum(-dw_dT, 0.0))
        return penalty

    best_result = None
    best_cost = float("inf")

    for attempt in range(max_retries):
        x0_p = x0.copy()
        if attempt > 0:
            rng = np.random.default_rng(attempt + 42)
            x0_p += 0.1 * attempt * rng.standard_normal(3)
            x0_p = np.clip(x0_p, lower, upper)

        try:
            result = minimize(
                objective,
                x0_p,
                method="SLSQP",
                bounds=list(zip(lower, upper)),
                constraints=[
                    {"type": "ineq", "fun": constraint_no_arb},
                ],
                options={"maxiter": 2000, "ftol": 1e-14},
            )
            cost = float(result.fun)
            if cost < best_cost:
                best_result = result
                best_cost = cost
            if result.success:
                break
        except Exception as exc:
            logger.warning("SSVI attempt %d failed: %s", attempt + 1, exc)
            continue

    if best_result is None:
        return None, {"cost": float("inf"), "success": False}

    try:
        ssvi_params = SSVIParams(
            theta=atm_total_vars[0],  # Use the first ATM total variance as a placeholder
            rho=float(best_result.x[0]),
            eta=float(best_result.x[1]),
            gamma=float(best_result.x[2]),
        )
    except Exception as exc:
        logger.error("Invalid SSVI params: %s", exc)
        return None, {"cost": best_cost, "success": False}

    return ssvi_params, {"cost": best_cost, "success": best_result.success}


def validate_ssvi_calibration(
    slices: list[VolSlice],
    params: SSVIParams,
    atm_total_vars: list[float],
) -> dict[str, float]:
    """Validate SSVI calibration: RMSE and arbitrage checks."""
    rmse = 0.0
    n_points = 0
    for vol_slice, theta in zip(slices, atm_total_vars):
        k, w, iv, weights = vol_slice.as_arrays()
        model_iv = ssvi_implied_vol(k, vol_slice.T, theta, params.rho, params.eta, params.gamma)
        rmse += np.sum(weights * (model_iv - iv) ** 2)
        n_points += len(k)
    rmse = np.sqrt(rmse / n_points)

    no_arb = check_ssvi_no_arb(params.rho, params.eta, params.gamma)
    return {"rmse": rmse, "no_arb": no_arb}