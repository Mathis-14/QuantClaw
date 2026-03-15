"""
Shared utilities for pricing models.
"""

import numpy as np
from numpy.typing import NDArray


def black_scholes(
    spot: float,
    strike: float,
    maturity: float,
    vol: float,
    risk_free_rate: float = 0.0,
    dividend_yield: float = 0.0,
    option_type: str = "call",
) -> float:
    """
    Compute the Black-Scholes price for a European option.

    Args:
        spot (float): Current spot price.
        strike (float): Strike price.
        maturity (float): Time to maturity (in years).n        vol (float): Volatility.
        risk_free_rate (float): Risk-free interest rate. Defaults to 0.0.
        dividend_yield (float): Dividend yield. Defaults to 0.0.
        option_type (str): "call" or "put". Defaults to "call".

    Returns:
        float: Option price.
    """
    from scipy.stats import norm
    
    d1 = (
        np.log(spot / strike)
        + (risk_free_rate - dividend_yield + 0.5 * vol ** 2) * maturity
    ) / (vol * np.sqrt(maturity))
    d2 = d1 - vol * np.sqrt(maturity)
    
    if option_type == "call":
        price = (
            spot * np.exp(-dividend_yield * maturity) * norm.cdf(d1)
            - strike * np.exp(-risk_free_rate * maturity) * norm.cdf(d2)
        )
    else:
        price = (
            strike * np.exp(-risk_free_rate * maturity) * norm.cdf(-d2)
            - spot * np.exp(-dividend_yield * maturity) * norm.cdf(-d1)
        )
    
    return price


def binary_payoff(
    paths: NDArray[np.float64],
    condition: str,
    strike: float,
    payoff: float,
) -> NDArray[np.float64]:
    """
    Compute binary payoff for a set of paths.

    Args:
        paths (NDArray[np.float64]): Simulated price paths. Shape: (num_paths, num_time_steps).
        condition (str): "above" or "below".
        strike (float): Strike price.
        payoff (float): Payoff if condition is met.

    Returns:
        NDArray[np.float64]: Binary payoff for each path. Shape: (num_paths,).
    """
    final_prices = paths[:, -1]
    if condition == "above":
        return np.where(final_prices >= strike,    payoff, 0.0)
    elif condition == "below":
        return np.where(final_prices <= strike, payoff, 0.0)
    else:
        raise ValueError(f"Unknown condition: {condition}")