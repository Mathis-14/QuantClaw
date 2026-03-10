"""Clean raw option chains into calibration-ready VolSlice objects."""

from __future__ import annotations

import logging
from datetime import date, datetime

import numpy as np

from vol_surface.data.schema import OptionChain, VolSlice

logger = logging.getLogger(__name__)

MIN_STRIKES = 5
MONEYNESS_BAND = 0.5  # |log(K/F)| < 0.5


def clean_chain(
    chain: OptionChain,
    valuation_date: date | None = None,
    min_strikes: int = MIN_STRIKES,
    moneyness_band: float = MONEYNESS_BAND,
    use_otm: bool = True,
) -> list[VolSlice]:
    """Convert a raw OptionChain into a list of VolSlice objects.

    For each expiry:
    1. Use OTM options (calls K>F, puts K<F) or all if *use_otm* is False.
    2. Compute forward from put-call parity at ATM, or approximate as spot.
    3. Filter by moneyness band and minimum IV.
    4. Compute log-moneyness and total variance.
    """
    if valuation_date is None:
        valuation_date = chain.timestamp.date()

    expiries = sorted({q.expiry for q in chain.quotes})
    slices: list[VolSlice] = []

    for expiry in expiries:
        T = _year_fraction(valuation_date, expiry)
        if T <= 0:
            continue

        expiry_quotes = [q for q in chain.quotes if q.expiry == expiry]
        forward = _estimate_forward(chain.spot, expiry_quotes, T)

        filtered = _filter_quotes(
            expiry_quotes, forward, moneyness_band, use_otm
        )
        if len(filtered) < min_strikes:
            logger.warning(
                "Expiry %s: only %d clean strikes (need %d), skipping",
                expiry,
                len(filtered),
                min_strikes,
            )
            continue

        filtered.sort(key=lambda q: q.strike)
        strikes = [q.strike for q in filtered]
        ivs = [q.implied_vol for q in filtered]  # type: ignore[misc]
        log_k = [float(np.log(k / forward)) for k in strikes]
        total_var = [iv**2 * T for iv in ivs]
        weights = _compute_weights(filtered)

        slices.append(
            VolSlice(
                expiry=expiry,
                T=T,
                forward=forward,
                strikes=strikes,
                log_moneyness=log_k,
                total_variance=total_var,
                implied_vols=ivs,  # type: ignore[arg-type]
                weights=weights,
            )
        )

    logger.info("Cleaned %d / %d expiry slices", len(slices), len(expiries))
    return slices


def _year_fraction(d1: date, d2: date) -> float:
    return (d2 - d1).days / 365.25


def _estimate_forward(
    spot: float, quotes: list, T: float, r: float = 0.0
) -> float:
    """Estimate forward price.  Uses put-call parity near ATM if possible,
    otherwise falls back to spot * exp(r*T)."""
    calls = {q.strike: q for q in quotes if q.option_type == "call"}
    puts = {q.strike: q for q in quotes if q.option_type == "put"}
    common = sorted(set(calls) & set(puts))
    if common:
        atm_k = min(common, key=lambda k: abs(k - spot))
        c = calls[atm_k].mid
        p = puts[atm_k].mid
        return atm_k + np.exp(r * T) * (c - p)
    return spot * np.exp(r * T)


def _filter_quotes(quotes, forward, moneyness_band, use_otm):
    out = []
    for q in quotes:
        if q.implied_vol is None or q.implied_vol <= 0.01:
            continue
        k = np.log(q.strike / forward)
        if abs(k) > moneyness_band:
            continue
        if use_otm:
            if q.option_type == "call" and q.strike < forward:
                continue
            if q.option_type == "put" and q.strike > forward:
                continue
        out.append(q)

    seen: dict[float, object] = {}
    deduped = []
    for q in out:
        if q.strike not in seen:
            seen[q.strike] = q
            deduped.append(q)
        else:
            existing = seen[q.strike]
            if q.open_interest > existing.open_interest:  # type: ignore[union-attr]
                seen[q.strike] = q
                deduped[deduped.index(existing)] = q  # type: ignore[arg-type]
    return deduped


def _compute_weights(quotes) -> list[float]:
    """Weight by open interest, fall back to uniform."""
    ois = [q.open_interest for q in quotes]
    total = sum(ois)
    if total > 0:
        return [oi / total for oi in ois]
    return [1.0 / len(quotes)] * len(quotes)
