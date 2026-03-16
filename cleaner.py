"""Data cleaning utilities."""

import pandas as pd


def _safe_float(val, default: float = 0.0) -> float:
    """Safely convert value to float."""
    try:
        v = float(val)
        return v if not pd.isna(v) else default
    except (TypeError, ValueError):
        return default


def _safe_int(val, default: int = 0) -> int:
    """Safely convert value to int."""
    try:
        v = float(val)
        return int(v) if not pd.isna(v) else default
    except (TypeError, ValueError):
        return default