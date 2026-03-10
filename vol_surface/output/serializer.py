"""Serialize VolSurface to JSON."""

from __future__ import annotations

import json
from pathlib import Path

from vol_surface.data.schema import VolSurface


def to_json(surface: VolSurface, indent: int = 2) -> str:
    return surface.model_dump_json(indent=indent)


def save_json(surface: VolSurface, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(to_json(surface))
    return p


def load_json(path: str | Path) -> VolSurface:
    p = Path(path)
    data = json.loads(p.read_text())
    return VolSurface.model_validate(data)
