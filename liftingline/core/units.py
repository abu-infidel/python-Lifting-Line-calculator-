"""Unit conversion.

Everything inside the solvers is SI; the GUI converts at its edges. Each *kind* of
quantity has a table of ``unit -> factor`` where ``value_SI = value_unit * factor``.
Temperature (offset units) and angle are handled separately.
"""

from __future__ import annotations

import math

import numpy as np

_G0 = 9.80665
_LBF = 4.4482216152605
_FT = 0.3048
_SLUG = 14.5939029372

_FACTORS: dict[str, dict[str, float]] = {
    "length": {"m": 1.0, "cm": 1e-2, "mm": 1e-3, "km": 1e3, "ft": _FT, "in": 0.0254, "nmi": 1852.0},
    "area": {"m²": 1.0, "cm²": 1e-4, "ft²": _FT**2, "in²": 0.0254**2},
    "speed": {
        "m/s": 1.0,
        "km/h": 1.0 / 3.6,
        "kt": 1852.0 / 3600.0,
        "mph": 1609.344 / 3600.0,
        "ft/s": _FT,
    },
    "force": {"N": 1.0, "kN": 1e3, "lbf": _LBF, "kgf": _G0},
    "moment": {"N·m": 1.0, "kN·m": 1e3, "lbf·ft": _LBF * _FT, "lbf·in": _LBF * 0.0254, "kgf·m": _G0},
    "density": {"kg/m³": 1.0, "g/cm³": 1e3, "slug/ft³": _SLUG / _FT**3},
    "pressure": {
        "Pa": 1.0,
        "kPa": 1e3,
        "hPa": 1e2,
        "atm": 101325.0,
        "psi": _LBF / 0.0254**2,
        "inHg": 3386.389,
    },
    "viscosity": {"Pa·s": 1.0, "slug/(ft·s)": _SLUG / _FT},
    "mass": {"kg": 1.0, "g": 1e-3, "lb": 0.45359237, "slug": _SLUG},
    "angle": {"rad": 1.0, "deg": math.pi / 180.0},
}

_TEMPERATURE_UNITS = ("K", "°C", "°F")

#: default display unit per quantity kind for each unit system
SYSTEMS: dict[str, dict[str, str]] = {
    "SI": {
        "length": "m",
        "area": "m²",
        "speed": "m/s",
        "force": "N",
        "moment": "N·m",
        "density": "kg/m³",
        "pressure": "Pa",
        "viscosity": "Pa·s",
        "mass": "kg",
        "angle": "deg",
        "temperature": "°C",
    },
    "Imperial": {
        "length": "ft",
        "area": "ft²",
        "speed": "kt",
        "force": "lbf",
        "moment": "lbf·ft",
        "density": "slug/ft³",
        "pressure": "psi",
        "viscosity": "slug/(ft·s)",
        "mass": "lb",
        "angle": "deg",
        "temperature": "°F",
    },
}


def kinds() -> tuple[str, ...]:
    return tuple(_FACTORS) + ("temperature",)


def units_of(kind: str) -> tuple[str, ...]:
    """All unit names available for a quantity kind."""
    if kind == "temperature":
        return _TEMPERATURE_UNITS
    return tuple(_FACTORS[kind])


def display_unit(system: str, kind: str) -> str:
    return SYSTEMS[system][kind]


def factor(kind: str, unit: str) -> float:
    """Multiplier that converts ``unit`` to SI (not valid for temperature)."""
    try:
        return _FACTORS[kind][unit]
    except KeyError as exc:
        raise ValueError(f"unknown {kind} unit {unit!r}") from exc


def to_si(kind: str, value, unit: str):
    """Convert ``value`` (scalar or array) expressed in ``unit`` to SI."""
    if kind == "temperature":
        v = np.asarray(value, dtype=float) if not np.isscalar(value) else float(value)
        if unit == "K":
            return v
        if unit == "°C":
            return v + 273.15
        if unit == "°F":
            return (v - 32.0) * 5.0 / 9.0 + 273.15
        raise ValueError(f"unknown temperature unit {unit!r}")
    return value * factor(kind, unit)


def from_si(kind: str, value, unit: str):
    """Convert an SI ``value`` (scalar or array) to ``unit``."""
    if kind == "temperature":
        v = np.asarray(value, dtype=float) if not np.isscalar(value) else float(value)
        if unit == "K":
            return v
        if unit == "°C":
            return v - 273.15
        if unit == "°F":
            return (v - 273.15) * 9.0 / 5.0 + 32.0
        raise ValueError(f"unknown temperature unit {unit!r}")
    return value / factor(kind, unit)


def convert(kind: str, value, unit_from: str, unit_to: str):
    return from_si(kind, to_si(kind, value, unit_from), unit_to)
