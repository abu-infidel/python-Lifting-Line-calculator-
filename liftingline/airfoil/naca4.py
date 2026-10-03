"""NACA 4-digit airfoils (exact closed-form geometry)."""

from __future__ import annotations

import numpy as np

from .designation import NacaCode, parse_naca
from .geometry import Airfoil, build_from_camber_thickness


def half_thickness(x, t: float, closed_te: bool = True) -> np.ndarray:
    """NACA 4-digit half-thickness distribution (fraction of chord).

    ``closed_te`` uses the -0.1036 x^4 coefficient so the trailing edge closes to a point;
    the original -0.1015 leaves a 0.21 %-chord gap.
    """
    x = np.asarray(x, dtype=float)
    a4 = -0.1036 if closed_te else -0.1015
    return 5.0 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 + a4 * x**4)


def camber_line(x, m: float, p: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if m == 0.0 or p == 0.0:
        return np.zeros_like(x)
    return np.where(x < p, m / p**2 * (2 * p * x - x**2), m / (1 - p) ** 2 * (1 - 2 * p + 2 * p * x - x**2))


def camber_slope(x, m: float, p: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if m == 0.0 or p == 0.0:
        return np.zeros_like(x)
    return np.where(x < p, 2 * m / p**2 * (p - x), 2 * m / (1 - p) ** 2 * (p - x))


def naca4_airfoil(code: NacaCode | str, *, n_per_surface: int = 100, closed_te: bool = True) -> Airfoil:
    c = parse_naca(code) if isinstance(code, str) else code
    if c.family != "4":
        raise ValueError(f"{c.name} is not a 4-digit airfoil")
    return build_from_camber_thickness(
        c.name,
        lambda x: half_thickness(x, c.thickness, closed_te),
        lambda x: camber_line(x, c.m, c.p),
        lambda x: camber_slope(x, c.m, c.p),
        n_per_surface=n_per_surface,
        source="NACA 4-digit (analytic)",
    )
