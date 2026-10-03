"""NACA 5-digit airfoils (standard and reflex mean lines, NACA TR-537).

Constants for design lift coefficient 0.3 (first digit 2); other first digits scale the
camber linearly. The thickness distribution is the 4-digit one.
"""

from __future__ import annotations

import numpy as np

from .designation import NacaCode, parse_naca
from .geometry import Airfoil, build_from_camber_thickness
from .naca4 import half_thickness

# position digit -> (r, k1)
_STANDARD = {1: (0.0580, 361.4), 2: (0.1260, 51.64), 3: (0.2025, 15.957), 4: (0.2900, 6.643), 5: (0.3910, 3.230)}
# position digit -> (r, k1, k2/k1)
_REFLEX = {2: (0.1300, 51.99, 0.000764), 3: (0.2170, 15.793, 0.00677), 4: (0.3180, 6.520, 0.0303), 5: (0.4410, 3.191, 0.1355)}


def _params(c: NacaCode) -> tuple[float, float, float]:
    scale = c.cl5 / 0.3
    if c.reflex:
        r, k1, ratio = _REFLEX[c.p5]
        return r, k1 * scale, ratio
    r, k1 = _STANDARD[c.p5]
    return r, k1 * scale, 0.0


def camber_line(x, c: NacaCode) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    r, k1, ratio = _params(c)
    if c.reflex:
        a = ratio * (1 - r) ** 3 * x + r**3 * x - r**3
        fore = k1 / 6 * ((x - r) ** 3 - a)
        aft = k1 / 6 * (ratio * (x - r) ** 3 - a)
    else:
        fore = k1 / 6 * (x**3 - 3 * r * x**2 + r**2 * (3 - r) * x)
        aft = np.full_like(x, k1 * r**3 / 6) * (1 - x)
    return np.where(x < r, fore, aft)


def camber_slope(x, c: NacaCode) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    r, k1, ratio = _params(c)
    if c.reflex:
        b = ratio * (1 - r) ** 3 + r**3
        fore = k1 / 6 * (3 * (x - r) ** 2 - b)
        aft = k1 / 6 * (3 * ratio * (x - r) ** 2 - b)
    else:
        fore = k1 / 6 * (3 * x**2 - 6 * r * x + r**2 * (3 - r))
        aft = np.full_like(x, -k1 * r**3 / 6)
    return np.where(x < r, fore, aft)


def naca5_airfoil(code: NacaCode | str, *, n_per_surface: int = 100, closed_te: bool = True) -> Airfoil:
    c = parse_naca(code) if isinstance(code, str) else code
    if c.family != "5":
        raise ValueError(f"{c.name} is not a 5-digit airfoil")
    return build_from_camber_thickness(
        c.name,
        lambda x: half_thickness(x, c.thickness, closed_te),
        lambda x: camber_line(x, c),
        lambda x: camber_slope(x, c),
        n_per_surface=n_per_surface,
        source="NACA 5-digit (analytic)",
    )
