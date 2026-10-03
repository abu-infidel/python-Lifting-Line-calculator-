"""NACA 6-series: exact analytic mean lines and a flagged approximate fallback geometry.

The *mean line* of the 6-series (the ``a`` family of Abbott & von Doenhoff) is a closed
formula, so zero-lift angle and pitching moment follow exactly from thin-airfoil theory.
The *thickness distributions* are not closed formulas (they come from NACA's tabulated
thickness forms), so the exact section shape is taken from the user's ``.dat`` library
(:mod:`liftingline.airfoil.library`). Only if the file is missing does
:func:`naca6_fallback_airfoil` build an **approximate** section (exact mean line + 4-digit
thickness), clearly flagged ``approximate=True``.

Mean line, loading uniform from the leading edge to ``x = a`` then falling linearly to zero
at the trailing edge (``c_li`` = design lift coefficient)::

    a = 1:   z = -(c_li / 4 pi) [ (1-x) ln(1-x) + x ln x ]
    a < 1:   z = c_li / (2 pi (a+1)) { 1/(1-a) [ (a-x)^2 ln|a-x| / 2 - (1-x)^2 ln(1-x) / 2
                                                 + (1-x)^2 / 4 - (a-x)^2 / 4 ]
                                       - x ln x + g - h x }
    g = -1/(1-a) [ a^2 (ln(a)/2 - 1/4) + 1/4 ]
    h =  1/(1-a) [ (1-a)^2 ln(1-a)/2 - (1-a)^2/4 ] + g
"""

from __future__ import annotations

import math

import numpy as np

from .designation import NacaCode, parse_naca
from .geometry import Airfoil, build_from_camber_thickness
from .naca4 import half_thickness


def _xlogx(u) -> np.ndarray:
    """u ln|u| with the limit 0 at u = 0."""
    u = np.asarray(u, dtype=float)
    out = np.zeros_like(u)
    nz = u != 0.0
    out[nz] = u[nz] * np.log(np.abs(u[nz]))
    return out


def _g_h(a: float) -> tuple[float, float]:
    g = -(1.0 / (1.0 - a)) * (a * a * (0.5 * math.log(a) - 0.25) + 0.25)
    h = (1.0 / (1.0 - a)) * (0.5 * (1.0 - a) ** 2 * math.log(1.0 - a) - 0.25 * (1.0 - a) ** 2) + g
    return g, h


def mean_line(x, cli: float, a: float = 1.0) -> np.ndarray:
    """Mean-line ordinate z(x)/c for design lift coefficient ``cli``."""
    x = np.asarray(x, dtype=float)
    if a >= 1.0 - 1e-9:
        return -(cli / (4.0 * math.pi)) * (_xlogx(1.0 - x) + _xlogx(x))
    g, h = _g_h(a)
    bracket = (1.0 / (1.0 - a)) * (
        0.5 * (a - x) * _xlogx(a - x) - 0.5 * (1.0 - x) * _xlogx(1.0 - x) + 0.25 * (1.0 - x) ** 2 - 0.25 * (a - x) ** 2
    ) - _xlogx(x) + g - h * x
    return cli / (2.0 * math.pi * (a + 1.0)) * bracket


def mean_line_slope(x, cli: float, a: float = 1.0) -> np.ndarray:
    """dz/dx of the mean line (integrable logarithmic singularity at the leading edge)."""
    x = np.clip(np.asarray(x, dtype=float), 1e-12, 1.0)
    if a >= 1.0 - 1e-9:
        return -(cli / (4.0 * math.pi)) * (np.log(x) - np.log(np.clip(1.0 - x, 1e-12, None)))
    _, h = _g_h(a)
    k = cli / (2.0 * math.pi * (a + 1.0))
    return k * ((1.0 / (1.0 - a)) * (_xlogx(1.0 - x) - _xlogx(a - x)) - np.log(x) - 1.0 - h)


def naca6_fallback_airfoil(code: NacaCode | str, *, n_per_surface: int = 100) -> Airfoil:
    """Approximate 6-series section: exact mean line with the 4-digit thickness form.

    Lift and moment characteristics follow the exact mean line, but the section shape (and
    therefore profile drag / laminar bucket) is only approximate - the result is flagged.
    """
    c = parse_naca(code) if isinstance(code, str) else code
    if not c.is_6_series:
        raise ValueError(f"{c.name} is not a 6-series airfoil")
    a = c.effective_a
    return build_from_camber_thickness(
        c.name,
        lambda x: half_thickness(x, c.thickness, True),
        lambda x: mean_line(x, c.cli, a),
        lambda x: mean_line_slope(x, c.cli, a),
        n_per_surface=n_per_surface,
        source="approximate 6-series (exact mean line + 4-digit thickness)",
        approximate=True,
        notes=(
            "approximate geometry: the exact 6-series thickness form needs the tabulated ordinates "
            "(load the matching .dat file from your library for exact results)",
        ),
    )
