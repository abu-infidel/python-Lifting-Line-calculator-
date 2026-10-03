"""Thin-airfoil theory for a given mean line.

With ``x = (c/2)(1 - cos(theta))`` and the camber slope ``dz/dx`` the Fourier coefficients

    A0 = alpha - (1/pi) int dz/dx dtheta,   A_n = (2/pi) int dz/dx cos(n theta) dtheta

give ``cl = pi (2 A0 + A1)``, ``alpha_L0 = -(1/pi) int dz/dx (cos(theta) - 1) dtheta`` and
``cm_c/4 = (pi/4)(A2 - A1)``. The *ideal* angle of attack (A0 = 0, no leading-edge singularity)
is ``(1/pi) int dz/dx dtheta`` and the lift coefficient there is ``pi A1``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .geometry import Airfoil


@dataclass(frozen=True)
class ThinAirfoilResult:
    alpha_L0: float  # zero-lift angle, rad
    cm_c4: float  # pitching moment about c/4 (angle-independent in thin-airfoil theory)
    alpha_ideal: float  # ideal angle of attack, rad
    cl_ideal: float  # lift coefficient at the ideal angle (the "design" cl of NACA mean lines)

    @property
    def alpha_L0_deg(self) -> float:
        return math.degrees(self.alpha_L0)

    @property
    def alpha_ideal_deg(self) -> float:
        return math.degrees(self.alpha_ideal)


def thin_airfoil(slope: Callable[[np.ndarray], np.ndarray], n: int = 20000) -> ThinAirfoilResult:
    """Thin-airfoil results from a camber-slope function ``dz/dx(x)`` (midpoint rule in theta)."""
    theta = (np.arange(n) + 0.5) * math.pi / n
    x = 0.5 * (1.0 - np.cos(theta))
    s = np.asarray(slope(x), dtype=float)
    d = math.pi / n
    i0, i1, i2 = float(np.sum(s) * d), float(np.sum(s * np.cos(theta)) * d), float(np.sum(s * np.cos(2 * theta)) * d)
    a1, a2 = 2.0 / math.pi * i1, 2.0 / math.pi * i2
    return ThinAirfoilResult(
        alpha_L0=-(i1 - i0) / math.pi,
        cm_c4=0.25 * math.pi * (a2 - a1),
        alpha_ideal=i0 / math.pi,
        cl_ideal=math.pi * a1,
    )


def thin_airfoil_from_airfoil(airfoil: Airfoil, n: int = 4000) -> ThinAirfoilResult:
    """Thin-airfoil results from the mean line extracted from coordinates."""
    return thin_airfoil(airfoil.camber_slope, n=n)
