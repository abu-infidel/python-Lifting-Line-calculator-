"""Section (2-D airfoil) aerodynamics interface used by the lifting-line solvers.

An *engine* turns airfoil geometry into aerodynamic data. Solvers only see the
:class:`SectionModel` protocol, evaluated at spanwise stations ``y`` (absolute lateral
position in metres) and the local Reynolds number ``re``. Engines work in
**incompressible** flow; compressibility (Prandtl-Glauert) is applied by the
pipeline through :class:`liftingline.core.corrections.CompressibleSection`.

Conventions: angles in radians, ``alpha`` measured from the section chord line,
pitching moment about the quarter chord, nose-up positive.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class LinearParams:
    """Attached-flow ("linear range") description of the sections at a set of stations."""

    a0: np.ndarray  # lift-curve slope, 1/rad
    alpha_L0: np.ndarray  # zero-lift angle, rad
    cm_c4: np.ndarray  # pitching-moment coefficient about c/4
    cl_max: np.ndarray  # maximum lift coefficient (inf if unknown)
    alpha_stall: np.ndarray  # angle of attack at cl_max, rad (inf if unknown)


@dataclass(frozen=True)
class SectionCoeffs:
    """Section coefficients at given angles of attack."""

    cl: np.ndarray
    cd: np.ndarray
    cm: np.ndarray  # about c/4
    cl_alpha: np.ndarray  # d(cl)/d(alpha), 1/rad


@runtime_checkable
class SectionModel(Protocol):
    label: str

    def linear(self, y: np.ndarray, re: np.ndarray) -> LinearParams: ...

    def evaluate(self, y: np.ndarray, alpha: np.ndarray, re: np.ndarray) -> SectionCoeffs: ...


ArrayOrFloat = float | Callable[[np.ndarray], np.ndarray]


def _eval(value: ArrayOrFloat, y: np.ndarray) -> np.ndarray:
    if callable(value):
        return np.asarray(value(y), dtype=float)
    return np.full_like(np.asarray(y, dtype=float), float(value))


@dataclass
class LinearSection:
    """Idealised section: ``cl = a0 (alpha - alpha_L0)``, ``cd = cd0 + cd_k cl^2``.

    Values may be constants or callables of the station ``y``. With the defaults this
    is the thin symmetric airfoil of classical lifting-line theory (a0 = 2 pi).
    """

    a0: ArrayOrFloat = 2.0 * math.pi
    alpha_L0: ArrayOrFloat = 0.0  # rad
    cm_c4: ArrayOrFloat = 0.0
    cd0: ArrayOrFloat = 0.0
    cd_k: float = 0.0
    cl_max: float = math.inf
    label: str = "linear section"

    def linear(self, y: np.ndarray, re: np.ndarray) -> LinearParams:
        y = np.asarray(y, dtype=float)
        a0 = _eval(self.a0, y)
        aL0 = _eval(self.alpha_L0, y)
        alpha_stall = np.full_like(y, math.inf) if not math.isfinite(self.cl_max) else aL0 + self.cl_max / a0
        return LinearParams(
            a0=a0,
            alpha_L0=aL0,
            cm_c4=_eval(self.cm_c4, y),
            cl_max=np.full_like(y, self.cl_max),
            alpha_stall=alpha_stall,
        )

    def evaluate(self, y: np.ndarray, alpha: np.ndarray, re: np.ndarray) -> SectionCoeffs:
        y = np.asarray(y, dtype=float)
        alpha = np.asarray(alpha, dtype=float)
        a0 = _eval(self.a0, y)
        cl = a0 * (alpha - _eval(self.alpha_L0, y))
        cd = _eval(self.cd0, y) + self.cd_k * cl**2
        return SectionCoeffs(cl=cl, cd=cd, cm=_eval(self.cm_c4, y), cl_alpha=a0)
