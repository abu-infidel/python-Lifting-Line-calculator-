"""Physical corrections applied around the section data: compressibility, ground effect."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .section import LinearParams, SectionCoeffs, SectionModel

#: Mach number above which the Prandtl-Glauert rule is no longer trustworthy
PG_MACH_LIMIT = 0.7


def pg_beta(mach: float) -> float:
    """Prandtl-Glauert factor sqrt(1 - M^2)."""
    if mach >= 1.0:
        raise ValueError("Prandtl-Glauert compressibility correction needs a subsonic Mach number")
    return math.sqrt(1.0 - mach * mach)


@dataclass
class CompressibleSection:
    """Applies the Prandtl-Glauert rule to an incompressible section model.

    Lift-curve slope and pitching moment scale with ``1/beta``; zero-lift angle and
    profile drag are unchanged. (Drag divergence and shock stall are *not* modelled.)
    """

    inner: SectionModel
    mach: float

    @property
    def label(self) -> str:
        return self.inner.label

    @property
    def beta(self) -> float:
        return pg_beta(self.mach)

    def linear(self, y, re) -> LinearParams:
        p = self.inner.linear(y, re)
        k = 1.0 / self.beta
        return LinearParams(a0=p.a0 * k, alpha_L0=p.alpha_L0, cm_c4=p.cm_c4 * k, cl_max=p.cl_max, alpha_stall=p.alpha_stall)

    def evaluate(self, y, alpha, re) -> SectionCoeffs:
        c = self.inner.evaluate(y, alpha, re)
        k = 1.0 / self.beta
        return SectionCoeffs(cl=c.cl * k, cd=c.cd, cm=c.cm * k, cl_alpha=c.cl_alpha * k)


def ground_effect_induced_drag_factor(height_over_span: float) -> float:
    """McCormick/Wieselsberger estimate of CDi(in ground effect)/CDi(free air) for a
    wing at height ``h`` (c/4 above ground) over span ``b``. Used for sanity checks of
    the image-vortex model, not by the solvers themselves."""
    k = (16.0 * height_over_span) ** 2
    return k / (1.0 + k)


def _as_array(x) -> np.ndarray:
    return np.asarray(x, dtype=float)
