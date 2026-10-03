"""Spanwise section model: the airfoils of a wing's sections, blended along the span.

:class:`WingSectionModel` implements :class:`liftingline.core.section.SectionModel`. At every
spanwise station the coefficients of the two neighbouring sections' airfoils are evaluated at the
**local Reynolds number** and blended linearly in the spanwise coordinate (coefficient space, not
coordinate space). Individual factors of a section can be overridden (wind-tunnel data):
an override replaces that section's polar by a smooth parametric one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..core.section import LinearParams, SectionCoeffs
from ..core.wing import Wing
from . import builtin_calibration as cal
from .resolver import AirfoilResolver, Resolution
from .section_base import AirfoilAero, SectionEngine
from .section_builtin import saturate


@dataclass(frozen=True)
class SectionOverride:
    """User-supplied factors for one wing section (``None`` keeps the engine's value)."""

    a0: float | None = None  # lift-curve slope, 1/rad
    alpha_L0: float | None = None  # zero-lift angle, rad
    cm_c4: float | None = None
    cl_max: float | None = None
    cd0: float | None = None  # minimum profile drag coefficient

    @property
    def active(self) -> bool:
        return any(v is not None for v in (self.a0, self.alpha_L0, self.cm_c4, self.cl_max, self.cd0))


def reference_cd0(base: AirfoilAero, re: np.ndarray) -> np.ndarray:
    """The ``cd0`` of the override model: the engine's profile drag about 3 degrees above its zero-lift angle
    with the parabolic ``K cl^2`` part taken out (the override polar adds that part back at every angle)."""
    re = np.asarray(re, dtype=float)
    alpha = base.linear(re).alpha_L0 + math.radians(3.0)
    cl, cd = base.coefficients(alpha, re)[:2]
    return np.maximum(cd - cal.K_CD * cl**2, 0.2 * cd)


class OverriddenAero:
    """An airfoil whose factors the user has overridden: smooth parametric polar built on the base data."""

    approximate = False

    def __init__(self, base: AirfoilAero, ov: SectionOverride) -> None:
        self.base, self.ov = base, ov
        self.label = f"{base.label} + overrides"
        self.approximate = base.approximate

    def linear(self, re: np.ndarray) -> LinearParams:
        p, ov = self.base.linear(re), self.ov
        a0 = np.full_like(p.a0, ov.a0) if ov.a0 is not None else p.a0
        aL0 = np.full_like(p.alpha_L0, ov.alpha_L0) if ov.alpha_L0 is not None else p.alpha_L0
        clm = np.full_like(p.cl_max, ov.cl_max) if ov.cl_max is not None else p.cl_max
        cm = np.full_like(p.cm_c4, ov.cm_c4) if ov.cm_c4 is not None else p.cm_c4
        return LinearParams(a0=a0, alpha_L0=aL0, cm_c4=cm, cl_max=clm, alpha_stall=aL0 + clm / a0 + math.radians(cal.STALL_DELTA_DEG))

    def coefficients(self, alpha: np.ndarray, re: np.ndarray):
        alpha = np.asarray(alpha, dtype=float)
        re = np.asarray(re, dtype=float) * np.ones_like(alpha)
        lin = self.linear(re)
        u = lin.a0 * (alpha - lin.alpha_L0)
        cl, dcl_du, sat, excess = saturate(u, lin.cl_max, 0.8 * lin.cl_max)
        if self.ov.cd0 is not None:
            cd0 = np.full_like(alpha, self.ov.cd0)
        else:  # the engine's own drag at a typical operating point, without its parabolic part
            cd0 = reference_cd0(self.base, re)
        cd = cd0 + cal.K_CD * cl**2 + 0.25 * excess**2
        return cl, cd, np.asarray(lin.cm_c4, dtype=float), lin.a0 * dcl_du

    def confidence(self, alpha: np.ndarray, re: np.ndarray):
        return self.base.confidence(alpha, re)


class WingSectionModel:
    """Section aerodynamics along a wing (a :class:`~liftingline.core.section.SectionModel`)."""

    has_polars = True  # real polars: the pipeline solves the nonlinear lifting-line equations by default

    def __init__(
        self,
        wing: Wing,
        aeros: Sequence[AirfoilAero],
        *,
        overrides: Sequence[SectionOverride | None] | None = None,
        label: str = "",
        notes: Sequence[str] = (),
        resolutions: Sequence[Resolution] = (),
    ) -> None:
        if len(aeros) != len(wing.sections):
            raise ValueError("need one airfoil per wing section")
        ovs = list(overrides) if overrides is not None else [None] * len(aeros)
        if len(ovs) != len(aeros):
            raise ValueError("need one override entry (or None) per wing section")
        self.wing = wing
        self.aeros: list[AirfoilAero] = [
            OverriddenAero(a, o) if (o is not None and o.active) else a for a, o in zip(aeros, ovs)
        ]
        self.label = label or ", ".join(dict.fromkeys(a.label for a in self.aeros))
        self.resolutions = list(resolutions)
        self.notes = list(notes)
        self.approximate = any(a.approximate for a in self.aeros)

    @classmethod
    def from_wing(
        cls,
        wing: Wing,
        resolver: AirfoilResolver,
        engine: SectionEngine,
        *,
        overrides: Sequence[SectionOverride | None] | None = None,
    ) -> "WingSectionModel":
        resolutions = [resolver.resolve(s.airfoil) for s in wing.sections]
        aeros = [engine.aero(r.airfoil) for r in resolutions]
        notes = list(dict.fromkeys(r.note for r in resolutions if r.note))
        return cls(wing, aeros, overrides=overrides, label=engine.name, notes=notes, resolutions=resolutions)

    # --------------------------------------------------------------------- blending
    def _weights(self, y: np.ndarray):
        i0, i1, xi = self.wing.airfoil_blend(y)
        for k in range(len(self.aeros)):
            w = np.where(i0 == k, 1.0 - xi, 0.0) + np.where(i1 == k, xi, 0.0)
            m = w > 0.0
            if m.any():
                yield k, w, m

    def linear(self, y: np.ndarray, re: np.ndarray) -> LinearParams:
        y = np.asarray(y, dtype=float)
        re = np.broadcast_to(np.asarray(re, dtype=float), y.shape).astype(float)
        acc = {k: np.zeros(y.shape) for k in ("a0", "alpha_L0", "cm_c4", "cl_max", "alpha_stall")}
        for k, w, m in self._weights(y):
            p = self.aeros[k].linear(re[m])
            for key in acc:
                acc[key][m] += w[m] * getattr(p, key)
        return LinearParams(**acc)

    def evaluate(self, y: np.ndarray, alpha: np.ndarray, re: np.ndarray) -> SectionCoeffs:
        y = np.asarray(y, dtype=float)
        alpha = np.broadcast_to(np.asarray(alpha, dtype=float), y.shape).astype(float)
        re = np.broadcast_to(np.asarray(re, dtype=float), y.shape).astype(float)
        cl, cd, cm, dcl = (np.zeros(y.shape) for _ in range(4))
        for k, w, m in self._weights(y):
            c, d, mm, s = self.aeros[k].coefficients(alpha[m], re[m])
            cl[m] += w[m] * c
            cd[m] += w[m] * d
            cm[m] += w[m] * mm
            dcl[m] += w[m] * s
        return SectionCoeffs(cl=cl, cd=cd, cm=cm, cl_alpha=dcl)

    def confidence(self, y: np.ndarray, alpha: np.ndarray, re: np.ndarray) -> np.ndarray | None:
        """Lowest engine confidence among the airfoils blended at each station (None if unreported)."""
        y = np.asarray(y, dtype=float)
        alpha = np.broadcast_to(np.asarray(alpha, dtype=float), y.shape).astype(float)
        re = np.broadcast_to(np.asarray(re, dtype=float), y.shape).astype(float)
        out, seen = np.ones(y.shape), False
        for k, _, m in self._weights(y):
            c = self.aeros[k].confidence(alpha[m], re[m])
            if c is not None:
                out[m] = np.minimum(out[m], c)
                seen = True
        return out if seen else None
