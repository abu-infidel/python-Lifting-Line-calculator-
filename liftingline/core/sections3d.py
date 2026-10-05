"""Where the wing's airfoil sections are in space: the shared geometry of the front/side views and the 3-D model.

A section at spanwise station ``y`` is the section airfoil (blended linearly between neighbouring wing sections, as in
the analysis) scaled to the local chord, turned nose-up by the local twist - plus the root incidence, optionally -
about its quarter-chord point, and placed on the quarter-chord line (sweep in x, dihedral in z). Frame: x aft (root
leading edge at 0), y to the right wing, z up.

GUI-free and light (numpy only): the calculator uses it for its views, the 3-D model exporter for its mesh, so the two
always show the same wing.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from ..airfoil.geometry import Airfoil, cosine_spacing
from .wing import Wing


class SectionShapeError(ValueError):
    """An airfoil whose surfaces cross (bad coordinates)."""


def closed_section(af: Airfoil, frac: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(x, z_upper, z_lower) of an airfoil at chord fractions ``frac`` (0 nose, 1 trailing edge), unit chord.

    The nose is a single point and a thin trailing-edge gap is blended away over the rear of the section, so both
    surfaces meet at the trailing edge (a closed, sharp section)."""
    x0 = float(af.coords[:, 0].min())
    x = x0 + (1.0 - x0) * frac
    zu, zl = np.asarray(af.y_upper(x), dtype=float), np.asarray(af.y_lower(x), dtype=float)
    nose = 0.5 * (zu[0] + zl[0])
    zu[0] = zl[0] = nose
    mid = 0.5 * (zu[-1] + zl[-1])
    ramp = frac**2
    zu = zu - (zu[-1] - mid) * ramp
    zl = zl - (zl[-1] - mid) * ramp
    zu[-1] = zl[-1] = mid
    if np.any(zu[1:-1] < zl[1:-1] - 1e-3):
        raise SectionShapeError(f"airfoil {af.name!r}: the upper surface crosses the lower one; check the coordinates")
    zl = np.minimum(zl, zu)  # digitising noise of a thin trailing edge must not turn the surface inside out
    return x, zu, zl


def section_shapes(airfoils: Sequence[Airfoil], n_chord: int) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """Chord fractions (``n_chord`` intervals, cosine-clustered) and the closed shape of every section airfoil."""
    frac = cosine_spacing(n_chord)
    return frac, [closed_section(af, frac) for af in airfoils]


def blended_shape(wing: Wing, shapes, y: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unit-chord (x, z_upper, z_lower) of the section at station ``y`` (airfoils blended between wing sections)."""
    i0, i1, xi = wing.airfoil_blend(np.array([y]))
    a, b, w = int(i0[0]), int(i1[0]), float(xi[0])
    return tuple((1.0 - w) * np.asarray(shapes[a][k]) + w * np.asarray(shapes[b][k]) for k in range(3))  # type: ignore[return-value]


def place(wing: Wing, y: float, xa: np.ndarray, za: np.ndarray, include_incidence: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Aircraft-frame (x, z) of unit-chord airfoil points (xa, za) at station ``y``."""
    c = float(wing.chord(y))
    theta = math.radians(float(wing.twist_deg(y)) + (wing.incidence_deg if include_incidence else 0.0))
    dx, dz = (np.asarray(xa) - 0.25) * c, np.asarray(za) * c
    ct, st = math.cos(theta), math.sin(theta)  # nose-up rotation about the quarter chord
    return float(wing.x_c4(y)) + dx * ct + dz * st, float(wing.z_c4(y)) - dx * st + dz * ct


def section_outline(wing: Wing, shapes, y: float, include_incidence: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Closed outline (x, z) of the section at station ``y``: trailing edge, upper surface, nose, lower surface, TE."""
    x, zu, zl = blended_shape(wing, shapes, y)
    xa = np.concatenate([x[::-1], x[1:]])
    za = np.concatenate([zu[::-1], zl[1:]])
    return place(wing, y, xa, za, include_incidence)


def front_silhouette(wing: Wing, airfoils: Sequence[Airfoil], ys: np.ndarray, include_incidence: bool = True,
                     n_chord: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """Lowest and highest z of the wing seen from the front at each station ``ys`` (thickness, camber, twist,
    incidence and dihedral all show; a zero-chord tip is a point on the quarter-chord line)."""
    _frac, shapes = section_shapes(airfoils, n_chord)
    lo, hi = np.empty(len(ys)), np.empty(len(ys))
    for k, y in enumerate(ys):
        if float(wing.chord(y)) <= 0.0:
            lo[k] = hi[k] = float(wing.z_c4(y))
            continue
        _x, z = section_outline(wing, shapes, float(y), include_incidence)
        lo[k], hi[k] = float(z.min()), float(z.max())
    return lo, hi


__all__ = ["SectionShapeError", "closed_section", "section_shapes", "blended_shape", "place", "section_outline",
           "front_silhouette"]
