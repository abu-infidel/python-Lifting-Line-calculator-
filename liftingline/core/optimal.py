"""Optimal twist: the geometric twist that makes a given wing carry an elliptic load (span efficiency 1).

The inverse of the classical lifting-line problem. For an elliptic circulation ``Gamma = Gamma_0 sin(theta)`` only the
first Fourier coefficient ``A_1 = CL / (pi AR)`` is non-zero and Prandtl's equation gives, station by station,

    alpha_geo(y) - alpha_L0(y) = (CL / (pi AR)) * (1 + 4 b sin(theta) / (c(y) a0(y))),    y = -(b/2) cos(theta).

The local geometric angle follows from the section's zero-lift angle and lift slope at its own Reynolds number; the twist
is that angle relative to the root. A pointed tip (zero chord) would need an unbounded twist, so the outermost station is
placed slightly inboard of the tip in that case and the result is flagged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

from .analysis import AnalysisRequest, run
from .corrections import CompressibleSection
from .wing import Section, Wing


@dataclass
class OptimalTwist:
    rows: list[dict[str, float]]  # section-table rows: y, chord, x_le, z, twist_deg (SI, degrees)
    alpha_root_deg: float  # root-chord angle of attack (flow + incidence) that gives the target CL with this twist
    cl_target: float
    max_abs_twist_deg: float
    notes: list[str] = field(default_factory=list)
    predicted: dict[str, float] = field(default_factory=dict)  # CL, CDi and e of the piecewise-linear wing at alpha_root


def optimal_twist(req: AnalysisRequest, cl_target: float, n_stations: int = 15, refine: bool = True) -> OptimalTwist:
    """Twist distribution for an elliptic load at ``cl_target`` on the planform of ``req.wing``.

    The ideal twist is a smooth curve; the returned section table is its piecewise-linear interpolation at
    ``n_stations`` stations (dense towards the tip). With ``refine`` the root angle is corrected by the actual solver so
    that the interpolated wing carries exactly ``cl_target``; ``predicted`` then holds its CL, CDi and span efficiency.
    """
    if not math.isfinite(cl_target):
        raise ValueError("the target lift coefficient must be a finite number")
    if n_stations < 3:
        raise ValueError("use at least three stations")
    wing, flow = req.wing, req.flow
    section = CompressibleSection(req.section, flow.mach) if req.compressibility else req.section
    b, AR = wing.span, wing.aspect_ratio
    hs = wing.half_span

    phi = 0.5 * math.pi * np.arange(n_stations) / (n_stations - 1)  # root -> tip, dense towards the tip
    y = hs * np.sin(phi)
    notes: list[str] = []
    if wing.tip_chord <= 1e-9 * wing.root_chord:
        y[-1] = hs * math.sin(0.5 * math.pi * (1.0 - 0.5 / (n_stations - 1)))  # half a step inboard of a pointed tip
        notes.append("the planform has a pointed tip, which would need an infinite twist: the outermost station is placed just inboard")
    chord = np.asarray(wing.chord(y), dtype=float)
    re = flow.reynolds(np.maximum(chord, 1e-6))
    lin = section.linear(y, re)
    sin_theta = np.sqrt(np.clip(1.0 - (2.0 * y / b) ** 2, 0.0, None))
    alpha_geo = lin.alpha_L0 + cl_target / (math.pi * AR) * (1.0 + 4.0 * b * sin_theta / (chord * lin.a0))  # rad
    twist = np.degrees(alpha_geo - alpha_geo[0])
    rows = [
        dict(y=float(y[i]), chord=float(chord[i]), x_le=float(wing.x_le(y[i])), z=float(wing.z_c4(y[i])), twist_deg=float(twist[i]))
        for i in range(n_stations)
    ]
    if rows[-1]["y"] < hs:  # keep the real tip so the span is unchanged: same chord, extrapolated twist
        rows.append(dict(y=float(hs), chord=0.0, x_le=float(wing.x_le(hs)), z=float(wing.z_c4(hs)), twist_deg=rows[-1]["twist_deg"]))
    peak = float(np.max(np.abs(twist)))
    if peak > 12.0:
        notes.append(f"the ideal twist reaches {peak:.1f} degrees, which is hard to build; a smaller twist gives a span efficiency slightly below 1")
    alpha_root = float(math.degrees(alpha_geo[0]))
    predicted: dict[str, float] = {}
    if refine:
        alpha_root, predicted = _refine_root_angle(req, rows, alpha_root, cl_target)
    return OptimalTwist(
        rows=rows,
        alpha_root_deg=alpha_root,
        cl_target=float(cl_target),
        max_abs_twist_deg=peak,
        notes=notes,
        predicted=predicted,
    )


def _refine_root_angle(req: AnalysisRequest, rows: list[dict[str, float]], alpha_root: float, cl_target: float):
    """Secant correction of the root angle so that the table's wing carries ``cl_target`` (a few cheap solves)."""
    wing = req.wing
    new = Wing(
        sections=tuple(Section(r["y"], r["chord"], r["x_le"], r["z"], r["twist_deg"]) for r in rows),
        incidence_deg=wing.incidence_deg,
        name=wing.name,
    )

    def solve(a_root: float):
        flow = replace(req.flow, alpha_deg=a_root - wing.incidence_deg)
        return run(replace(req, wing=new, flow=flow, convergence_check=False)).totals

    slope = 2.0 * math.pi * new.aspect_ratio / (new.aspect_ratio + 2.0) * math.pi / 180.0  # CL per degree, crude start
    a, t = alpha_root, solve(alpha_root)
    for _ in range(4):
        err = t.CL - cl_target
        if abs(err) < 1e-8:
            break
        a_next = a - err / slope
        t_next = solve(a_next)
        if abs(a_next - a) > 1e-12 and abs(t_next.CL - t.CL) > 1e-14:
            slope = (t_next.CL - t.CL) / (a_next - a)
        a, t = a_next, t_next
    return float(a), dict(CL=float(t.CL), CD_induced=float(t.CD_induced), e=float(t.e))
