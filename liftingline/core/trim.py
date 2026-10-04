"""Trim: the angle of attack or the airspeed that gives a target lift (every evaluation is a full analysis)."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import brentq

from .analysis import AnalysisRequest, run
from .forces import Totals


class TrimError(ValueError):
    """The target cannot be reached; the message says why."""


@dataclass
class TrimResult:
    variable: str  # "alpha_deg" or "speed"
    value: float  # degrees or m/s
    totals: Totals
    target: str
    evaluations: int
    warnings: list[str]


class _Counter:
    def __init__(self, fn):
        self.fn, self.n, self.last = fn, 0, None

    def __call__(self, x: float) -> float:
        self.n += 1
        value, self.last = self.fn(x)
        return value


def _analyse(req: AnalysisRequest, **flow_changes) -> "tuple[Totals, list[str]]":
    r = run(replace(req, flow=replace(req.flow, **flow_changes), convergence_check=False))
    return r.totals, r.warnings


def trim_alpha(
    req: AnalysisRequest,
    *,
    CL: float | None = None,
    lift: float | None = None,
    alpha_range: tuple[float, float] = (-10.0, 28.0),
    step: float = 4.0,
) -> TrimResult:
    """Angle of attack (deg, of the flow: root incidence is not included) for a target ``CL`` or a target ``lift`` (N).

    The lowest angle that reaches the target is returned (the rising branch, before stall).
    """
    if (CL is None) == (lift is None):
        raise TrimError("give exactly one of CL or lift")
    qS = req.flow.q * req.wing.area
    target = float(CL) if CL is not None else float(lift) / qS
    if not math.isfinite(target):
        raise TrimError("the target must be a finite number")

    cache: dict[float, Totals] = {}

    def f(a: float) -> float:
        if a not in cache:
            cache[a] = _analyse(req, alpha_deg=a)[0]
        return cache[a].CL - target

    lo, hi = alpha_range
    grid = list(np.arange(lo, hi + 0.5 * step, step))
    prev_a, prev_f = grid[0], f(grid[0])
    if prev_f == 0.0:
        root = prev_a
    else:
        root = None
        for a in grid[1:]:
            fa = f(a)
            if prev_f * fa <= 0.0:
                root = brentq(f, prev_a, a, xtol=1e-9, rtol=1e-12) if fa != 0.0 else a
                break
            prev_a, prev_f = a, fa
    if root is None:
        cls = [t.CL for t in cache.values()]
        raise TrimError(
            f"CL = {target:.3f} is not reached for angles of attack between {lo:.0f} and {hi:.0f} degrees "
            f"(CL ranges from {min(cls):.3f} to {max(cls):.3f}); the wing stalls first or the target is too low."
        )
    totals, warnings = _analyse(req, alpha_deg=float(root))
    what = f"CL = {target:.4f}" if CL is not None else f"lift = {lift:.6g} N (CL = {target:.4f})"
    return TrimResult("alpha_deg", float(root), totals, what, len(cache) + 1, warnings)


def trim_speed(
    req: AnalysisRequest,
    lift: float,
    *,
    speed_range: tuple[float, float] | None = None,
) -> TrimResult:
    """True airspeed (m/s) at which the wing makes ``lift`` (N) at the flow's angle of attack."""
    if not lift > 0.0:
        raise TrimError("the lift to be carried must be positive")
    lo, hi = speed_range if speed_range is not None else (3.0, 0.92 * req.flow.a)
    if not lo < hi:
        raise TrimError("empty speed range")
    n = [0]

    def f(v: float) -> float:
        n[0] += 1
        return _analyse(req, V=v)[0].L - lift

    f_lo, f_hi = f(lo), f(hi)
    if f_lo > 0.0:
        raise TrimError(f"even at {lo:.1f} m/s the wing carries more than {lift:.6g} N at this angle of attack")
    if f_hi < 0.0:
        raise TrimError(f"at {hi:.0f} m/s the wing still carries less than {lift:.6g} N at this angle of attack; increase the angle of attack")
    root = brentq(f, lo, hi, xtol=1e-9, rtol=1e-12)
    totals, warnings = _analyse(req, V=float(root))
    return TrimResult("speed", float(root), totals, f"lift = {lift:.6g} N", n[0] + 1, warnings)


def stall_speed(weight: float, rho: float, area: float, cl_max: float) -> float:
    """Level-flight stall speed sqrt(2 W / (rho S CL_max)), m/s."""
    if not (weight > 0 and rho > 0 and area > 0 and cl_max > 0):
        raise TrimError("weight, density, area and CL_max must be positive")
    return math.sqrt(2.0 * weight / (rho * area * cl_max))
