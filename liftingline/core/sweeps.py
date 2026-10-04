"""Parameter sweeps (polars) and the quantities derived from them (first stall, maximum lift)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Callable, Sequence

import numpy as np

from .analysis import AnalysisRequest, run
from .controls import ControlError
from .flow import FlowCondition
from .forces import Totals

ALPHA, SPEED, ALTITUDE, BETA, DEFLECTION = "alpha", "speed", "altitude", "beta", "deflection"
VARIABLES = (ALPHA, SPEED, ALTITUDE, BETA, DEFLECTION)

#: a station counts as stalled when its lift coefficient is within 2 % of cl_max (the built-in engine's saturating
#: polar approaches cl_max only asymptotically; NeuralFoil's peaks are sharper)
STALL_MARGIN = 0.98

LABELS = {
    ALPHA: "angle of attack [deg]",
    SPEED: "airspeed [m/s]",
    ALTITUDE: "altitude [m]",
    BETA: "sideslip [deg]",
    DEFLECTION: "control deflection [deg]",
}

#: Totals attributes available as columns of a sweep
COLUMNS = ("CL", "CD", "CD_induced", "CD_profile", "CY", "Cm", "C_roll", "C_yaw", "L_over_D", "e", "L", "D")


@dataclass
class SweepPoint:
    value: float  # the swept variable (degrees for angles, SI otherwise)
    totals: Totals | None
    converged: bool = True
    max_margin: float = math.nan  # highest local cl / cl_max (nan when the section data has no cl_max)
    first_stall_y: float = math.nan  # |y| of the station with the highest margin, m
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.totals is not None


@dataclass
class SweepResult:
    variable: str
    points: list[SweepPoint]
    solver: str = ""
    engine: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def values(self) -> np.ndarray:
        return np.array([p.value for p in self.points], dtype=float)

    def column(self, name: str) -> np.ndarray:
        """Values of a ``Totals`` attribute at every point (nan where the point failed)."""
        return np.array([getattr(p.totals, name) if p.totals is not None else math.nan for p in self.points], dtype=float)

    @property
    def margin(self) -> np.ndarray:
        return np.array([p.max_margin for p in self.points], dtype=float)

    @property
    def converged(self) -> np.ndarray:
        return np.array([p.converged and p.ok for p in self.points], dtype=bool)

    def to_csv(self) -> str:
        head = ["value"] + list(COLUMNS) + ["max_cl_over_clmax", "converged"]
        lines = [",".join(head)]
        for p in self.points:
            t = p.totals
            vals = [repr(float(p.value))] + [("" if t is None else repr(float(getattr(t, c)))) for c in COLUMNS]
            lines.append(",".join(vals + [("" if math.isnan(p.max_margin) else repr(float(p.max_margin))), str(p.converged and p.ok)]))
        return "\n".join(lines) + "\n"


def linspace_values(start: float, stop: float, n: int) -> list[float]:
    if n < 2:
        raise ValueError("a sweep needs at least two points")
    return [float(v) for v in np.linspace(start, stop, n)]


def _modified_request(req: AnalysisRequest, variable: str, v: float, *, control: str | None, delta_T: float, hold: str) -> AnalysisRequest:
    flow = req.flow
    if variable == ALPHA:
        return replace(req, flow=replace(flow, alpha_deg=v))
    if variable == SPEED:
        return replace(req, flow=replace(flow, V=v))
    if variable == BETA:
        solver = "auto" if req.solver == "classical" else req.solver  # sideslip needs the numerical solver
        return replace(req, flow=replace(flow, beta_deg=v), solver=solver)
    if variable == ALTITUDE:
        speed = dict(V=flow.V) if hold == "speed" else dict(mach=flow.mach)
        new = FlowCondition.from_isa(v, alpha_deg=flow.alpha_deg, beta_deg=flow.beta_deg, delta_T=delta_T, **speed)
        return replace(req, flow=new)
    if variable == DEFLECTION:
        if req.controls is None or control is None:
            raise ControlError("define a control surface to sweep its deflection")
        return replace(req, controls=req.controls.with_deflection(control, v))
    raise ValueError(f"unknown sweep variable {variable!r}")


def run_sweep(
    req: AnalysisRequest,
    variable: str,
    values: Sequence[float],
    *,
    control: str | None = None,
    delta_T: float = 0.0,
    hold: str = "speed",
    progress: Callable[[int, int], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> SweepResult:
    """Run the analysis at each value of ``variable`` (``alpha``, ``speed``, ``altitude``, ``beta`` or ``deflection``).

    The convergence re-solve is skipped (it doubles the cost of every point); points that fail (supersonic speed,
    a solver that cannot converge past stall ...) are kept with an error message instead of aborting the sweep.
    ``hold`` is what an altitude sweep keeps fixed: ``"speed"`` (true airspeed) or ``"mach"``.
    """
    if variable not in VARIABLES:
        raise ValueError(f"unknown sweep variable {variable!r}; choose one of {VARIABLES}")
    if hold not in ("speed", "mach"):
        raise ValueError("hold must be 'speed' or 'mach'")
    if variable == DEFLECTION:
        if req.controls is None or not req.controls.surfaces:
            raise ControlError("define a control surface to sweep its deflection")
        names = [s.name for s in req.controls.surfaces]
        if control not in names:
            raise ControlError(f"choose which control surface to sweep (available: {', '.join(names)})")
    points: list[SweepPoint] = []
    solver_used, engine, notes = "", "", []
    n = len(values)
    for i, v in enumerate(values):
        if cancelled is not None and cancelled():
            notes.append(f"cancelled after {i} of {n} points")
            break
        try:
            r = run(replace(_modified_request(req, variable, float(v), control=control, delta_T=delta_T, hold=hold), convergence_check=False))
        except (ValueError, ControlError) as exc:
            points.append(SweepPoint(float(v), None, False, error=str(exc)))
        else:
            margin = r.load.cl_margin
            k = int(np.argmax(margin)) if margin.size else 0
            has_margin = bool(np.any(np.isfinite(r.load.cl_max)))
            converged = bool(getattr(r.solution, "converged", True))
            points.append(
                SweepPoint(
                    float(v), r.totals, converged,
                    max_margin=float(margin[k]) if has_margin else math.nan,
                    first_stall_y=abs(float(r.load.y[k])) if has_margin else math.nan,
                )
            )
            solver_used, engine = r.solver_used, str(r.info.get("section_engine", ""))
        if progress is not None:
            progress(i + 1, n)
    return SweepResult(variable, points, solver_used, engine, notes)


# ===================================================================================== stall
@dataclass
class StallInfo:
    """Where the wing first stalls and how much lift it can make (from an angle-of-attack sweep)."""

    available: bool  # the section data has a maximum lift coefficient
    reached: bool  # some station reaches cl_max within the swept range
    alpha_first_stall: float = math.nan  # deg, where the highest cl/cl_max crosses STALL_MARGIN
    y_first_stall: float = math.nan  # |y| of that station, m
    CL_first_stall: float = math.nan
    CL_max: float = math.nan  # highest CL of the sweep
    alpha_CL_max: float = math.nan
    CL_max_is_peak: bool = False  # CL falls after the maximum (otherwise the sweep ended before the peak)
    note: str = ""


def stall_info(sweep: SweepResult) -> StallInfo:
    """First-stall angle and maximum lift from an angle-of-attack sweep (ascending)."""
    if sweep.variable != ALPHA:
        raise ValueError("stall analysis needs an angle-of-attack sweep")
    a, cl, mg = sweep.values, sweep.column("CL"), sweep.margin
    ok = np.isfinite(cl)
    if not ok.any():
        return StallInfo(False, False, note="no valid points")
    if not np.any(np.isfinite(mg)):
        return StallInfo(False, False, note="the section data (linear section) has no maximum lift coefficient")
    a, cl, mg = a[ok], cl[ok], mg[ok]
    k = int(np.argmax(cl))
    info = StallInfo(True, False, CL_max=float(cl[k]), alpha_CL_max=float(a[k]), CL_max_is_peak=bool(k < len(cl) - 1 and cl[k + 1:].max() < cl[k] - 1e-6))
    over = np.where(mg >= STALL_MARGIN)[0]
    if over.size:
        j = int(over[0])
        if j > 0 and mg[j] != mg[j - 1]:  # interpolate the crossing of the stall threshold
            f = (STALL_MARGIN - mg[j - 1]) / (mg[j] - mg[j - 1])
            a0, cl0 = a[j - 1] + f * (a[j] - a[j - 1]), cl[j - 1] + f * (cl[j] - cl[j - 1])
        else:
            a0, cl0 = a[j], cl[j]
        pts = [p for p in sweep.points if p.ok and p.value == a[j]]
        info.reached = True
        info.alpha_first_stall, info.CL_first_stall = float(a0), float(cl0)
        info.y_first_stall = float(pts[0].first_stall_y) if pts else math.nan
    else:
        info.note = f"no station reached cl_max up to {a[-1]:.1f} deg (highest cl/cl_max = {np.nanmax(mg):.2f})"
    return info
