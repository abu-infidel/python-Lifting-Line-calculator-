"""The single analysis pipeline: ``AnalysisRequest -> run() -> AnalysisResult``.

The GUI, sweeps, trim solver and validation suite all go through :func:`run`.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace

import numpy as np

from .corrections import PG_MACH_LIMIT, CompressibleSection
from .flow import FlowCondition
from .forces import LoadDistribution, Totals, compute_totals
from .section import SectionModel
from .solver_classical import ClassicalSolution, ControlEffects, build_load_classical, solve_classical
from .wing import Wing

SOLVER_AUTO = "auto"
SOLVER_CLASSICAL = "classical"
SOLVER_NUMERICAL = "numerical"


@dataclass
class MomentReference:
    """Where pitching/rolling/yawing moments are taken about.

    By default the point at ``x_frac_mac`` (25 %) of the mean aerodynamic chord, in the
    wing's plane of symmetry. ``x_abs`` (m aft of the root leading edge) overrides it.
    """

    x_frac_mac: float = 0.25
    x_abs: float | None = None
    z_abs: float = 0.0

    def resolve(self, wing: Wing) -> tuple[float, float]:
        x = self.x_abs if self.x_abs is not None else wing.moment_reference_x(self.x_frac_mac)
        return float(x), float(self.z_abs)


@dataclass
class AnalysisRequest:
    wing: Wing
    flow: FlowCondition
    section: SectionModel
    solver: str = SOLVER_AUTO
    n_terms: int = 80  # Fourier terms (classical solver)
    nonlinear: bool | None = None  # full section polars instead of (a0, alpha_L0); None = auto (on for real polars)
    compressibility: bool = True  # Prandtl-Glauert correction of section lift slope and moment
    ref: MomentReference = field(default_factory=MomentReference)
    extra_cd0: float = 0.0  # additional parasite drag coefficient (fuselage, tail, ...) on S
    controls: ControlEffects | None = None
    convergence_check: bool = True
    ground_height: float | None = None  # c/4 height above ground, m (numerical solver only)
    n_panels: int = 80  # horseshoe vortices per half-wing (numerical solver)


@dataclass
class AnalysisResult:
    request: AnalysisRequest
    solver_used: str
    totals: Totals
    load: LoadDistribution
    solution: object
    warnings: list[str]
    info: dict[str, object]


def needs_numerical_solver(req: AnalysisRequest) -> list[str]:
    """Reasons why the classical straight-lifting-line solver is not applicable."""
    reasons = []
    if not req.wing.is_planar_unswept():
        reasons.append("sweep or dihedral")
    if abs(req.flow.beta_deg) > 1e-9:
        reasons.append("sideslip")
    if req.ground_height is not None:
        reasons.append("ground effect")
    return reasons


def run(req: AnalysisRequest) -> AnalysisResult:
    t0 = time.perf_counter()
    wing, flow = req.wing, req.flow
    warnings: list[str] = []
    info: dict[str, object] = {}

    if flow.mach >= 1.0:
        raise ValueError(f"Mach {flow.mach:.2f}: lifting-line theory needs subsonic flow")
    if flow.mach > PG_MACH_LIMIT:
        warnings.append(
            f"Mach {flow.mach:.2f} exceeds {PG_MACH_LIMIT}: Prandtl-Glauert compressibility is unreliable and "
            "wave drag is not modelled."
        )
    section = CompressibleSection(req.section, flow.mach) if req.compressibility else req.section
    nonlinear = bool(req.nonlinear) if req.nonlinear is not None else bool(getattr(req.section, "has_polars", False))
    info["nonlinear"] = nonlinear
    for note in getattr(req.section, "notes", ()):
        warnings.append(note)

    reasons = needs_numerical_solver(req)
    solver = req.solver
    if solver == SOLVER_AUTO:
        solver = SOLVER_NUMERICAL if reasons else SOLVER_CLASSICAL
        info["solver_reason"] = ("numerical solver: " + ", ".join(reasons)) if reasons else "straight planar wing"
    elif solver == SOLVER_CLASSICAL and reasons:
        warnings.append(
            "Classical lifting-line theory assumes an unswept, planar wing in symmetric flow; "
            f"ignoring: {', '.join(reasons)}. Use the numerical solver for these effects."
        )

    x_ref, z_ref = req.ref.resolve(wing)

    if solver == SOLVER_NUMERICAL:
        from .solver_numerical import run_numerical  # imported lazily (milestone M4)

        solution, load = run_numerical(req, section, warnings, info)
    else:
        solution = solve_classical(
            wing, flow, section, n_terms=req.n_terms, nonlinear=nonlinear, controls=req.controls
        )
        if not solution.converged:
            warnings.append(
                f"Nonlinear solve did not fully converge ({solution.iterations} iterations, residual "
                f"{solution.residual:.2e}); the wing is probably at or beyond stall."
            )
        load = build_load_classical(solution, wing, flow, section, controls=req.controls)
        info.update(n_terms=solution.n_terms, iterations=solution.iterations, residual=solution.residual)

    totals = compute_totals(load, wing, flow, x_ref=x_ref, z_ref=z_ref, extra_cd0=req.extra_cd0)

    if solver == SOLVER_CLASSICAL and req.convergence_check:
        _check_term_convergence(req, section, nonlinear, totals, warnings, info)
    _validity_warnings(req, load, totals, warnings)
    _confidence_warning(req, load, warnings, info)

    info["solve_time_s"] = time.perf_counter() - t0
    info["mach"] = flow.mach
    info["section_engine"] = getattr(req.section, "label", type(req.section).__name__)
    return AnalysisResult(
        request=req, solver_used=solver, totals=totals, load=load, solution=solution, warnings=warnings, info=info
    )


# ---------------------------------------------------------------------------- checks
def _check_term_convergence(req, section, nonlinear: bool, totals: Totals, warnings: list[str], info: dict) -> None:
    """Re-solve with twice the Fourier terms and report the change in CL and CDi."""
    sol2 = solve_classical(
        req.wing, req.flow, section, n_terms=2 * req.n_terms, nonlinear=nonlinear, controls=req.controls
    )
    load2 = build_load_classical(sol2, req.wing, req.flow, section, controls=req.controls)
    t2 = compute_totals(load2, req.wing, req.flow, x_ref=0.0, extra_cd0=req.extra_cd0)
    dCL = abs(t2.CL - totals.CL) / max(abs(t2.CL), 1e-12)
    dCDi = abs(t2.CD_induced - totals.CD_induced) / max(abs(t2.CD_induced), 1e-12)
    info["convergence_dCL"] = dCL
    info["convergence_dCDi"] = dCDi
    if dCL > 5e-3 or dCDi > 2e-2:
        warnings.append(
            f"Result not converged in the number of Fourier terms (CL changes {dCL:.2%}, CDi {dCDi:.2%} "
            f"when doubling {req.n_terms} terms); increase the number of terms."
        )


def _validity_warnings(req: AnalysisRequest, load: LoadDistribution, totals: Totals, warnings: list[str]) -> None:
    wing = req.wing
    if wing.aspect_ratio < 4.0:
        warnings.append(
            f"Low aspect ratio (AR = {wing.aspect_ratio:.2f}): lifting-line theory over-predicts lift and loses "
            "accuracy below AR of about 4."
        )
    if wing.max_abs_sweep_c4_deg > 30.0:
        warnings.append(
            f"Quarter-chord sweep of {wing.max_abs_sweep_c4_deg:.0f} deg is beyond the range where lifting-line "
            "theory is dependable (about 30 deg)."
        )
    margin = load.cl_margin
    finite = np.isfinite(load.cl_max)
    if np.any(finite):
        worst = float(np.max(margin[finite]))
        if worst > 1.0:
            where = float(np.abs(load.y[finite][np.argmax(margin[finite])]))
            warnings.append(
                f"Sections reach/exceed cl_max (worst {worst:.2f} x cl_max near |y| = {where:.2f} m): results "
                "beyond stall are indicative only."
            )
        elif worst > 0.9:
            warnings.append(f"Close to stall: highest local cl is {worst:.0%} of cl_max.")


def _confidence_warning(req: AnalysisRequest, load: LoadDistribution, warnings: list[str], info: dict) -> None:
    """Surface the section engine's own confidence (NeuralFoil) where it is low."""
    conf_fn = getattr(req.section, "confidence", None)
    if conf_fn is None:
        return
    inner = np.abs(load.y) < 0.95 * req.wing.half_span  # the last few percent of span carry almost no load
    conf = conf_fn(np.abs(load.y[inner]), load.alpha_eff[inner], load.re[inner])
    if conf is None:
        return
    worst = float(np.min(conf))
    info["section_confidence_min"] = worst
    if worst < 0.5:
        where = float(np.abs(load.y[inner][int(np.argmin(conf))]))
        warnings.append(
            f"Low section-data confidence ({worst:.2f}) near |y| = {where:.2f} m: the airfoil is at an extreme "
            "angle of attack or Reynolds number for the section engine; treat drag and stall results with care."
        )
