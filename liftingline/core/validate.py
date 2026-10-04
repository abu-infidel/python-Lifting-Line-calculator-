"""Built-in validation suite: analytic benchmarks and invariants that the program must reproduce.

``python -m liftingline --validate`` (or Help > Run self-test in the window) runs it in a few seconds, offline.
Every check compares a computed number with a known exact value, a published reference or an independent calculation,
and states the tolerance. A failing check means something is wrong with the installation or the code, not with a wing.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from . import atmosphere, units
from .analysis import AnalysisRequest, run
from .controls import ControlSet, ControlSurface, flap_effectiveness
from .flow import FlowCondition
from .section import LinearSection
from .wing import Wing

DEG = math.pi / 180.0
TWO_PI = 2.0 * math.pi

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"


@dataclass
class Check:
    group: str
    name: str
    status: str  # PASS / FAIL / SKIP
    expected: str = ""
    got: str = ""
    tolerance: str = ""
    note: str = ""

    @property
    def ok(self) -> bool:
        return self.status != FAIL


class _Collector:
    def __init__(self) -> None:
        self.checks: list[Check] = []
        self.group = ""

    def close(self, name: str, expected: float, got: float, *, rel: float | None = None, abs_tol: float | None = None, note: str = "") -> None:
        """Pass when ``got`` is within ``rel`` (relative) or ``abs_tol`` (absolute) of ``expected``."""
        err = abs(got - expected)
        if rel is not None:
            ok = err <= rel * max(abs(expected), 1e-300)
            tol = f"{rel:.1e} rel"
        else:
            ok = err <= (abs_tol or 0.0)
            tol = f"{abs_tol:.1e} abs"
        self.checks.append(Check(self.group, name, PASS if ok else FAIL, f"{expected:.10g}", f"{got:.10g}", tol, note))

    def true(self, name: str, condition: bool, note: str = "") -> None:
        self.checks.append(Check(self.group, name, PASS if condition else FAIL, note=note))

    def skip(self, name: str, why: str) -> None:
        self.checks.append(Check(self.group, name, SKIP, note=why))

    def error(self, name: str, exc: Exception) -> None:
        self.checks.append(Check(self.group, name, FAIL, note=f"{type(exc).__name__}: {exc}"))


def _totals(wing: Wing, alpha: float, section=None, V: float = 40.0, solver: str = "classical", **kw):
    flow = FlowCondition.from_isa(0.0, V=V, alpha_deg=alpha, beta_deg=kw.pop("beta", 0.0))
    kw.setdefault("compressibility", False)
    kw.setdefault("convergence_check", False)
    return run(AnalysisRequest(wing, flow, section or LinearSection(), solver=solver, **kw)).totals


# ======================================================================================== the groups
def _classical(c: _Collector) -> None:
    c.group = "Classical lifting line (exact and published results)"
    AR, a0, alpha = 8.0, TWO_PI, 5.0
    ell = Wing.elliptic(10.0, 4.0 * 10.0 / (math.pi * AR))
    t = _totals(ell, alpha, n_terms=100)
    cl_exact = a0 * math.radians(alpha) / (1.0 + a0 / (math.pi * AR))
    c.close("elliptic wing, AR 8: CL = a0 alpha / (1 + a0/(pi AR))", cl_exact, t.CL, rel=1e-9)
    c.close("elliptic wing: CDi = CL^2 / (pi AR)", cl_exact**2 / (math.pi * AR), t.CD_induced, rel=1e-9)
    c.close("elliptic wing: span efficiency e = 1", 1.0, t.e, abs_tol=1e-9)

    anchors = (
        ("rectangular AR 6", Wing.from_area_ar_taper(1.0, 6.0, 1.0), 0.39535, 0.95393),
        ("taper 0.4, AR 8", Wing.from_area_ar_taper(1.0, 8.0, 0.4), 0.43451, 0.98720),
        ("taper 0.25, AR 10", Wing.from_area_ar_taper(1.0, 10.0, 0.25), 0.45296, 0.97800),
    )
    for name, wing, cl_ref, e_ref in anchors:
        t = _totals(wing, 5.0, n_terms=100)
        c.close(f"{name}: CL (reference solution)", cl_ref, t.CL, abs_tol=2e-5)
        c.close(f"{name}: span efficiency", e_ref, t.e, abs_tol=2e-5)

    wing = Wing.from_area_ar_taper(1.0, 8.0, 0.4)
    c.close("convergence: 40 terms against 100", _totals(wing, 5.0, n_terms=100).CL, _totals(wing, 5.0, n_terms=40).CL, rel=3e-4)
    sec = LinearSection(alpha_L0=-2.0 * DEG)
    c.close("linearity: CL(2 alpha - alpha_L0) = 2 CL(alpha - alpha_L0)", 2.0 * _totals(wing, 2.0, sec).CL, _totals(wing, 6.0, sec).CL, rel=1e-12)
    flipped = Wing(sections=tuple(wing.sections), name="same")
    c.close("mirror symmetry: roll and yaw vanish", 0.0, abs(_totals(flipped, 4.0).C_roll) + abs(_totals(flipped, 4.0).C_yaw), abs_tol=1e-12)

    from .optimal import optimal_twist
    from .wing import Section

    base = Wing.from_area_ar_taper(12.0, 9.0, 0.45)
    ot = optimal_twist(AnalysisRequest(base, FlowCondition.from_isa(0.0, V=40.0), LinearSection(a0=5.9, alpha_L0=-1.8 * DEG), compressibility=False), 0.55, 25)
    twisted = Wing(sections=tuple(Section(r["y"], r["chord"], r["x_le"], r["z"], r["twist_deg"]) for r in ot.rows))
    t = _totals(twisted, ot.alpha_root_deg, LinearSection(a0=5.9, alpha_L0=-1.8 * DEG))
    c.close("optimal twist gives an elliptic load (e = 1)", 1.0, t.e, abs_tol=2e-3)
    c.close("optimal twist: CL equals the design value", 0.55, t.CL, rel=1e-6)

    flap = ControlSurface("f", "flap", 0.0, 1.0, 0.3, 8.0, 0.9)
    rect = Wing.rectangular(10.0, 1.2)
    t_flap = _totals(rect, 2.0, controls=ControlSet((flap,), rect.half_span))
    c.close("full-span flap equals an angle-of-attack change", _totals(rect, 2.0 + math.degrees(flap.tau * 8.0 * DEG)).CL, t_flap.CL, rel=1e-12)
    c.close("thin-airfoil flap effectiveness, 25 % chord", 0.6090, flap_effectiveness(0.25), abs_tol=1e-3)


def _numerical(c: _Collector) -> None:
    c.group = "Numerical lifting line (sweep, dihedral, sideslip)"
    for name, wing in (("rectangular AR 6", Wing.from_area_ar_taper(1.0, 6.0, 1.0)), ("taper 0.4, AR 10", Wing.from_area_ar_taper(1.0, 10.0, 0.4))):
        a, b = _totals(wing, 5.0), _totals(wing, 5.0, solver="numerical", n_panels=80)
        c.close(f"{name}: CL against the classical solver", a.CL, b.CL, rel=5e-3)
        c.close(f"{name}: induced drag against the classical solver", a.CD_induced, b.CD_induced, rel=1e-2)
    swept = Wing.from_area_ar_taper(1.0, 8.0, 0.5, sweep_deg=30.0)
    c.close("swept wing converges with the element count (80 vs 160)", _totals(swept, 4.0, solver="numerical", n_panels=160).CL, _totals(swept, 4.0, solver="numerical", n_panels=80).CL, rel=3e-3)

    def slope(w):
        return (_totals(w, 4.0, solver="numerical", n_panels=120).CL - _totals(w, 2.0, solver="numerical", n_panels=120).CL) / math.radians(2.0)

    AR = 60.0
    base = slope(Wing.from_area_ar_taper(1.0, AR, 1.0))
    sw = slope(Wing.from_area_ar_taper(1.0, AR, 1.0, sweep_deg=30.0))
    k = math.cos(math.radians(30.0))
    c.close("infinite-wing limit: lift slope falls with cos(sweep) (AR 60, 30 deg)", (TWO_PI * k / (1 + TWO_PI * k / (math.pi * AR))) / (TWO_PI / (1 + TWO_PI / (math.pi * AR))), sw / base, rel=5e-3)
    t = _totals(Wing.from_area_ar_taper(1.0, 8.0, 0.5, sweep_deg=25.0, dihedral_deg=4.0), 4.0, solver="numerical")
    c.close("symmetric flow: no roll, yaw or side force", 0.0, abs(t.C_roll) + abs(t.C_yaw) + abs(t.CY), abs_tol=1e-12)
    wing = Wing.from_area_ar_taper(1.0, 8.0, 0.5, sweep_deg=20.0, dihedral_deg=4.0)
    p, m = _totals(wing, 4.0, solver="numerical", beta=6.0), _totals(wing, 4.0, solver="numerical", beta=-6.0)
    c.close("sideslip: roll moment is odd in beta", -p.C_roll, m.C_roll, rel=1e-9)
    c.true("dihedral and sweepback are stabilising (roll opposes the sideslip)", p.C_roll < 0.0)
    rect = Wing.from_area_ar_taper(1.0, 8.0, 1.0)
    free = _totals(rect, 4.0, solver="numerical")
    far = _totals(rect, 4.0, solver="numerical", ground_height=30.0 * rect.span)
    c.close("ground effect vanishes far from the ground", free.CL, far.CL, rel=1e-3)
    near = _totals(rect, 4.0, solver="numerical", ground_height=0.25 * rect.span)
    c.true("ground effect raises lift and lowers induced drag", near.CL > free.CL and near.CD_induced < free.CD_induced)


def _atmosphere_and_units(c: _Collector) -> None:
    c.group = "Atmosphere and units"
    s = atmosphere.isa(0.0)
    c.close("ISA sea level: density", 1.2250, s.rho, rel=1e-4)
    c.close("ISA sea level: viscosity (Sutherland)", 1.7894e-5, s.mu, rel=2e-4)
    c.close("ISA sea level: speed of sound", 340.29, s.a, rel=1e-4)
    t = atmosphere.isa(11000.0)
    c.close("ISA 11 km: temperature", 216.65, t.T, abs_tol=1e-6)
    c.close("ISA 11 km: pressure", 22632.0, t.p, rel=2e-4)
    c.close("ISA 11 km: density", 0.3639, t.rho, rel=5e-4)
    c.close("ISA + 15 K at sea level: density", 1.2250 * 288.15 / 303.15, atmosphere.isa(0.0, 15.0).rho, rel=1e-4)
    c.close("unit round trip m -> ft -> m", 3.1415926535, units.convert("length", units.convert("length", 3.1415926535, "m", "ft"), "ft", "m"), rel=1e-14)
    c.close("1 kt in m/s", 0.514444, units.to_si("speed", 1.0, "kt"), abs_tol=1e-6)
    c.close("1 lbf in N", 4.448222, units.to_si("force", 1.0, "lbf"), abs_tol=1e-6)


def _airfoils(c: _Collector) -> None:
    c.group = "Airfoil geometry, panel method and parsers"
    from ..airfoil import naca4
    from ..airfoil.dat_io import airfoil_from_text, to_dat_text
    from ..airfoil.naca6 import mean_line_slope
    from ..airfoil.panel2d import PanelSolver
    from ..airfoil.thin_airfoil import thin_airfoil

    af = naca4.naca4_airfoil("2412")
    c.close("NACA 2412: maximum thickness", 0.12, af.t_over_c, abs_tol=1e-3)
    c.close("NACA 2412: maximum camber", 0.02, af.max_camber, abs_tol=1e-4)
    ps = PanelSolver(af, 160)
    c.close("NACA 2412 panel method: zero-lift angle [deg]", -2.08, math.degrees(ps.alpha_L0), abs_tol=0.15)
    c.close("NACA 2412 panel method: c/4 moment at zero lift", -0.05, ps.cm_c4_zero_lift, abs_tol=0.01)
    plate = naca4.naca4_airfoil("0003")
    c.close("thin symmetric airfoil: lift slope close to 2 pi (per radian)", TWO_PI, PanelSolver(plate, 160).lift_slope, rel=0.03)
    ta = thin_airfoil(lambda x: mean_line_slope(x, 1.0, 1.0))
    c.close("6-series mean line a = 1, cl = 1: zero-lift angle [deg] (thin-airfoil theory)", -9.11, ta.alpha_L0_deg, abs_tol=0.05)
    c.close("6-series mean line a = 1, cl = 1: c/4 moment", -0.25, ta.cm_c4, abs_tol=2e-3)
    for fmt_name, text in (("Selig", to_dat_text(af, 8)),):
        back = airfoil_from_text(text, "x")
        c.close(f".dat {fmt_name} round trip: maximum thickness", af.t_over_c, back.t_over_c, abs_tol=1e-5)
    lines = to_dat_text(af, 8).splitlines()
    name, pts = lines[0], [tuple(map(float, ln.split())) for ln in lines[1:] if ln.strip()]
    i_le = int(np.argmin([p[0] for p in pts]))
    upper, lower = pts[: i_le + 1][::-1], pts[i_le:]
    lednicer = [name, f"{len(upper):.0f}. {len(lower):.0f}.", ""] + [f"{x:.8f} {y:.8f}" for x, y in upper] + [""] + [f"{x:.8f} {y:.8f}" for x, y in lower]
    back = airfoil_from_text("\n".join(lednicer), "x")
    c.close(".dat Lednicer parse: maximum camber", af.max_camber, back.max_camber, abs_tol=1e-4)


def _engines(c: _Collector) -> None:
    c.group = "Section data engines"
    from ..airfoil import naca4
    from ..airfoil.engines import available_engines, make_engine

    re = np.array([3e6])
    for kind, label in (("builtin", "built-in"), ("neuralfoil", "NeuralFoil")):
        if not available_engines()[kind][0]:
            c.skip(f"{label} engine", "not installed")
            continue
        eng = make_engine(kind)
        sym = eng.aero(naca4.naca4_airfoil("0012"))
        cam = eng.aero(naca4.naca4_airfoil("2412"))
        lin = cam.linear(re)
        cl_p = sym.coefficients(np.radians([5.0]), re)[0][0]
        cl_m = sym.coefficients(np.radians([-5.0]), re)[0][0]
        c.close(f"{label}: symmetric airfoil lift is odd in alpha", -cl_m, cl_p, abs_tol=2e-3)
        c.close(f"{label}: NACA 2412 zero-lift angle [deg]", -2.1, math.degrees(float(lin.alpha_L0[0])), abs_tol=0.35)
        c.true(f"{label}: NACA 2412 lift slope within 5.4 .. 6.9 per radian", 5.4 < float(lin.a0[0]) < 6.9, f"a0 = {float(lin.a0[0]):.3f}")
        c.true(f"{label}: NACA 2412 maximum lift coefficient within 1.2 .. 2.0", 1.2 < float(lin.cl_max[0]) < 2.0, f"cl_max = {float(lin.cl_max[0]):.3f}")
        cd = cam.coefficients(np.radians([2.0]), re)[1][0]
        c.true(f"{label}: NACA 2412 drag coefficient plausible (0.004 .. 0.012)", 0.004 < cd < 0.012, f"cd = {cd:.5f}")


def _pipeline(c: _Collector) -> None:
    c.group = "Whole pipeline"
    from ..io.backend import Backend
    from ..io.presets import PRESETS

    be = Backend()
    st = PRESETS["Elliptic wing, AR 8"]()
    st.solver.engine = "builtin"
    t = run(be.build_request(st)).totals
    c.close("example 'elliptic wing': span efficiency", 1.0, t.e, abs_tol=0.02)
    st = PRESETS["Rectangular wing, AR 8"]()
    st.solver.engine = "builtin"
    res = run(be.build_request(st))
    t = res.totals
    c.close("force balance: drag = induced + profile + extra", t.D, t.D_induced + t.D_profile + t.D_extra, rel=1e-12)
    c.close("lift = CL q S", t.L, t.CL * t.q * t.S, rel=1e-12)
    sec = res.load
    c.close("lift is the integral of the spanwise load", t.L, float(np.sum((sec.force_kj + sec.force_profile)[:, 2] * math.cos(math.radians(t.alpha_deg)))) - float(np.sum((sec.force_kj + sec.force_profile)[:, 0] * math.sin(math.radians(t.alpha_deg)))), rel=1e-9)


GROUPS: tuple[tuple[str, Callable[[_Collector], None]], ...] = (
    ("classical", _classical),
    ("numerical", _numerical),
    ("atmosphere", _atmosphere_and_units),
    ("airfoils", _airfoils),
    ("engines", _engines),
    ("pipeline", _pipeline),
)


def run_validation(progress: Callable[[int, int], None] | None = None) -> list[Check]:
    """Run every group; a crash inside a group becomes a FAIL entry instead of aborting the suite."""
    c = _Collector()
    for i, (name, fn) in enumerate(GROUPS):
        try:
            fn(c)
        except Exception as exc:  # report, keep going
            c.error(f"group '{name}' could not run", exc)
        if progress is not None:
            progress(i + 1, len(GROUPS))
    return c.checks


def format_report(checks: list[Check], elapsed: float | None = None) -> str:
    lines: list[str] = []
    group = None
    for ch in checks:
        if ch.group != group:
            group = ch.group
            lines += ["", group, "-" * len(group)]
        detail = f"expected {ch.expected}, got {ch.got} (tolerance {ch.tolerance})" if ch.expected else ch.note
        lines.append(f"  [{ch.status}] {ch.name}" + (f"\n         {detail}" if detail and ch.status != PASS else ""))
    n_fail = sum(1 for c in checks if c.status == FAIL)
    n_skip = sum(1 for c in checks if c.status == SKIP)
    summary = f"{len(checks) - n_fail - n_skip} passed, {n_fail} failed, {n_skip} skipped"
    if elapsed is not None:
        summary += f" in {elapsed:.1f} s"
    lines += ["", ("ALL CHECKS PASSED - " if n_fail == 0 else "SOME CHECKS FAILED - ") + summary]
    return "\n".join(lines).lstrip("\n")


def main() -> int:
    t0 = time.perf_counter()
    print("Lifting-line wing calculator: self-test\n")
    checks = run_validation()
    print(format_report(checks, time.perf_counter() - t0))
    return 0 if all(ch.ok for ch in checks) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
