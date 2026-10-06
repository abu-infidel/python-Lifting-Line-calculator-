"""Comparison with real aircraft: published wing geometry and Pilot's Operating Handbook (POH) performance.

For each aircraft the wing is built from published data, then the program's results are set against the handbook:

* **Stall** (flaps up and flaps down): the handbook stall speed (calibrated airspeed at maximum weight, power off) gives
  the airplane's maximum lift coefficient ``CL_max = 2 W / (rho0 V_S^2 S)``; the program finds the wing's maximum lift
  from an angle-of-attack sweep at that speed and the stall speed it implies.
* **Level flight** (cruise and maximum speed): the angle of attack at which the wing carries the weight, measured from
  the fuselage reference line (the wing's root incidence included) - a real aircraft cruises nearly level.

What to expect: the program models a clean, smooth wing on its own, while the handbook values belong to the whole
airplane - its tail carries a download to trim (more lift needed from the wing), the fuselage interrupts the wing,
and production wings are rougher than wind-tunnel models. The wing's maximum lift therefore comes out a few per cent
*above* the airplane's, and more so with flaps (the handbook flap increments are for a fully deflected flap; Cessna
single-slotted flaps stop at 30 degrees).

Data sources: Cessna 172S and 152 Pilot's Operating Handbooks (weights, wing areas, spans, stall speeds in KCAS,
cruise speeds), Cessna wing data (NACA 2412 section, 64-inch chord constant to 100 in from the centreline tapering
to 44 in at 208 in, incidence +1 deg 30 min at the root and -1 deg 30 min at the tip, dihedral 1 deg 44 min).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

from .project import ProjectState

INCH = 0.0254
LBF = 4.4482216152605
KNOT = 1852.0 / 3600.0
FT = 0.3048
FT2 = FT * FT
RHO0 = 1.225


def cessna_wing(span_in: float, area_ft2: float, root_chord_in: float | None = None, break_frac: float = 0.462,
                taper: float = 0.6875, name: str = "Cessna wing", engine: str = "auto") -> ProjectState:
    """A Cessna 150/152/172-type wing as a section table: constant chord to ``break_frac`` of the semi-span, straight
    taper to the tip; NACA 2412; root incidence +1.5 deg and 3 deg washout on the tapered panel; dihedral 1 deg 44 min.

    With ``root_chord_in`` the tip chord is chosen so that the area is the published one (the rounded tip fairing is
    folded into it); without it the root chord follows from the area and ``taper``."""
    st = ProjectState()
    w = st.wing
    hs = 0.5 * span_in * INCH
    yb = break_frac * hs
    half_area = 0.5 * area_ft2 * FT2
    if root_chord_in is None:
        cr = half_area / (yb + (hs - yb) * (1.0 + taper) / 2.0)
        ct = taper * cr
    else:
        cr = root_chord_in * INCH
        ct = 2.0 * (half_area - cr * yb) / (hs - yb) - cr
    dih = math.radians(1.0 + 44.0 / 60.0)
    w.mode = "sections"
    w.sections = [
        dict(y=0.0, chord=cr, x_le=0.0, z=0.0, twist_deg=0.0),
        dict(y=yb, chord=cr, x_le=0.0, z=yb * math.tan(dih), twist_deg=0.0),
        dict(y=hs, chord=ct, x_le=0.25 * (cr - ct), z=hs * math.tan(dih), twist_deg=-3.0),
    ]
    w.incidence_deg = 1.5
    w.name = name
    st.name = name
    st.airfoils.keys = ["NACA 2412"] * 3
    st.airfoils.fit(3)
    st.solver.engine = engine
    st.high_lift.flap_type, st.high_lift.flap_y_in, st.high_lift.flap_y_out = "slotted", 0.0, break_frac
    return st


@dataclass(frozen=True)
class StallPoint:
    label: str
    weight_lb: float
    v_kcas: float
    flaps: bool


@dataclass(frozen=True)
class LevelPoint:
    label: str
    weight_lb: float
    v_ktas: float
    altitude_ft: float


@dataclass(frozen=True)
class ReferenceAircraft:
    name: str
    build: Callable[[str], ProjectState]  # engine -> project state of the wing
    stalls: tuple[StallPoint, ...]
    level: tuple[LevelPoint, ...] = ()
    notes: str = ""
    source: str = ""


AIRCRAFT: tuple[ReferenceAircraft, ...] = (
    ReferenceAircraft(
        "Cessna 172S Skyhawk",
        lambda engine: cessna_wing(433.0, 174.0, root_chord_in=64.0, name="Cessna 172S", engine=engine),
        (StallPoint("stall, flaps up", 2550, 53, False), StallPoint("stall, flaps 30 deg", 2550, 48, True)),
        (LevelPoint("cruise 75 % power", 2550, 124, 8500), LevelPoint("maximum speed", 2550, 126, 0)),
        "wing: published planform and incidence",
        "Cessna 172S POH (2550 lb; stall 53/48 KCAS; 124 KTAS at 8500 ft, 126 KTAS at sea level); span 36 ft 1 in, 174 ft2",
    ),
    ReferenceAircraft(
        "Cessna 152",
        lambda engine: cessna_wing(400.0, 160.0, name="Cessna 152", engine=engine),
        (StallPoint("stall, flaps up", 1670, 48, False), StallPoint("stall, flaps 30 deg", 1670, 43, True)),
        (),
        "wing: published span, area and section; planform and incidence assumed like the 172's",
        "Cessna 152 POH (1670 lb; stall 48/43 KCAS); span 33 ft 4 in, 160 ft2",
    ),
)


@dataclass
class ComparisonRow:
    aircraft: str
    point: str
    handbook: str
    program: str
    ratio: float = math.nan  # program / handbook (maximum lift rows)
    detail: str = ""


@dataclass
class Comparison:
    rows: list[ComparisonRow] = field(default_factory=list)
    engine: str = ""


def compare(backend, engine: str = "auto", aircraft: tuple[ReferenceAircraft, ...] = AIRCRAFT,
            progress: Callable[[int, int], None] | None = None,
            cancelled: Callable[[], bool] | None = None) -> Comparison:
    """Run every handbook point of every aircraft (``backend``: a :class:`~liftingline.io.backend.Backend`)."""
    from ..core.highlift import find_max_lift
    from ..core.trim import trim_alpha

    out = Comparison()
    total = sum(len(a.stalls) + len(a.level) for a in aircraft)
    done = 0
    for ac in aircraft:
        for sp in ac.stalls:
            st = ac.build(engine)
            st.flow.altitude_m, st.flow.V, st.flow.alpha_deg = 0.0, sp.v_kcas * KNOT, 5.0
            st.high_lift.flaps_on, st.high_lift.phase = sp.flaps, "landing"
            req = backend.build_request(st)
            out.engine = str(getattr(getattr(req.section, "base", req.section), "label", ""))
            cfg = find_max_lift(req, sp.label, cancelled=cancelled)
            W, S = sp.weight_lb * LBF, req.wing.area
            cl_poh = W / (0.5 * RHO0 * (sp.v_kcas * KNOT) ** 2 * S)
            vs = math.sqrt(2.0 * W / (RHO0 * S * cfg.CL_max)) / KNOT if cfg.CL_max > 0 else math.nan
            out.rows.append(ComparisonRow(
                ac.name, sp.label, f"CL_max {cl_poh:.3f} (V_S {sp.v_kcas:.0f} kt)",
                f"CL_max {cfg.CL_max:.3f} (V_S {vs:.1f} kt)", cfg.CL_max / cl_poh,
                f"{cfg.definition} at {cfg.alpha_CL_max:.1f} deg; first section stall at {cfg.alpha_first_stall:.1f} deg",
            ))
            done += 1
            if progress is not None:
                progress(done, total)
        for lp in ac.level:
            st = ac.build(engine)
            st.flow.altitude_m, st.flow.V = lp.altitude_ft * FT, lp.v_ktas * KNOT
            req = backend.build_request(st)
            W = lp.weight_lb * LBF
            res = trim_alpha(req, lift=W)
            q = req.flow.q
            out.rows.append(ComparisonRow(
                ac.name, lp.label, f"{lp.v_ktas:.0f} KTAS at {lp.altitude_ft:.0f} ft, {lp.weight_lb:.0f} lb",
                f"level at {res.value:.2f} deg (CL {res.totals.CL:.3f})", math.nan,
                f"angle of attack of the fuselage reference line (root incidence +1.5 deg); q = {q:.0f} Pa",
            ))
            done += 1
            if progress is not None:
                progress(done, total)
    return out


def format_comparison(c: Comparison) -> str:
    lines = [f"Comparison with real aircraft (section data: {c.engine})"]
    current = None
    for r in c.rows:
        if r.aircraft != current:
            current = r.aircraft
            ac = next((a for a in AIRCRAFT if a.name == current), None)
            lines += ["", current + (f"  ({ac.notes})" if ac and ac.notes else "")]
            if ac and ac.source:
                lines.append(f"  data: {ac.source}")
        ratio = f"  program/handbook {r.ratio:.3f}" if math.isfinite(r.ratio) else ""
        lines.append(f"  {r.point:20s} handbook {r.handbook:30s} program {r.program:30s}{ratio}")
        if r.detail:
            lines.append(f"  {'':20s} {r.detail}")
    lines += [
        "",
        "Reading it: the program's wing is clean, smooth and on its own; the handbook figures are for the whole airplane",
        "(tail download to trim, fuselage, production surface finish). Expect the wing's maximum lift a few per cent above",
        "the airplane's, more with flaps (the handbook flap increments assume a full landing deflection; these flaps stop",
        "at 30 deg). In level flight the wing should carry the weight at a small positive angle of attack.",
    ]
    return "\n".join(lines).strip("\n")


__all__ = ["AIRCRAFT", "ReferenceAircraft", "StallPoint", "LevelPoint", "Comparison", "ComparisonRow", "cessna_wing",
           "compare", "format_comparison"]
