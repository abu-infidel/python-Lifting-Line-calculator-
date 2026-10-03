"""Ready-made example wings (all use analytic NACA 4-digit sections, so they work without an airfoil library)."""

from __future__ import annotations

import math
from typing import Callable

from .project import (
    DEFINE_AREA_AR_TAPER,
    DEFINE_SPAN_CHORDS,
    MODE_ELLIPTIC,
    AirfoilSpec,
    FlowSpec,
    ProjectState,
    SolverSpec,
    WingSpec,
)


def _state(name: str, wing: WingSpec, keys: list[str], flow: FlowSpec, notes: str = "") -> ProjectState:
    wing.name = name
    st = ProjectState(wing=wing, airfoils=AirfoilSpec(keys=list(keys)), flow=flow, solver=SolverSpec(), name=name, notes=notes)
    st.wing.sync()
    st.airfoils.fit(st.wing.n_sections())
    return st


def light_aircraft() -> ProjectState:
    return _state(
        "Light aircraft (Cessna 172-like)",
        WingSpec(span=11.0, root_chord=1.63, tip_chord=1.13, dihedral_deg=1.7, twist_tip_deg=-3.0),
        ["NACA 2412", "NACA 2412"],
        FlowSpec(V=55.0, alpha_deg=3.0),
        "Straight tapered wing with a little dihedral and washout; cruise-like speed at sea level.",
    )


def rectangular_wing() -> ProjectState:
    return _state(
        "Rectangular wing, AR 8",
        WingSpec(span=8.0, root_chord=1.0, tip_chord=1.0),
        ["NACA 0012", "NACA 0012"],
        FlowSpec(V=40.0, alpha_deg=5.0),
        "The textbook rectangular wing. Compare the span efficiency with 1/(1+delta).",
    )


def tapered_wing() -> ProjectState:
    return _state(
        "Tapered wing, AR 10",
        WingSpec(define_by=DEFINE_AREA_AR_TAPER, area=12.0, aspect_ratio=10.0, taper_ratio=0.4, twist_tip_deg=-2.0),
        ["NACA 4415", "NACA 4412"],
        FlowSpec(V=45.0, alpha_deg=4.0),
        "Taper 0.4 is close to the span-efficiency optimum for a straight wing.",
    )


def elliptic_wing() -> ProjectState:
    return _state(
        "Elliptic wing, AR 8",
        WingSpec(mode=MODE_ELLIPTIC, define_by=DEFINE_SPAN_CHORDS, span=10.0, root_chord=4.0 * (10.0**2 / 8.0) / (math.pi * 10.0)),
        ["NACA 0009", "NACA 0009"],
        FlowSpec(V=40.0, alpha_deg=5.0),
        "Prandtl's optimum: the span efficiency comes out as exactly 1 with a constant-section wing.",
    )


def swept_wing() -> ProjectState:
    return _state(
        "Swept wing, 30 degrees",
        WingSpec(define_by=DEFINE_AREA_AR_TAPER, area=16.0, aspect_ratio=8.0, taper_ratio=0.4, sweep_deg=30.0, twist_tip_deg=-3.0),
        ["NACA 0012", "NACA 0012"],
        FlowSpec(V=70.0, alpha_deg=6.0),
        "Needs the numerical solver (selected automatically). Lifting-line accuracy drops with sweep.",
    )


def sailplane_wing() -> ProjectState:
    return _state(
        "Sailplane wing, AR 22",
        WingSpec(span=18.0, root_chord=1.1, tip_chord=0.5, dihedral_deg=3.0, twist_tip_deg=-1.5),
        ["NACA 4415", "NACA 4412"],
        FlowSpec(V=28.0, alpha_deg=3.0),
        "High aspect ratio: induced drag matters less, profile drag and Reynolds number matter more.",
    )


def ground_effect_wing() -> ProjectState:
    return _state(
        "Wing in ground effect",
        WingSpec(span=8.0, root_chord=1.3, tip_chord=1.3),
        ["NACA 2412", "NACA 2412"],
        FlowSpec(V=30.0, alpha_deg=4.0, ground_effect=True, ground_height_m=2.0),
        "Quarter-chord line 2 m above the ground. Image vortices reduce the induced drag.",
    )


PRESETS: dict[str, Callable[[], ProjectState]] = {
    "Light aircraft (Cessna 172-like)": light_aircraft,
    "Rectangular wing, AR 8": rectangular_wing,
    "Tapered wing, AR 10": tapered_wing,
    "Elliptic wing, AR 8": elliptic_wing,
    "Swept wing, 30 degrees": swept_wing,
    "Sailplane wing, AR 22": sailplane_wing,
    "Wing in ground effect": ground_effect_wing,
}

__all__ = ["PRESETS", "light_aircraft"]
