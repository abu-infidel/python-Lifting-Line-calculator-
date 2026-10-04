"""PDF report: inputs, results, validity notes and the plots, on a light theme suitable for printing."""

from __future__ import annotations

import datetime as _dt
import textwrap
from typing import Sequence

import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from .. import __version__
from ..core import units
from ..core.analysis import AnalysisResult
from ..core.sweeps import StallInfo, SweepResult
from ..io.backend import SectionInfo
from ..io.project import ProjectState
from ..io.report import totals_rows
from .plots import plot_airfoils, plot_planform, plot_spanwise, plot_sweep
from .theme import LIGHT

PAGE = (8.27, 11.69)  # A4 portrait, inches
MONO = {"family": "monospace"}


def _page(title: str, constrained: bool = False) -> Figure:
    fig = Figure(figsize=PAGE, dpi=100, facecolor="white", layout="constrained" if constrained else None)
    fig.text(0.07, 0.965, title, fontsize=15, fontweight="bold", va="top", color=LIGHT.fg)
    fig.text(0.93, 0.965, f"Lifting-Line Wing Calculator {__version__}", fontsize=7, ha="right", va="top", color=LIGHT.muted)
    return fig


def _block(fig: Figure, y: float, heading: str, lines: Sequence[str], size: float = 7.6, step: float = 0.0143) -> float:
    """Draw a heading and monospaced lines starting at figure height ``y``; returns the next free height."""
    fig.text(0.07, y, heading, fontsize=10, fontweight="bold", va="top", color=LIGHT.accent)
    y -= 0.024
    for ln in lines:
        fig.text(0.07, y, ln, fontsize=size, va="top", color=LIGHT.fg, **MONO)
        y -= step
    return y - 0.012


def _wrap(text: str, width: int = 104, indent: str = "  ") -> list[str]:
    return textwrap.wrap(text, width=width, initial_indent=indent, subsequent_indent=indent + "  ") or [indent]


def _summary_page(state: ProjectState, result: AnalysisResult, infos: Sequence[SectionInfo], system: str) -> Figure:
    fig = _page(state.name or "Wing analysis")
    req = result.request
    wing, flow = req.wing, req.flow
    uL, uA, uP = (units.display_unit(system, k) for k in ("length", "area", "pressure"))
    uV = "m/s" if system == "SI" else "kt"
    L = lambda v: units.from_si("length", v, uL)  # noqa: E731
    sweep, dih = wing.sweep_c4_deg_by_segment(), wing.dihedral_deg_by_segment()
    meta = f"{_dt.datetime.now():%Y-%m-%d %H:%M}   solver: {result.solver_used}   section data: {result.info.get('section_engine', '')}"
    fig.text(0.07, 0.94, meta, fontsize=8, va="top", color=LIGHT.muted)
    y = 0.915
    wing_lines = [
        f"Area S {units.from_si('area', wing.area, uA):.4g} {uA}   span b {L(wing.span):.4g} {uL}   aspect ratio {wing.aspect_ratio:.3g}   taper {wing.taper_ratio:.3g}",
        f"MAC {L(wing.mac):.4g} {uL}   root chord {L(wing.root_chord):.4g} {uL}   tip chord {L(wing.tip_chord):.4g} {uL}",
        f"quarter-chord sweep {sweep[0]:.2f} deg ({'varies' if abs(sweep[-1] - sweep[0]) > 0.05 else 'constant'})   dihedral {dih[0]:.2f} deg   root incidence {wing.incidence_deg:.2f} deg",
        "",
        f"{'y [' + uL + ']':>9} {'chord [' + uL + ']':>11} {'twist [deg]':>12}  airfoil / section data",
    ]
    for s, info in zip(wing.sections, infos):
        src = info.data_label if info.data_label else ""
        wing_lines.append(f"{L(s.y):9.4g} {L(s.chord):11.4g} {s.twist_deg:12.2f}  {info.airfoil_name}  ({info.source}; {src})")
    y = _block(fig, y, "Wing", wing_lines)
    flow_lines = [
        f"altitude {flow.altitude_m if flow.altitude_m is not None else float('nan'):.0f} m   density {flow.rho:.4g} kg/m3   temperature {flow.T - 273.15:.1f} C   viscosity {flow.mu:.4g} Pa s",
        f"airspeed {units.from_si('speed', flow.V, uV):.4g} {uV}   Mach {flow.mach:.3f}   dynamic pressure {units.from_si('pressure', flow.q, uP):.5g} {uP}",
        f"angle of attack {flow.alpha_deg:.3f} deg   sideslip {flow.beta_deg:.3f} deg   Reynolds number (MAC) {float(flow.reynolds(wing.mac)):.3g}",
    ]
    if req.ground_height is not None:
        flow_lines.append(f"ground effect: quarter-chord line {L(req.ground_height):.4g} {uL} above the ground")
    if req.controls is not None and req.controls.surfaces:
        for s in req.controls.surfaces:
            flow_lines.append(f"control: {s.name} ({s.kind}), {s.y_in * 100:.0f}-{s.y_out * 100:.0f} % semi-span, {s.chord_fraction * 100:.0f} % chord, {s.deflection_deg:+.2f} deg")
    y = _block(fig, y, "Flow", flow_lines)
    rows = totals_rows(result, system)
    res_lines = [f"{'quantity':<26}{'value':>18}   coefficient"]
    for r in rows:
        value = f"{r.value} {r.unit}".strip()
        coeff = f"{r.coeff_name} = {r.coeff}" if r.coeff_name and r.coeff else ""
        res_lines.append(f"{r.label:<26}{value:>18}   {coeff}")
    y = _block(fig, y, "Results", res_lines)
    notes: list[str] = []
    for w in result.warnings:
        notes += _wrap("- " + w, 108, "")
    y = _block(fig, y, "Validity notes", notes or ["  none: this case is inside the range of the theory"], size=7.2)
    solver = ["Classical Prandtl lifting line (Glauert Fourier solution)" if result.solver_used == "classical" else "Numerical lifting line (horseshoe vortices)"]
    solver.append(f"{result.info.get('n_terms', result.info.get('n_elements', '?'))} {'terms' if result.solver_used == 'classical' else 'elements'}, "
                  f"residual {float(result.info.get('residual', 0.0)):.1e}, {1000 * float(result.info.get('solve_time_s', 0.0)):.0f} ms")
    _block(fig, y, "Method", [f"  {s}" for s in solver])
    return fig


def _plot_page(title: str, draw) -> Figure:
    """A page with ``draw(subfigure)`` below the title; the plots stay vector graphics."""
    fig = _page(title, constrained=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[0.07, 0.93])
    draw(fig.add_subfigure(gs[1]))
    return fig


def export_pdf(
    path: str,
    state: ProjectState,
    result: AnalysisResult,
    infos: Sequence[SectionInfo],
    system: str = "SI",
    sweep: SweepResult | None = None,
    stall: StallInfo | None = None,
) -> int:
    """Write the report; returns the number of pages."""
    matplotlib.rcParams["pdf.fonttype"] = 42  # keep text as text
    theme = LIGHT
    wing = result.request.wing
    pages = 0
    with PdfPages(path, metadata={"Title": f"{state.name} - lifting-line analysis", "Creator": f"Lifting-Line Wing Calculator {__version__}"}) as pdf:
        for fig in (
            _summary_page(state, result, infos, system),
            _plot_page("Planform and lift loading",
                       lambda f: plot_planform(f, result, wing, system, theme, getattr(result.request.section, "high_lift", None))),
            _plot_page("Spanwise distributions", lambda f: plot_spanwise(f, result, system, theme)),
            _plot_page(
                "Section airfoils",
                lambda f: plot_airfoils(f, [(i.airfoil_name, i.airfoil) for i in infos], 0, theme),
            ),
        ):
            pdf.savefig(fig)
            pages += 1
        if sweep is not None:
            current = {"alpha": result.totals.alpha_deg - wing.incidence_deg, "speed": result.request.flow.V}.get(sweep.variable)
            pdf.savefig(_plot_page("Sweep", lambda f: plot_sweep(f, sweep, theme, system, current, stall)))
            pages += 1
    return pages
