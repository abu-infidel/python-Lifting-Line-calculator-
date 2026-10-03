"""Plots: pure matplotlib drawing functions (testable without Qt) and the Qt canvas that hosts them."""

from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.figure import Figure

from ..airfoil.geometry import Airfoil
from ..core import units
from ..core.analysis import AnalysisResult
from .theme import Theme, current_theme


# ============================================================================== styling
def style_axes(ax, theme: Theme, grid: bool = True) -> None:
    ax.set_facecolor(theme.bg)
    for spine in ax.spines.values():
        spine.set_color(theme.grid)
    ax.tick_params(colors=theme.muted, labelsize=8)
    ax.xaxis.label.set_color(theme.fg)
    ax.yaxis.label.set_color(theme.fg)
    ax.title.set_color(theme.fg)
    if grid:
        ax.grid(True, color=theme.grid, linewidth=0.6, alpha=0.8)
        ax.set_axisbelow(True)


def message(fig: Figure, text: str, theme: Theme) -> None:
    fig.clear()
    fig.set_facecolor(theme.bg)
    fig.text(0.5, 0.5, text, ha="center", va="center", color=theme.muted, fontsize=11, wrap=True)


def _length(system: str):
    unit = units.display_unit(system, "length")
    return unit, (lambda v: units.from_si("length", np.asarray(v, dtype=float), unit))


# ================================================================================ planform
def _loading(result: AnalysisResult):
    """(y sorted, c cl, elliptic c cl of the same total lift) of a result, SI."""
    load, t = result.load, result.totals
    order = np.argsort(load.y)
    y = load.y[order]
    ccl = load.chord[order] * load.cl[order]
    ell = 4.0 * t.S * t.CL / (math.pi * t.b) * np.sqrt(np.clip(1.0 - (2.0 * y / t.b) ** 2, 0.0, None))
    return y, ccl, ell


def plot_planform(fig: Figure, result: AnalysisResult | None, wing, system: str, theme: Theme) -> None:
    """Top view with the lift coefficient (or stall margin) painted on the wing, the loading under it and a
    front view for dihedral. Without a result only the geometry is drawn."""
    fig.clear()
    fig.set_facecolor(theme.bg)
    unit, L = _length(system)
    if result is not None:
        gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.15, 0.8])
        ax = fig.add_subplot(gs[0])
        axl = fig.add_subplot(gs[1], sharex=ax)
        axf = fig.add_subplot(gs[2], sharex=ax)
        axes = (ax, axl, axf)
    else:
        gs = fig.add_gridspec(2, 1, height_ratios=[1.3, 1.0])
        ax = fig.add_subplot(gs[0])
        axf = fig.add_subplot(gs[1], sharex=ax)
        axl = None
        axes = (ax, axf)
    for a in axes:
        style_axes(a, theme)

    out = wing.planform_outline()
    y, x_le, x_te, x_c4 = (out[k] for k in ("y", "x_le", "x_te", "x_c4"))
    ax.fill(np.r_[L(y), L(y[::-1])], -np.r_[L(x_le), L(x_te[::-1])], facecolor=theme.panel, edgecolor="none", zorder=1)

    mappable = None
    if result is not None:
        load = result.load
        yc = load.y
        order = np.argsort(yc)
        yc = yc[order]
        edges = np.r_[-wing.half_span, 0.5 * (yc[1:] + yc[:-1]), wing.half_span]
        finite = np.isfinite(load.cl_max[order]) & (load.cl_max[order] > 0)
        use_margin = bool(np.all(finite))
        vals = (load.cl[order] / load.cl_max[order]) if use_margin else load.cl[order]
        polys = [
            [
                (L(y0), -L(wing.x_le(y0))),
                (L(y1), -L(wing.x_le(y1))),
                (L(y1), -L(wing.x_le(y1) + wing.chord(y1))),
                (L(y0), -L(wing.x_le(y0) + wing.chord(y0))),
            ]
            for y0, y1 in zip(edges[:-1], edges[1:])
        ]
        pc = PolyCollection(polys, array=np.asarray(vals, dtype=float), cmap="viridis", edgecolors="none", zorder=2)
        ax.add_collection(pc)
        mappable = (pc, "cl / cl,max" if use_margin else "section lift coefficient cl")
    ax.plot(L(y), -L(x_le), color=theme.fg, lw=1.2, zorder=3)
    ax.plot(L(y), -L(x_te), color=theme.fg, lw=1.2, zorder=3)
    ax.plot([L(y[0]), L(y[0])], [-L(x_le[0]), -L(x_te[0])], color=theme.fg, lw=1.2, zorder=3)
    ax.plot([L(y[-1]), L(y[-1])], [-L(x_le[-1]), -L(x_te[-1])], color=theme.fg, lw=1.2, zorder=3)
    ax.plot(L(y), -L(x_c4), "--", color=theme.accent2, lw=1.0, zorder=4, label="quarter-chord line")
    for sec in wing.sections:  # section stations
        for s_ in (-1, 1):
            ys = s_ * sec.y
            ax.plot([L(ys)] * 2, [-L(wing.x_le(ys)), -L(wing.x_le(ys) + wing.chord(ys))], color=theme.muted, lw=0.7, zorder=3)
    ymac, xmac = wing.y_mac, wing.x_le_mac
    for s_ in (-1, 1):
        ax.plot([L(s_ * ymac)] * 2, [-L(xmac), -L(xmac + wing.mac)], color=theme.bad, lw=2.0, zorder=5, label="mean aerodynamic chord" if s_ == 1 else None)
    if result is not None:
        ax.plot([0.0], [-L(result.totals.x_ref)], marker="X", color=theme.accent, markersize=8, zorder=6, linestyle="none", label="moment reference")
    ax.autoscale_view()
    ax.margins(x=0.04, y=0.2)
    ax.set_aspect("equal", adjustable="box")
    ax.set_ylabel(f"x [{unit}]  (aft down)", fontsize=8)
    sweep = wing.sweep_c4_deg_by_segment()
    ax.set_title(
        f"Planform: S = {units.from_si('area', wing.area, units.display_unit(system, 'area')):.3g} "
        f"{units.display_unit(system, 'area')}, AR = {wing.aspect_ratio:.3g}, c/4 sweep {sweep[0]:.1f}\N{DEGREE SIGN}",
        fontsize=10,
    )
    handles, labels = ax.get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=7, frameon=False)
    for t in leg.get_texts():
        t.set_color(theme.muted)
    if mappable is not None:
        cb = fig.colorbar(mappable[0], ax=ax, orientation="vertical", fraction=0.025, pad=0.01, aspect=12)
        cb.set_label(mappable[1], color=theme.fg, fontsize=7)
        cb.ax.tick_params(colors=theme.muted, labelsize=7)
        cb.outline.set_edgecolor(theme.grid)

    if axl is not None:
        yl, ccl, ell = _loading(result)
        axl.plot(L(yl), L(ccl), color=theme.series[0], lw=1.6, label="this wing")
        axl.plot(L(yl), L(ell), "--", color=theme.series[3], lw=1.2, label="elliptic, same lift")
        axl.set_ylabel(f"c cl [{unit}]", fontsize=8)
        axl.set_title(f"Lift loading (span efficiency e = {result.totals.e:.3f})", fontsize=9)
        leg2 = axl.legend(fontsize=7, frameon=False, loc="lower center")
        for t in leg2.get_texts():
            t.set_color(theme.muted)
        axl.tick_params(labelbottom=False)
        axl.set_ylim(bottom=0.0)
    ax.tick_params(labelbottom=False)

    ys = np.linspace(-wing.half_span, wing.half_span, 201)
    z = L(wing.z_c4(ys))
    axf.plot(L(ys), z, color=theme.accent, lw=1.6)
    axf.set_xlabel(f"y [{unit}]  (right wing to the right)")
    axf.set_ylabel(f"z c/4 [{unit}]", fontsize=8)
    axf.set_title("Front view (quarter-chord line)", fontsize=9)
    zr = float(np.ptp(z))
    if zr < 1e-9:
        pad = 0.05 * float(np.ptp(L(ys)))
        axf.set_ylim(-pad, pad)
    else:
        axf.margins(y=0.25)


# ================================================================================ spanwise
def plot_spanwise(fig: Figure, result: AnalysisResult, system: str, theme: Theme) -> None:
    """Six panels: cl, loading versus the elliptic ideal, angles, drag, Reynolds number, circulation."""
    fig.clear()
    fig.set_facecolor(theme.bg)
    unit, L = _length(system)
    wing, flow, load, t = result.request.wing, result.request.flow, result.load, result.totals
    order = np.argsort(load.y)
    y = L(load.y[order])
    hs = wing.half_span
    axes = fig.subplots(2, 3)
    for ax in axes.ravel():
        style_axes(ax, theme)
        ax.set_xlim(-L(hs), L(hs))
    c = theme.series

    ax = axes[0, 0]
    ax.plot(y, load.cl[order], color=c[0], lw=1.6, label="cl")
    clmax = load.cl_max[order]
    if np.all(np.isfinite(clmax)):
        ax.plot(y, clmax, "--", color=c[1], lw=1.2, label="cl,max")
    ax.set_title("Section lift coefficient", fontsize=9)
    ax.legend(fontsize=7, frameon=False, labelcolor=theme.muted)

    ax = axes[0, 1]
    ccl = load.chord[order] * load.cl[order]
    ell = 4.0 * t.S * t.CL / (math.pi * t.b) * np.sqrt(np.clip(1.0 - (2.0 * load.y[order] / t.b) ** 2, 0.0, None))
    ax.plot(y, L(ccl), color=c[0], lw=1.6, label="c cl (this wing)")
    ax.plot(y, L(ell), "--", color=c[3], lw=1.2, label="elliptic, same lift")
    ax.set_title(f"Lift loading (span efficiency e = {t.e:.3f})", fontsize=9)
    ax.set_ylabel(f"c cl [{unit}]", fontsize=8)
    ax.legend(fontsize=7, frameon=False, labelcolor=theme.muted)

    ax = axes[0, 2]
    a_eff, a_i = np.degrees(load.alpha_eff[order]), np.degrees(load.alpha_induced[order])
    ax.plot(y, a_eff + a_i, color=c[3], lw=1.2, label="geometric")
    ax.plot(y, a_eff, color=c[0], lw=1.6, label="effective")
    ax.plot(y, a_i, color=c[1], lw=1.4, label="induced")
    ax.set_title("Angles of attack [deg]", fontsize=9)
    ax.legend(fontsize=7, frameon=False, labelcolor=theme.muted)

    ax = axes[1, 0]
    ax.plot(y, load.cd[order], color=c[2], lw=1.6)
    ax.set_title("Section profile drag cd", fontsize=9)
    ax.set_xlabel(f"y [{unit}]", fontsize=8)

    ax = axes[1, 1]
    ax.plot(y, load.re[order] / 1e6, color=c[4], lw=1.6)
    ax.set_title("Reynolds number [million]", fontsize=9)
    ax.set_xlabel(f"y [{unit}]", fontsize=8)

    ax = axes[1, 2]
    ax.plot(y, load.gamma[order] / (flow.V * t.b), color=c[5], lw=1.6)
    ax.set_title("Circulation  \N{GREEK CAPITAL LETTER GAMMA}/(V b)", fontsize=9)
    ax.set_xlabel(f"y [{unit}]", fontsize=8)
    for ax in axes.ravel():
        ax.axvline(0.0, color=theme.grid, lw=0.8)


# ================================================================================ airfoils
def plot_airfoils(
    fig: Figure,
    entries: Sequence[tuple[str, Airfoil | None]],
    selected: int,
    theme: Theme,
) -> None:
    """Outlines of the section airfoils (the selected one filled, with its camber line)."""
    fig.clear()
    fig.set_facecolor(theme.bg)
    ax = fig.add_subplot(111)
    style_axes(ax, theme)
    if not any(a is not None for _, a in entries):
        message(fig, "No airfoil to show.", theme)
        return
    for i, (label, af) in enumerate(entries):
        if af is None:
            continue
        col = theme.series[i % len(theme.series)]
        x, yv = af.coords[:, 0], af.coords[:, 1]
        sel = i == selected
        if sel:
            ax.fill(x, yv, facecolor=col, alpha=0.18, edgecolor="none")
        ax.plot(x, yv, color=col, lw=2.0 if sel else 1.1, label=f"{label}: {af.name}", alpha=1.0 if sel else 0.8)
        if sel:
            xs = np.linspace(0.0, 1.0, 200)
            ax.plot(xs, af.camber(xs), "--", color=col, lw=1.0)
            ax.annotate(
                f"t/c {100 * af.t_over_c:.1f} % at {100 * af.x_tmax:.0f} % chord\ncamber {100 * af.max_camber:.2f} % at {100 * af.x_camber:.0f} % chord",
                xy=(0.5, 0.97), xycoords="axes fraction", ha="center", va="top", fontsize=8, color=theme.fg,
            )
    ax.plot([0, 1], [0, 0], color=theme.muted, lw=0.6)
    top = max((float(np.max(np.abs(a.coords[:, 1]))) for _, a in entries if a is not None), default=0.1)
    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(-max(2.2 * top, 0.16), max(2.2 * top, 0.16))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x/c")
    ax.set_ylabel("y/c")
    ax.set_title("Section airfoils", fontsize=10)
    leg = ax.legend(loc="lower right", fontsize=8, frameon=False)
    for tx in leg.get_texts():
        tx.set_color(theme.muted)


# =================================================================================== widget
def make_canvas(parent=None, toolbar: bool = True):
    """A Qt widget with a matplotlib figure (and optionally the navigation toolbar)."""
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
    from PySide6.QtWidgets import QVBoxLayout, QWidget

    class PlotCanvas(QWidget):
        def __init__(self) -> None:
            super().__init__(parent)
            self.figure = Figure(figsize=(6, 4), dpi=100, layout="constrained")
            self.canvas = FigureCanvasQTAgg(self.figure)
            lay = QVBoxLayout(self)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.setSpacing(0)
            self.toolbar = NavigationToolbar2QT(self.canvas, self) if toolbar else None
            if self.toolbar is not None:
                lay.addWidget(self.toolbar)
            lay.addWidget(self.canvas, 1)
            self._last: Callable[[Figure, Theme], None] | None = None
            self.error: str = ""

        def draw_with(self, painter: Callable[[Figure, Theme], None] | None, placeholder: str = "") -> None:
            """Redraw with ``painter(fig, theme)``; a failure shows a message instead of raising."""
            theme = current_theme()
            self._last = painter
            self.error = ""
            try:
                if painter is None:
                    message(self.figure, placeholder, theme)
                else:
                    painter(self.figure, theme)
            except Exception as exc:  # a plotting bug must not take the window down
                self.error = f"{type(exc).__name__}: {exc}"
                message(self.figure, f"Could not draw this plot:\n{self.error}", theme)
            self.canvas.draw_idle()

        def redraw(self) -> None:
            if self._last is not None:
                self.draw_with(self._last)

    return PlotCanvas()
