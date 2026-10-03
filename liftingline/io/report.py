"""Turning an analysis result into display rows, text and CSV (shared by the window and the exporters)."""

from __future__ import annotations

import csv
import io
import math
from dataclasses import dataclass, replace

import numpy as np

from ..core import units
from ..core.analysis import AnalysisResult


def fmt(x: float | None, sig: int = 4) -> str:
    """Compact number: ``sig`` significant digits, scientific notation for very large/small values."""
    if x is None:
        return ""
    if isinstance(x, float) and math.isnan(x):
        return "-"
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    if x == 0.0:
        return "0"
    a = abs(x)
    if a >= 1e6 or a < 1e-3:
        return f"{x:.{sig - 1}e}"
    digits = max(sig - 1 - int(math.floor(math.log10(a))), 0)
    return f"{x:.{digits}f}"


def _snap(x: float, scale: float, tol: float = 1e-9) -> float:
    return 0.0 if abs(x) < tol * scale else x


@dataclass(frozen=True)
class Row:
    group: str
    label: str
    value: str = ""  # dimensional value in the display units
    unit: str = ""
    coeff_name: str = ""
    coeff: str = ""
    tip: str = ""  # explanation shown as a tooltip
    key: str = ""  # machine-readable name (CSV)
    si_value: float | None = None
    coeff_value: float | None = None


def totals_rows(result: AnalysisResult, system: str = "SI") -> list[Row]:
    """The result table: forces/moments with coefficients, efficiency and reference quantities."""
    t = result.totals
    req = result.request
    scale_f = abs(t.L) + abs(t.D) + 1e-300  # round-off in symmetric cases (1e-11 N m) must not look like a result
    scale_m = scale_f * max(t.b, t.MAC)
    t = replace(
        t,
        Y=_snap(t.Y, scale_f), CY=_snap(t.CY, 1.0),
        M_roll=_snap(t.M_roll, scale_m), C_roll=_snap(t.C_roll, 1.0),
        M_yaw=_snap(t.M_yaw, scale_m), C_yaw=_snap(t.C_yaw, 1.0),
        M_pitch=_snap(t.M_pitch, scale_m), Cm=_snap(t.Cm, 1.0),
    )
    uF, uM = units.display_unit(system, "force"), units.display_unit(system, "moment")
    uL, uA = units.display_unit(system, "length"), units.display_unit(system, "area")
    uP = units.display_unit(system, "pressure")
    F = lambda v: fmt(units.from_si("force", v, uF))  # noqa: E731
    M = lambda v: fmt(units.from_si("moment", v, uM))  # noqa: E731
    rows: list[Row] = []
    g = "Forces and moments"
    rows += [
        Row(g, "Lift L", F(t.L), uF, "CL", fmt(t.CL), "Force perpendicular to the freestream (up positive)", "L", t.L, t.CL),
        Row(g, "Drag D", F(t.D), uF, "CD", fmt(t.CD, 4), "Total drag along the freestream: induced + profile + extra", "D", t.D, t.CD),
        Row(g, "   induced drag", F(t.D_induced), uF, "CDi", fmt(t.CD_induced), "Drag due to lift (from the lifting-line solution)", "D_induced", t.D_induced, t.CD_induced),
        Row(g, "   profile drag", F(t.D_profile), uF, "CD0", fmt(t.CD_profile), "Skin-friction and pressure drag from the section polars at the local Reynolds number and angle of attack", "D_profile", t.D_profile, t.CD_profile),
    ]
    if t.CD_extra:
        rows.append(Row(g, "   extra drag", F(t.D_extra), uF, "CD,extra", fmt(t.CD_extra), "Drag of the rest of the aircraft that you entered", "D_extra", t.D_extra, t.CD_extra))
    if abs(t.beta_deg) > 1e-12 or abs(t.CY) > 1e-9:
        rows.append(Row(g, "Side force Y", F(t.Y), uF, "CY", fmt(t.CY), "Positive towards the right wing", "Y", t.Y, t.CY))
    rows += [
        Row(g, "Pitching moment", M(t.M_pitch), uM, "Cm", fmt(t.Cm), f"About x = {fmt(units.from_si('length', t.x_ref, uL))} {uL} behind the root leading edge; nose-up positive; Cm uses the mean aerodynamic chord", "M_pitch", t.M_pitch, t.Cm),
        Row(g, "Rolling moment", M(t.M_roll), uM, "Cl", fmt(t.C_roll), "Right wing down positive; Cl uses the span", "M_roll", t.M_roll, t.C_roll),
        Row(g, "Yawing moment", M(t.M_yaw), uM, "Cn", fmt(t.C_yaw), "Nose right positive; Cn uses the span", "M_yaw", t.M_yaw, t.C_yaw),
        Row(g, "Root bending moment", M(max(t.root_bending_right, t.root_bending_left)), uM, "", "", "Bending moment of the larger semi-span about the root due to lift (aerodynamic only, no inertia relief)", "root_bending", max(t.root_bending_right, t.root_bending_left), None),
    ]
    g = "Efficiency"
    rows += [
        Row(g, "Lift-to-drag L/D", "", "", "L/D", fmt(t.L_over_D, 4), "", "L_over_D", None, t.L_over_D),
        Row(g, "Span efficiency e", "", "", "e", fmt(t.e, 4), "CL^2 / (pi AR CDi): 1 for an elliptic load distribution", "e", None, t.e),
    ]
    ld = result.load
    margin = ld.cl_margin
    if np.any(np.isfinite(ld.cl_max)) and np.any(margin > 0):
        k = int(np.argmax(margin))
        rows.append(
            Row("Stall", "Highest cl / cl,max", "", "", "cl/cl,max", fmt(float(margin[k]), 3),
                f"At |y| = {fmt(units.from_si('length', abs(float(ld.y[k])), uL))} {uL}; 1 means that section is at its maximum lift", "stall_margin", None, float(margin[k]))
        )
    g = "Reference"
    flow = req.flow
    rows += [
        Row(g, "Wing area S", fmt(units.from_si("area", t.S, uA)), uA, "", "", "Projected planform area", "S", t.S, None),
        Row(g, "Span b", fmt(units.from_si("length", t.b, uL)), uL, "", "", "Projected span", "b", t.b, None),
        Row(g, "Aspect ratio AR", "", "", "AR", fmt(t.AR, 4), "b^2 / S", "AR", None, t.AR),
        Row(g, "Mean chord (MAC)", fmt(units.from_si("length", t.MAC, uL)), uL, "", "", "", "MAC", t.MAC, None),
        Row(g, "Dynamic pressure q", fmt(units.from_si("pressure", t.q, uP)), uP, "", "", "", "q", t.q, None),
        Row(g, "Re (MAC)", "", "", "Re", fmt(float(flow.reynolds(t.MAC)), 3), "", "Re_MAC", None, float(flow.reynolds(t.MAC))),
        Row(g, "Mach number", "", "", "M", fmt(flow.mach, 3), "", "Mach", None, flow.mach),
        Row(g, "Root angle of attack", fmt(t.alpha_deg, 4), "deg", "", "", "Flow angle of attack plus root incidence", "alpha_deg", t.alpha_deg, None),
    ]
    return rows


def totals_text(result: AnalysisResult, system: str = "SI") -> str:
    """Plain text (tab separated, pastes into spreadsheets) of the result table."""
    lines = ["Quantity\tValue\tUnit\tCoefficient\tCoefficient value"]
    for r in totals_rows(result, system):
        lines.append(f"{r.label.strip()}\t{r.value}\t{r.unit}\t{r.coeff_name}\t{r.coeff}")
    return "\n".join(lines)


# --------------------------------------------------------------------------------- spanwise data
SPANWISE_COLUMNS = (
    ("y_m", "y [m]"),
    ("chord_m", "chord [m]"),
    ("gamma_m2s", "circulation [m^2/s]"),
    ("cl", "cl"),
    ("cd", "cd"),
    ("cm", "cm"),
    ("alpha_eff_deg", "alpha_eff [deg]"),
    ("alpha_induced_deg", "alpha_induced [deg]"),
    ("re", "Re"),
    ("cl_max", "cl_max"),
)


def spanwise_table(result: AnalysisResult) -> dict[str, np.ndarray]:
    ld = result.load
    return dict(
        y_m=ld.y,
        chord_m=ld.chord,
        gamma_m2s=ld.gamma,
        cl=ld.cl,
        cd=ld.cd,
        cm=ld.cm,
        alpha_eff_deg=np.degrees(ld.alpha_eff),
        alpha_induced_deg=np.degrees(ld.alpha_induced),
        re=ld.re,
        cl_max=np.where(np.isfinite(ld.cl_max), ld.cl_max, np.nan),
    )


def _num(x) -> str:
    """Full-precision number for CSV (plain float repr: NumPy scalars would print as ``np.float64(...)``)."""
    return "" if x is None else repr(float(x))


def result_to_csv(result: AnalysisResult, system: str = "SI", project_name: str = "") -> str:
    """Totals (SI values and coefficients) followed by the spanwise distribution, as CSV text."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["# Lifting-line wing calculator"])
    if project_name:
        w.writerow(["# project", project_name])
    w.writerow(["# solver", result.solver_used, "section data", result.info.get("section_engine", "")])
    for note in result.warnings:
        w.writerow(["# warning", note])
    w.writerow([])
    w.writerow(["quantity", "SI value", "coefficient"])
    for r in totals_rows(result, system):
        w.writerow([r.key or r.label.strip(), _num(r.si_value), _num(r.coeff_value)])
    w.writerow([])
    table = spanwise_table(result)
    w.writerow([label for _, label in SPANWISE_COLUMNS])
    for i in range(len(table["y_m"])):
        w.writerow([_num(table[key][i]) for key, _ in SPANWISE_COLUMNS])
    return buf.getvalue()
