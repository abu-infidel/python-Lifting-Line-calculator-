"""Programmatic interface: one call per task, JSON-ready results (used by the command line and the MCP server).

No Qt is imported here. Every task takes a *project* - a :class:`~liftingline.io.project.ProjectState`, a project file
(``File > Save`` in the GUI) or the name of a built-in example - plus optional overrides of the flight condition, runs
the same pipeline as the GUI and returns an :class:`ApiResult`:

* ``data``: a plain dictionary for ``json.dumps`` - SI units, the unit in every key name (``L_N``, ``V_mps``,
  ``M_pitch_Nm`` ...), ``None`` where a value does not exist (NaN), and ``"schema"`` naming the format
  (``"liftingline.analysis/1"`` ...; README_FOR_CLAUDE.md describes each one);
* ``text``: the same result as a readable report;
* ``csv``: a table, for the tasks that have one (analysis, sweep, scenario).

Conventions (as in the GUI): x forward, y to starboard, z down; lift up +, drag aft +, pitching moment nose-up +,
rolling moment right-wing-down +, yawing moment nose-right +. Angles of attack are those of the fuselage reference
line (the Flow tab's); the wing's root incidence is added inside the analysis.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .core import units
from .io.presets import PRESETS, light_aircraft
from .io.project import SPEED_BY_MACH, SPEED_BY_VELOCITY, ProjectError, ProjectState

SCHEMA = {
    "analysis": "liftingline.analysis/1",
    "sweep": "liftingline.sweep/1",
    "max_lift": "liftingline.max_lift/1",
    "trim": "liftingline.trim/1",
    "scenario": "liftingline.scenario/1",
    "examples": "liftingline.examples/1",
    "project": "liftingline.project/1",
    "compare": "liftingline.compare/1",
    "stl": "liftingline.stl/1",
    "validate": "liftingline.validate/1",
}
ENGINES = ("auto", "neuralfoil", "builtin")
SOLVERS = ("auto", "classical", "numerical")
SWEEP_VARIABLES = {"alpha": "deg", "speed": "m/s", "altitude": "m", "beta": "deg", "deflection": "deg"}
EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


class ApiError(ValueError):
    """Bad input or a task that cannot be done; the message is meant for the user (or the calling agent)."""


@dataclass
class ApiResult:
    data: dict[str, Any]
    text: str
    csv: str | None = None

    def render(self, fmt: str = "text") -> str:
        """The result as ``json``, ``text`` or ``csv`` (``csv`` falls back to JSON where there is no table)."""
        if fmt == "json":
            return to_json(self.data)
        if fmt == "csv":
            return self.csv if self.csv is not None else to_json(self.data)
        return self.text


# =============================================================================================== JSON
def clean(obj: Any) -> Any:
    """Plain JSON types: NumPy scalars and arrays become numbers and lists, NaN and infinity become ``None``."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [clean(v) for v in obj.tolist()]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        x = float(obj)
        return x if math.isfinite(x) else None
    if obj is None or isinstance(obj, str):
        return obj
    return str(obj)


def to_json(data: dict, indent: int | None = 2) -> str:
    import json

    return json.dumps(clean(data), indent=indent, ensure_ascii=False)


def _program() -> dict:
    return {"name": "liftingline", "version": __version__}


# =========================================================================================== projects
def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def example_names() -> list[str]:
    return list(PRESETS)


def find_example(name: str) -> str:
    """The built-in example meant by ``name``: its exact name, its file name (slug), or a unique part of either."""
    key = name.strip().lower()
    for full in PRESETS:
        if key in (full.lower(), slug(full), slug(full) + ".json"):
            return full
    hits = [full for full in PRESETS if key in full.lower() or slug(key) in slug(full)]
    if len(hits) == 1:
        return hits[0]
    choices = "; ".join(PRESETS)
    if not hits:
        raise ApiError(f"no example called {name!r}; the examples are: {choices}")
    raise ApiError(f"{name!r} matches several examples ({'; '.join(hits)}); be more specific")


def load_project(project: ProjectState | dict | str | Path | None = None, *, example: str | None = None) -> ProjectState:
    """A project state from a state, a project dictionary, a project file or an example name.

    With neither ``project`` nor ``example`` the GUI's start-up wing (the Cessna 172-like light aircraft) is used.
    A string that is not an existing file is tried as an example name.
    """
    if project is not None and example is not None:
        raise ApiError("give a project or an example, not both")
    if example is not None:
        return PRESETS[find_example(example)]()
    if project is None:
        return light_aircraft()
    if isinstance(project, ProjectState):
        return project.copy()
    try:
        if isinstance(project, dict):
            return ProjectState.from_dict(project)
        path = Path(project)
        if path.is_file():
            return ProjectState.load(path)
    except (ProjectError, OSError, ValueError, TypeError, KeyError) as exc:
        raise ApiError(f"cannot read the project: {exc}") from exc
    if isinstance(project, str) and not project.lower().endswith(".json"):
        return PRESETS[find_example(project)]()
    raise ApiError(f"project file not found: {project}")


def apply_overrides(
    state: ProjectState,
    *,
    alpha_deg: float | None = None,
    speed: float | None = None,
    speed_unit: str = "m/s",
    mach: float | None = None,
    altitude_m: float | None = None,
    beta_deg: float | None = None,
    engine: str | None = None,
    solver: str | None = None,
    flaps: bool | None = None,
    slats: bool | None = None,
    phase: str | None = None,
    extra_cd0: float | None = None,
    ground_height_m: float | None = None,
    convergence_check: bool | None = None,
) -> ProjectState:
    """A copy of ``state`` with the given flight-condition and solver settings replaced (``None`` keeps a value).

    ``ground_height_m``: a height switches ground effect on, ``0`` switches it off.
    """
    st = state.copy()
    f, s, hl = st.flow, st.solver, st.high_lift
    if alpha_deg is not None:
        f.alpha_deg = float(alpha_deg)
    if speed is not None and mach is not None:
        raise ApiError("give the airspeed or the Mach number, not both")
    if speed is not None:
        if speed_unit not in units.units_of("speed"):
            raise ApiError(f"unknown speed unit {speed_unit!r}; use one of {', '.join(units.units_of('speed'))}")
        f.speed_mode, f.V = SPEED_BY_VELOCITY, float(units.to_si("speed", float(speed), speed_unit))
    if mach is not None:
        f.speed_mode, f.mach = SPEED_BY_MACH, float(mach)
    if altitude_m is not None:
        f.atmosphere, f.altitude_m = "isa", float(altitude_m)
    if beta_deg is not None:
        f.beta_deg = float(beta_deg)
    if extra_cd0 is not None:
        f.extra_cd0 = float(extra_cd0)
    if ground_height_m is not None:
        f.ground_effect = ground_height_m > 0.0
        if f.ground_effect:
            f.ground_height_m = float(ground_height_m)
    if engine is not None:
        if engine not in ENGINES:
            raise ApiError(f"unknown section engine {engine!r}; use one of {', '.join(ENGINES)}")
        s.engine = engine
    if solver is not None:
        if solver not in SOLVERS:
            raise ApiError(f"unknown solver {solver!r}; use one of {', '.join(SOLVERS)}")
        s.solver = solver
    if convergence_check is not None:
        s.convergence_check = bool(convergence_check)
    if flaps is not None:
        hl.flaps_on = bool(flaps)
    if slats is not None:
        hl.slats_on = bool(slats)
    if phase is not None:
        if phase not in ("takeoff", "landing"):
            raise ApiError("the high-lift setting is 'takeoff' or 'landing'")
        hl.phase = phase
    return st


def make_backend(library: str | None = None):
    """The analysis back end; ``library`` is a folder of ``.dat`` airfoils (UIUC style) for names that need it."""
    from .io.backend import Backend

    be = Backend()
    if library:
        try:
            be.scan_library(library)
        except (OSError, ValueError) as exc:
            raise ApiError(f"airfoil library folder {library!r}: {exc}") from exc
    return be


def _request(state: ProjectState, backend=None):
    be = backend or make_backend()
    try:
        return be.build_request(state)
    except ProjectError as exc:
        raise ApiError(str(exc)) from exc


# ========================================================================================== describing
def _wing_dict(wing) -> dict:
    return {
        "name": wing.name,
        "span_m": wing.span,
        "area_m2": wing.area,
        "aspect_ratio": wing.aspect_ratio,
        "taper_ratio": wing.taper_ratio,
        "mac_m": wing.mac,
        "root_chord_m": wing.root_chord,
        "tip_chord_m": wing.tip_chord,
        "incidence_deg": wing.incidence_deg,
        "max_sweep_c4_deg": wing.max_abs_sweep_c4_deg,
        "max_dihedral_deg": wing.max_abs_dihedral_deg,
        "sections": [
            {"y_m": s.y, "chord_m": s.chord, "x_le_m": s.x_le, "z_m": s.z, "twist_deg": s.twist_deg, "airfoil": s.airfoil}
            for s in wing.sections
        ],
    }


def _flow_dict(flow, wing) -> dict:
    return {
        "V_mps": flow.V,
        "alpha_deg": flow.alpha_deg,
        "beta_deg": flow.beta_deg,
        "altitude_m": flow.altitude_m,
        "rho_kgm3": flow.rho,
        "mu_Pas": flow.mu,
        "T_K": flow.T,
        "speed_of_sound_mps": flow.a,
        "mach": flow.mach,
        "q_Pa": flow.q,
        "re_mac": float(flow.reynolds(wing.mac)),
    }


TOTALS_KEYS = {
    "L": "L_N", "D": "D_N", "D_induced": "D_induced_N", "D_profile": "D_profile_N", "D_extra": "D_extra_N",
    "Y": "Y_N", "M_pitch": "M_pitch_Nm", "M_roll": "M_roll_Nm", "M_yaw": "M_yaw_Nm",
    "CL": "CL", "CD": "CD", "CD_induced": "CD_induced", "CD_profile": "CD_profile", "CD_extra": "CD_extra", "CY": "CY",
    "Cm": "Cm", "C_roll": "C_roll", "C_yaw": "C_yaw", "S": "S_m2", "b": "b_m", "AR": "AR", "MAC": "MAC_m", "q": "q_Pa",
    "e": "e", "L_over_D": "L_over_D", "root_bending_right": "root_bending_right_Nm",
    "root_bending_left": "root_bending_left_Nm", "root_shear": "root_shear_N", "x_ref": "x_ref_m", "z_ref": "z_ref_m",
    "alpha_deg": "alpha_root_deg", "beta_deg": "beta_deg",
}


def totals_dict(t) -> dict:
    """``Totals`` with unit-suffixed keys (``alpha_root_deg`` is the root chord's angle: flow angle + incidence)."""
    return {TOTALS_KEYS[f.name]: getattr(t, f.name) for f in fields(t)}


def _project_meta(state: ProjectState) -> dict:
    return {"name": state.name, "notes": state.notes}


def describe_project(state: ProjectState) -> ApiResult:
    """The full project (every input, as saved in a project file) plus the derived wing geometry."""
    try:
        wing, flow = state.build_wing(), state.build_flow()
    except (ProjectError, ValueError) as exc:
        raise ApiError(str(exc)) from exc
    data = {"schema": SCHEMA["project"], "program": _program(), "project": state.to_dict(),
            "wing": _wing_dict(wing), "flow": _flow_dict(flow, wing)}
    w = data["wing"]
    text = "\n".join([
        f"{state.name}: span {w['span_m']:.3f} m, area {w['area_m2']:.3f} m2, AR {w['aspect_ratio']:.2f}, "
        f"MAC {w['mac_m']:.3f} m, incidence {w['incidence_deg']:.2f} deg",
        "airfoils: " + ", ".join(dict.fromkeys(s["airfoil"] for s in w["sections"])),
        f"flow: V {flow.V:.2f} m/s, alpha {flow.alpha_deg:.2f} deg, altitude {flow.altitude_m} m, rho {flow.rho:.4f} kg/m3",
    ] + ([state.notes] if state.notes else []))
    return ApiResult(clean(data), text)


def examples() -> ApiResult:
    rows = []
    for name, make in PRESETS.items():
        st = make()
        wing = st.build_wing()
        rows.append({
            "name": name, "slug": slug(name), "file": f"examples/{slug(name)}.json", "notes": st.notes,
            "span_m": wing.span, "area_m2": wing.area, "aspect_ratio": wing.aspect_ratio,
            "airfoils": list(dict.fromkeys(s.airfoil for s in wing.sections)),
            "V_mps": st.flow.V, "alpha_deg": st.flow.alpha_deg, "altitude_m": st.flow.altitude_m,
            "high_lift_devices": st.high_lift.active, "ground_effect": st.flow.ground_effect,
        })
    text = "\n".join(f"{r['name']:34s} span {r['span_m']:6.2f} m, area {r['area_m2']:6.2f} m2, AR {r['aspect_ratio']:5.2f}, "
                     f"{', '.join(r['airfoils'])}" for r in rows)
    return ApiResult(clean({"schema": SCHEMA["examples"], "program": _program(), "examples": rows}), text)


# ============================================================================================= analysis
def analyze(state: ProjectState, *, spanwise: bool = True, backend=None) -> ApiResult:
    """Lift, drag and moments of the wing at the project's flight condition."""
    from .core.analysis import run
    from .io.report import result_to_csv, spanwise_table, totals_rows

    req = _request(state, backend)
    try:
        r = run(req)
    except ValueError as exc:
        raise ApiError(str(exc)) from exc
    t = r.totals
    info = {k: v for k, v in r.info.items() if isinstance(v, (int, float, str, bool, np.floating, np.integer))}
    data = {
        "schema": SCHEMA["analysis"],
        "program": _program(),
        "project": _project_meta(state),
        "wing": _wing_dict(req.wing),
        "flow": _flow_dict(req.flow, req.wing),
        "solver_used": r.solver_used,
        "section_data": str(r.info.get("section_engine", "")),
        "nonlinear": bool(r.info.get("nonlinear", False)),
        "high_lift": r.info.get("high_lift"),
        "totals": totals_dict(t),
        "max_cl_over_clmax": float(np.max(r.load.cl_margin)) if np.any(np.isfinite(r.load.cl_max)) else None,
        "warnings": list(r.warnings),
        "info": info,
    }
    if spanwise:
        data["spanwise"] = spanwise_table(r)
    lines = [f"{state.name}: {r.solver_used} solver, section data {data['section_data']}"
             + (f", high-lift devices: {data['high_lift']}" if data["high_lift"] else ""),
             f"V {req.flow.V:.3f} m/s, alpha {req.flow.alpha_deg:.3f} deg (root {t.alpha_deg:.3f} deg), "
             f"beta {req.flow.beta_deg:.3f} deg, rho {req.flow.rho:.4f} kg/m3, Mach {req.flow.mach:.3f}", ""]
    for row in totals_rows(r, "SI"):
        if row.key == "high_lift":
            continue
        val = f"{row.value} {row.unit}".strip()
        coeff = f"{row.coeff_name} = {row.coeff}" if row.coeff_name else ""
        lines.append(f"  {row.label.strip():28s}{val:>20s}   {coeff}")
    lines += ([""] + [f"warning: {w}" for w in r.warnings]) if r.warnings else ["", "no validity warnings"]
    return ApiResult(clean(data), "\n".join(lines), result_to_csv(r, "SI", state.name))


def sweep(state: ProjectState, variable: str, start: float, stop: float, n: int = 11, *, control: str | None = None,
          hold: str = "speed", backend=None) -> ApiResult:
    """The analysis at ``n`` values of ``variable`` (alpha [deg], speed [m/s], altitude [m], beta [deg],
    deflection [deg] of the control surface ``control``) from ``start`` to ``stop``."""
    from .core.controls import ControlError
    from .core.sweeps import COLUMNS, linspace_values, run_sweep, stall_info

    if variable not in SWEEP_VARIABLES:
        raise ApiError(f"unknown sweep variable {variable!r}; use one of {', '.join(SWEEP_VARIABLES)}")
    if not 2 <= int(n) <= 400:
        raise ApiError("the number of points must be between 2 and 400")
    req = _request(state, backend)
    try:
        res = run_sweep(req, variable, linspace_values(float(start), float(stop), int(n)), control=control, hold=hold)
    except (ValueError, ControlError) as exc:
        raise ApiError(str(exc)) from exc
    unit = SWEEP_VARIABLES[variable]
    points = []
    for p in res.points:
        row = {"value": p.value, "ok": p.ok, "converged": bool(p.converged and p.ok),
               "max_cl_over_clmax": p.max_margin, "first_stall_y_m": p.first_stall_y, "error": p.error or None}
        row.update({TOTALS_KEYS[c]: (getattr(p.totals, c) if p.totals is not None else None) for c in COLUMNS})
        points.append(row)
    data = {"schema": SCHEMA["sweep"], "program": _program(), "project": _project_meta(state), "variable": variable,
            "unit": unit, "solver_used": res.solver, "section_data": res.engine, "points": points, "notes": res.notes}
    if variable == "alpha" and int(n) >= 3:
        si = stall_info(res)
        data["stall"] = {"available": si.available, "reached": si.reached, "alpha_first_stall_deg": si.alpha_first_stall,
                         "y_first_stall_m": si.y_first_stall, "CL_first_stall": si.CL_first_stall, "CL_max": si.CL_max,
                         "alpha_CL_max_deg": si.alpha_CL_max, "CL_max_is_peak": si.CL_max_is_peak, "note": si.note}
    head = f"{variable} [{unit}]"
    lines = [f"{state.name}: sweep of {variable}, {res.solver} solver, section data {res.engine}", "",
             f"{head:>14s} {'CL':>8s} {'CD':>9s} {'CDi':>9s} {'Cm':>8s} {'L/D':>7s} {'L [N]':>10s} {'D [N]':>9s} "
             f"{'cl/clmax':>8s}"]
    for p in points:
        if not p["ok"]:
            lines.append(f"{p['value']:14.4g}  failed: {p['error']}")
            continue
        lines.append(f"{p['value']:14.4g} {p['CL']:8.4f} {p['CD']:9.5f} {p['CD_induced']:9.5f} {p['Cm']:8.4f} "
                     f"{p['L_over_D']:7.2f} {p['L_N']:10.1f} {p['D_N']:9.2f} {p['max_cl_over_clmax']:8.3f}")
    if "stall" in data and data["stall"]["reached"]:
        s = data["stall"]
        lines += ["", f"first section stall at alpha {s['alpha_first_stall_deg']:.2f} deg (CL {s['CL_first_stall']:.4f}, "
                      f"|y| {s['y_first_stall_m']:.2f} m); highest CL of the sweep {s['CL_max']:.4f} at {s['alpha_CL_max_deg']:.2f} deg"]
    lines += [f"note: {n_}" for n_ in res.notes]
    return ApiResult(clean(data), "\n".join(lines), res.to_csv())


def max_lift(state: ProjectState, *, weight_N: float | None = None, backend=None) -> ApiResult:
    """Maximum lift (CL_max) and first section stall; with high-lift devices deployed in the project, the clean
    wing and the wing with the devices out side by side (the GUI's takeoff / landing stall analysis). A weight gives
    stall speeds."""
    from .core.highlift import find_max_lift, format_report, high_lift_report

    be = backend or make_backend()
    weight = float(weight_N) if weight_N else (state.high_lift.weight_N or None)

    def cfg_dict(c) -> dict:
        return {"label": c.label, "CL_max": c.CL_max, "alpha_CL_max_deg": c.alpha_CL_max, "definition": c.definition,
                "CL_first_stall": c.CL_first_stall, "alpha_first_stall_deg": c.alpha_first_stall,
                "y_first_stall_m": c.y_first_stall, "note": c.note}

    try:
        if state.high_lift.active:
            clean_state = state.copy()
            clean_state.high_lift.flaps_on = clean_state.high_lift.slats_on = False
            system = state.high_lift.system()
            rep = high_lift_report(_request(clean_state, be), _request(state, be), system, weight=weight)
            data = {"clean": cfg_dict(rep.clean), "deployed": cfg_dict(rep.deployed), "dCL_max": rep.dCL_max,
                    "handbook_dCL_max": rep.estimate.dCL_max, "devices": system.short_label(),
                    "speeds_mps": dict(rep.speeds), "notes": rep.notes}
            text = format_report(rep)
        else:
            req = _request(state, be)
            c = find_max_lift(req, "clean wing")
            data = {"clean": cfg_dict(c), "deployed": None, "speeds_mps": {}, "notes": [c.note] if c.note else []}
            text_lines = [f"{state.name}: CL_max {c.CL_max:.4f} at alpha {c.alpha_CL_max:.2f} deg ({c.definition}); "
                          f"first section stall at {c.alpha_first_stall:.2f} deg (CL {c.CL_first_stall:.4f}, "
                          f"|y| {c.y_first_stall:.2f} m)"]
            if weight and c.CL_max > 0:
                vs = math.sqrt(2.0 * weight / (req.flow.rho * req.wing.area * c.CL_max))
                data["speeds_mps"]["stall speed (clean)"] = vs
                text_lines.append(f"stall speed at {weight:.0f} N: {vs:.2f} m/s ({vs / 0.514444:.1f} kt)")
            text = "\n".join(text_lines)
    except (ProjectError, InterruptedError) as exc:
        raise ApiError(str(exc)) from exc
    data.update(schema=SCHEMA["max_lift"], program=_program(), project=_project_meta(state), weight_N=weight,
                altitude_m=state.flow.altitude_m)
    return ApiResult(clean(data), text)


def trim(state: ProjectState, *, CL: float | None = None, lift_N: float | None = None, mass_kg: float | None = None,
         solve_for: str = "alpha", backend=None) -> ApiResult:
    """The angle of attack (``solve_for="alpha"``, at the project's airspeed) or the airspeed (``"speed"``, at its
    angle of attack) at which the wing makes a target CL, a target lift, or carries a mass (lift = m g)."""
    from .core.trim import TrimError, trim_alpha, trim_speed

    given = [x is not None for x in (CL, lift_N, mass_kg)]
    if sum(given) != 1:
        raise ApiError("give exactly one target: CL, lift_N or mass_kg")
    if mass_kg is not None:
        lift_N = float(mass_kg) * 9.80665
    req = _request(state, backend)
    try:
        if solve_for == "alpha":
            res = trim_alpha(req, CL=CL, lift=lift_N)
        elif solve_for == "speed":
            if lift_N is None:
                raise ApiError("solving for the airspeed needs a lift or a mass (a CL fixes the angle, not the speed)")
            res = trim_speed(req, lift_N)
        else:
            raise ApiError("solve_for is 'alpha' or 'speed'")
    except TrimError as exc:
        raise ApiError(str(exc)) from exc
    unit = "deg" if res.variable == "alpha_deg" else "m/s"
    data = {"schema": SCHEMA["trim"], "program": _program(), "project": _project_meta(state),
            "solved_for": "alpha_deg" if res.variable == "alpha_deg" else "V_mps", "value": res.value, "unit": unit,
            "target": res.target, "evaluations": res.evaluations, "totals": totals_dict(res.totals),
            "warnings": res.warnings}
    what = "angle of attack" if unit == "deg" else "airspeed"
    text = (f"{state.name}: {res.target} needs {what} {res.value:.4f} {unit}"
            + (f" ({res.value / 0.514444:.2f} kt)" if unit == "m/s" else "")
            + f"; CL {res.totals.CL:.4f}, CD {res.totals.CD:.5f}, L {res.totals.L:.1f} N, D {res.totals.D:.2f} N")
    return ApiResult(clean(data), text + "".join(f"\nwarning: {w}" for w in res.warnings))


def scenario_spec(
    base=None,
    *,
    speed_unit: str | None = None,
    v_start: float | None = None,
    v_end: float | None = None,
    v_rotate: float | None = None,
    alpha_ground_deg: float | None = None,
    ramp: str | None = None,
    ramp_target_deg: float | None = None,
    ramp_rate_deg_per_s: float | None = None,
    ramp_rate_deg_per_speed_unit: float | None = None,
    mass_kg: float | None = None,
    time_model: bool | None = None,
    thrust_static_N: float | None = None,
    thrust_end_N: float | None = None,
    rolling_friction: float | None = None,
    ground_height_m: float | None = None,
    points: int | None = None,
):
    """A :class:`~liftingline.core.scenario.ScenarioSpec` from inputs in a speed unit (``None`` keeps the value of
    ``base``, by default the scenario defaults). The speeds and the per-speed rate are in ``speed_unit`` (default: the
    unit of ``base``, mph for the defaults); ``ground_height_m`` > 0 switches ground effect on, 0 off."""
    from .core.scenario import RAMP_MODES, ScenarioSpec

    spec = base if base is not None else ScenarioSpec()
    u = speed_unit or spec.speed_unit
    if u not in units.units_of("speed"):
        raise ApiError(f"unknown speed unit {u!r}; use one of {', '.join(units.units_of('speed'))}")
    per = float(units.to_si("speed", 1.0, u))  # m/s per unit
    ch: dict[str, Any] = {"speed_unit": u}
    for name, val in (("v_start", v_start), ("v_end", v_end), ("v_rotate", v_rotate)):
        if val is not None:
            ch[name] = float(val) * per
    if ramp is not None:
        if ramp not in RAMP_MODES:
            raise ApiError(f"unknown angle-of-attack rise {ramp!r}; use one of {', '.join(RAMP_MODES)}")
        ch["ramp_mode"] = ramp
    if ramp_rate_deg_per_speed_unit is not None:
        ch["ramp_rate_deg_per_mps"] = float(ramp_rate_deg_per_speed_unit) / per
    for name, val in (("alpha_ground_deg", alpha_ground_deg), ("ramp_target_deg", ramp_target_deg),
                      ("ramp_rate_deg_s", ramp_rate_deg_per_s), ("mass_kg", mass_kg),
                      ("thrust_static_N", thrust_static_N), ("thrust_end_N", thrust_end_N),
                      ("rolling_friction", rolling_friction)):
        if val is not None:
            ch[name] = float(val)
    if time_model is not None:
        ch["time_model"] = bool(time_model)
    if points is not None:
        ch["points"] = int(points)
    if ground_height_m is not None:
        ch["ground_effect"] = ground_height_m > 0.0
        if ground_height_m > 0.0:
            ch["ground_height_m"] = float(ground_height_m)
    return replace(spec, **ch)


def scenario(state: ProjectState, spec=None, *, max_frames: int | None = None, backend=None, **changes) -> ApiResult:
    """A scenario run (see :mod:`liftingline.core.scenario`): ``spec`` is a ScenarioSpec or a dict of its fields (SI);
    without it the project's saved scenario is used. ``changes`` replace single fields."""
    from .core.scenario import ScenarioError, ScenarioSpec, result_to_dict, run_scenario, to_csv, to_text

    if spec is None:
        spec = state.scenario
    elif isinstance(spec, dict):
        known = {f.name for f in fields(ScenarioSpec)}
        bad = sorted(set(spec) - known)
        if bad:
            raise ApiError(f"unknown scenario fields {bad}; the fields are {sorted(known)}")
        spec = replace(state.scenario, **spec)
    if changes:
        spec = replace(spec, **{k: v for k, v in changes.items() if v is not None})
    problems = spec.problems()
    if problems:
        raise ApiError("; ".join(problems))
    req = _request(state, backend)
    try:
        res = run_scenario(req, spec)
    except ScenarioError as exc:
        raise ApiError(str(exc)) from exc
    data = result_to_dict(res, max_frames=max_frames)
    data["program"], data["project"] = _program(), _project_meta(state)
    lof = res.liftoff
    data["liftoff"] = None if lof is None else {"t_s": lof.t if res.has_time else None, "airspeed_mps": lof.V,
                                                "distance_m": lof.x if res.has_time else None,
                                                "alpha_deg": lof.alpha_deg}
    return ApiResult(clean(data), to_text(res), to_csv(res))


# ============================================================================================ others
def compare_aircraft(engine: str = "auto", backend=None) -> ApiResult:
    """The program against published handbook data of real aircraft (Cessna 172S and 152 stall and cruise)."""
    from .io.reference_aircraft import compare, format_comparison

    if engine not in ENGINES:
        raise ApiError(f"unknown section engine {engine!r}; use one of {', '.join(ENGINES)}")
    c = compare(backend or make_backend(), engine=engine)
    rows = [{"aircraft": r.aircraft, "point": r.point, "handbook": r.handbook, "program": r.program,
             "ratio_program_over_handbook": r.ratio, "detail": r.detail} for r in c.rows]
    return ApiResult(clean({"schema": SCHEMA["compare"], "program": _program(), "section_data": c.engine, "rows": rows}),
                     format_comparison(c))


def export_stl(state: ProjectState, path: str | Path, *, quality: str = "fine", unit: str = "mm", scale: float = 1.0,
               half: bool = False, ascii: bool = False, include_incidence: bool = True,
               library: str | None = None) -> ApiResult:
    """Write the wing as a watertight STL surface model (real airfoils, twist, sweep, dihedral)."""
    from .io.stl import QUALITY, UNITS, StlError, export_wing_stl
    from .stl_app import describe, load_wing

    if quality not in QUALITY:
        raise ApiError(f"unknown mesh quality {quality!r}; use one of {', '.join(QUALITY)}")
    if unit not in UNITS:
        raise ApiError(f"unknown unit {unit!r}; use one of {', '.join(UNITS)}")
    try:
        lw = load_wing(state, library or "")
        info = export_wing_stl(path, lw.wing, lw.airfoils, quality=quality, unit=unit, scale=scale, half=half,
                               include_incidence=include_incidence, ascii=ascii)
    except (ProjectError, StlError, OSError) as exc:
        raise ApiError(str(exc)) from exc
    data = {"schema": SCHEMA["stl"], "program": _program(), "project": _project_meta(state),
            "path": str(Path(path).resolve()), "unit": unit, "scale": scale, "quality": quality, "half": half,
            "ascii": ascii, "n_triangles": info.n_triangles, "n_vertices": info.n_vertices,
            "watertight": info.watertight, "volume_m3": info.volume_m3, "area_m2": info.area_m2,
            "bounds_m": {"min": list(info.bounds_m[0]), "max": list(info.bounds_m[1])},
            "file_bytes": info.file_bytes, "airfoils": lw.sources, "notes": lw.notes}
    return ApiResult(clean(data), f"wrote {path}: {describe(info, unit, scale)}")


def validate() -> ApiResult:
    """The built-in self-test (analytic benchmarks and cross-checks); ``data["passed"]`` is False on any failure."""
    from .core.validate import FAIL, PASS, SKIP, format_report, run_validation

    t0 = time.perf_counter()
    checks = run_validation()
    elapsed = time.perf_counter() - t0
    rows = [{"group": c.group, "name": c.name, "status": c.status, "expected": c.expected, "got": c.got,
             "tolerance": c.tolerance, "note": c.note} for c in checks]
    data = {"schema": SCHEMA["validate"], "program": _program(), "passed": all(c.ok for c in checks),
            "counts": {"pass": sum(c.status == PASS for c in checks), "fail": sum(c.status == FAIL for c in checks),
                       "skip": sum(c.status == SKIP for c in checks)},
            "elapsed_s": elapsed, "checks": rows}
    return ApiResult(clean(data), format_report(checks, elapsed))


__all__ = ["SCHEMA", "ApiError", "ApiResult", "analyze", "apply_overrides", "clean", "compare_aircraft",
           "describe_project", "example_names", "examples", "export_stl", "find_example", "load_project",
           "make_backend", "max_lift", "scenario", "scenario_spec", "sweep", "to_json", "totals_dict", "trim", "validate"]
