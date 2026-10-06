"""MCP server: the calculator's tasks as tools for AI agents (Model Context Protocol, stdio transport).

    python -m liftingline mcp [--library FOLDER]

Needs the ``mcp`` package (``pip install "mcp>=1.25,<2"``). Every tool returns the JSON of :mod:`liftingline.api`
(SI units, unit-suffixed keys, ``null`` for values that do not exist); README_FOR_CLAUDE.md, also served as the
resource ``liftingline://docs/readme-for-claude`` and by the tool ``get_documentation``, describes every field.

Nothing is printed to stdout while a tool runs (stdout carries the protocol): stray output goes to stderr. The GUI is
never imported.
"""

from __future__ import annotations

import contextlib
import os
import sys
import threading
from pathlib import Path
from typing import Annotated, Any, Literal

from . import __version__, api

try:
    from pydantic import Field  # installed with mcp
except ImportError:  # pragma: no cover - without mcp, build_server() explains what to install

    def Field(*_args, **_kwargs):  # noqa: N802 - stands in for pydantic.Field
        return None

DOC_PATH = Path(__file__).resolve().parents[1] / "README_FOR_CLAUDE.md"
DOC_URI = "liftingline://docs/readme-for-claude"

INSTRUCTIONS = f"""Lifting-line wing calculator {__version__} (Prandtl lifting-line theory, offline).
Describe a wing by a built-in example (list_examples), a project file written by the GUI (File > Save), or a project
JSON object; override the flight condition (alpha_deg, speed + speed_unit, altitude_m ...) per call.
Tools: analyze_wing (lift, drag, moments at one condition), sweep (polar over alpha/speed/altitude/beta), max_lift
(CL_max, first stall, stall speeds, flaps/slats), trim (angle or speed for a CL, lift or mass), run_scenario (forces
along a start-to-end airspeed run such as a takeoff roll: lift-off point, time, distance), compare_real_aircraft,
export_stl, describe_project, self_test, get_documentation.
Results are JSON in SI units with the unit in the key name (L_N, V_mps, M_pitch_Nm); null means 'does not exist'.
Angles of attack are those of the fuselage reference line: the wing's root incidence is added (totals.alpha_root_deg).
Read 'warnings' in every result: they say when the theory is outside its range. Section data: engine 'builtin' is fast
(about 0.1-1 s per analysis), 'neuralfoil' (default when installed) is more accurate for drag and stall but slower.
Read get_documentation() for every output field."""

_lock = threading.Lock()  # one computation at a time (they are CPU-bound; it also keeps stdout redirection sane)
_backend = None
_library: str | None = None

Example = Annotated[str | None, Field(description="built-in example name (or a unique part, e.g. 'Cessna 172S'); see list_examples")]
ProjectFile = Annotated[str | None, Field(description="path of a project JSON file written by the GUI (File > Save)")]
ProjectJson = Annotated[dict[str, Any] | None, Field(description="a whole project object, as in a project file / describe_project()['project']")]
Alpha = Annotated[float | None, Field(description="angle of attack of the fuselage reference line, deg (root incidence is added)")]
Speed = Annotated[float | None, Field(description="true airspeed in speed_unit")]
SpeedUnit = Annotated[Literal["m/s", "km/h", "kt", "mph", "ft/s"], Field(description="unit of the speeds given")]
Altitude = Annotated[float | None, Field(description="ISA altitude, m")]
Engine = Annotated[Literal["auto", "neuralfoil", "builtin"] | None, Field(description="section data: builtin (fast) or neuralfoil (max fidelity)")]
Solver = Annotated[Literal["auto", "classical", "numerical"] | None, Field(description="lifting-line solver (auto picks)")]
Flaps = Annotated[bool | None, Field(description="deploy (true) or retract (false) the project's trailing-edge flaps")]
Slats = Annotated[bool | None, Field(description="deploy (true) or retract (false) the project's leading-edge slats")]
Setting = Annotated[Literal["takeoff", "landing"] | None, Field(description="flap / slat setting")]
Output = Annotated[Literal["json", "text", "csv"], Field(description="json (default): structured result; text: readable report; csv: table where available")]


def _get_backend():
    global _backend
    if _backend is None:
        _backend = api.make_backend(_library)
    return _backend


def _state(example=None, project_file=None, project=None, **overrides):
    given = [x is not None for x in (example, project_file, project)]
    if sum(given) > 1:
        raise api.ApiError("give one of example, project_file or project")
    st = api.load_project(project_file if project_file is not None else project, example=example)
    return api.apply_overrides(st, **overrides)


def _flight(alpha_deg, speed, speed_unit, altitude_m, engine, solver, flaps, slats, high_lift_setting, **more) -> dict:
    return dict(alpha_deg=alpha_deg, speed=speed, speed_unit=speed_unit, altitude_m=altitude_m, engine=engine,
                solver=solver, flaps=flaps, slats=slats, phase=high_lift_setting, **more)


def _out(result: api.ApiResult, output: str = "json") -> dict[str, Any]:
    if output == "json":
        return result.data
    if output == "csv" and result.csv is None:
        raise api.ApiError("this result has no CSV table; use output='json' or 'text'")
    return {"schema": result.data.get("schema"), "format": output, "content": result.render(output)}


def _call(fn, *args, **kwargs):
    """Run a calculation with stdout pointed at stderr (stdout is the protocol channel)."""
    with _lock, contextlib.redirect_stdout(sys.stderr):
        return fn(*args, **kwargs)


async def _run(fn, *args, **kwargs):
    import anyio

    return await anyio.to_thread.run_sync(lambda: _call(fn, *args, **kwargs))


def documentation() -> str:
    try:
        return DOC_PATH.read_text(encoding="utf-8")
    except OSError:
        return "README_FOR_CLAUDE.md was not found next to the liftingline package."


def build_server():
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover - depends on the installation
        raise SystemExit('The MCP server needs the "mcp" package:  pip install "mcp>=1.25,<2"') from exc

    server = FastMCP("liftingline", instructions=INSTRUCTIONS)

    @server.tool()
    async def list_examples() -> dict[str, Any]:
        """The built-in example wings (name, span, area, aspect ratio, airfoils, default flight condition)."""
        return (await _run(api.examples)).data

    @server.tool()
    async def describe_project(example: Example = None, project_file: ProjectFile = None,
                               project: ProjectJson = None) -> dict[str, Any]:
        """Every input of a project (as saved in a project file) plus the derived wing geometry and flow.
        Edit the returned 'project' object and pass it back as 'project' to analyse a modified wing."""
        return (await _run(lambda: api.describe_project(_state(example, project_file, project)))).data

    @server.tool()
    async def analyze_wing(
        example: Example = None, project_file: ProjectFile = None, project: ProjectJson = None,
        alpha_deg: Alpha = None, speed: Speed = None, speed_unit: SpeedUnit = "m/s", altitude_m: Altitude = None,
        engine: Engine = None, solver: Solver = None, flaps: Flaps = None, slats: Slats = None,
        high_lift_setting: Setting = None,
        beta_deg: Annotated[float | None, Field(description="sideslip angle, deg")] = None,
        ground_height_m: Annotated[float | None, Field(description="quarter-chord height above the ground (ground effect), m; 0 = off")] = None,
        extra_cd0: Annotated[float | None, Field(description="parasite drag coefficient of the rest of the aircraft")] = None,
        include_spanwise: Annotated[bool, Field(description="include the spanwise distribution (cl, circulation ...)")] = False,
        output: Output = "json",
    ) -> dict[str, Any]:
        """Lift, drag (induced + profile), pitching/rolling/yawing moments, coefficients, L/D, span efficiency and
        validity warnings of the wing at one flight condition (the project's, with the given overrides)."""
        def work():
            st = _state(example, project_file, project, **_flight(
                alpha_deg, speed, speed_unit, altitude_m, engine, solver, flaps, slats, high_lift_setting,
                beta_deg=beta_deg, ground_height_m=ground_height_m, extra_cd0=extra_cd0))
            return api.analyze(st, spanwise=include_spanwise or output == "csv", backend=_get_backend())
        return _out(await _run(work), output)

    @server.tool()
    async def sweep(
        variable: Annotated[Literal["alpha", "speed", "altitude", "beta"], Field(description="what to vary")],
        start: Annotated[float, Field(description="first value: deg for alpha/beta, speed_unit for speed, m for altitude")],
        stop: Annotated[float, Field(description="last value (same unit as start)")],
        points: Annotated[int, Field(ge=2, le=200, description="number of values")] = 11,
        example: Example = None, project_file: ProjectFile = None, project: ProjectJson = None,
        alpha_deg: Alpha = None, speed: Speed = None, speed_unit: SpeedUnit = "m/s", altitude_m: Altitude = None,
        engine: Engine = None, solver: Solver = None, flaps: Flaps = None, slats: Slats = None,
        high_lift_setting: Setting = None, output: Output = "json",
    ) -> dict[str, Any]:
        """The analysis at evenly spaced values of one variable (a polar: CL, CD, Cm, L/D ... per point); an alpha
        sweep also reports the first section stall and the highest CL reached."""
        def work():
            st = _state(example, project_file, project, **_flight(
                alpha_deg, speed, speed_unit, altitude_m, engine, solver, flaps, slats, high_lift_setting))
            lo, hi = start, stop
            if variable == "speed":
                from .core import units

                lo, hi = (float(units.to_si("speed", v, speed_unit)) for v in (start, stop))
            return api.sweep(st, variable, lo, hi, points, backend=_get_backend())
        return _out(await _run(work), output)

    @server.tool()
    async def max_lift(
        example: Example = None, project_file: ProjectFile = None, project: ProjectJson = None,
        speed: Speed = None, speed_unit: SpeedUnit = "m/s", altitude_m: Altitude = None, engine: Engine = None,
        solver: Solver = None, flaps: Flaps = None, slats: Slats = None, high_lift_setting: Setting = None,
        weight_N: Annotated[float | None, Field(description="weight for the stall speeds, N")] = None,
        mass_kg: Annotated[float | None, Field(description="mass for the stall speeds, kg (instead of weight_N)")] = None,
        output: Output = "json",
    ) -> dict[str, Any]:
        """Maximum lift coefficient (CL_max), its angle, the first section stall and, given a weight, the stall speed.
        With flaps or slats deployed: clean wing and wing with the devices out, plus the handbook comparison and the
        takeoff/landing reference speeds. Takes some seconds (an angle-of-attack search)."""
        if weight_N is not None and mass_kg is not None:
            raise api.ApiError("give weight_N or mass_kg, not both")
        weight = weight_N if weight_N is not None else (mass_kg * 9.80665 if mass_kg is not None else None)

        def work():
            st = _state(example, project_file, project, **_flight(
                None, speed, speed_unit, altitude_m, engine, solver, flaps, slats, high_lift_setting))
            return api.max_lift(st, weight_N=weight, backend=_get_backend())
        return _out(await _run(work), output)

    @server.tool()
    async def trim(
        example: Example = None, project_file: ProjectFile = None, project: ProjectJson = None,
        target_CL: Annotated[float | None, Field(description="target lift coefficient")] = None,
        target_lift_N: Annotated[float | None, Field(description="target lift, N")] = None,
        mass_kg: Annotated[float | None, Field(description="mass to carry: lift = m g")] = None,
        solve_for: Annotated[Literal["alpha", "speed"], Field(description="alpha: angle at the given speed; speed: airspeed at the given angle")] = "alpha",
        alpha_deg: Alpha = None, speed: Speed = None, speed_unit: SpeedUnit = "m/s", altitude_m: Altitude = None,
        engine: Engine = None, solver: Solver = None, flaps: Flaps = None, slats: Slats = None,
        high_lift_setting: Setting = None, output: Output = "json",
    ) -> dict[str, Any]:
        """The angle of attack (or the airspeed) at which the wing makes a target CL, a target lift, or carries a
        mass in level flight. Exactly one of target_CL, target_lift_N, mass_kg."""
        def work():
            st = _state(example, project_file, project, **_flight(
                alpha_deg, speed, speed_unit, altitude_m, engine, solver, flaps, slats, high_lift_setting))
            return api.trim(st, CL=target_CL, lift_N=target_lift_N, mass_kg=mass_kg, solve_for=solve_for,
                            backend=_get_backend())
        return _out(await _run(work), output)

    @server.tool()
    async def run_scenario(
        example: Example = None, project_file: ProjectFile = None, project: ProjectJson = None,
        speed_unit: Annotated[Literal["m/s", "km/h", "kt", "mph", "ft/s"] | None, Field(description="unit of v_start, v_end, v_rotate and the per-speed rate (default: the project's scenario unit, mph)")] = None,
        v_start: Annotated[float | None, Field(description="start airspeed (default 0)")] = None,
        v_end: Annotated[float | None, Field(description="end airspeed (default 90 mph)")] = None,
        v_rotate: Annotated[float | None, Field(description="rotation airspeed: the angle of attack starts rising here (default 60 mph)")] = None,
        alpha_ground_deg: Annotated[float | None, Field(description="angle of attack before rotation, deg (default 0)")] = None,
        ramp: Annotated[Literal["target", "per_speed", "per_second"] | None, Field(description="rise after rotation: linear to ramp_target_deg at v_end; ramp_rate_per_speed deg per speed unit; or ramp_rate_per_second deg/s (needs time_model)")] = None,
        ramp_target_deg: Annotated[float | None, Field(description="angle at v_end for ramp='target' (default 10)")] = None,
        ramp_rate_per_speed: Annotated[float | None, Field(description="deg per speed_unit for ramp='per_speed'")] = None,
        ramp_rate_per_second: Annotated[float | None, Field(description="deg/s for ramp='per_second'")] = None,
        mass_kg: Annotated[float | None, Field(description="aircraft mass (weight m g; default 1157 kg)")] = None,
        time_model: Annotated[bool | None, Field(description="compute time, distance, acceleration of a takeoff roll")] = None,
        thrust_static_N: Annotated[float | None, Field(description="thrust at standstill, N (time model)")] = None,
        thrust_end_N: Annotated[float | None, Field(description="thrust at v_end, N (linear in between)")] = None,
        rolling_friction: Annotated[float | None, Field(description="rolling-friction coefficient (0.02 paved, 0.05 grass)")] = None,
        ground_height_m: Annotated[float | None, Field(description="wing height above the runway for ground effect, m; 0 = off")] = None,
        points: Annotated[int | None, Field(ge=5, le=400, description="analysis points along the run")] = None,
        altitude_m: Altitude = None, engine: Engine = None, solver: Solver = None, flaps: Flaps = None,
        slats: Slats = None, high_lift_setting: Setting = None,
        max_frames: Annotated[int, Field(ge=2, le=1000, description="at most this many rows in 'frames' (json)")] = 61,
        output: Output = "json",
    ) -> dict[str, Any]:
        """Forces along a run from a start to an end airspeed (e.g. a takeoff from 0 to 90 mph): angle of attack
        constant until v_rotate, then rising linearly. Reports when lift first equals weight (lift-off: airspeed,
        angle and - with time_model - time and distance), plus lift, weight, drag, thrust, acceleration per point.
        Unset inputs keep the project's saved scenario. Takes several seconds (one analysis per point)."""
        def work():
            st = _state(example, project_file, project, altitude_m=altitude_m, engine=engine, solver=solver,
                        flaps=flaps, slats=slats, phase=high_lift_setting)
            spec = api.scenario_spec(
                st.scenario, speed_unit=speed_unit, v_start=v_start, v_end=v_end, v_rotate=v_rotate,
                alpha_ground_deg=alpha_ground_deg, ramp=ramp, ramp_target_deg=ramp_target_deg,
                ramp_rate_deg_per_s=ramp_rate_per_second, ramp_rate_deg_per_speed_unit=ramp_rate_per_speed,
                mass_kg=mass_kg, time_model=time_model, thrust_static_N=thrust_static_N, thrust_end_N=thrust_end_N,
                rolling_friction=rolling_friction, ground_height_m=ground_height_m, points=points)
            return api.scenario(st, spec, max_frames=max_frames, backend=_get_backend())
        return _out(await _run(work), output)

    @server.tool()
    async def compare_real_aircraft(engine: Engine = "auto", output: Output = "json") -> dict[str, Any]:
        """The program against published handbook data (Cessna 172S and 152: stall speeds clean and with flaps,
        cruise). Takes a minute."""
        return _out(await _run(lambda: api.compare_aircraft(engine or "auto", backend=_get_backend())), output)

    @server.tool()
    async def export_stl(
        path: Annotated[str, Field(description="STL file to write (absolute path recommended)")],
        example: Example = None, project_file: ProjectFile = None, project: ProjectJson = None,
        quality: Annotated[Literal["coarse", "medium", "fine", "very_fine"], Field(description="mesh fineness")] = "fine",
        unit: Annotated[Literal["mm", "cm", "m", "in", "ft"], Field(description="length unit of the file")] = "mm",
        scale: Annotated[float, Field(gt=0, description="model scale (0.05 = 1:20)")] = 1.0,
        half: Annotated[bool, Field(description="right half only, closed at the root")] = False,
        ascii: Annotated[bool, Field(description="ASCII STL instead of binary")] = False,
    ) -> dict[str, Any]:
        """Write the wing as a watertight STL surface model (real airfoils, twist, sweep, dihedral, incidence)."""
        return (await _run(lambda: api.export_stl(_state(example, project_file, project), path, quality=quality,
                                                  unit=unit, scale=scale, half=half, ascii=ascii,
                                                  library=_library))).data

    @server.tool()
    async def self_test() -> dict[str, Any]:
        """The built-in validation suite (analytic benchmarks, cross-checks); 'passed' is false on any failure."""
        return (await _run(api.validate)).data

    @server.tool()
    def get_documentation() -> str:
        """README_FOR_CLAUDE.md: how to call the tools and what every output field means (units, conventions)."""
        return documentation()

    @server.resource(DOC_URI, name="readme-for-claude", mime_type="text/markdown",
                     description="How the calculator's outputs are structured (units, conventions, schemas)")
    def readme_for_claude() -> str:
        return documentation()

    return server


def main(library: str | None = None) -> int:
    global _library, _backend
    _library = library or os.environ.get("LIFTINGLINE_LIBRARY") or None
    _backend = None
    server = build_server()
    server.run("stdio")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
