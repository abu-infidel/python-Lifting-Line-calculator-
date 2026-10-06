"""Command-line entry point:  ``python -m liftingline``  (or ``python main.py``).

Without a command the GUI starts (as before); ``--validate`` and ``--version`` are unchanged. Commands run the
calculator without a window and print a report (``--format text``), JSON (``--format json``) or CSV::

    python -m liftingline analyze --example "Cessna 172S" --speed 100 --speed-unit kt --alpha 2
    python -m liftingline sweep my_wing.json --variable alpha --start -4 --stop 16 -n 11 --format csv -o polar.csv
    python -m liftingline scenario --example "Cessna 172S" --speed-unit mph --v-end 90 --v-rotate 60 --time-model
    python -m liftingline mcp            # MCP server on stdio, for AI agents

``python -m liftingline <command> --help`` lists every option; README_FOR_CLAUDE.md describes the outputs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__

# ============================================================================================ options
def _project_options(p: argparse.ArgumentParser, flight: bool = True) -> None:
    g = p.add_argument_group("project")
    g.add_argument("project", nargs="?", help="project file (File > Save in the GUI) or an example name; "
                                              "default: the GUI's start-up wing")
    g.add_argument("-e", "--example", help="built-in example (name, file name or a unique part: 'Cessna 172S')")
    g.add_argument("--library", help="folder of .dat airfoil files for airfoil names that need it")
    g.add_argument("--engine", choices=("auto", "neuralfoil", "builtin"),
                   help="section data: NeuralFoil (max fidelity) or the built-in engine (fast)")
    g.add_argument("--solver", choices=("auto", "classical", "numerical"))
    g.add_argument("--altitude", type=float, metavar="M", help="ISA altitude, m")
    g.add_argument("--altitude-ft", type=float, metavar="FT", help="ISA altitude, ft")
    g.add_argument("--flaps", action=argparse.BooleanOptionalAction, default=None,
                   help="deploy (or retract) the project's trailing-edge flaps")
    g.add_argument("--slats", action=argparse.BooleanOptionalAction, default=None,
                   help="deploy (or retract) the project's leading-edge slats")
    g.add_argument("--high-lift-setting", choices=("takeoff", "landing"), help="flap / slat setting")
    if flight:
        g = p.add_argument_group("flight condition (replaces the project's)")
        g.add_argument("--alpha", type=float, metavar="DEG", help="angle of attack of the fuselage reference line")
        g.add_argument("--speed", type=float, help="true airspeed in --speed-unit")
        g.add_argument("--mach", type=float, help="Mach number (instead of --speed)")
        g.add_argument("--beta", type=float, metavar="DEG", help="sideslip angle")
        g.add_argument("--extra-cd0", type=float, help="parasite drag coefficient of the rest of the aircraft")
        g.add_argument("--ground-height", type=float, metavar="M",
                       help="quarter-chord height above the ground (ground effect); 0 switches it off")
    p.add_argument("--speed-unit", default=None, choices=("m/s", "km/h", "kt", "mph", "ft/s"),
                   help="unit of the speeds given on the command line (default m/s; scenario: the project's, mph)")


def _output_options(p: argparse.ArgumentParser, formats=("text", "json")) -> None:
    g = p.add_argument_group("output")
    g.add_argument("-f", "--format", choices=formats, default="text")
    g.add_argument("-o", "--output", help="write to this file instead of the terminal")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="liftingline",
        description="Lifting-line wing calculator (Prandtl LLT). Without a command the GUI starts.",
        epilog="Run 'liftingline <command> --help' for the options of a command. Outputs: README_FOR_CLAUDE.md.",
    )
    parser.add_argument("--version", action="store_true", help="print the version and exit")
    parser.add_argument("--validate", action="store_true",
                        help="run the built-in validation suite (analytic benchmarks) and exit")
    sub = parser.add_subparsers(dest="command", metavar="command")

    p = sub.add_parser("analyze", help="lift, drag and moments at one flight condition",
                       description="Lift, drag and moments of the wing at the project's (or the given) flight condition.")
    _project_options(p)
    p.add_argument("--no-spanwise", action="store_true", help="leave the spanwise distribution out of the JSON")
    _output_options(p, ("text", "json", "csv"))

    p = sub.add_parser("sweep", help="the analysis over a range of one variable (polar)",
                       description="Sweep alpha [deg], speed [--speed-unit], altitude [m], beta [deg] or the "
                                   "deflection [deg] of a control surface.")
    _project_options(p)
    p.add_argument("--variable", default="alpha", choices=("alpha", "speed", "altitude", "beta", "deflection"))
    p.add_argument("--start", type=float, required=True)
    p.add_argument("--stop", type=float, required=True)
    p.add_argument("-n", "--points", type=int, default=11)
    p.add_argument("--control", help="control surface name (deflection sweeps)")
    _output_options(p, ("text", "json", "csv"))

    p = sub.add_parser("max-lift", help="maximum lift, first stall, stall speed (clean and with flaps/slats)",
                       description="CL_max and the first section stall; with --flaps/--slats the clean wing and the "
                                   "wing with the devices out (takeoff / landing stall analysis).")
    _project_options(p)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--weight", type=float, metavar="N", help="weight for the stall speeds, N")
    g.add_argument("--mass", type=float, metavar="KG", help="mass for the stall speeds, kg")
    _output_options(p)

    p = sub.add_parser("trim", help="angle of attack or airspeed for a target CL, lift or mass",
                       description="The angle of attack (at the given airspeed) or the airspeed (at the given angle "
                                   "of attack) at which the wing makes a target CL, lift, or carries a mass.")
    _project_options(p)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--cl", type=float, help="target lift coefficient")
    g.add_argument("--lift", type=float, metavar="N", help="target lift, N")
    g.add_argument("--mass", type=float, metavar="KG", help="mass to carry (lift = m g), kg")
    p.add_argument("--solve-for", choices=("alpha", "speed"), default="alpha")
    _output_options(p)

    p = sub.add_parser("scenario", help="forces along a run from a start to an end airspeed (takeoff run)",
                       description="Forces along an airspeed schedule: angle of attack constant until the rotation "
                                   "speed, then a linear rise; lift-off where lift = weight; optional time model. "
                                   "Unset options keep the project's saved scenario (defaults: 0 -> 90 mph, rotation "
                                   "at 60 mph, 0 deg rising to 10 deg, 1157 kg).")
    _project_options(p, flight=False)
    g = p.add_argument_group("scenario (speeds in --speed-unit)")
    g.add_argument("--v-start", type=float)
    g.add_argument("--v-end", type=float)
    g.add_argument("--v-rotate", type=float, help="rotation airspeed: the angle of attack starts rising here")
    g.add_argument("--alpha-ground", type=float, metavar="DEG", help="angle of attack before rotation")
    g.add_argument("--ramp", choices=("target", "per_speed", "per_second"),
                   help="rise after rotation: to a target angle at the end speed, degrees per speed unit, or degrees "
                        "per second (needs --time-model)")
    g.add_argument("--target", type=float, metavar="DEG", help="angle at the end speed (--ramp target)")
    g.add_argument("--rate", type=float, help="rise rate: deg per --speed-unit (per_speed) or deg/s (per_second)")
    g.add_argument("--mass", type=float, metavar="KG")
    g.add_argument("--time-model", action=argparse.BooleanOptionalAction, default=None,
                   help="compute time, distance and acceleration (takeoff roll)")
    g.add_argument("--thrust-static", type=float, metavar="N")
    g.add_argument("--thrust-end", type=float, metavar="N", help="thrust at the end speed (linear in between)")
    g.add_argument("--friction", type=float, help="rolling-friction coefficient (0.02 paved, 0.05 grass)")
    g.add_argument("--ground-height", type=float, metavar="M", help="wing height above the runway; 0 = no ground effect")
    g.add_argument("--points", type=int)
    p.add_argument("--max-frames", type=int, default=None, help="JSON: thin the rows out to at most this many")
    _output_options(p, ("text", "json", "csv"))

    p = sub.add_parser("examples", help="list the built-in example wings")
    _output_options(p)

    p = sub.add_parser("show", help="every input of a project and its wing geometry")
    _project_options(p)
    _output_options(p)

    p = sub.add_parser("compare-aircraft", help="the program against handbook data of real aircraft")
    p.add_argument("--engine", choices=("auto", "neuralfoil", "builtin"), default="auto")
    _output_options(p)

    p = sub.add_parser("stl", help="write the wing as an STL 3-D model")
    _project_options(p, flight=False)
    p.add_argument("--stl", required=True, metavar="FILE", help="the STL file to write")
    p.add_argument("--quality", choices=("coarse", "medium", "fine", "very_fine"), default="fine")
    p.add_argument("--unit", choices=("mm", "cm", "m", "in", "ft"), default="mm")
    p.add_argument("--scale", type=float, default=1.0, help="model scale (0.05 = 1:20)")
    p.add_argument("--half", action="store_true", help="right half only, closed at the root")
    p.add_argument("--ascii", action="store_true", help="ASCII STL instead of binary")
    p.add_argument("--no-incidence", action="store_true", help="do not tilt the wing by its root incidence")
    _output_options(p)

    p = sub.add_parser("validate", help="the built-in self-test (exit code 1 on a failure)")
    _output_options(p)

    p = sub.add_parser("mcp", help="run the MCP server on stdio (for AI agents such as Claude)")
    p.add_argument("--library", help="folder of .dat airfoil files for airfoil names that need it")
    return parser


# ============================================================================================ running
def _state(a: argparse.Namespace):
    from . import api

    st = api.load_project(a.project, example=a.example)
    altitude = a.altitude if a.altitude is not None else (a.altitude_ft * 0.3048 if a.altitude_ft is not None else None)
    kw = dict(engine=a.engine, solver=a.solver, altitude_m=altitude, flaps=a.flaps, slats=a.slats,
              phase=a.high_lift_setting)
    if hasattr(a, "alpha"):
        kw.update(alpha_deg=a.alpha, speed=a.speed, speed_unit=a.speed_unit or "m/s", mach=a.mach, beta_deg=a.beta,
                  extra_cd0=a.extra_cd0, ground_height_m=a.ground_height)
    return api.apply_overrides(st, **kw)


def _emit(result, a: argparse.Namespace) -> None:
    text = result.render(a.format)
    if a.output:
        Path(a.output).write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
        print(f"wrote {a.output}", file=sys.stderr)
    else:
        sys.stdout.write(text if text.endswith("\n") else text + "\n")


def run_command(a: argparse.Namespace) -> int:
    from . import api

    cmd = a.command
    if cmd == "mcp":
        from .mcp_server import main as mcp_main

        return mcp_main(library=a.library)
    if cmd == "examples":
        _emit(api.examples(), a)
        return 0
    if cmd == "compare-aircraft":
        _emit(api.compare_aircraft(a.engine), a)
        return 0
    if cmd == "validate":
        res = api.validate()
        _emit(res, a)
        return 0 if res.data["passed"] else 1
    st = _state(a)
    be = api.make_backend(a.library)
    if cmd == "show":
        res = api.describe_project(st)
    elif cmd == "analyze":
        res = api.analyze(st, spanwise=not a.no_spanwise, backend=be)
    elif cmd == "sweep":
        start, stop = a.start, a.stop
        if a.variable == "speed" and a.speed_unit:
            from .core import units

            start, stop = (float(units.to_si("speed", v, a.speed_unit)) for v in (start, stop))
        res = api.sweep(st, a.variable, start, stop, a.points, control=a.control, backend=be)
    elif cmd == "max-lift":
        weight = a.weight if a.weight is not None else (a.mass * 9.80665 if a.mass is not None else None)
        res = api.max_lift(st, weight_N=weight, backend=be)
    elif cmd == "trim":
        res = api.trim(st, CL=a.cl, lift_N=a.lift, mass_kg=a.mass, solve_for=a.solve_for, backend=be)
    elif cmd == "scenario":
        per_speed = a.rate if a.ramp == "per_speed" or (a.ramp is None and st.scenario.ramp_mode == "per_speed") else None
        per_second = a.rate if per_speed is None else None
        spec = api.scenario_spec(
            st.scenario, speed_unit=a.speed_unit, v_start=a.v_start, v_end=a.v_end, v_rotate=a.v_rotate,
            alpha_ground_deg=a.alpha_ground, ramp=a.ramp, ramp_target_deg=a.target, ramp_rate_deg_per_s=per_second,
            ramp_rate_deg_per_speed_unit=per_speed, mass_kg=a.mass, time_model=a.time_model,
            thrust_static_N=a.thrust_static, thrust_end_N=a.thrust_end, rolling_friction=a.friction,
            ground_height_m=a.ground_height, points=a.points,
        )
        res = api.scenario(st, spec, max_frames=a.max_frames, backend=be)
    elif cmd == "stl":
        res = api.export_stl(st, a.stl, quality=a.quality, unit=a.unit, scale=a.scale, half=a.half, ascii=a.ascii,
                             include_incidence=not a.no_incidence, library=a.library)
    else:  # pragma: no cover - argparse rejects unknown commands
        raise api.ApiError(f"unknown command {cmd!r}")
    _emit(res, a)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.version:
        print(f"liftingline {__version__}")
        return 0
    if args.validate:
        from .core.validate import main as validate_main

        return validate_main()
    if args.command:
        from .api import ApiError

        try:
            return run_command(args)
        except ApiError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

    from .gui.app import run_gui

    return run_gui()


if __name__ == "__main__":
    sys.exit(main())
