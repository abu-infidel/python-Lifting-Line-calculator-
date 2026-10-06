"""Scenario: the wing's forces along a run from a start to an end airspeed (a takeoff run, for example).

The wing, airfoils, atmosphere, high-lift devices and extra drag come from an ordinary
:class:`~liftingline.core.analysis.AnalysisRequest`; the scenario replaces its airspeed and angle of attack by a schedule:

* the airspeed goes from ``v_start`` to ``v_end``;
* the angle of attack (of the fuselage reference line, like the Flow tab's; the wing's root incidence is added) is
  ``alpha_ground_deg`` until the rotation speed ``v_rotate`` and then rises linearly - by a rate in degrees per second
  (needs the time model), by a rate in degrees per unit of airspeed, or so as to reach a target angle at ``v_end``.

At every point the full lifting-line analysis gives lift, drag and moment; the weight ``m g`` is compared with the lift
and the **lift-off** point (lift first equal to the weight) is found to 0.1 % of the speed range by bisection.

Optional **time model** (a takeoff roll on a level runway, no wind): the aircraft accelerates with
``m dV/dt = T(V) - D - mu (W - L)`` while on the ground (``mu`` rolling friction, thrust varying linearly from its static
value to its value at ``v_end``), which gives time, distance and acceleration. After lift-off the friction is gone; the
flight path itself (climb) is not modelled - the run continues along the same schedule. Ground effect (image vortices of
the numerical solver) can be switched on with the height of the wing's quarter-chord line above the runway.

Below a small floor speed (10 % of ``v_end``) the coefficients are those at the floor speed and the forces scale with
the dynamic pressure, so the run can start from standstill.
"""

from __future__ import annotations

import csv
import io
import math
from dataclasses import asdict, dataclass, field, replace
from typing import Callable

import numpy as np

from . import units
from .analysis import AnalysisRequest, run

G0 = 9.80665
RAMP_PER_SECOND, RAMP_PER_SPEED, RAMP_TARGET = "per_second", "per_speed", "target"
RAMP_MODES = (RAMP_PER_SECOND, RAMP_PER_SPEED, RAMP_TARGET)
#: a station at this fraction of its cl_max counts as stalled (same threshold as the sweeps)
STALL_RATIO = 0.98
MAX_STEPS = 4000


class ScenarioError(ValueError):
    """A scenario that cannot be run (the message is meant for the user)."""


class ScenarioCancelled(Exception):
    pass


@dataclass
class ScenarioSpec:
    """The scenario inputs, SI units (m/s, kg, N, deg). Saved with the project."""

    v_start: float = 0.0
    v_end: float = 40.2336  # 90 mph
    v_rotate: float = 26.8224  # 60 mph
    alpha_ground_deg: float = 0.0
    ramp_mode: str = RAMP_TARGET
    ramp_rate_deg_s: float = 3.0
    ramp_rate_deg_per_mps: float = 0.75  # = 0.335 deg per mph
    ramp_target_deg: float = 10.0
    mass_kg: float = 1157.0  # 2550 lb
    time_model: bool = False
    thrust_static_N: float = 2400.0
    thrust_end_N: float = 1800.0
    rolling_friction: float = 0.02
    ground_effect: bool = False
    ground_height_m: float = 1.5
    points: int = 61
    speed_unit: str = "mph"  # display only

    def problems(self) -> list[str]:
        out = []
        if not (self.v_start >= 0.0 and self.v_end > self.v_start):
            out.append("the end speed must be above the start speed (and the start speed not negative)")
        if not (self.v_start <= self.v_rotate <= self.v_end):
            out.append("the rotation speed must lie between the start and the end speed")
        if self.ramp_mode not in RAMP_MODES:
            out.append(f"unknown angle-of-attack rise {self.ramp_mode!r}")
        if self.ramp_mode == RAMP_PER_SECOND and not self.time_model:
            out.append("a rise in degrees per second needs the time model (tick 'Compute time and distance')")
        if not (-15.0 <= self.alpha_ground_deg <= 25.0) or not (-15.0 <= self.ramp_target_deg <= 35.0):
            out.append("angles of attack must be between -15 and 35 degrees")
        if not (-20.0 <= self.ramp_rate_deg_s <= 20.0) or not (-10.0 <= self.ramp_rate_deg_per_mps <= 10.0):
            out.append("the rate of the angle-of-attack rise is out of range")
        if not self.mass_kg > 0.0:
            out.append("the mass must be positive")
        if self.time_model:
            if not (self.thrust_static_N > 0.0 and self.thrust_end_N >= 0.0):
                out.append("the thrust must be positive")
            if not (0.0 <= self.rolling_friction <= 0.5):
                out.append("the rolling-friction coefficient must be between 0 and 0.5")
        if self.ground_effect and not self.ground_height_m > 0.0:
            out.append("the height above the runway must be positive")
        if not (5 <= int(self.points) <= 400):
            out.append("the number of points must be between 5 and 400")
        if self.speed_unit not in units.units_of("speed"):
            out.append(f"unknown speed unit {self.speed_unit!r}")
        return out

    @property
    def weight_N(self) -> float:
        return self.mass_kg * G0

    def alpha_at(self, v: float, t_since_rotation: float | None = None) -> float:
        """Scheduled angle of attack, deg, at airspeed ``v`` (and time since rotation, for the per-second rise)."""
        if v < self.v_rotate:
            return self.alpha_ground_deg
        if self.ramp_mode == RAMP_PER_SPEED:
            return self.alpha_ground_deg + self.ramp_rate_deg_per_mps * (v - self.v_rotate)
        if self.ramp_mode == RAMP_TARGET:
            span = self.v_end - self.v_rotate
            if span <= 0.0:  # rotation at the end speed: no rise within the run
                return self.alpha_ground_deg
            f = min((v - self.v_rotate) / span, 1.0)
            return self.alpha_ground_deg + (self.ramp_target_deg - self.alpha_ground_deg) * f
        return self.alpha_ground_deg + self.ramp_rate_deg_s * (t_since_rotation or 0.0)

    def thrust_at(self, v: float) -> float:
        return self.thrust_static_N + (self.thrust_end_N - self.thrust_static_N) * min(max(v / self.v_end, 0.0), 1.0)

    def to_dict(self) -> dict:
        return asdict(self)


#: columns of a scenario result (SI): key, label, unit
COLUMNS = (
    ("t_s", "time", "s"),
    ("V_mps", "airspeed", "m/s"),
    ("alpha_deg", "angle of attack", "deg"),
    ("q_Pa", "dynamic pressure", "Pa"),
    ("L_N", "lift", "N"),
    ("W_N", "weight", "N"),
    ("L_minus_W_N", "lift - weight", "N"),
    ("L_over_W", "lift / weight", "-"),
    ("D_N", "drag", "N"),
    ("Di_N", "induced drag", "N"),
    ("Dp_N", "profile drag", "N"),
    ("Dx_N", "extra drag", "N"),
    ("CL", "CL", "-"),
    ("CD", "CD", "-"),
    ("Cm", "Cm", "-"),
    ("M_pitch_Nm", "pitching moment", "N m"),
    ("T_N", "thrust", "N"),
    ("F_roll_N", "rolling friction", "N"),
    ("F_net_N", "net force along the runway", "N"),
    ("a_mps2", "acceleration", "m/s^2"),
    ("x_m", "distance", "m"),
    ("cl_ratio", "highest cl / cl_max", "-"),
    ("on_ground", "on the ground", "0/1"),
    ("stalled", "a section stalled", "0/1"),
)


@dataclass
class ScenarioEvent:
    name: str
    t: float
    V: float
    x: float
    alpha_deg: float
    text: str = ""


@dataclass
class ScenarioResult:
    spec: ScenarioSpec
    data: dict[str, np.ndarray]
    events: list[ScenarioEvent] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    has_time: bool = False
    solver: str = ""
    engine: str = ""
    wing_area: float = math.nan

    @property
    def n(self) -> int:
        return int(self.data["V_mps"].size)

    def event(self, name: str) -> ScenarioEvent | None:
        return next((e for e in self.events if e.name == name), None)

    @property
    def liftoff(self) -> ScenarioEvent | None:
        return self.event("lift-off")

    def frame(self, i: int) -> dict[str, float]:
        return {k: float(v[i]) for k, v in self.data.items()}


# ===================================================================================== running
class _Aero:
    """Lifting-line analysis at (airspeed, angle of attack), with the floor-speed rule and a call counter."""

    def __init__(self, req: AnalysisRequest, spec: ScenarioSpec, cancelled) -> None:
        self.req = replace(req, convergence_check=False,
                           ground_height=spec.ground_height_m if spec.ground_effect else None)
        self.floor = max(0.1 * spec.v_end, 0.5)
        self.cancelled = cancelled
        self.calls = 0
        self.warnings: list[str] = []
        self.solver = self.engine = ""

    def __call__(self, v: float, alpha: float) -> dict[str, float]:
        if self.cancelled is not None and self.cancelled():
            raise ScenarioCancelled()
        vv = max(v, self.floor)
        r = run(replace(self.req, flow=replace(self.req.flow, V=vv, alpha_deg=alpha)))
        self.calls += 1
        self.solver, self.engine = r.solver_used, str(r.info.get("section_engine", ""))
        for w in r.warnings:
            if w not in self.warnings and not w.startswith("Sections reach") and not w.startswith("Close to stall"):
                self.warnings.append(w)
        t = r.totals
        k = (v / vv) ** 2  # below the floor speed: same coefficients, forces scale with q
        margin = r.load.cl_margin
        ratio = float(np.max(margin)) if np.any(np.isfinite(r.load.cl_max)) else math.nan
        return dict(q_Pa=t.q * k, L_N=t.L * k, D_N=t.D * k, Di_N=t.D_induced * k, Dp_N=t.D_profile * k,
                    Dx_N=t.D_extra * k, CL=t.CL, CD=t.CD, Cm=t.Cm, M_pitch_Nm=t.M_pitch * k, cl_ratio=ratio)


def _row(spec: ScenarioSpec, v: float, alpha: float, aero: dict, on_ground: bool) -> dict[str, float]:
    W = spec.weight_N
    L = aero["L_N"]
    T = spec.thrust_at(v) if spec.time_model else math.nan
    friction = spec.rolling_friction * max(W - L, 0.0) if (spec.time_model and on_ground) else (0.0 if spec.time_model else math.nan)
    net = T - aero["D_N"] - friction if spec.time_model else math.nan
    row = dict(V_mps=v, alpha_deg=alpha, W_N=W, L_minus_W_N=L - W, L_over_W=L / W, T_N=T, F_roll_N=friction,
               F_net_N=net, a_mps2=net / spec.mass_kg if spec.time_model else math.nan,
               on_ground=1.0 if on_ground else 0.0,
               stalled=1.0 if (math.isfinite(aero["cl_ratio"]) and aero["cl_ratio"] >= STALL_RATIO) else 0.0)
    row.update(aero)
    return row


def run_scenario(req: AnalysisRequest, spec: ScenarioSpec, *, progress: Callable[[int, int], None] | None = None,
                 cancelled: Callable[[], bool] | None = None) -> ScenarioResult:
    """Run the scenario on the wing / flow of ``req`` (its airspeed and angle of attack are replaced)."""
    msgs = spec.problems()
    if msgs:
        raise ScenarioError("; ".join(msgs))
    aero = _Aero(req, spec, cancelled)
    n = int(spec.points)
    expected = n + 12

    def tick() -> None:
        if progress is not None:
            progress(min(aero.calls, expected - 1), expected)

    if spec.ramp_mode == RAMP_PER_SECOND:
        rows, events = _run_per_second(spec, aero, tick)
    else:
        rows, events = _run_speed_schedule(spec, aero, tick, n)
    data = {k: np.array([r.get(k, math.nan) for r in rows], dtype=float) for k, _l, _u in COLUMNS}
    res = ScenarioResult(spec, data, events, aero.warnings, spec.time_model, aero.solver, aero.engine, req.wing.area)
    _add_summary_events(res)
    if progress is not None:
        progress(expected, expected)
    return res


def _speed_grid(spec: ScenarioSpec, n: int) -> np.ndarray:
    v = np.linspace(spec.v_start, spec.v_end, n)
    return np.unique(np.concatenate([v, [spec.v_rotate]]))


def _run_speed_schedule(spec: ScenarioSpec, aero: _Aero, tick, n: int):
    """Angle of attack scheduled on airspeed: evaluate on a speed grid, refine the lift-off, integrate time if asked."""
    rows: list[dict] = []
    lifted = False
    for v in _speed_grid(spec, n):
        a = spec.alpha_at(float(v))
        r = _row(spec, float(v), a, aero(float(v), a), on_ground=not lifted)
        rows.append(r)
        tick()
        if not lifted and r["L_N"] >= spec.weight_N:
            lifted = True
            if len(rows) > 1:  # refine where lift first equals the weight
                lo, hi = rows[-2]["V_mps"], r["V_mps"]
                for _ in range(10):
                    mid = 0.5 * (lo + hi)
                    am = spec.alpha_at(mid)
                    if aero(mid, am)["L_N"] >= spec.weight_N:
                        hi = mid
                    else:
                        lo = mid
                    tick()
                a_lof = spec.alpha_at(hi)
                lof = _row(spec, hi, a_lof, aero(hi, a_lof), on_ground=False)
                r["on_ground"] = 0.0
                rows.insert(len(rows) - 1, lof)
            else:
                r["on_ground"] = 0.0
            r["on_ground"] = 0.0
    for r in rows:  # friction applies only while on the ground (recompute after the lift-off refinement)
        if spec.time_model:
            on = r["on_ground"] > 0.5
            r["F_roll_N"] = spec.rolling_friction * max(spec.weight_N - r["L_N"], 0.0) if on else 0.0
            r["F_net_N"] = r["T_N"] - r["D_N"] - r["F_roll_N"]
            r["a_mps2"] = r["F_net_N"] / spec.mass_kg
    events = []
    if spec.time_model:
        rows, events = _integrate_over_speed(spec, rows)
    else:
        for r in rows:
            r["t_s"] = r["x_m"] = math.nan
    return rows, events


def _integrate_over_speed(spec: ScenarioSpec, rows: list[dict]):
    """t(V) = integral of dV/a and x(V) = integral of V dV/a (trapezoids in 1/a); stops where the thrust runs out."""
    events = []
    rows[0]["t_s"] = rows[0]["x_m"] = 0.0
    if rows[0]["a_mps2"] <= 0.0:
        raise ScenarioError(
            f"the thrust ({rows[0]['T_N']:.0f} N) cannot overcome drag and rolling friction "
            f"({rows[0]['D_N'] + rows[0]['F_roll_N']:.0f} N) at the start speed"
        )
    kept = [rows[0]]
    for prev, cur in zip(rows[:-1], rows[1:]):
        if cur["a_mps2"] <= 0.0:  # the aircraft stops accelerating between prev and cur
            f = prev["a_mps2"] / (prev["a_mps2"] - cur["a_mps2"])
            v_lim = prev["V_mps"] + f * (cur["V_mps"] - prev["V_mps"])
            events.append(ScenarioEvent("thrust limit", math.inf, v_lim, math.inf, spec.alpha_at(v_lim),
                                        f"net force reaches zero at {v_lim:.2f} m/s: the end speed is not reached"))
            break
        dv = cur["V_mps"] - prev["V_mps"]
        cur["t_s"] = prev["t_s"] + 0.5 * dv * (1.0 / prev["a_mps2"] + 1.0 / cur["a_mps2"])
        cur["x_m"] = prev["x_m"] + 0.5 * dv * (prev["V_mps"] / prev["a_mps2"] + cur["V_mps"] / cur["a_mps2"])
        kept.append(cur)
    return kept, events


def _run_per_second(spec: ScenarioSpec, aero: _Aero, tick):
    """Rise in degrees per second: speed-scheduled up to rotation, then a time-stepped run (Heun) to the end speed."""
    pre = replace(spec, v_end=spec.v_rotate if spec.v_rotate > spec.v_start else spec.v_start + 1e-6,
                  ramp_mode=RAMP_TARGET)
    pre_rows: list[dict] = []
    if spec.v_rotate > spec.v_start:
        n_pre = max(5, int(round(spec.points * (spec.v_rotate - spec.v_start) / (spec.v_end - spec.v_start))))
        for v in np.linspace(spec.v_start, spec.v_rotate, n_pre):
            a = spec.alpha_ground_deg
            pre_rows.append(_row(spec, float(v), a, aero(float(v), a), on_ground=True))
            tick()
        pre_rows, events = _integrate_over_speed(pre, pre_rows)
        if events:  # thrust ran out before rotation
            return pre_rows, events
    else:
        a = spec.alpha_ground_deg
        r = _row(spec, spec.v_start, a, aero(spec.v_start, a), on_ground=True)
        r["t_s"] = r["x_m"] = 0.0
        pre_rows = [r]
    rows = pre_rows
    t_rot = rows[-1]["t_s"]
    dt = min(0.25, 0.5 / max(abs(spec.ramp_rate_deg_s), 1e-6))
    lifted = any(r["on_ground"] < 0.5 for r in rows)
    cur = rows[-1]
    events: list[ScenarioEvent] = []

    def state_row(t: float, v: float, x: float, on_ground: bool) -> dict:
        a = spec.alpha_at(v, t - t_rot)
        r = _row(spec, v, a, aero(v, a), on_ground)
        r["t_s"], r["x_m"] = t, x
        tick()
        return r

    for _ in range(MAX_STEPS):
        if cur["V_mps"] >= spec.v_end:
            break
        a1 = cur["a_mps2"]
        if a1 <= 0.0:
            events.append(ScenarioEvent("thrust limit", cur["t_s"], cur["V_mps"], cur["x_m"], cur["alpha_deg"],
                                        "net force reaches zero: the end speed is not reached"))
            break
        t1 = cur["t_s"] + dt
        pred = state_row(t1, cur["V_mps"] + a1 * dt, cur["x_m"] + cur["V_mps"] * dt, not lifted)
        v_new = cur["V_mps"] + 0.5 * (a1 + pred["a_mps2"]) * dt
        x_new = cur["x_m"] + 0.5 * (cur["V_mps"] + v_new) * dt
        if v_new >= spec.v_end:  # last, shortened step onto the end speed
            frac = (spec.v_end - cur["V_mps"]) / max(v_new - cur["V_mps"], 1e-12)
            t1, x_new, v_new = cur["t_s"] + frac * dt, cur["x_m"] + frac * (x_new - cur["x_m"]), spec.v_end
        new = state_row(t1, v_new, x_new, not lifted)
        if not lifted and new["L_N"] >= spec.weight_N:
            lifted = True
            new["on_ground"] = 0.0
            new["F_roll_N"] = 0.0
            new["F_net_N"] = new["T_N"] - new["D_N"]
            new["a_mps2"] = new["F_net_N"] / spec.mass_kg
        rows.append(new)
        cur = new
    else:
        events.append(ScenarioEvent("step limit", cur["t_s"], cur["V_mps"], cur["x_m"], cur["alpha_deg"],
                                    "the run did not reach the end speed within the step limit"))
    return rows, events


def _add_summary_events(res: ScenarioResult) -> None:
    d, spec = res.data, res.spec
    t = d["t_s"]

    def at(i: int, name: str, text: str) -> ScenarioEvent:
        return ScenarioEvent(name, float(t[i]), float(d["V_mps"][i]), float(d["x_m"][i]), float(d["alpha_deg"][i]), text)

    ev = [at(0, "start", "start of the run")]
    rot = np.where(d["V_mps"] >= spec.v_rotate - 1e-9)[0]
    if rot.size:
        ev.append(at(int(rot[0]), "rotation", "the angle of attack starts to rise"))
    above = np.where(d["L_N"] >= spec.weight_N * (1.0 - 1e-9))[0]
    if above.size:
        i = int(above[0])
        if i > 0 and d["L_N"][i] > spec.weight_N:  # interpolate the exact crossing between two frames
            j = i - 1
            f = (spec.weight_N - d["L_N"][j]) / (d["L_N"][i] - d["L_N"][j])
            vals = {k: float(d[k][j] + f * (d[k][i] - d[k][j])) for k in ("t_s", "V_mps", "x_m", "alpha_deg")}
            ev.append(ScenarioEvent("lift-off", vals["t_s"], vals["V_mps"], vals["x_m"], vals["alpha_deg"],
                                    "lift equals the weight"))
        else:
            ev.append(at(i, "lift-off", "lift equals the weight"))
    stalled = np.where(d["stalled"] > 0.5)[0]
    if stalled.size:
        ev.append(at(int(stalled[0]), "stall", "a wing section reaches its maximum lift"))
    ev.append(at(res.n - 1, "end", "end of the run"))
    res.events = ev + [e for e in res.events if e.name not in {x.name for x in ev}]
    res.events.sort(key=lambda e: (e.V, 0 if e.name == "start" else 1))
    if res.liftoff is None:
        res.warnings.insert(0, f"No lift-off: the lift stays below the weight up to {d['V_mps'][-1]:.2f} m/s "
                               f"(highest lift/weight {np.nanmax(d['L_over_W']):.2f}).")


# ===================================================================================== export
def _speed(v: float, unit: str) -> float:
    return float(units.from_si("speed", v, unit))


def summary_lines(res: ScenarioResult, speed_unit: str | None = None) -> list[str]:
    s, u = res.spec, speed_unit or res.spec.speed_unit
    lines = [
        f"Scenario: {_speed(s.v_start, u):.1f} -> {_speed(s.v_end, u):.1f} {u}, rotation at {_speed(s.v_rotate, u):.1f} {u}, "
        f"angle {s.alpha_ground_deg:.1f} deg before rotation, rise: "
        + {RAMP_PER_SECOND: f"{s.ramp_rate_deg_s:g} deg/s",
           RAMP_PER_SPEED: f"{s.ramp_rate_deg_per_mps / _speed(1.0, u):.4g} deg per {u}",
           RAMP_TARGET: f"to {s.ramp_target_deg:g} deg at the end speed"}[s.ramp_mode],
        f"Mass {s.mass_kg:.1f} kg (weight {s.weight_N:.0f} N)"
        + (f"; thrust {s.thrust_static_N:.0f} N static, {s.thrust_end_N:.0f} N at the end speed, rolling friction {s.rolling_friction:g}"
           if s.time_model else "; no time model (forces against airspeed)")
        + (f"; ground effect at {s.ground_height_m:g} m" if s.ground_effect else ""),
        f"Solver {res.solver}, section data {res.engine}",
    ]
    for e in res.events:
        when = f"t = {e.t:.2f} s, distance {e.x:.1f} m, " if (res.has_time and math.isfinite(e.t)) else ""
        lines.append(f"  {e.name:12s} {when}airspeed {_speed(e.V, u):.2f} {u}, angle of attack {e.alpha_deg:.2f} deg"
                     + (f" - {e.text}" if e.text else ""))
    lines += [f"  note: {w}" for w in res.warnings]
    return lines


def to_csv(res: ScenarioResult, speed_unit: str | None = None) -> str:
    """CSV: ``#`` comment lines with the inputs and events, then one row per point (SI, plus airspeed in the unit)."""
    u = speed_unit or res.spec.speed_unit
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    for line in summary_lines(res, u):  # written raw (not quoted), so readers can skip them: read_csv(comment="#")
        buf.write("# " + " ".join(line.split()) + "\n")
    keys = [k for k, _l, _u in COLUMNS if not (k in ("t_s", "x_m", "T_N", "F_roll_N", "F_net_N", "a_mps2") and not res.has_time)]
    head = []
    for k in keys:
        head.append(k)
        if k == "V_mps" and u != "m/s":  # the airspeed also in the chosen unit
            head.append(f"V_{u.replace('/', '_per_')}")
    w.writerow(head)
    for i in range(res.n):
        row = []
        for k in keys:
            row.append(repr(float(res.data[k][i])))
            if k == "V_mps" and u != "m/s":
                row.append(repr(_speed(float(res.data[k][i]), u)))
        w.writerow(row)
    return buf.getvalue()


def to_text(res: ScenarioResult, speed_unit: str | None = None) -> str:
    """Readable report: the summary and a fixed-width table."""
    u = speed_unit or res.spec.speed_unit
    cols = [("t_s", "t [s]", "{:8.2f}"), ("V_mps", f"V [{u}]", "{:9.2f}"), ("alpha_deg", "alpha", "{:7.2f}"),
            ("L_N", "lift [N]", "{:10.0f}"), ("L_minus_W_N", "L-W [N]", "{:10.0f}"), ("D_N", "drag [N]", "{:9.1f}"),
            ("CL", "CL", "{:7.3f}"), ("T_N", "thrust", "{:8.0f}"), ("a_mps2", "a [m/s2]", "{:9.3f}"),
            ("x_m", "x [m]", "{:8.1f}"), ("cl_ratio", "cl/clmax", "{:8.2f}"), ("on_ground", "ground", "{:>7s}")]
    if not res.has_time:
        cols = [c for c in cols if c[0] not in ("t_s", "T_N", "a_mps2", "x_m")]
    lines = summary_lines(res, u) + [""]
    widths = [len(fmt.format(0.0)) if "s}" not in fmt else 7 for _k, _h, fmt in cols]
    lines.append(" ".join(h.rjust(wd) for (_k, h, _f), wd in zip(cols, widths)))
    for i in range(res.n):
        cells = []
        for k, _h, fmt in cols:
            val = float(res.data[k][i])
            if k == "V_mps":
                val = _speed(val, u)
            if k == "on_ground":
                cells.append(("yes" if val > 0.5 else "no").rjust(7))
            else:
                cells.append(fmt.format(val))
        lines.append(" ".join(cells))
    return "\n".join(lines) + "\n"


def result_to_dict(res: ScenarioResult, max_frames: int | None = None) -> dict:
    """JSON-ready dictionary (used by the command line and the MCP server)."""
    idx = np.arange(res.n)
    if max_frames is not None and res.n > max_frames:
        idx = np.unique(np.round(np.linspace(0, res.n - 1, max_frames)).astype(int))

    def clean(x: float):
        return None if not math.isfinite(x) else x

    return {
        "schema": "liftingline.scenario/1",
        "inputs": res.spec.to_dict(),
        "solver": res.solver,
        "section_data": res.engine,
        "has_time": res.has_time,
        "weight_N": res.spec.weight_N,
        "events": [{"name": e.name, "t_s": clean(e.t), "airspeed_mps": e.V, "distance_m": clean(e.x),
                    "alpha_deg": e.alpha_deg, "text": e.text} for e in res.events],
        "warnings": list(res.warnings),
        "columns": {k: {"label": lab, "unit": un} for k, lab, un in COLUMNS},
        "frames": [{k: clean(float(res.data[k][i])) for k, _l, _u in COLUMNS} for i in idx],
    }


__all__ = ["ScenarioSpec", "ScenarioResult", "ScenarioEvent", "ScenarioError", "ScenarioCancelled", "COLUMNS",
           "RAMP_PER_SECOND", "RAMP_PER_SPEED", "RAMP_TARGET", "RAMP_MODES", "run_scenario", "to_csv", "to_text",
           "summary_lines", "result_to_dict", "G0"]
