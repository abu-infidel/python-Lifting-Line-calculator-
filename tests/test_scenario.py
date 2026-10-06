"""Scenario runs (forces along a start-to-end airspeed schedule): exact cases first, then a real wing."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import replace

import numpy as np
import pytest

from liftingline.core.analysis import AnalysisRequest
from liftingline.core.flow import FlowCondition
from liftingline.core.scenario import (
    G0,
    RAMP_PER_SECOND,
    RAMP_PER_SPEED,
    RAMP_TARGET,
    ScenarioError,
    ScenarioSpec,
    result_to_dict,
    run_scenario,
    to_csv,
    to_text,
)
from liftingline.core.section import LinearSection
from liftingline.core.wing import Wing
from liftingline.io.backend import Backend
from liftingline.io.presets import CESSNA_172S, PRESETS
from liftingline.io.project import ProjectState

MPH = 0.44704


def _linear_req(alpha_L0_deg: float = -3.0, cd0: float = 0.0) -> AnalysisRequest:
    wing = Wing.rectangular(10.0, 1.5)
    sec = LinearSection(a0=2 * math.pi, alpha_L0=math.radians(alpha_L0_deg), cd0=cd0)
    return AnalysisRequest(wing, FlowCondition.from_isa(0.0, V=30.0), sec, nonlinear=False, compressibility=False,
                           convergence_check=False)


def test_lift_off_speed_of_a_constant_angle_run_is_the_textbook_value():
    req = _linear_req()
    spec = ScenarioSpec(v_start=0.0, v_end=60.0, v_rotate=60.0, alpha_ground_deg=2.0, mass_kg=1000.0, points=31)
    res = run_scenario(req, spec)
    d = res.data
    S, rho = req.wing.area, req.flow.rho
    CL = float(d["CL"][-1])
    assert np.allclose(d["CL"], CL, rtol=1e-9)  # linear sections: CL does not depend on the speed
    assert np.allclose(d["L_N"], 0.5 * rho * d["V_mps"] ** 2 * S * CL, rtol=1e-9)  # L = q S CL at every point
    assert d["L_N"][0] == 0.0  # standstill
    v_lof = math.sqrt(2 * 1000.0 * G0 / (rho * S * CL))
    assert res.liftoff is not None and res.liftoff.V == pytest.approx(v_lof, rel=1e-3)
    on = d["on_ground"] > 0.5
    assert np.all(d["V_mps"][on] < v_lof + 1e-9) and np.all(d["V_mps"][~on] >= v_lof * (1 - 1e-3))


def test_time_and_distance_of_a_constant_thrust_run_are_exact():
    req = _linear_req(alpha_L0_deg=0.0)  # zero angle, zero camber: no lift and no drag at all
    m, T, mu = 900.0, 3000.0, 0.03
    spec = ScenarioSpec(v_start=0.0, v_end=40.0, v_rotate=40.0, alpha_ground_deg=0.0, mass_kg=m, time_model=True,
                        thrust_static_N=T, thrust_end_N=T, rolling_friction=mu, points=21)
    res = run_scenario(req, spec)
    a = (T - mu * m * G0) / m
    assert np.allclose(res.data["a_mps2"], a, rtol=1e-9)
    assert res.data["t_s"][-1] == pytest.approx(40.0 / a, rel=1e-9)
    assert res.data["x_m"][-1] == pytest.approx(40.0**2 / (2 * a), rel=1e-9)
    assert res.liftoff is None and any("No lift-off" in w for w in res.warnings)


def test_angle_of_attack_schedules():
    req = _linear_req()
    base = ScenarioSpec(v_start=0.0, v_end=40.0, v_rotate=25.0, alpha_ground_deg=1.0, mass_kg=5000.0, points=17)
    per_speed = run_scenario(req, replace(base, ramp_mode=RAMP_PER_SPEED, ramp_rate_deg_per_mps=0.4)).data
    v, a = per_speed["V_mps"], per_speed["alpha_deg"]
    assert np.allclose(a[v < 25.0], 1.0) and np.allclose(a[v >= 25.0], 1.0 + 0.4 * (v[v >= 25.0] - 25.0))
    target = run_scenario(req, replace(base, ramp_mode=RAMP_TARGET, ramp_target_deg=9.0)).data
    assert target["alpha_deg"][-1] == pytest.approx(9.0) and target["alpha_deg"][0] == 1.0
    with pytest.raises(ScenarioError, match="time model"):
        run_scenario(req, replace(base, ramp_mode=RAMP_PER_SECOND))
    timed = replace(base, ramp_mode=RAMP_PER_SECOND, ramp_rate_deg_s=2.0, time_model=True, thrust_static_N=8000.0,
                    thrust_end_N=8000.0, rolling_friction=0.0)
    d = run_scenario(req, timed).data
    after = d["V_mps"] > 25.0 + 1e-6
    t_rot = float(np.interp(25.0, d["V_mps"], d["t_s"]))
    assert np.allclose(d["alpha_deg"][after], 1.0 + 2.0 * (d["t_s"][after] - t_rot), atol=1e-6)
    assert d["V_mps"][-1] == pytest.approx(40.0) and np.all(np.diff(d["t_s"]) > 0)


def test_thrust_that_cannot_beat_the_drag_stops_the_run():
    req = _linear_req(cd0=0.05)
    spec = ScenarioSpec(v_start=0.0, v_end=60.0, v_rotate=60.0, alpha_ground_deg=0.0, mass_kg=900.0, time_model=True,
                        thrust_static_N=800.0, thrust_end_N=800.0, rolling_friction=0.02, points=31)
    res = run_scenario(req, spec)
    lim = res.event("thrust limit")
    assert lim is not None and 0.0 < lim.V < 60.0 and res.data["V_mps"][-1] < 60.0
    with pytest.raises(ScenarioError, match="cannot overcome"):
        run_scenario(req, replace(spec, thrust_static_N=50.0, thrust_end_N=50.0))


def test_invalid_scenarios_are_refused_with_readable_messages():
    req = _linear_req()
    with pytest.raises(ScenarioError, match="end speed"):
        run_scenario(req, ScenarioSpec(v_start=30.0, v_end=20.0))
    with pytest.raises(ScenarioError, match="rotation speed"):
        run_scenario(req, ScenarioSpec(v_start=0.0, v_end=20.0, v_rotate=30.0))
    with pytest.raises(ScenarioError, match="mass"):
        run_scenario(req, ScenarioSpec(mass_kg=0.0))


# ================================================================================ a real wing
@pytest.fixture(scope="module")
def cessna():
    st = PRESETS[CESSNA_172S]()
    st.solver.engine = "builtin"
    st.flow.altitude_m = 0.0
    return st


def test_the_users_example_takeoff_run(cessna):
    """0 to 90 mph, angle 0 until 60 mph then rising linearly: when does the lift overcome the weight?"""
    be = Backend()
    spec = ScenarioSpec(v_start=0.0, v_end=90 * MPH, v_rotate=60 * MPH, alpha_ground_deg=0.0, ramp_mode=RAMP_TARGET,
                        ramp_target_deg=10.0, mass_kg=1157.0, time_model=True, thrust_static_N=2400.0,
                        thrust_end_N=1800.0, rolling_friction=0.02, points=61)
    res = run_scenario(be.build_request(cessna), spec)
    lof = res.liftoff
    assert lof is not None and 60 * MPH < lof.V < 90 * MPH  # not before rotation: at 0 deg the wing needs ~150 mph
    i = int(np.argmin(np.abs(res.data["V_mps"] - lof.V)))
    assert res.data["L_N"][i] == pytest.approx(spec.weight_N, rel=0.01)
    assert 5.0 < lof.t < 60.0 and 50.0 < lof.x < 1500.0  # a light-aircraft takeoff roll
    on, off = res.data["on_ground"] > 0.5, res.data["on_ground"] < 0.5
    assert np.all(res.data["L_N"][on] < spec.weight_N) and np.all(res.data["F_roll_N"][off] == 0.0)
    assert np.all(np.diff(res.data["x_m"]) > 0)


def test_flaps_and_ground_effect_bring_the_lift_off_forward(cessna):
    be = Backend()
    spec = ScenarioSpec(v_end=90 * MPH, v_rotate=50 * MPH, ramp_target_deg=8.0, points=41)
    clean = run_scenario(be.build_request(cessna), spec).liftoff
    flapped_state = cessna.copy()
    flapped_state.high_lift.flaps_on, flapped_state.high_lift.phase = True, "takeoff"
    flapped = run_scenario(be.build_request(flapped_state), spec).liftoff
    ground = run_scenario(be.build_request(cessna), replace(spec, ground_effect=True, ground_height_m=2.3)).liftoff
    assert clean is not None and flapped is not None and ground is not None
    assert flapped.V < clean.V and ground.V < clean.V


def test_exports_csv_text_and_json(cessna):
    be = Backend()
    spec = ScenarioSpec(v_end=90 * MPH, v_rotate=60 * MPH, time_model=True, points=21)
    res = run_scenario(be.build_request(cessna), spec)
    text_csv = to_csv(res, "mph")
    comments = [ln for ln in text_csv.splitlines() if ln.startswith("# ") or ln.startswith('"# ')]
    rows = list(csv.DictReader(ln for ln in text_csv.splitlines() if not (ln.startswith("# ") or ln.startswith('"# '))))
    assert len(rows) == res.n and any("lift-off" in c for c in comments)
    assert float(rows[-1]["V_mph"]) == pytest.approx(90.0) and float(rows[-1]["V_mps"]) == pytest.approx(90 * MPH)
    assert {"t_s", "L_N", "W_N", "D_N", "x_m", "alpha_deg"} <= set(rows[0])
    txt = to_text(res, "mph")
    assert "lift-off" in txt and "V [mph]" in txt and len(txt.splitlines()) > res.n
    d = result_to_dict(res, max_frames=10)
    json.dumps(d)  # plain JSON (no NaN objects)
    assert d["schema"] == "liftingline.scenario/1" and len(d["frames"]) <= 10 and d["events"][0]["name"] == "start"


def test_scenario_settings_round_trip_and_leave_the_analysis_alone(tmp_path):
    st = ProjectState()
    st.scenario = ScenarioSpec(v_end=50.0, v_rotate=30.0, mass_kg=800.0, time_model=True, ramp_mode=RAMP_PER_SECOND)
    st.save(tmp_path / "p.json")
    back = ProjectState.load(tmp_path / "p.json")
    assert back.scenario == st.scenario
    plain = ProjectState()
    st.solver.engine = plain.solver.engine = "builtin"
    from liftingline.core.analysis import run

    be = Backend()
    assert run(be.build_request(st)).totals.CL == run(be.build_request(plain)).totals.CL
    old = json.loads((tmp_path / "p.json").read_text())
    del old["scenario"]  # a project written by an earlier version
    assert ProjectState.from_dict(old).scenario == ScenarioSpec()
