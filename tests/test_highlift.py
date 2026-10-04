"""High-lift devices (deployable flaps and slats): published data, section mapping, wing results, project wiring.

The handbook numbers (Raymer, Aircraft Design: A Conceptual Approach, ch. 12) are pinned, the clean wing must be
untouched when the devices are retracted, and the lifting-line results with the devices out must agree with the
handbook's wing-level estimate and with the physics of each device.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace

import numpy as np
import pytest

from liftingline.core.analysis import AnalysisRequest, run
from liftingline.core.controls import ControlSurface
from liftingline.core.flow import FlowCondition
from liftingline.core.highlift import (
    DALPHA_L0_DEG,
    EDGE_BLEND,
    FLAP_TYPES,
    LANDING,
    SLAT_TYPES,
    SPEED_FACTORS,
    TAKEOFF,
    TAKEOFF_FRACTION,
    Device,
    HighLiftError,
    HighLiftSection,
    HighLiftSystem,
    format_report,
    high_lift_report,
    section_saturates,
)
from liftingline.core.section import LinearSection
from liftingline.core.wing import Wing
from liftingline.io.backend import Backend
from liftingline.io.project import HighLiftSpec, ProjectState
from liftingline.io.report import result_to_csv, totals_rows


# ============================================================================ published data
def test_section_increments_are_the_published_handbook_values():
    # Raymer, Table 12.2 (landing): plain/split 0.9, slotted 1.3, Fowler 1.3 c'/c, double 1.6 c'/c, triple 1.9 c'/c
    expected_flaps = {"plain": (0.9, False), "split": (0.9, False), "slotted": (1.3, False), "fowler": (1.3, True),
                      "double_slotted": (1.6, True), "triple_slotted": (1.9, True)}
    # fixed slot 0.2, leading-edge flap 0.3, Krueger 0.3, slat 0.4 c'/c
    expected_slats = {"fixed_slot": (0.2, False), "le_flap": (0.3, False), "krueger": (0.3, False), "slat": (0.4, True)}
    assert {k: (t.dcl_max, t.times_extension) for k, t in FLAP_TYPES.items()} == expected_flaps
    assert {k: (t.dcl_max, t.times_extension) for k, t in SLAT_TYPES.items()} == expected_slats
    assert DALPHA_L0_DEG == {TAKEOFF: -10.0, LANDING: -15.0}
    assert 0.6 <= TAKEOFF_FRACTION <= 0.8
    assert FLAP_TYPES["plain"].drag_factor == 0.0144 and FLAP_TYPES["slotted"].drag_factor == 0.0074
    for t in FLAP_TYPES.values():  # typical settings: takeoff 15-20 deg, landing 40-60 deg
        assert 15.0 <= t.deflection_deg[0] <= 20.0 and 40.0 <= t.deflection_deg[1] <= 60.0


def test_device_increments_follow_type_phase_and_chord_extension():
    fow = HighLiftSystem(LANDING, flaps=Device("fowler", 0.0, 0.6, 0.3, 1.2))
    assert fow.flap_dcl_max() == pytest.approx(1.3 * 1.2)
    assert fow.flap_slope_factor() == pytest.approx(1.2)
    assert replace(fow, phase=TAKEOFF).flap_dcl_max() == pytest.approx(TAKEOFF_FRACTION * 1.3 * 1.2)
    assert replace(fow, phase=TAKEOFF).flap_dalpha_L0_deg() == -10.0
    slot = HighLiftSystem(LANDING, flaps=Device("slotted", 0.0, 0.6, 0.3, 1.3))
    assert slot.flap_dcl_max() == pytest.approx(1.3) and slot.flap_slope_factor() == 1.0  # no extension credit
    sl = HighLiftSystem(TAKEOFF, slats=Device("slat", 0.0, 1.0, chord_extension=1.1))
    assert sl.slat_dcl_max() == pytest.approx(0.44)  # leading-edge devices: same in both phases
    assert sl.flap_dcl_max() == 0.0 and sl.flap_dalpha_L0_deg() == 0.0
    # flap drag, Raymer: F (cf/c) (delta - 10)
    assert slot.flap_dcd() == pytest.approx(0.0074 * 0.3 * (40.0 - 10.0))
    assert HighLiftSystem(LANDING, flaps=Device("split", 0.0, 0.6, 0.25)).flap_dcd() == pytest.approx(0.0144 * 0.25 * 50.0)
    assert slot.flap_dcm() < 0.0  # a flap pitches the nose down


def test_invalid_devices_are_refused_with_readable_messages():
    with pytest.raises(HighLiftError, match="span"):
        HighLiftSystem(LANDING, flaps=Device("slotted", 0.6, 0.4))
    with pytest.raises(HighLiftError, match="flap chord"):
        HighLiftSystem(LANDING, flaps=Device("slotted", 0.0, 0.5, chord_fraction=0.8))
    with pytest.raises(HighLiftError, match="unknown"):
        HighLiftSystem(LANDING, slats=Device("slotted", 0.0, 0.5))
    with pytest.raises(HighLiftError, match="phase"):
        HighLiftSystem("cruise", flaps=Device("plain", 0.0, 0.5))
    with pytest.raises(HighLiftError, match="c'/c"):
        HighLiftSystem(LANDING, slats=Device("slat", 0.0, 0.5, chord_extension=2.0))


def test_spanwise_layout_blends_the_device_ends_and_keeps_its_area():
    sys = HighLiftSystem(LANDING, flaps=Device("slotted", 0.2, 0.6, 0.3))
    eta = np.linspace(0.0, 1.0, 100001)
    inc = sys.increments(eta * 5.0, 5.0)
    w = inc.dcl_max / sys.flap_dcl_max()
    assert w[eta < 0.2 - EDGE_BLEND / 2].max() == 0.0 and w[eta > 0.6 + EDGE_BLEND / 2].max() == 0.0
    assert w[(eta > 0.2 + EDGE_BLEND / 2) & (eta < 0.6 - EDGE_BLEND / 2)].min() == 1.0
    assert np.trapezoid(w, eta) == pytest.approx(0.4, abs=1e-6)  # the blend neither adds nor removes flap span
    assert np.allclose(inc.dalpha_L0, w * math.radians(-15.0))
    full = HighLiftSystem(LANDING, slats=Device("slat", 0.0, 1.0)).increments(np.array([0.0, 5.0]), 5.0)
    assert np.all(full.dcl_max > 0.0)  # ends at the root and the tip are not blended


# ========================================================================== section mapping
@pytest.fixture(scope="module")
def backend():
    return Backend()


def _state(**hl) -> ProjectState:
    st = ProjectState()
    st.wing.span, st.wing.root_chord, st.wing.tip_chord = 12.0, 1.8, 1.0
    st.wing.sync()
    st.solver.engine = "builtin"
    st.flow.V, st.flow.alpha_deg = 30.0, 4.0
    for k, v in hl.items():
        setattr(st.high_lift, k, v)
    return st


def test_flapped_section_has_the_published_zero_lift_angle_slope_and_maximum_lift(backend):
    req = backend.build_request(_state(flaps_on=True, flap_type="fowler", flap_chord_extension=1.2, flap_y_out=0.6))
    hl: HighLiftSection = req.section
    base = hl.base
    y, re = np.array([1.0]), np.array([3.0e6])
    lin0, lin = base.linear(y, re), hl.linear(y, re)
    assert lin.alpha_L0[0] == pytest.approx(lin0.alpha_L0[0] + math.radians(-15.0))
    assert lin.a0[0] == pytest.approx(1.2 * lin0.a0[0])
    assert lin.cl_max[0] == pytest.approx(lin0.cl_max[0] + 1.3 * 1.2)
    assert lin.alpha_stall[0] < lin0.alpha_stall[0]  # flaps stall the section earlier
    assert lin.cm_c4[0] < lin0.cm_c4[0]
    # the section polar itself: zero lift at the new zero-lift angle, slope k a0 in the attached range,
    # the clean polar's shape scaled to the new maximum
    a = np.radians(np.linspace(-20.0, 30.0, 2001))
    ones = np.ones_like(a)
    c_hl = hl.evaluate(y * ones, a, re * ones)
    c_0 = base.evaluate(y * ones, a, re * ones)
    assert np.interp(0.0, c_hl.cl, a) == pytest.approx(lin.alpha_L0[0], abs=2e-4)
    da = math.radians(1.0)
    a1 = lin.alpha_L0[0] + np.array([-da, da])
    slope = np.diff(hl.evaluate(y * np.ones(2), a1, re * np.ones(2)).cl)[0] / (2 * da)
    assert slope == pytest.approx(lin.a0[0], rel=2e-3)
    s = lin.cl_max[0] / lin0.cl_max[0]
    mapped = hl.mapped_alpha(y * ones, a, re * ones)
    assert np.allclose(c_hl.cl, s * base.evaluate(y * ones, mapped, re * ones).cl, rtol=1e-12, atol=0.0)
    assert c_hl.cl.max() <= lin.cl_max[0] and c_0.cl.max() <= lin0.cl_max[0]
    assert np.all(c_hl.cd >= c_0.cd.min())


def test_slat_extends_the_lift_curve_without_changing_it_below_the_stall(backend):
    req = backend.build_request(_state(slats_on=True))
    hl, base = req.section, req.section.base
    y, re = np.array([2.0]), np.array([3.0e6])
    lin0, lin = base.linear(y, re), hl.linear(y, re)
    assert lin.alpha_L0[0] == lin0.alpha_L0[0] and lin.a0[0] == lin0.a0[0]
    assert lin.cl_max[0] == pytest.approx(lin0.cl_max[0] + 0.44)
    assert math.degrees(lin.alpha_stall[0] - lin0.alpha_stall[0]) > 2.0  # stall angle raised
    a = np.radians(np.linspace(-2.0, 6.0, 9))
    ones = np.ones_like(a)
    assert np.allclose(hl.evaluate(y * ones, a, re * ones).cl, base.evaluate(y * ones, a, re * ones).cl, rtol=0.02)


def test_stations_without_a_device_are_evaluated_bit_for_bit_unchanged(backend):
    req = backend.build_request(_state(flaps_on=True, flap_y_in=0.2, flap_y_out=0.5))
    hl, base = req.section, req.section.base
    y = np.array([0.1, 0.5, 4.0, 5.9]) * np.ones(1)  # semi-span 6 m: outside 0.2-0.5 (+- the blend)
    a = np.radians(np.array([2.0, 8.0, 14.0, 25.0]))
    re = np.full(4, 2.5e6)
    c1, c0 = hl.evaluate(y, a, re), base.evaluate(y, a, re)
    for f in ("cl", "cd", "cm", "cl_alpha"):
        assert np.array_equal(getattr(c1, f), getattr(c0, f))
    l1, l0 = hl.linear(y, re), base.linear(y, re)
    for f in ("a0", "alpha_L0", "cm_c4", "cl_max", "alpha_stall"):
        assert np.array_equal(getattr(l1, f), getattr(l0, f))


# ================================================================================ wing results
def test_retracted_devices_leave_every_result_exactly_unchanged(backend):
    ref = run(backend.build_request(_state()))
    # devices configured but both toggles off: no wrapper, identical numbers
    st = _state(flap_type="triple_slotted", flap_y_in=0.3, slat_type="krueger", phase="takeoff", weight_N=5000.0)
    req = backend.build_request(st)
    assert not isinstance(req.section, HighLiftSection)
    off = run(req)
    for f in ("CL", "CD", "CD_induced", "CD_profile", "Cm", "C_roll", "C_yaw", "e"):
        assert getattr(off.totals, f) == getattr(ref.totals, f)
    assert off.warnings == ref.warnings and "high_lift" not in off.info
    # toggling on and off again returns to the clean wing
    on = run(backend.build_request(_state(flaps_on=True, slats_on=True)))
    assert on.totals.CL != ref.totals.CL
    again = run(backend.build_request(_state(flaps_on=False, slats_on=False)))
    assert again.totals.CL == ref.totals.CL and again.totals.CD == ref.totals.CD


def test_full_span_flap_shifts_the_whole_lift_curve_by_the_zero_lift_angle():
    """Linear sections: a full-span flap is exactly the clean wing at 15 deg more angle of attack."""
    wing = Wing.trapezoid(10.0, 1.6, 0.8)
    flow = FlowCondition.from_isa(0.0, V=30.0, alpha_deg=2.0)
    sec = LinearSection(a0=6.0, alpha_L0=math.radians(-2.0), cl_max=1.6)
    hl = HighLiftSection(sec, HighLiftSystem(LANDING, flaps=Device("slotted", 0.0, 1.0)), wing.half_span)
    r_hl = run(AnalysisRequest(wing, flow, hl, nonlinear=False, compressibility=False, convergence_check=False))
    r_0 = run(AnalysisRequest(wing, replace(flow, alpha_deg=17.0), sec, nonlinear=False, compressibility=False,
                              convergence_check=False))
    assert r_hl.totals.CL == pytest.approx(r_0.totals.CL, rel=1e-9)
    assert r_hl.totals.CD_induced == pytest.approx(r_0.totals.CD_induced, rel=1e-9)
    # and the wing-level drag increment of the flap is exactly the handbook flap-drag formula (full span: S'/S = 1)
    est = hl.high_lift.wing_estimate(wing)
    assert r_hl.totals.CD_profile - r_0.totals.CD_profile == pytest.approx(est.dCD0_flaps, rel=1e-3)
    assert est.dCD0_flaps == pytest.approx(0.0074 * 0.3 * 30.0)


def test_partial_span_flap_lift_increment_matches_the_handbook_zero_lift_angle_estimate(backend):
    st = _state(flaps_on=True, flap_y_in=0.0, flap_y_out=0.6)
    st.flow.alpha_deg = 0.0
    clean = run(backend.build_request(replace_flow(_state(), alpha_deg=0.0)))
    flapped = run(backend.build_request(st))
    c2 = run(backend.build_request(replace_flow(_state(), alpha_deg=2.0)))
    cla = (c2.totals.CL - clean.totals.CL) / math.radians(2.0)  # clean wing lift slope, 1/rad
    est = flapped.request.section.high_lift.wing_estimate(flapped.request.wing)
    expected = -cla * math.radians(est.dalpha_L0_deg)  # Raymer: d(alpha_L0, wing) = d(alpha_L0) S'/S cos(hinge)
    assert flapped.totals.CL - clean.totals.CL == pytest.approx(expected, rel=0.15)


def replace_flow(st: ProjectState, **kw) -> ProjectState:
    for k, v in kw.items():
        setattr(st.flow, k, v)
    return st


def test_devices_are_symmetric_pitch_the_nose_down_and_add_drag(backend):
    ref = run(backend.build_request(_state()))
    r = run(backend.build_request(_state(flaps_on=True, slats_on=True)))
    assert abs(r.totals.C_roll) < 1e-9 and abs(r.totals.C_yaw) < 1e-9 and abs(r.totals.CY) < 1e-9
    assert r.totals.CL > ref.totals.CL + 0.3
    assert r.totals.Cm < ref.totals.Cm - 0.05
    assert r.totals.CD_profile > ref.totals.CD_profile and r.totals.CD_induced > ref.totals.CD_induced
    assert r.info["high_lift"].startswith("landing setting")
    assert any(w.startswith("High-lift devices deployed, landing setting") for w in r.warnings)


def test_both_solvers_agree_with_devices_out_and_the_fourier_series_converges(backend):
    req = backend.build_request(_state(flaps_on=True, slats_on=True, flap_y_in=0.2, flap_y_out=0.7))
    rc, rn = run(req), run(replace(req, solver="numerical"))
    assert rc.solver_used == "classical" and rn.solver_used == "numerical"
    assert rn.totals.CL == pytest.approx(rc.totals.CL, rel=0.01)
    assert rn.totals.CD_induced == pytest.approx(rc.totals.CD_induced, rel=0.02)
    assert rc.info["convergence_dCL"] < 5e-3
    assert not any("not converged" in w for w in rc.warnings + rn.warnings)


def test_ailerons_still_work_with_the_flaps_down(backend):
    def roll(**hl):
        st = _state(**hl)
        st.controls = [ControlSurface("Aileron", "aileron", 0.7, 0.95, 0.25, 10.0).to_dict()]
        return run(backend.build_request(st)).totals.C_roll
    clean, flapped = roll(), roll(flaps_on=True, flap_y_out=0.65)
    assert clean < 0.0 and flapped < 0.0  # right aileron down: right wing up (roll left, negative)
    assert flapped == pytest.approx(clean, rel=0.15)


def test_high_mach_with_devices_out_is_flagged(backend):
    st = _state(flaps_on=True)
    st.flow.V = 150.0
    assert any("takeoff and landing devices" in w for w in run(backend.build_request(st)).warnings)
    st.flow.V = 60.0
    assert not any("takeoff and landing devices" in w for w in run(backend.build_request(st)).warnings)
    st.high_lift.flaps_on = False
    st.flow.V = 150.0
    assert not any("takeoff and landing devices" in w for w in run(backend.build_request(st)).warnings)


# ================================================================ maximum lift and stall speed
@pytest.mark.parametrize(
    "hl",
    [
        dict(flaps_on=True, flap_type="plain"),
        dict(flaps_on=True, flap_type="slotted"),
        dict(flaps_on=True, flap_type="fowler", flap_chord_extension=1.25),
        dict(flaps_on=True, flap_type="slotted", phase="takeoff"),
        dict(slats_on=True, slat_type="slat"),
        dict(slats_on=True, slat_type="krueger", slat_chord_extension=1.0),
        dict(flaps_on=True, slats_on=True, flap_type="double_slotted", flap_chord_extension=1.2),
    ],
)
def test_wing_maximum_lift_gain_agrees_with_the_handbook_estimate(backend, hl):
    """Raymer's wing-level correlation d(CL_max) = 0.9 d(cl_max) S'/S cos(sweep) condenses wind-tunnel data; the
    lifting-line result must land within its accuracy (+-25 %)."""
    st = _state(**hl)
    st.high_lift.flap_y_out = 0.6
    clean = _state()
    rep = high_lift_report(backend.build_request(clean), backend.build_request(st), st.high_lift.system(), weight=12000.0)
    assert rep.clean.CL_max == pytest.approx(rep.deployed.CL_max - rep.dCL_max)
    assert rep.estimate.dCL_max > 0.0
    assert 0.8 <= rep.dCL_max / rep.estimate.dCL_max <= 1.25, (rep.dCL_max, rep.estimate.dCL_max)
    if st.high_lift.flaps_on:  # flaps stall the wing at a lower angle, slats at a higher one
        assert rep.deployed.alpha_first_stall < rep.clean.alpha_first_stall
    else:
        assert rep.deployed.alpha_first_stall > rep.clean.alpha_first_stall
    vs = rep.stall_speed()
    assert vs == pytest.approx(math.sqrt(2 * 12000.0 / (rep.rho * rep.area * rep.deployed.CL_max)))
    assert vs < rep.stall_speed("clean")
    for name, factor in SPEED_FACTORS[st.high_lift.phase]:
        assert rep.speeds[f"{name} ({factor:g} Vs)"] == pytest.approx(factor * vs)
    text = format_report(rep)
    assert "Raymer" in text and "stall speed" in text


@pytest.mark.neuralfoil
def test_maximum_lift_with_neuralfoil_polars_uses_the_lift_peak(backend):
    st = _state(flaps_on=True, slats_on=True)
    st.solver.engine = "neuralfoil"
    clean = _state()
    clean.solver.engine = "neuralfoil"
    req_c, req_d = backend.build_request(clean), backend.build_request(st)
    assert not section_saturates(req_c) and not section_saturates(req_d)
    rep = high_lift_report(req_c, req_d, st.high_lift.system())
    assert rep.clean.definition == rep.deployed.definition == "peak of the lift curve"
    assert 0.8 <= rep.dCL_max / rep.estimate.dCL_max <= 1.25
    assert 1.3 < rep.clean.CL_max < 1.8  # NACA 2412 wing at Re ~ 3e6


def test_builtin_section_data_are_recognised_as_saturating(backend):
    assert section_saturates(backend.build_request(_state()))


# =============================================================================== project wiring
def test_high_lift_settings_round_trip_through_a_project_file(tmp_path):
    st = _state(flaps_on=True, flap_type="fowler", flap_chord_extension=1.3, slats_on=True, slat_type="krueger",
                phase="takeoff", weight_N=9000.0)
    path = tmp_path / "p.json"
    st.save(path)
    back = ProjectState.load(path)
    assert back.high_lift == st.high_lift
    data = json.loads(path.read_text())
    del data["high_lift"]  # a project written before the High lift tab existed
    old = ProjectState.from_dict(data)
    assert old.high_lift == HighLiftSpec() and not old.high_lift.active


def test_project_validation_reports_bad_devices_only_when_deployed():
    st = _state(flap_y_in=0.8, flap_y_out=0.2)
    assert st.validate() == []  # retracted: not checked
    st.high_lift.flaps_on = True
    assert any(m.startswith("High lift:") and "span" in m for m in st.validate())


def test_result_table_and_csv_name_the_configuration(backend):
    r = run(backend.build_request(_state(flaps_on=True)))
    rows = {row.key: row for row in totals_rows(r)}
    assert "single-slotted flap" in rows["high_lift"].value
    csv = result_to_csv(r)
    assert "# high-lift devices" in csv and "\nhigh_lift," not in csv
    clean = run(backend.build_request(_state()))
    assert "high_lift" not in {row.key for row in totals_rows(clean)}
