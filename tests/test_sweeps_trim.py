"""Sweeps, stall analysis, trim and optimal twist."""

import math

import numpy as np
import pytest

from liftingline.core.analysis import AnalysisRequest, run
from liftingline.core.controls import ControlError, ControlSet, ControlSurface
from liftingline.core.flow import FlowCondition
from liftingline.core.optimal import optimal_twist
from liftingline.core.section import LinearSection
from liftingline.core.sweeps import ALPHA, ALTITUDE, BETA, DEFLECTION, SPEED, linspace_values, run_sweep, stall_info
from liftingline.core.trim import TrimError, stall_speed, trim_alpha, trim_speed
from liftingline.core.wing import Section, Wing
from liftingline.io.backend import Backend
from liftingline.io.presets import PRESETS

TWO_PI = 2.0 * math.pi
DEG = math.pi / 180.0


def request(wing=None, alpha=3.0, V=40.0, section=None, **kw):
    wing = wing or Wing.rectangular(10.0, 1.25)
    kw.setdefault("compressibility", False)
    return AnalysisRequest(wing, FlowCondition.from_isa(0.0, V=V, alpha_deg=alpha), section or LinearSection(), **kw)


# =========================================================================================== sweeps
def test_alpha_sweep_of_a_linear_wing_is_a_straight_line_through_the_zero_lift_angle():
    sec = LinearSection(alpha_L0=-2.0 * DEG)
    sw = run_sweep(request(section=sec), ALPHA, linspace_values(-4.0, 12.0, 9))
    cl = sw.column("CL")
    assert np.all(sw.converged) and sw.solver == "classical"
    slope = np.diff(cl) / np.diff(sw.values)
    assert slope == pytest.approx(slope[0], rel=1e-9)  # exactly linear
    assert np.interp(0.0, cl, sw.values) == pytest.approx(-2.0, abs=1e-9)  # CL = 0 at the zero-lift angle
    assert np.all(np.isnan(sw.margin))  # a linear section has no cl_max
    assert stall_info(sw).available is False


def test_speed_and_altitude_sweeps_scale_the_lift_like_q_and_density():
    req = request()
    sp = run_sweep(req, SPEED, [20.0, 40.0, 80.0])
    assert sp.column("L")[1] / sp.column("L")[0] == pytest.approx(4.0, rel=1e-9)
    assert sp.column("CL") == pytest.approx(sp.column("CL")[0])  # CL independent of speed (linear section, no Mach effect)
    al = run_sweep(req, ALTITUDE, [0.0, 3000.0, 6000.0])
    rho = [FlowCondition.from_isa(h, V=40.0).rho for h in (0.0, 3000.0, 6000.0)]
    assert al.column("L") / al.column("L")[0] == pytest.approx(np.array(rho) / rho[0], rel=1e-9)
    hold_mach = run_sweep(req, ALTITUDE, [0.0, 6000.0], hold="mach")
    mach0 = FlowCondition.from_isa(0.0, V=40.0).mach
    assert hold_mach.points[1].totals.q / hold_mach.points[0].totals.q == pytest.approx(
        (FlowCondition.from_isa(6000.0, mach=mach0).q) / FlowCondition.from_isa(0.0, mach=mach0).q
    )


def test_sideslip_sweep_uses_the_numerical_solver_and_is_odd_in_beta():
    sw = run_sweep(request(Wing.trapezoid(10.0, 1.2, 0.9, sweep_deg=15.0), solver="classical"), BETA, [-6.0, 0.0, 6.0])
    assert sw.solver == "numerical"
    roll = sw.column("C_roll")
    assert roll[0] == pytest.approx(-roll[2], rel=1e-9) and roll[2] < 0.0 and abs(roll[1]) < 1e-12
    cl = sw.column("CL")
    assert cl[0] == pytest.approx(cl[2], rel=1e-9)


def test_control_deflection_sweep_gives_a_linear_roll_moment():
    wing = Wing.rectangular(10.0, 1.2)
    cs = ControlSet((ControlSurface("ail", "aileron", 0.55, 0.95, 0.25, 0.0, 0.85),), wing.half_span)
    sw = run_sweep(request(wing, controls=cs), DEFLECTION, [-5.0, 0.0, 5.0], control="ail")
    roll = sw.column("C_roll")
    assert roll[0] == pytest.approx(-roll[2], rel=1e-9) and roll[2] < 0 and abs(roll[1]) < 1e-12
    with pytest.raises(ControlError, match="define a control surface"):
        run_sweep(request(wing), DEFLECTION, [1.0], control="ail")  # no controls defined
    with pytest.raises(ControlError, match="available: ail"):
        run_sweep(request(wing, controls=cs), DEFLECTION, [1.0], control="missing")


def test_failing_points_are_kept_and_the_sweep_goes_on():
    sw = run_sweep(request(), SPEED, [40.0, 500.0, 60.0])  # 500 m/s is supersonic at sea level
    assert [p.ok for p in sw.points] == [True, False, True]
    assert "Mach" in sw.points[1].error and math.isnan(sw.column("CL")[1])
    assert sw.column("CL")[0] == pytest.approx(sw.column("CL")[2])


def test_progress_and_cancel_callbacks():
    seen, stop = [], []
    sw = run_sweep(request(), ALPHA, linspace_values(0, 8, 9), progress=lambda i, n: (seen.append((i, n)), stop.append(i >= 3)))
    assert seen[-1] == (9, 9) and len(sw.points) == 9
    seen.clear()
    count = [0]

    def cancelled():
        count[0] += 1
        return count[0] > 4

    sw = run_sweep(request(), ALPHA, linspace_values(0, 8, 9), cancelled=cancelled)
    assert len(sw.points) == 4 and any("cancelled" in n for n in sw.notes)


def test_unknown_variables_and_short_ranges_are_refused():
    with pytest.raises(ValueError):
        run_sweep(request(), "mass", [1.0])
    with pytest.raises(ValueError):
        linspace_values(0.0, 1.0, 1)


def test_csv_has_one_line_per_point():
    sw = run_sweep(request(), ALPHA, [0.0, 2.0, 4.0])
    lines = sw.to_csv().strip().splitlines()
    assert len(lines) == 4 and lines[0].startswith("value,CL,CD") and float(lines[2].split(",")[0]) == 2.0


# ===================================================================================== stall analysis
@pytest.fixture(scope="module")
def stall_sweep():
    st = PRESETS["Rectangular wing, AR 8"]()
    st.solver.engine = "builtin"
    req = Backend().build_request(st)
    return run_sweep(req, ALPHA, linspace_values(0.0, 26.0, 27))


def test_stall_analysis_of_a_real_wing(stall_sweep):
    info = stall_info(stall_sweep)
    assert info.available and info.reached
    assert 12.0 < info.alpha_first_stall < 24.0 and 0.0 <= info.y_first_stall <= 4.0
    assert 0.9 < info.CL_first_stall <= info.CL_max < 1.8
    cl, mg, a = stall_sweep.column("CL"), stall_sweep.margin, stall_sweep.values
    assert np.all(np.diff(cl[a < info.alpha_first_stall - 1.0]) > 0)  # rising branch before stall
    assert np.all(mg[a < info.alpha_first_stall - 0.5] < 1.0)


def test_stall_info_when_stall_is_not_reached():
    st = PRESETS["Rectangular wing, AR 8"]()
    st.solver.engine = "builtin"
    sw = run_sweep(Backend().build_request(st), ALPHA, linspace_values(0.0, 6.0, 4))
    info = stall_info(sw)
    assert info.available and not info.reached and "no station reached cl_max" in info.note
    with pytest.raises(ValueError):
        stall_info(run_sweep(request(), SPEED, [20.0, 30.0]))


# ============================================================================================ trim
def test_trim_alpha_for_a_target_cl_and_for_a_target_lift():
    sec = LinearSection(alpha_L0=-1.5 * DEG)
    req = request(section=sec)
    tr = trim_alpha(req, CL=0.6)
    again = run(request(section=sec, alpha=tr.value)).totals
    assert again.CL == pytest.approx(0.6, abs=1e-9) and tr.totals.CL == pytest.approx(0.6, abs=1e-9)
    assert tr.evaluations < 30 and tr.variable == "alpha_deg"
    W = 9000.0
    tl = trim_alpha(req, lift=W)
    assert tl.totals.L == pytest.approx(W, rel=1e-8) and tl.totals.CL == pytest.approx(W / (req.flow.q * req.wing.area), rel=1e-9)


def test_trim_alpha_finds_the_lowest_angle_before_stall_and_refuses_the_unreachable():
    st = PRESETS["Rectangular wing, AR 8"]()
    st.solver.engine = "builtin"
    req = Backend().build_request(st)
    tr = trim_alpha(req, CL=1.0)
    assert 8.0 < tr.value < 16.0 and tr.totals.CL == pytest.approx(1.0, abs=1e-8)
    with pytest.raises(TrimError, match="not reached"):
        trim_alpha(req, CL=3.0)
    with pytest.raises(TrimError):
        trim_alpha(req, CL=-3.0)
    with pytest.raises(TrimError):
        trim_alpha(req)
    with pytest.raises(TrimError):
        trim_alpha(req, CL=1.0, lift=1.0)


def test_trim_speed_matches_the_level_flight_formula():
    sec = LinearSection(alpha_L0=-1.5 * DEG)
    req = request(section=sec, alpha=4.0)
    W = 12000.0
    tr = trim_speed(req, W)
    CL = run(req).totals.CL
    assert tr.value == pytest.approx(math.sqrt(2.0 * W / (req.flow.rho * req.wing.area * CL)), rel=1e-7)
    assert tr.totals.L == pytest.approx(W, rel=1e-8)
    with pytest.raises(TrimError):
        trim_speed(req, 1e9)
    with pytest.raises(TrimError):
        trim_speed(req, 1e-3)
    with pytest.raises(TrimError):
        trim_speed(req, -5.0)


def test_stall_speed_formula():
    assert stall_speed(10000.0, 1.225, 15.0, 1.6) == pytest.approx(math.sqrt(2 * 10000 / (1.225 * 15 * 1.6)))
    with pytest.raises(TrimError):
        stall_speed(0.0, 1.225, 15.0, 1.6)


# ======================================================================================= optimal twist
def _twisted(base: Wing, ot) -> Wing:
    return Wing(sections=tuple(Section(r["y"], r["chord"], r["x_le"], r["z"], r["twist_deg"]) for r in ot.rows))


@pytest.mark.parametrize("taper", [1.0, 0.6, 0.35])
def test_optimal_twist_gives_an_elliptic_load(taper):
    base = Wing.from_area_ar_taper(12.0, 9.0, taper)
    sec = LinearSection(a0=5.9, alpha_L0=-1.8 * DEG)
    req = request(base, section=sec)
    ot = optimal_twist(req, cl_target=0.55, n_stations=25)
    assert ot.rows[0]["twist_deg"] == 0.0 and len(ot.rows) == 25
    wing = _twisted(base, ot)
    t = run(request(wing, alpha=ot.alpha_root_deg, section=sec)).totals
    assert t.CL == pytest.approx(0.55, rel=1e-6)  # the root angle is refined to hit the target exactly
    assert t.e == pytest.approx(1.0, abs=1e-3)  # the interpolated twist is a fine approximation of the ideal curve
    assert t.CD_induced == pytest.approx(0.55**2 / (math.pi * base.aspect_ratio), rel=2e-3)
    assert ot.predicted["CL"] == pytest.approx(0.55, rel=1e-6) and ot.predicted["e"] == pytest.approx(t.e, rel=1e-9)
    ideal = optimal_twist(req, cl_target=0.55, n_stations=25, refine=False)
    assert ideal.predicted == {} and ideal.alpha_root_deg == pytest.approx(ot.alpha_root_deg, abs=0.05)
    if taper < 1.0:
        assert ot.rows[-1]["twist_deg"] < 0.0  # washout towards the tip
    assert wing.area == pytest.approx(base.area, rel=1e-12)


def test_optimal_twist_follows_a_changing_airfoil_along_the_span():
    base = Wing.from_area_ar_taper(12.0, 9.0, 0.5)
    sec = LinearSection(a0=lambda y: 6.1 - 0.25 * np.abs(y), alpha_L0=lambda y: (-2.5 + 0.3 * np.abs(y)) * DEG)
    ot = optimal_twist(request(base, section=sec), cl_target=0.7, n_stations=25)
    t = run(request(_twisted(base, ot), alpha=ot.alpha_root_deg, section=sec)).totals
    assert t.CL == pytest.approx(0.7, rel=1e-6) and t.e == pytest.approx(1.0, abs=1.5e-3)


def test_an_elliptic_planform_needs_no_twist():
    ell = Wing.elliptic(10.0, 1.6)
    ot = optimal_twist(request(ell, section=LinearSection(alpha_L0=-2.0 * DEG)), cl_target=0.5, n_stations=9)
    assert max(abs(r["twist_deg"]) for r in ot.rows) < 1e-6
    assert any("pointed tip" in n for n in ot.notes)  # the chord vanishes at the tip: the last station sits just inboard


def test_optimal_twist_with_compressibility_matches_the_solver():
    base = Wing.from_area_ar_taper(12.0, 9.0, 0.5)
    sec = LinearSection(alpha_L0=-1.0 * DEG)
    req = request(base, V=120.0, section=sec, compressibility=True)
    ot = optimal_twist(req, cl_target=0.5, n_stations=25)
    t = run(request(_twisted(base, ot), alpha=ot.alpha_root_deg, V=120.0, section=sec, compressibility=True)).totals
    assert t.CL == pytest.approx(0.5, rel=1e-6) and t.e == pytest.approx(1.0, abs=1e-3)


def test_a_pointed_tip_wing_is_flagged_and_a_huge_twist_is_warned_about():
    base = Wing.trapezoid(10.0, 2.0, 0.0)
    ot = optimal_twist(request(base), cl_target=0.5, n_stations=9)
    assert any("pointed tip" in n for n in ot.notes) and ot.rows[-1]["chord"] == 0.0
    big = optimal_twist(request(Wing.from_area_ar_taper(8.0, 20.0, 0.15)), cl_target=2.0)
    assert any("hard to build" in n for n in big.notes) and big.max_abs_twist_deg > 12.0
    with pytest.raises(ValueError):
        optimal_twist(request(), cl_target=float("nan"))
