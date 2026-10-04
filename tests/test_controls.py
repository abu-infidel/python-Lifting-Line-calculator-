"""Control surfaces: thin-airfoil flap theory, span/antisymmetry rules, and the effect on the lifting-line totals."""

import math

import numpy as np
import pytest

from liftingline.core.analysis import AnalysisRequest, run
from liftingline.core.controls import (
    ControlError,
    ControlSet,
    ControlSurface,
    flap_effectiveness,
    flap_moment_slope,
    hinge_angle,
    validate_rows,
)
from liftingline.core.flow import FlowCondition
from liftingline.core.section import LinearSection
from liftingline.core.wing import Wing

DEG = math.pi / 180.0


# ======================================================================== thin-airfoil flap theory
def test_flap_effectiveness_limits_and_the_textbook_value():
    assert flap_effectiveness(0.999) == pytest.approx(1.0, abs=1e-3)
    assert flap_effectiveness(1e-8) == pytest.approx(0.0, abs=1e-3)  # tau ~ 4 sqrt(cf) / pi for a vanishing flap
    assert flap_effectiveness(0.25) == pytest.approx(0.609, abs=1e-3)  # the familiar tau for a 25 % flap
    assert flap_effectiveness(0.5) == pytest.approx(1.0 - (math.pi / 2 - 1.0) / math.pi)
    cf = np.linspace(0.05, 0.6, 30)
    assert np.all(np.diff([flap_effectiveness(c) for c in cf]) > 0)  # bigger flap, more effective


@pytest.mark.parametrize("cf", [0.15, 0.25, 0.4])
def test_flap_theory_agrees_with_a_direct_glauert_integration(cf):
    """Independent route: integrate the Glauert coefficients A0, A1, A2 of a flapped flat plate numerically."""
    delta = 0.1
    th = np.linspace(0.0, math.pi, 200001)
    slope = np.where(th > hinge_angle(cf), -delta, 0.0)  # dz/dx = -delta on the flap (trailing edge down)
    integ = lambda f: float(np.trapezoid(f, th))  # noqa: E731
    A0 = -(1.0 / math.pi) * integ(slope)
    A1 = (2.0 / math.pi) * integ(slope * np.cos(th))
    A2 = (2.0 / math.pi) * integ(slope * np.cos(2.0 * th))
    cl = math.pi * (2.0 * A0 + A1)
    cm = 0.25 * math.pi * (A2 - A1)
    assert cl == pytest.approx(2.0 * math.pi * flap_effectiveness(cf) * delta, rel=1e-4)
    assert cm == pytest.approx(flap_moment_slope(cf) * delta, rel=1e-3)


def test_a_flap_down_pitches_the_nose_down():
    assert flap_moment_slope(0.25) == pytest.approx(-0.6495, abs=1e-3)
    assert all(flap_moment_slope(c) < 0 for c in (0.1, 0.3, 0.5))


# ============================================================================== the control set
def test_flaps_act_on_both_wings_alike_and_ailerons_oppositely():
    flap = ControlSurface("flap", "flap", 0.1, 0.6, 0.3, 10.0, 1.0)
    ail = ControlSurface("ail", "aileron", 0.6, 1.0, 0.25, 8.0, 1.0)
    cs = ControlSet((flap, ail), half_span=5.0)
    y = np.array([-4.8, -2.0, -0.2, 0.0, 0.2, 2.0, 4.8])
    da = cs.d_alpha(y)
    tau_f, tau_a = flap_effectiveness(0.3) * 10 * DEG, flap_effectiveness(0.25) * 8 * DEG
    assert da == pytest.approx([-tau_a, tau_f, 0.0, 0.0, 0.0, tau_f, tau_a])  # stations by |y|/hs: 0.96 .. 0.04 .. 0.96
    assert cs.d_cm(np.array([2.0]))[0] == pytest.approx(flap_moment_slope(0.3) * 10 * DEG)
    assert cs.d_cm(np.array([-4.8]))[0] == pytest.approx(-flap_moment_slope(0.25) * 8 * DEG)


def test_effectiveness_factor_scales_both_terms():
    full, part = ControlSurface("f", "flap", 0.0, 1.0, 0.3, 5.0, 1.0), ControlSurface("f", "flap", 0.0, 1.0, 0.3, 5.0, 0.8)
    y = np.array([1.0])
    for fn in ("d_alpha", "d_cm"):
        assert getattr(ControlSet((part,), 4.0), fn)(y) == pytest.approx(0.8 * getattr(ControlSet((full,), 4.0), fn)(y))


def test_with_deflection_replaces_by_name_and_unknown_names_fail():
    cs = ControlSet((ControlSurface("ail", deflection_deg=0.0), ControlSurface("fl", "flap", 0.0, 0.5, 0.3, 0.0)), 5.0)
    assert not cs.active and cs.with_deflection("ail", 10.0).active
    assert cs.with_deflection("ail", 10.0).surfaces[1].deflection_deg == 0.0
    with pytest.raises(ControlError):
        cs.with_deflection("nope", 1.0)


@pytest.mark.parametrize(
    "kw, text",
    [
        (dict(y_in=0.8, y_out=0.5), "span range"),
        (dict(chord_fraction=0.01), "control chord"),
        (dict(deflection_deg=60.0), "deflection"),
        (dict(kind="elevon"), "unknown"),
        (dict(effectiveness=1.5), "effectiveness"),
    ],
)
def test_invalid_surfaces_are_rejected_with_a_reason(kw, text):
    with pytest.raises(ControlError, match=text):
        ControlSet((ControlSurface(**kw),), 5.0)
    assert any(text in m for m in validate_rows([ControlSurface(**kw).to_dict()]))


def test_rows_roundtrip_and_duplicate_names_are_flagged():
    s = ControlSurface("a", "flap", 0.1, 0.5, 0.3, 4.0, 0.9)
    assert ControlSurface.from_dict(s.to_dict()) == s
    assert validate_rows([s.to_dict(), s.to_dict()]) == ["control surface names must be unique"]
    assert ControlSet.from_dicts([s.to_dict()], 3.0).surfaces == (s,)
    with pytest.raises(ControlError):
        ControlSet.from_dicts([dict(name="x", y_in="oops")], 3.0)


# ============================================================================= in the solvers
def flow(alpha=3.0):
    return FlowCondition.from_isa(0.0, V=40.0, alpha_deg=alpha)


def totals(wing, controls=None, solver="classical", alpha=3.0, **kw):
    req = AnalysisRequest(wing, flow(alpha), LinearSection(), solver=solver, compressibility=False,
                          convergence_check=False, controls=controls, **kw)
    return run(req).totals


def test_a_full_span_flap_is_exactly_an_angle_of_attack_change():
    """With linear sections a uniform d_alpha is the same as more alpha: CL(alpha, flap) == CL(alpha + tau delta)."""
    wing = Wing.rectangular(10.0, 1.2)
    flap = ControlSurface("flap", "flap", 0.0, 1.0, 0.3, 8.0, 0.9)
    with_flap = totals(wing, ControlSet((flap,), wing.half_span), alpha=2.0)
    shifted = totals(wing, alpha=2.0 + math.degrees(flap.tau * 8.0 * DEG))
    assert with_flap.CL == pytest.approx(shifted.CL, rel=1e-12) and with_flap.CD_induced == pytest.approx(shifted.CD_induced, rel=1e-12)
    assert abs(with_flap.C_roll) < 1e-12


def test_flap_pitching_moment_change_equals_the_section_value_on_a_rectangular_wing():
    wing = Wing.rectangular(10.0, 1.2)
    flap = ControlSurface("flap", "flap", 0.0, 1.0, 0.3, 8.0, 1.0)
    base = totals(wing)
    dcm = totals(wing, ControlSet((flap,), wing.half_span)).Cm - base.Cm
    # lift acts at c/4 = the reference point, so only the section moment changes it (alpha shift adds lift, no arm)
    assert dcm == pytest.approx(flap_moment_slope(0.3) * 8.0 * DEG, rel=1e-6)


@pytest.mark.parametrize("solver", ["classical", "numerical"])
def test_a_right_aileron_down_rolls_the_wing_left_and_the_solvers_agree(solver):
    wing = Wing.rectangular(10.0, 1.2)
    ail = ControlSurface("ail", "aileron", 0.55, 0.95, 0.25, 6.0, 0.85)
    cs = ControlSet((ail,), wing.half_span)
    t = totals(wing, cs, solver=solver)
    assert t.C_roll < 0.0  # more lift on the right: right wing up = negative roll in the right-wing-down convention
    base = totals(wing, solver=solver)
    assert t.CL == pytest.approx(base.CL, rel=2e-3)  # an aileron changes the roll moment, hardly the net lift


def test_roll_power_is_linear_in_the_deflection_and_the_two_solvers_agree():
    wing = Wing.rectangular(10.0, 1.2)
    cs = ControlSet((ControlSurface("ail", "aileron", 0.55, 0.95, 0.25, 0.0, 0.85),), wing.half_span)
    rolls = {s: [totals(wing, cs.with_deflection("ail", d), solver=s).C_roll for d in (2.0, 4.0, -4.0)] for s in ("classical", "numerical")}
    for s, (r2, r4, rm4) in rolls.items():
        assert r4 == pytest.approx(2.0 * r2, rel=2e-3) and rm4 == pytest.approx(-r4, rel=1e-6), s
    assert rolls["numerical"][1] == pytest.approx(rolls["classical"][1], rel=0.06)
    assert -0.01 < rolls["classical"][1] / 4.0 < -0.002  # roll power of a typical aileron: 0.002 ... 0.01 per degree


def test_adverse_yaw_has_the_physical_sign_and_grows_with_lift():
    """The lowered (right) aileron raises the lift and the induced drag of the right wing: the nose is pulled to the
    right (positive yaw) while the wing rolls to the left - adverse yaw. It vanishes at zero lift."""
    wing = Wing.rectangular(10.0, 1.2)
    cs = ControlSet((ControlSurface("ail", "aileron", 0.55, 0.95, 0.25, 5.0, 0.85),), wing.half_span)
    for solver in ("classical", "numerical"):
        t4, t0 = totals(wing, cs, solver=solver, alpha=4.0), totals(wing, cs, solver=solver, alpha=0.0)
        assert t4.C_roll < 0.0 and t4.C_yaw > 0.0
        assert abs(t0.C_yaw) < 1e-9 < t4.C_yaw
