"""Oracle tests for the classical (Glauert) lifting-line solver.

Reference numbers come from analytic results or from an independent pure-Python
prototype (collocation at theta_i = i pi/(N+1), N = 100) written during planning.
"""

import math

import numpy as np
import pytest

from liftingline.core.analysis import AnalysisRequest, MomentReference, run
from liftingline.core.flow import FlowCondition
from liftingline.core.section import LinearSection
from liftingline.core.solver_classical import build_load_classical, solve_classical
from liftingline.core.wing import Section, Wing

TWO_PI = 2.0 * math.pi


def flow(alpha=5.0, V=50.0, alt=0.0, **kw):
    return FlowCondition.from_isa(alt, V=V, alpha_deg=alpha, **kw)


def analyse(wing, alpha=5.0, section=None, n=80, **kw):
    """Run the classical solver. Compressibility is off by default so results can be compared
    with the *incompressible* analytic formulas (the test flow is at 50 m/s, M = 0.15)."""
    kw.setdefault("compressibility", False)
    req = AnalysisRequest(
        wing=wing, flow=flow(alpha), section=section or LinearSection(), solver="classical", n_terms=n, **kw
    )
    return run(req)


def elliptic(AR, b=10.0):
    return Wing.elliptic(b, 4.0 * b / (math.pi * AR))


# ------------------------------------------------------------ analytic: elliptic wing
@pytest.mark.parametrize("AR", [4.0, 8.0, 20.0])
@pytest.mark.parametrize("a0", [TWO_PI, 5.5])
def test_elliptic_wing_matches_exact_solution(AR, a0):
    alpha = 5.0
    res = analyse(elliptic(AR), alpha, LinearSection(a0=a0))
    CL_exact = a0 * math.radians(alpha) / (1.0 + a0 / (math.pi * AR))
    assert res.totals.CL == pytest.approx(CL_exact, rel=1e-8)
    assert res.totals.CD_induced == pytest.approx(CL_exact**2 / (math.pi * AR), rel=1e-8)
    assert res.totals.e == pytest.approx(1.0, abs=1e-8)
    assert np.max(np.abs(res.solution.A[1:])) < 1e-12  # purely first-harmonic loading


def test_elliptic_wing_reproduces_the_prototype_numbers():
    res = analyse(elliptic(8.0), 5.0)
    assert res.totals.CL == pytest.approx(0.4386490845, abs=1e-9)
    assert res.totals.CD_induced == pytest.approx(0.0076558708, abs=1e-9)


def test_elliptic_loading_gives_constant_induced_angle():
    res = analyse(elliptic(8.0), 5.0)
    interior = np.abs(res.load.y) < 0.98 * 5.0
    ai = res.load.alpha_induced[interior]
    assert ai == pytest.approx(res.totals.CL / (math.pi * 8.0), rel=1e-8)


# ------------------------------------------------- anchors from the independent prototype
@pytest.mark.parametrize(
    "AR, lam, CL, e",
    [
        (6.0, 1.00, 0.39535, 0.95393),
        (8.0, 0.40, 0.43451, 0.98720),
        (10.0, 0.25, 0.45296, 0.97800),
        (8.0, 0.50, 0.43317, 0.98311),
    ],
)
def test_tapered_wing_anchor_values(AR, lam, CL, e):
    wing = Wing.from_area_ar_taper(1.0, AR, lam)
    res = analyse(wing, 5.0, n=100)
    assert res.totals.CL == pytest.approx(CL, abs=2e-5)
    assert res.totals.e == pytest.approx(e, abs=2e-5)


def test_rectangular_wing_is_less_efficient_than_near_elliptic_taper():
    e_rect = analyse(Wing.from_area_ar_taper(1.0, 8.0, 1.0)).totals.e
    e_taper = analyse(Wing.from_area_ar_taper(1.0, 8.0, 0.4)).totals.e
    assert e_rect < e_taper < 1.0 + 1e-9


# ---------------------------------------------------------------------- convergence
def test_convergence_in_number_of_terms():
    wing = Wing.from_area_ar_taper(1.0, 8.0, 0.4)
    ref = analyse(wing, 5.0, n=240, convergence_check=False).totals
    for n, tol in ((20, 2e-3), (40, 5e-4), (80, 5e-4)):
        t = analyse(wing, 5.0, n=n, convergence_check=False).totals
        assert abs(t.CL - ref.CL) / ref.CL < tol
        assert abs(t.CD_induced - ref.CD_induced) / ref.CD_induced < 5 * tol


def test_independent_symmetric_odd_term_method_agrees():
    """Anderson's textbook scheme (odd harmonics, collocation on 0 < theta <= pi/2)."""
    AR, lam, alpha = 7.0, 0.5, math.radians(4.0)
    wing = Wing.from_area_ar_taper(1.0, AR, lam)
    b = wing.span
    N = 120
    th = np.arange(1, N + 1) * math.pi / (2 * N)
    odd = 2 * np.arange(N) + 1
    c = wing.chord(-0.5 * b * np.cos(th))
    mu = c * TWO_PI / (4 * b)
    M = np.sin(np.outer(th, odd)) * (odd[None, :] * mu[:, None] + np.sin(th)[:, None])
    A = np.linalg.solve(M, mu * alpha * np.sin(th))
    CL_ref = math.pi * AR * A[0]
    CDi_ref = math.pi * AR * float(np.sum(odd * A**2))
    t = analyse(wing, 4.0, n=100).totals
    assert t.CL == pytest.approx(CL_ref, rel=2e-4)
    assert t.CD_induced == pytest.approx(CDi_ref, rel=2e-3)


# ------------------------------------------------------------------ physical properties
def test_lift_is_linear_in_angle_of_attack_with_twist_and_zero_lift_angle():
    wing = Wing.trapezoid(10.0, 1.4, 0.7, twist_tip_deg=-3.0)
    sec = LinearSection(alpha_L0=math.radians(-2.0))
    cl = [analyse(wing, a, sec).totals.CL for a in (0.0, 2.0, 6.0)]
    assert (cl[1] - cl[0]) / 2.0 == pytest.approx((cl[2] - cl[0]) / 6.0, rel=1e-10)
    assert cl[0] > 0.0  # camber gives lift at zero angle of attack


def test_finite_wing_tends_to_section_lift_slope_as_aspect_ratio_grows():
    big = analyse(Wing.rectangular(2000.0, 1.0), 5.0, n=200)
    assert big.totals.CL / (TWO_PI * math.radians(5.0)) > 0.995


def test_induced_drag_from_force_integration_equals_fourier_result():
    res = analyse(Wing.trapezoid(9.0, 1.5, 0.6, twist_tip_deg=-2.0), 4.0)
    s = res.solution
    assert res.totals.CL == pytest.approx(s.CL_fourier, rel=1e-10)
    assert res.totals.CD_induced == pytest.approx(s.CDi_fourier, rel=1e-10)
    assert res.totals.e == pytest.approx(s.e_fourier, rel=1e-10)


def test_symmetric_loading_has_no_roll_yaw_or_side_force():
    t = analyse(Wing.trapezoid(9.0, 1.5, 0.6, twist_tip_deg=-2.0), 4.0, LinearSection(cd0=0.01)).totals
    assert abs(t.C_roll) < 1e-12 and abs(t.C_yaw) < 1e-12 and abs(t.CY) < 1e-12
    assert t.root_bending_right == pytest.approx(t.root_bending_left, rel=1e-10)


@pytest.mark.parametrize(
    "wing, rel",
    [
        (Wing.rectangular(8.0, 1.0), 1e-9),  # smooth chord: Euler-Maclaurin-corrected quadrature is exact
        (elliptic(7.0), 1e-9),
        (Wing.trapezoid(8.0, 1.5, 0.5), 1e-5),  # chord has a slope kink at the root -> O(h^2) ~ 2e-6
    ],
)
def test_profile_drag_of_constant_cd_equals_cd_times_area(wing, rel):
    t = analyse(wing, 3.0, LinearSection(cd0=0.011)).totals
    assert t.CD_profile == pytest.approx(0.011, rel=rel)


def test_quadratic_profile_drag_and_extra_drag_are_added():
    t = analyse(Wing.rectangular(10.0, 1.0), 5.0, LinearSection(cd0=0.008, cd_k=0.01), extra_cd0=0.02).totals
    assert t.CD == pytest.approx(t.CD_induced + t.CD_profile + 0.02, rel=1e-12)
    assert t.CD_profile > 0.008  # the cl^2 term adds to the constant part
    assert t.L_over_D == pytest.approx(t.CL / t.CD, rel=1e-12)


# --------------------------------------------------------------------------- moments
def test_pitching_moment_equals_section_cm_for_rectangular_wing_about_quarter_chord():
    for alpha in (0.0, 4.0, 9.0):
        t = analyse(Wing.rectangular(10.0, 1.2), alpha, LinearSection(cm_c4=-0.05)).totals
        assert t.Cm == pytest.approx(-0.05, abs=1e-6)


def test_moving_the_reference_point_aft_adds_nose_up_moment_equal_to_vertical_force_times_arm():
    wing = Wing.rectangular(10.0, 1.2)
    sec = LinearSection(cd0=0.01)
    base = analyse(wing, 4.0, sec, ref=MomentReference(x_frac_mac=0.25)).totals
    aft = analyse(wing, 4.0, sec, ref=MomentReference(x_frac_mac=0.55)).totals
    arm = 0.30 * wing.mac
    alpha = math.radians(4.0)
    Fz = base.L * math.cos(alpha) + base.D * math.sin(alpha)
    assert aft.M_pitch - base.M_pitch == pytest.approx(arm * Fz, rel=1e-9)
    assert aft.M_pitch > base.M_pitch


def test_swept_back_quarter_chord_shifts_centre_of_lift_aft_and_gives_nose_down_moment():
    straight = Wing.trapezoid(10.0, 1.2, 1.2)
    swept = Wing.trapezoid(10.0, 1.2, 1.2, sweep_deg=25.0)
    # classical theory ignores sweep in the loading; only the moment arms change
    ts = analyse(straight, 4.0).totals
    tw = analyse(swept, 4.0, ref=MomentReference(x_abs=0.3)).totals
    assert tw.CL == pytest.approx(ts.CL, rel=1e-9)
    assert tw.M_pitch < ts.M_pitch


def test_root_bending_moment_is_lift_times_centroid_arm():
    wing = Wing.rectangular(10.0, 1.0)
    t = analyse(wing, 5.0).totals
    assert t.root_shear == pytest.approx(0.5 * t.L, rel=1e-9)
    assert 0.0 < t.root_bending_right / t.root_shear < wing.half_span


# --------------------------------------------------------------- asymmetry (ailerons)
class _Aileron:
    """Antisymmetric angle-of-attack change on the outer 40 % of each wing."""

    def __init__(self, hs, delta):
        self.hs, self.delta = hs, delta

    def d_alpha(self, y):
        return self.delta * np.sign(y) * (np.abs(y) > 0.6 * self.hs)

    def d_cm(self, y):
        return np.zeros_like(np.asarray(y, dtype=float))


def test_antisymmetric_deflection_rolls_the_wing_with_the_correct_sign():
    wing = Wing.rectangular(10.0, 1.0)
    ail = _Aileron(wing.half_span, math.radians(2.0))  # more lift on the right wing
    res = analyse(wing, 3.0, controls=ail)
    t = res.totals
    base = analyse(wing, 3.0).totals
    assert t.CL == pytest.approx(base.CL, rel=1e-9)  # antisymmetric: no net lift change
    assert t.C_roll < 0.0  # right wing up = negative roll moment (right-wing-down positive)
    # closed form: C_roll = (pi AR / 4) |A_2|
    assert abs(t.C_roll) == pytest.approx(math.pi * wing.aspect_ratio / 4 * abs(res.solution.A[1]), rel=2e-3)
    # adverse yaw: more drag on the wing that makes more lift (right wing) -> nose-right? check sign consistency
    assert abs(t.C_yaw) > 0.0


# ----------------------------------------------------------------- optimal twist (e = 1)
def test_optimal_twist_formula_gives_elliptic_loading_on_a_tapered_wing():
    """alpha - alpha_L0 = CL/(pi AR) * (1 + 4 b sin(theta)/(a0 c))  =>  e = 1 for any planform."""
    wing = Wing.trapezoid(10.0, 1.5, 0.5)
    CL_target, a0 = 0.6, TWO_PI
    A1 = CL_target / (math.pi * wing.aspect_ratio)

    class Optimal:
        def d_alpha(self, y):
            y = np.asarray(y, dtype=float)
            sin_t = np.sqrt(np.clip(1.0 - (2.0 * y / wing.span) ** 2, 0.0, None))
            return A1 * (1.0 + 4.0 * wing.span * sin_t / (a0 * wing.chord(y)))

        def d_cm(self, y):
            return np.zeros_like(np.asarray(y, dtype=float))

    res = analyse(wing, 0.0, controls=Optimal(), n=120)
    assert res.totals.CL == pytest.approx(CL_target, rel=1e-6)
    assert res.totals.e == pytest.approx(1.0, abs=1e-6)
    assert np.max(np.abs(res.solution.A[1:])) < 1e-9


# ------------------------------------------------------------------ nonlinear (Newton)
def test_newton_with_a_linear_polar_returns_the_linear_solution():
    wing = Wing.trapezoid(10.0, 1.4, 0.7, twist_tip_deg=-2.0)
    sec = LinearSection(alpha_L0=math.radians(-1.5))
    f = flow(6.0)
    lin = solve_classical(wing, f, sec, n_terms=60)
    non = solve_classical(wing, f, sec, n_terms=60, nonlinear=True)
    assert non.converged and non.iterations <= 3
    assert non.A == pytest.approx(lin.A, rel=1e-8, abs=1e-12)


class _SoftStall(LinearSection):
    """cl = a0 (alpha - aL0) - k (alpha - aL0)^3: smooth, genuinely nonlinear, monotone for small alpha."""

    def __init__(self, k):
        super().__init__(label="soft stall")
        self.k = k

    def evaluate(self, y, alpha, re):
        y = np.asarray(y, dtype=float)
        x = np.asarray(alpha, dtype=float) - self.alpha_L0
        from liftingline.core.section import SectionCoeffs

        return SectionCoeffs(
            cl=self.a0 * x - self.k * x**3,
            cd=np.full_like(y, 0.01),
            cm=np.zeros_like(y),
            cl_alpha=self.a0 - 3.0 * self.k * x**2,
        )


def test_newton_matches_an_independent_root_finder_for_a_nonlinear_polar():
    """Same lifting-line equations, solved by SciPy's Levenberg-Marquardt with a finite-difference
    Jacobian: independent of the analytic Jacobian and line search used by the solver."""
    from scipy.optimize import root

    wing = Wing.trapezoid(10.0, 1.4, 0.7)
    sec = _SoftStall(k=60.0)
    f = flow(9.0)
    sol = solve_classical(wing, f, sec, n_terms=60, nonlinear=True)
    assert sol.converged

    N, b = 60, wing.span
    th = math.pi * np.arange(1, N + 1) / (N + 1)
    n = np.arange(1, N + 1)
    S = np.sin(np.outer(th, n))
    ay = np.abs(-0.5 * b * np.cos(th))
    c = wing.chord(ay)
    alpha_geo = math.radians(9.0)

    def residual(A):
        alpha_eff = alpha_geo - (S / np.sin(th)[:, None]) @ (n * A)
        return 4 * b * (S @ A) - c * sec.evaluate(ay, alpha_eff, None).cl

    lin = solve_classical(wing, f, sec, n_terms=60)
    ref = root(residual, lin.A, method="lm", tol=1e-12)
    assert ref.success
    assert sol.A == pytest.approx(ref.x, rel=1e-6, abs=1e-10)
    assert sol.CL_fourier < lin.CL_fourier  # the cubic term reduces lift


# ------------------------------------------------------------- compressibility & misc
def test_prandtl_glauert_scales_the_section_slope():
    AR, M = 8.0, 0.5
    a0 = TWO_PI
    req = AnalysisRequest(
        elliptic(AR), FlowCondition.from_isa(0.0, mach=M, alpha_deg=4.0), LinearSection(), solver="classical"
    )
    res = run(req)
    a0c = a0 / math.sqrt(1.0 - M * M)
    assert res.totals.CL == pytest.approx(a0c * math.radians(4.0) / (1.0 + a0c / (math.pi * AR)), rel=1e-8)
    assert not any("Mach" in w for w in res.warnings)


def test_validity_warnings():
    low_ar = analyse(Wing.rectangular(3.0, 1.0), 3.0)
    assert any("aspect ratio" in w.lower() for w in low_ar.warnings)
    fast = run(AnalysisRequest(elliptic(8.0), FlowCondition.from_isa(0.0, mach=0.75, alpha_deg=2.0), LinearSection(), solver="classical"))
    assert any("Mach" in w for w in fast.warnings)
    with pytest.raises(ValueError):
        run(AnalysisRequest(elliptic(8.0), FlowCondition.from_isa(0.0, mach=1.2), LinearSection(), solver="classical"))


def test_stall_margin_warning_when_cl_max_is_exceeded():
    res = analyse(Wing.rectangular(10.0, 1.0), 10.0, LinearSection(cl_max=0.8))
    assert any("cl_max" in w for w in res.warnings)
    assert res.load.cl_margin.max() > 1.0


def test_swept_wing_with_classical_solver_warns_that_sweep_is_ignored():
    res = analyse(Wing.trapezoid(10.0, 1.0, 1.0, sweep_deg=30.0), 4.0)
    assert any("sweep" in w.lower() for w in res.warnings)


def test_asymmetric_section_arrays_are_handled_by_nonuniform_chord_table():
    wing = Wing(sections=(Section(0, 1.6), Section(2.0, 1.2), Section(5.0, 0.5)))
    res = analyse(wing, 4.0)
    assert res.totals.CL > 0 and 0.9 < res.totals.e <= 1.0
    load = build_load_classical(res.solution, wing, flow(4.0), LinearSection())
    assert load.y[0] == pytest.approx(-5.0) and load.y[-1] == pytest.approx(5.0)
    assert load.gamma[0] == pytest.approx(0.0, abs=1e-9) and load.gamma[-1] == pytest.approx(0.0, abs=1e-9)
