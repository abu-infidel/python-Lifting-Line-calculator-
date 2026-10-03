"""Solver B (numerical lifting line): influence functions, agreement with the classical solver, sweep /
dihedral / sideslip / ground effect, and an independent vortex-lattice reference."""

import math

import numpy as np
import pytest

from liftingline.airfoil.resolver import AirfoilResolver
from liftingline.airfoil.section_builtin import BuiltinEngine
from liftingline.airfoil.section_model import WingSectionModel
from liftingline.core.analysis import AnalysisRequest, run
from liftingline.core.flow import FlowCondition
from liftingline.core.section import LinearSection
from liftingline.core.solver_numerical import (
    NumericalProblem,
    bound_segment_velocity,
    build_load_numerical,
    horseshoe_velocity,
    solve_numerical,
    trailing_leg_velocity,
)
from liftingline.core.wing import Wing
from tests import ref_vlm

TWO_PI = 2.0 * math.pi


def flow(alpha=5.0, beta=0.0, V=50.0):
    return FlowCondition.from_isa(0.0, V=V, alpha_deg=alpha, beta_deg=beta)


def totals(wing, alpha=5.0, beta=0.0, solver="numerical", section=None, n_panels=80, **kw):
    req = AnalysisRequest(
        wing,
        flow(alpha, beta),
        section or LinearSection(),
        solver=solver,
        compressibility=False,
        convergence_check=False,
        n_panels=n_panels,
        **kw,
    )
    return run(req).totals


def rect(AR=8.0, **kw):
    return Wing.from_area_ar_taper(1.0, AR, 1.0, **kw)


# =============================================================================== influence functions
def test_finite_segment_tends_to_the_infinite_line_value():
    A, B = np.array([0.0, -500.0, 0.0]), np.array([0.0, 500.0, 0.0])
    P = np.array([[0.0, 0.0, 0.5]])  # 0.5 m above the middle of a very long segment
    v = bound_segment_velocity(P, A, B)
    assert np.linalg.norm(v) == pytest.approx(1.0 / (TWO_PI * 0.5), rel=1e-5)  # Gamma / (2 pi d)
    assert v[0] == pytest.approx([1.0 / (TWO_PI * 0.5), 0.0, 0.0], rel=1e-5)  # along +x above a +y vortex


def test_semi_infinite_leg_induces_half_the_infinite_line_value():
    v = trailing_leg_velocity(np.array([[0.0, 0.5, 0.0]]), np.zeros(3), np.array([1.0, 0.0, 0.0]))
    assert np.linalg.norm(v) == pytest.approx(1.0 / (4.0 * math.pi * 0.5), rel=1e-12)


def test_a_vortex_induces_nothing_on_its_own_line():
    A, B = np.array([0.0, -1.0, 0.0]), np.array([0.0, 1.0, 0.0])
    assert np.linalg.norm(bound_segment_velocity(np.array([[0.0, 0.3, 0.0]]), A, B)) == 0.0
    assert np.linalg.norm(bound_segment_velocity(np.array([[0.0, 3.0, 0.0]]), A, B)) == 0.0


def test_horseshoe_gives_downwash_between_the_trailing_legs_and_upwash_outboard():
    A, B = np.array([0.0, -1.0, 0.0]), np.array([0.0, 1.0, 0.0])
    u = np.array([1.0, 0.0, 0.0])
    inside = horseshoe_velocity(np.array([[0.0, 0.0, 0.0]]), A, B, u)  # on the bound line, at the centre
    outside = horseshoe_velocity(np.array([[0.0, 2.0, 0.0]]), A, B, u)
    assert inside[0, 2] < 0.0 < outside[0, 2]  # downwash for positive (lift-producing) circulation


def test_influence_functions_agree_with_the_independently_coded_angle_formula():
    rng = np.random.default_rng(7)
    P = rng.uniform(-2, 2, size=(40, 3))
    A, B = np.array([0.3, -1.1, 0.2]), np.array([-0.2, 0.9, 0.4])
    u = np.array([math.cos(0.2), -0.1, math.sin(0.2)])
    u /= np.linalg.norm(u)
    assert bound_segment_velocity(P, A, B) == pytest.approx(ref_vlm.segment_velocity(P, A, B), rel=1e-9, abs=1e-12)
    assert trailing_leg_velocity(P, A, u) == pytest.approx(ref_vlm.leg_velocity(P, A, u), rel=1e-9, abs=1e-12)


def test_analytic_jacobian_matches_finite_differences_for_a_swept_dihedral_wing_in_sideslip():
    wing = rect(6.0, sweep_deg=15.0, dihedral_deg=4.0, twist_tip_deg=-2.0)
    prob = NumericalProblem(wing, flow(5.0, 3.0), LinearSection(), n_panels=12)
    rng = np.random.default_rng(1)
    G = 0.5 * prob.flow.V * prob.chord_n * TWO_PI * math.radians(3) * (1 + 0.1 * rng.standard_normal(prob.N))
    _, aux = prob.residual(G)
    J = prob.jacobian(G, aux)
    h = 1e-6 * np.abs(G).max()
    Jfd = np.zeros_like(J)
    for k in range(prob.N):
        e = np.zeros(prob.N)
        e[k] = h
        Jfd[:, k] = (prob.residual(G + e)[0] - prob.residual(G - e)[0]) / (2 * h)
    assert np.max(np.abs(J - Jfd)) / np.max(np.abs(J)) < 1e-8


# ================================================================ agreement with the classical solver
@pytest.mark.parametrize(
    "wing",
    [
        Wing.elliptic(10.0, 4.0 * 10.0 / (math.pi * 8.0)),
        rect(6.0),
        rect(10.0),
        Wing.from_area_ar_taper(1.0, 8.0, 0.4),
        Wing.from_area_ar_taper(1.0, 10.0, 0.25),
        Wing.trapezoid(10.0, 1.2, 0.6, twist_tip_deg=-3.0),
        Wing.trapezoid(10.0, 1.2, 0.6, twist_tip_deg=2.0, incidence_deg=1.5),
    ],
    ids=["elliptic8", "rect6", "rect10", "taper0.4", "taper0.25", "twist-3", "twist+2 inc"],
)
def test_planar_unswept_wings_agree_with_the_classical_solver(wing):
    sec = LinearSection(alpha_L0=math.radians(-1.5))
    a = totals(wing, 5.0, solver="classical", section=sec)
    b = totals(wing, 5.0, solver="numerical", section=sec)
    assert b.CL == pytest.approx(a.CL, rel=5e-3)  # plan: CL within 0.5 %
    assert b.CD_induced == pytest.approx(a.CD_induced, rel=1e-2)  # plan: CDi within 1 %
    assert b.e == pytest.approx(a.e, rel=1.2e-2)
    assert b.Cm == pytest.approx(a.Cm, abs=2e-3)


def test_the_elliptic_wing_is_recovered_with_the_exact_efficiency_and_lift():
    AR, a0 = 8.0, TWO_PI
    t = totals(Wing.elliptic(10.0, 4.0 * 10.0 / (math.pi * AR)), 5.0)
    assert t.e == pytest.approx(1.0, abs=1e-2)
    assert t.CL == pytest.approx(a0 * math.radians(5.0) / (1 + a0 / (math.pi * AR)), rel=3e-3)


def test_solution_converges_with_the_number_of_elements():
    ref = totals(rect(6.0), n_panels=320)
    # horseshoe lifting lines converge about first order in the element count (measured errors against the
    # 320-panel run: CL 1.9e-2, 9.5e-3, 4.5e-3, 1.9e-3 for 10/20/40/80 panels; CDi 1.0e-2 ... 6.9e-4)
    errs = [abs(totals(rect(6.0), n_panels=n).CL - ref.CL) / ref.CL for n in (10, 20, 40, 80)]
    assert errs[-1] < errs[0] / 4 and errs[-1] < 3e-3
    erri = [abs(totals(rect(6.0), n_panels=n).CD_induced - ref.CD_induced) / ref.CD_induced for n in (10, 20, 40, 80)]
    assert erri[-1] < erri[0] / 3 and erri[-1] < 1.5e-3


def test_newton_converges_quadratically_with_the_classical_start():
    sol = solve_numerical(rect(8.0, sweep_deg=20.0), flow(5.0), LinearSection(), n_panels=60)
    assert sol.converged and sol.iterations <= 8 and sol.residual < 1e-10


# ======================================================================= symmetry and sideslip physics
@pytest.mark.parametrize(
    "wing",
    [rect(8.0), rect(8.0, sweep_deg=25.0), rect(8.0, sweep_deg=-15.0, dihedral_deg=5.0, twist_tip_deg=-2.0)],
    ids=["planar", "swept", "swept+dihedral+twist"],
)
def test_symmetric_flow_gives_symmetric_loads(wing):
    t = totals(wing, 4.0)
    assert abs(t.C_roll) < 1e-12 and abs(t.C_yaw) < 1e-12 and abs(t.CY) < 1e-12


def test_sideslip_results_are_odd_in_beta_and_lift_and_drag_even():
    wing = rect(8.0, sweep_deg=20.0, dihedral_deg=4.0)
    p, m = totals(wing, 4.0, beta=6.0), totals(wing, 4.0, beta=-6.0)
    assert p.CL == pytest.approx(m.CL, rel=1e-9) and p.CD == pytest.approx(m.CD, rel=1e-9)
    for key in ("C_roll", "C_yaw", "CY"):
        assert getattr(p, key) == pytest.approx(-getattr(m, key), rel=1e-9, abs=1e-12)


def test_dihedral_and_sweepback_are_stabilising_and_forward_sweep_is_not():
    beta = 5.0
    planar = totals(rect(8.0), 4.0, beta=beta).C_roll
    dihedral = totals(rect(8.0, dihedral_deg=6.0), 4.0, beta=beta).C_roll
    back = totals(rect(8.0, sweep_deg=30.0), 4.0, beta=beta).C_roll
    forward = totals(rect(8.0, sweep_deg=-30.0), 4.0, beta=beta).C_roll
    # right-wing-down positive: wind from the right (beta > 0) should roll a stable wing to the left (negative)
    assert dihedral < 0.0 and back < 0.0 < forward
    assert dihedral < planar and back < planar < forward


def _vlm(wing, **kw):
    """Reference lattice at a resolution where the sweep effects are converged (cached: it is the slow part)."""
    h = 0.5 * wing.half_span
    key = (wing.span, wing.area, float(wing.chord(0.0)), float(wing.chord(h)), float(wing.x_le(h)), tuple(sorted(kw.items())))
    if key not in _VLM_CACHE:
        _VLM_CACHE[key] = ref_vlm.vlm(wing, kw.pop("alpha", 4.0), kw.pop("n_span", 60), kw.pop("n_chord", 12), **kw)
    return _VLM_CACHE[key]


_VLM_CACHE: dict = {}


@pytest.mark.parametrize("sweep", [30.0, -30.0])
def test_roll_due_to_sideslip_from_sweep_agrees_with_the_vortex_lattice(sweep):
    """The sweep contribution to the roll moment in sideslip (swept minus unswept wing). The lattice uses a
    body-aligned wake, the only choice that converges in sideslip; the lifting line over-predicts it by 20-30 %."""
    wing, planar = rect(8.0, sweep_deg=sweep), rect(8.0)
    ll = totals(wing, 4.0, beta=5.0).C_roll - totals(planar, 4.0, beta=5.0).C_roll
    vl = _vlm(wing, beta_deg=5.0, wake="body")["C_roll"]
    assert np.sign(ll) == np.sign(vl) == -np.sign(sweep)  # sweepback is stabilising, forward sweep is not
    assert 0.9 < ll / vl < 1.5


def test_roll_due_to_sideslip_of_an_unswept_wing_is_small_and_converged():
    """Regression guard: with a bare (unregularised) lifting line the skewed-wake term grows without bound as
    the element count grows (2.6e-3 -> 4.3e-3 between 20 and 320 panels) and was ten times too large."""
    wing = rect(8.0)
    rolls = [totals(wing, 4.0, beta=5.0, n_panels=n).C_roll for n in (20, 40, 80, 160)]
    assert all(0.0 < r < 1.3e-3 for r in rolls)  # an order of magnitude below the swept-wing values (3-4e-3)
    assert rolls[-1] == pytest.approx(rolls[-2], rel=0.02)


# ====================================================================================== sweep and dihedral
def test_lift_slope_of_a_swept_wing_follows_the_classical_result_at_high_aspect_ratio():
    """AR 60: the swept-wing lift slope must approach the infinite-wing result a0 cos(L) (with the finite-AR
    correction): the check that the normal-plane section closure and the swept Biot-Savart geometry are right."""

    def slope(w):
        return (totals(w, 4.0, n_panels=120).CL - totals(w, 2.0, n_panels=120).CL) / math.radians(2.0)

    AR, a = 60.0, TWO_PI
    base = slope(rect(AR))
    for sweep in (15.0, 30.0, 45.0):
        c = math.cos(math.radians(sweep))
        expect = (a * c / (1 + a * c / (math.pi * AR))) / (a / (1 + a / (math.pi * AR)))
        assert slope(rect(AR, sweep_deg=sweep)) / base == pytest.approx(expect, rel=4e-3)


def test_sweep_reduces_the_lift_slope_monotonically_and_dihedral_modestly():
    cls = [totals(rect(8.0, sweep_deg=s), 4.0).CL for s in (0.0, 15.0, 30.0, 45.0)]
    assert np.all(np.diff(cls) < 0)
    flat = totals(rect(8.0), 4.0).CL
    ratios = [totals(rect(8.0, dihedral_deg=d), 4.0).CL / flat for d in (5.0, 10.0, 20.0, 30.0)]
    assert ratios[-1] < ratios[-2] < ratios[-3]  # more dihedral, less lift once past the small-angle regime
    assert all(math.cos(math.radians(d)) ** 2 < r < 1.02 for d, r in zip((5.0, 10.0, 20.0, 30.0), ratios))


WING_8_05 = Wing.from_area_ar_taper(1.0, 8.0, 0.5)


@pytest.mark.parametrize("sweep, cl_tol, k_tol", [(15.0, 0.03, 0.10), (30.0, 0.05, 0.10), (45.0, 0.07, 0.10)])
def test_sweep_effect_agrees_with_the_vortex_lattice(sweep, cl_tol, k_tol):
    """Lift and induced-drag factor K = CDi / CL^2 of the swept wing *relative to the same wing unswept*, so that
    the planar Prandtl baseline cancels. The core radii were calibrated on this quantity (see the module
    docstring of the solver): errors of 0-5 % in CL and below 7 % in K across sweeps of +-45 deg."""
    wing = Wing.from_area_ar_taper(1.0, 8.0, 0.5, sweep_deg=sweep)
    ll, ll0 = totals(wing, 4.0), totals(WING_8_05, 4.0)
    vl, vl0 = _vlm(wing), _vlm(WING_8_05)
    assert ll.CL / ll0.CL == pytest.approx(vl["CL"] / vl0["CL"], rel=cl_tol)
    k_ll = (ll.CD_induced / ll.CL**2) / (ll0.CD_induced / ll0.CL**2)
    k_vl = (vl["CDi"] / vl["CL"] ** 2) / (vl0["CDi"] / vl0["CL"] ** 2)
    assert k_ll == pytest.approx(k_vl, rel=k_tol)


@pytest.mark.parametrize("AR, taper, sweep", [(4.0, 0.5, 30.0), (6.0, 0.3, 40.0), (12.0, 1.0, 30.0), (8.0, 0.6, -20.0)])
def test_other_swept_planforms_and_the_spanwise_load_shape_agree_with_the_vortex_lattice(AR, taper, sweep):
    wing = Wing.from_area_ar_taper(1.0, AR, taper, sweep_deg=sweep)
    base = Wing.from_area_ar_taper(1.0, AR, taper)
    r = run(AnalysisRequest(wing, flow(4.0), LinearSection(), solver="numerical", compressibility=False, convergence_check=False))
    r0 = totals(base, 4.0)
    vl, vl0 = _vlm(wing), _vlm(base)
    assert r.totals.CL / r0.CL == pytest.approx(vl["CL"] / vl0["CL"], rel=0.07)
    # normalised circulation distribution (shape only), away from the tips
    y = r.load.y
    g = r.load.gamma / r.load.gamma.max()
    gv = np.interp(y, vl["y"], vl["gamma"] / vl["gamma"].max())
    inner = np.abs(y) < 0.9 * wing.half_span
    assert np.max(np.abs(g[inner] - gv[inner])) < 0.09


@pytest.mark.parametrize("AR, taper, sweep", [(8.0, 0.5, 15.0), (8.0, 0.5, 30.0), (4.0, 0.5, 30.0), (6.0, 0.3, 20.0), (12.0, 1.0, 30.0)])
def test_swept_lift_slope_is_close_to_the_helmbold_datcom_estimate(AR, taper, sweep):
    """Independent empirical check (not a lattice): CL_alpha = 2 pi AR / (2 + sqrt(4 + AR^2 (1 + tan^2 L_c/2)))."""
    tan_half = math.tan(math.radians(sweep)) - (1.0 / AR) * (1.0 - taper) / (1.0 + taper)
    cla = 2.0 * math.pi * AR / (2.0 + math.sqrt(4.0 + AR**2 * (1.0 + tan_half**2)))
    wing = Wing.from_area_ar_taper(1.0, AR, taper, sweep_deg=sweep)
    assert totals(wing, 4.0).CL == pytest.approx(cla * math.radians(4.0), rel=0.05)


@pytest.mark.parametrize("sweep", [15.0, 30.0, 45.0, -20.0])
def test_swept_solution_converges_with_the_element_count(sweep):
    """With the chord-scaled cores the solution has a limit: the change between 80 and 160 panels is tiny."""
    wing = Wing.from_area_ar_taper(1.0, 8.0, 0.5, sweep_deg=sweep)
    a, b, c = (totals(wing, 4.0, n_panels=n) for n in (40, 80, 160))
    assert abs(b.CL / c.CL - 1.0) < 2e-3 and abs(a.CL / c.CL - 1.0) < 6e-3
    assert abs(b.CD_induced / c.CD_induced - 1.0) < 6e-3


def test_without_the_cores_a_swept_lifting_line_has_no_limit():
    """Why the regularisation exists: a bare lifting line loses ~4 % of its lift per doubling of the elements."""
    wing = Wing.from_area_ar_taper(1.0, 8.0, 0.5, sweep_deg=30.0)

    def cl(n, **kw):
        sol = solve_numerical(wing, flow(4.0), LinearSection(), n_panels=n, **kw)
        return build_load_numerical(sol, wing, flow(4.0), LinearSection()).force_kj[:, 2].sum()

    bare = [cl(n, leg_core=0.0, bound_core=0.0) for n in (40, 80, 160, 320)]
    assert all(bare[i + 1] < 0.97 * bare[i] for i in range(3))
    cored = [cl(n) for n in (40, 80, 160, 320)]
    assert abs(cored[-1] / cored[-2] - 1.0) < 3e-3


@pytest.mark.parametrize("wing", [rect(8.0), Wing.from_area_ar_taper(1.0, 8.0, 0.4), Wing.elliptic(10.0, 1.0)])
def test_cores_do_not_change_planar_unswept_wings(wing):
    """Collinear bound segments induce nothing and the symmetric leg field is untouched: continuity with Prandtl."""
    a = solve_numerical(wing, flow(5.0), LinearSection(), n_panels=40)
    b = solve_numerical(wing, flow(5.0), LinearSection(), n_panels=40, leg_core=0.0, bound_core=0.0)
    assert a.gamma == pytest.approx(b.gamma, rel=1e-10)


# ======================================================================================== ground effect
def test_ground_effect_limits_and_monotone_trends():
    wing = rect(8.0)
    free = totals(wing, 4.0)
    far = totals(wing, 4.0, ground_height=3.0 * wing.span)
    assert far.CL == pytest.approx(free.CL, rel=2e-3) and far.CD_induced == pytest.approx(free.CD_induced, rel=5e-3)
    cl, cdi = [], []
    for hb in (1.0, 0.5, 0.25, 0.15):
        g = totals(wing, 4.0, ground_height=hb * wing.span)
        cl.append(g.CL / free.CL)
        cdi.append(g.CD_induced / free.CD_induced)
    assert np.all(np.diff(cl) > 0) and cl[-1] > 1.0  # lift rises as the ground is approached
    assert np.all(np.diff(cdi) < 0) and cdi[-1] < 0.85  # induced drag falls


@pytest.mark.parametrize("ar, hb", [(8.0, 1.0), (8.0, 0.5), (16.0, 0.25)], ids=["h=8c", "h=4c", "h=4c-high-AR"])
def test_ground_effect_agrees_with_the_vortex_lattice_with_images(ar, hb):
    """Where the ground is several chords away the image-vortex model matches the lifting-surface reference."""
    wing = rect(ar)
    free_l, free_v = totals(wing, 4.0), ref_vlm.vlm(wing, 4.0, 30, 6)
    h = hb * wing.span
    ll = totals(wing, 4.0, ground_height=h)
    vl = ref_vlm.vlm(wing, 4.0, 30, 6, ground_height=h)
    assert ll.CL / free_l.CL == pytest.approx(vl["CL"] / free_v["CL"], abs=0.015)
    assert ll.CD_induced / free_l.CD_induced == pytest.approx(vl["CDi"] / free_v["CDi"], abs=0.03)


def test_lifting_line_misses_the_chordwise_ground_effect_by_the_documented_amount():
    """A lifting line has no chordwise extent, so near the ground it under-predicts the lift gain by about
    7 % x (chord/height)^2 of CL (the figure quoted in the warning); the gap depends on h/c, not on the span."""
    gap = {}
    for ar in (8.0, 12.0):
        wing = rect(ar)
        free_l, free_v = totals(wing, 4.0), ref_vlm.vlm(wing, 4.0, 30, 6)
        for hc in (1.0, 2.0, 4.0):
            h = hc * wing.mac
            ll = totals(wing, 4.0, ground_height=h).CL / free_l.CL
            vl = ref_vlm.vlm(wing, 4.0, 30, 6, ground_height=h)["CL"] / free_v["CL"]
            gap[ar, hc] = vl - ll
    for ar in (8.0, 12.0):
        assert gap[ar, 1.0] > gap[ar, 2.0] > gap[ar, 4.0] > 0  # lifting line under-predicts, more so when closer
        for hc in (1.0, 2.0, 4.0):
            assert gap[ar, hc] == pytest.approx(0.07 / hc**2, rel=0.4)


def test_low_ground_clearance_warns_and_quantifies_the_missing_chordwise_effect():
    wing = rect(8.0)

    def warnings_at(h_over_mac):
        req = AnalysisRequest(wing, flow(4.0), LinearSection(), solver="numerical",
                              ground_height=h_over_mac * wing.mac, convergence_check=False)
        return [w for w in run(req).warnings if "Ground effect" in w]

    assert "more than 15 %" in warnings_at(0.3)[0]
    assert "about 7 %" in warnings_at(1.0)[0] and "about 2 %" in warnings_at(1.9)[0]
    assert warnings_at(2.5) == []


# ======================================================================= the pipeline, loads, controls
def test_auto_selection_uses_the_numerical_solver_for_sweep_dihedral_sideslip_and_ground():
    base = dict(section=LinearSection(), convergence_check=False)
    for wing, beta, gh, reason in (
        (rect(8.0, sweep_deg=25.0), 0.0, None, "sweep"),
        (rect(8.0, dihedral_deg=5.0), 0.0, None, "dihedral"),
        (rect(8.0), 4.0, None, "sideslip"),
        (rect(8.0), 0.0, 5.0, "ground"),
    ):
        r = run(AnalysisRequest(wing, flow(4.0, beta), ground_height=gh, **base))
        assert r.solver_used == "numerical" and reason in r.info["solver_reason"]
    r = run(AnalysisRequest(rect(8.0), flow(4.0), **base))
    assert r.solver_used == "classical"


def test_sweep_warning_quantifies_the_lifting_line_accuracy():
    def sweep_warnings(sweep):
        r = run(AnalysisRequest(rect(8.0, sweep_deg=sweep), flow(4.0), LinearSection(), convergence_check=False))
        return [w for w in r.warnings if "Quarter-chord sweep" in w]

    assert "2-3 % at 30 deg" in sweep_warnings(30.0)[0]
    assert sweep_warnings(10.0) == [] and sweep_warnings(-10.0) == []


def test_loads_are_consistent_with_the_solution():
    wing = rect(8.0, sweep_deg=15.0, dihedral_deg=3.0)
    r = run(AnalysisRequest(wing, flow(4.0), LinearSection(cd0=0.01), solver="numerical", compressibility=False))
    load = r.load
    assert load.cl == pytest.approx(load.extras["cl_section"], rel=1e-7)  # converged: circulation lift = section lift
    assert np.all(load.alpha_induced > 0) and r.totals.root_shear > 0 and r.totals.root_bending_right > 0
    assert r.totals.CD_profile == pytest.approx(0.01, rel=2e-2)  # cd0 on the (projected) area, local dynamic pressure
    assert "convergence_dCL" in r.info and r.info["convergence_dCL"] < 5e-3


class _Aileron:
    def __init__(self, hs, delta):
        self.hs, self.delta = hs, delta

    def d_alpha(self, y):
        return self.delta * np.sign(y) * (np.abs(y) > 0.6 * self.hs)

    def d_cm(self, y):
        return np.zeros_like(np.asarray(y, dtype=float))


def test_aileron_roll_moment_matches_the_classical_solver():
    wing = rect(8.0)
    ail = _Aileron(wing.half_span, math.radians(2.0))
    a = totals(wing, 3.0, solver="classical", controls=ail)
    b = totals(wing, 3.0, solver="numerical", controls=ail)
    assert a.C_roll < 0 and b.C_roll < 0  # more lift on the right wing: right wing up
    assert b.C_roll == pytest.approx(a.C_roll, rel=0.06)
    assert b.CL == pytest.approx(totals(wing, 3.0, solver="numerical").CL, rel=2e-3)


# ================================================================================== real section engines
@pytest.fixture(scope="module")
def c172_model():
    wing = Wing.trapezoid(11.0, 1.63, 1.13, twist_tip_deg=-3.0, airfoil_root="NACA 2412", airfoil_tip="NACA 2412")
    return wing, WingSectionModel.from_wing(wing, AirfoilResolver(), BuiltinEngine())


def test_numerical_and_classical_agree_with_real_polars_on_a_planar_wing(c172_model):
    wing, model = c172_model
    for alpha in (2.0, 6.0, 12.0):
        a = totals(wing, alpha, solver="classical", section=model)
        b = totals(wing, alpha, solver="numerical", section=model)
        assert b.CL == pytest.approx(a.CL, rel=0.012)
        assert b.CD == pytest.approx(a.CD, rel=0.05)
        assert b.Cm == pytest.approx(a.Cm, abs=3e-3)


def test_numerical_solver_handles_a_swept_wing_with_real_polars_up_to_stall(c172_model):
    _, model = c172_model
    wing = Wing.trapezoid(11.0, 1.63, 1.13, sweep_deg=15.0, twist_tip_deg=-3.0, airfoil_root="NACA 2412", airfoil_tip="NACA 2412")
    sec = WingSectionModel.from_wing(wing, AirfoilResolver(), BuiltinEngine())
    cls = []
    for alpha in (0.0, 5.0, 10.0, 15.0, 20.0, 24.0):
        r = run(AnalysisRequest(wing, flow(alpha), sec, convergence_check=False))
        assert r.solver_used == "numerical" and r.solution.converged
        cls.append(r.totals.CL)
    assert np.all(np.diff(cls) > 0) and 1.2 < cls[-1] < 1.9


def test_linearised_mode_uses_the_linear_lift_but_real_drag(c172_model):
    wing, model = c172_model
    lin = totals(wing, 4.0, solver="numerical", section=model, nonlinear=False)
    non = totals(wing, 4.0, solver="numerical", section=model, nonlinear=True)
    assert lin.CL == pytest.approx(non.CL, rel=0.015)
    assert lin.CD_profile == pytest.approx(non.CD_profile, rel=0.03) and lin.CD_profile > 0.004
