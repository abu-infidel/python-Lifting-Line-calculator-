"""Panel method validation against an exact solution (Joukowski airfoil) and thin-airfoil theory."""

import math

import numpy as np
import pytest

from liftingline.airfoil import naca4, naca6
from liftingline.airfoil.geometry import Airfoil
from liftingline.airfoil.panel2d import PanelMethodError, PanelSolver
from liftingline.airfoil.thin_airfoil import thin_airfoil


def joukowski(mu_x=-0.08, mu_y=0.05, b=1.0, n=1441):
    """Joukowski airfoil z = zeta + b^2/zeta (circle through zeta = b, centre (mu_x, mu_y)).

    Returns the unit-chord airfoil (TE at x = 1, no rotation) and the exact potential-flow results:
    circulation G = 4 pi U R sin(alpha + beta), beta = asin(mu_y / R)  =>  cl = 8 pi R sin(alpha + beta) / c.
    """
    R = math.hypot(b - mu_x, mu_y)
    phi_te = math.atan2(-mu_y, b - mu_x)
    phi = phi_te + 2.0 * math.pi * np.arange(n) / n
    zeta = complex(mu_x, mu_y) + R * np.exp(1j * phi)
    z = zeta + b * b / zeta
    pts = np.column_stack([z.real, z.imag])
    pts = np.vstack([pts, pts[:1]])  # close at the cusped trailing edge
    x_min = pts[:, 0].min()
    chord = 2.0 * b - x_min
    pts = (pts - np.array([x_min, 0.0])) / chord
    beta = math.asin(mu_y / R)
    exact = dict(
        cl=lambda alpha: 8.0 * math.pi * R * math.sin(alpha + beta) / chord,
        slope=8.0 * math.pi * R / chord,
        alpha_L0=-beta,
    )
    return Airfoil("joukowski", pts), exact


@pytest.mark.parametrize("mu_x, mu_y", [(-0.02, 0.0), (-0.08, 0.0), (-0.08, 0.05), (-0.12, 0.1), (-0.15, 0.15)])
def test_matches_exact_joukowski_lift_and_zero_lift_angle(mu_x, mu_y):
    af, ex = joukowski(mu_x, mu_y)
    ps = PanelSolver(af, n_panels=240)
    assert ps.lift_slope == pytest.approx(ex["slope"], rel=2e-3)  # measured ~1e-3, first-order in 1/N
    assert ps.alpha_L0 == pytest.approx(ex["alpha_L0"], abs=math.radians(0.03))
    for alpha_deg in (-6.0, 0.0, 3.0, 9.0):
        a = math.radians(alpha_deg)
        s = ps.solve(a)
        assert s.cl_circulation == pytest.approx(ex["cl"](a), abs=8e-3)
        assert s.cl == pytest.approx(ex["cl"](a), abs=1.5e-2)  # pressure integration is ~2x less accurate


def test_error_decreases_with_panel_count():
    af, ex = joukowski(-0.1, 0.06)
    errs = [abs(PanelSolver(af, n_panels=n).lift_slope - ex["slope"]) / ex["slope"] for n in (60, 120, 240, 480)]
    assert errs[-1] < errs[0] / 6  # first-order convergence
    assert errs[-1] < 7e-4


def test_zero_thickness_plate_is_rejected_instead_of_returning_garbage():
    flat, _ = joukowski(0.0, 0.0)  # mu_x = 0: the circle degenerates to a flat plate
    with pytest.raises(PanelMethodError):
        PanelSolver(flat, n_panels=240)


def test_circulation_lift_equals_pressure_lift_and_no_drag():
    ps = PanelSolver(naca4.naca4_airfoil("4412"), n_panels=240)
    for a in (-2.0, 4.0, 10.0):
        s = ps.solve(math.radians(a))
        # pressure integration is ~2x less accurate than the circulation (measured vs Joukowski)
        assert s.cl == pytest.approx(s.cl_circulation, rel=1.2e-2, abs=3e-3)
        assert abs(s.cd) < 4e-3  # d'Alembert: no inviscid drag


def test_lift_is_an_exact_sinusoid_of_the_angle_of_attack():
    ps = PanelSolver(naca4.naca4_airfoil("2412"), n_panels=160)
    for a in np.radians([-8.0, -1.0, 2.5, 7.0, 15.0]):
        assert ps.solve(a).cl_circulation == pytest.approx(ps.lift_slope * math.sin(a - ps.alpha_L0), abs=1e-10)


def test_symmetric_section_has_zero_lift_angle_zero_moment_and_odd_symmetry():
    ps = PanelSolver(naca4.naca4_airfoil("0012"), n_panels=200)
    assert abs(ps.alpha_L0) < math.radians(0.02)
    assert abs(ps.cm_c4_zero_lift) < 1e-3
    a = math.radians(5.0)
    assert ps.solve(-a).cl == pytest.approx(-ps.solve(a).cl, abs=2e-3)


def test_aerodynamic_centre_is_near_quarter_chord_and_moves_aft_with_thickness():
    xs = [PanelSolver(naca4.naca4_airfoil(c), n_panels=240).x_aerodynamic_centre for c in ("0006", "0012", "0018")]
    assert all(0.24 < x < 0.28 for x in xs)
    assert xs[0] < xs[1] < xs[2]  # thickness shifts the potential-flow a.c. slightly aft of c/4
    assert xs[0] == pytest.approx(0.25, abs=0.008)


def test_naca0012_lift_slope_is_close_to_the_thick_airfoil_estimate():
    ps = PanelSolver(naca4.naca4_airfoil("0012"), n_panels=200)
    estimate = 2 * math.pi * (1 + 0.77 * 0.12)  # ~6.86 /rad
    assert 6.6 < ps.lift_slope < 7.2
    assert ps.lift_slope == pytest.approx(estimate, rel=0.03)


@pytest.mark.parametrize("code", ["2412", "4412", "2415"])
def test_cambered_naca_sections_agree_with_thin_airfoil_theory(code):
    c = naca4.parse_naca(code)
    tat = thin_airfoil(lambda x: naca4.camber_slope(x, c.m, c.p))
    ps = PanelSolver(naca4.naca4_airfoil(code), n_panels=240)
    # thickness moves alpha_L0 by ~0.06-0.12 deg (more negative); cm about c/4 stays within ~3 %
    assert math.degrees(ps.alpha_L0) == pytest.approx(tat.alpha_L0_deg, abs=0.2)
    assert math.degrees(ps.alpha_L0) < tat.alpha_L0_deg  # sign of the thickness effect
    assert ps.cm_c4_zero_lift == pytest.approx(tat.cm_c4, abs=0.01)
    assert ps.cm_c4_zero_lift < 0.0  # positive camber -> nose-down moment
    assert ps.lift_slope > 2 * math.pi


def test_six_series_section_zero_lift_angle_follows_the_exact_mean_line():
    af = naca6.naca6_fallback_airfoil("63-412", n_per_surface=200)
    ps = PanelSolver(af, n_panels=240)
    # exact mean line (c_li = 0.4, a = 1): alpha_L0 = -3.65 deg, cm = -c_li/4; the approximate
    # thickness form perturbs both slightly
    assert math.degrees(ps.alpha_L0) == pytest.approx(-0.4 * 9.119, abs=0.25)
    assert ps.cm_c4_zero_lift == pytest.approx(-0.4 / 4, abs=0.02)


def test_stagnation_pressure_and_surface_cp_distribution():
    ps = PanelSolver(naca4.naca4_airfoil("0012"), n_panels=200)
    s = ps.solve(math.radians(4.0))
    assert s.cp.max() == pytest.approx(1.0, abs=0.03)  # stagnation point
    upper = s.y_mid > 0
    assert s.cp[upper].min() < s.cp[~upper].min()  # suction peak on the upper surface at positive alpha
    assert s.cp.min() < -1.0


def test_open_trailing_edge_is_closed_automatically_and_barely_changes_the_result():
    open_te = naca4.naca4_airfoil("0012", closed_te=False)
    closed = naca4.naca4_airfoil("0012", closed_te=True)
    a, b = PanelSolver(open_te, n_panels=200), PanelSolver(closed, n_panels=200)
    assert a.te_closed
    assert a.lift_slope == pytest.approx(b.lift_slope, rel=0.01)


def test_rotated_scaled_and_shifted_input_is_aligned_back():
    """Input rotated by more than ~1 deg is re-aligned to the nose->TE chord, so rotation, scale and
    offset of the input do not matter. (Exact for a symmetric section; a cambered one can only be
    aligned to its nose->TE line, i.e. within the ~0.1 deg nominal-axis tilt of its nose.)"""
    for code, tol_deg in (("0012", 0.01), ("2412", 0.25)):
        base = naca4.naca4_airfoil(code)
        ang = math.radians(6.0)
        rot = np.array([[math.cos(ang), math.sin(ang)], [-math.sin(ang), math.cos(ang)]])
        a = Airfoil.from_raw("a", base.coords)
        b = Airfoil.from_raw("b", 3.0 * (base.coords @ rot.T) + np.array([1.0, -2.0]))
        assert PanelSolver(b, 200).alpha_L0 == pytest.approx(PanelSolver(a, 200).alpha_L0, abs=math.radians(tol_deg))
        assert PanelSolver(b, 200).lift_slope == pytest.approx(PanelSolver(a, 200).lift_slope, rel=2e-3)


def test_well_formed_files_keep_their_own_axes_so_code_and_file_agree():
    """A normalised .dat copy of a NACA section keeps the file's axes (as XFOIL does), so typing the
    code and loading the equivalent file give the same zero-lift angle."""
    nominal = naca4.naca4_airfoil("2412")
    from_file = Airfoil.from_raw("file", nominal.coords)
    assert not any("rotated" in n for n in from_file.notes)
    shift = math.degrees(PanelSolver(from_file, 240).alpha_L0 - PanelSolver(nominal, 240).alpha_L0)
    assert abs(shift) < 0.01
    # ... and a heavily cambered section too
    nominal = naca4.naca4_airfoil("6412")
    shift = math.degrees(PanelSolver(Airfoil.from_raw("f", nominal.coords), 240).alpha_L0 - PanelSolver(nominal, 240).alpha_L0)
    assert abs(shift) < 0.01
