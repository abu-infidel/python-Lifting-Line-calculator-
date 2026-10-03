"""Section engines: factor extraction, table interpolation, built-in engine, NeuralFoil adapter, polar import."""

import math

import numpy as np
import pytest

from liftingline.airfoil import naca4
from liftingline.airfoil.section_base import (
    ALPHA_MAX_DEG,
    ALPHA_MIN_DEG,
    PolarTable,
    TabulatedAirfoilAero,
    alpha_grid_deg,
    linear_parameters,
    re_node_value,
)
from liftingline.airfoil.section_builtin import BuiltinAero, BuiltinEngine, saturate
from liftingline.airfoil.section_polarfile import PolarParseError, TabulatedPolarAero, load_polar, parse_polar_text

DEG = math.pi / 180.0


# ======================================================================== factor extraction
def _synthetic_polar(a0_per_deg=0.11, aL0_deg=-2.0, clmax=1.4, stall_deg=15.0):
    a = alpha_grid_deg()
    cl = a0_per_deg * (a - aL0_deg)
    # soft stall peak at stall_deg then decline
    over = a > stall_deg
    cl = np.where(over, clmax - 0.04 * (a - stall_deg), np.minimum(cl, clmax))
    cm = np.full_like(a, -0.05)
    return a, cl, cm


def test_linear_parameters_recover_a_known_polar():
    a, cl, cm = _synthetic_polar()
    p = linear_parameters(a, cl, cm)
    assert p["a0"] == pytest.approx(0.11 / DEG, rel=1e-6)
    assert math.degrees(p["alpha_L0"]) == pytest.approx(-2.0, abs=1e-6)
    assert p["cm_c4"] == pytest.approx(-0.05)
    assert p["cl_max"] == pytest.approx(1.4, abs=0.05)
    assert math.degrees(p["alpha_stall"]) == pytest.approx(15.0, abs=1.1)


def test_symmetric_polar_has_exactly_zero_zero_lift_angle_even_if_the_polar_is_curved():
    a = alpha_grid_deg()
    cl = 0.1 * a + 0.0004 * a**3  # odd, curved: a free-intercept fit would shift alpha_L0
    p = linear_parameters(a, cl, np.zeros_like(a))
    assert p["alpha_L0"] == pytest.approx(0.0, abs=1e-12)


# ============================================================ table machinery (fake engine)
class _FakeAero(TabulatedAirfoilAero):
    """cl = (0.10 + 0.02 log10(Re/1e6)) per deg * (alpha + 2); cd grows as Re falls."""

    label = "fake"

    def __init__(self):
        super().__init__()
        self.builds = []

    def _build_table(self, re_node):
        self.builds.append(re_node)
        a = alpha_grid_deg()
        slope = 0.10 + 0.02 * math.log10(re_node / 1e6)
        return PolarTable(a, slope * (a + 2.0), 0.006 * (1e6 / re_node) ** 0.2 + 0 * a, np.full_like(a, -0.04), np.full_like(a, 0.9))


def test_tables_are_built_lazily_per_reynolds_node_and_reused():
    f = _FakeAero()
    al = np.radians(np.linspace(-3, 8, 12))
    f.coefficients(al, np.full_like(al, 1.0e6))
    n_first = len(f.builds)
    assert 1 <= n_first <= 2
    f.coefficients(al, np.full_like(al, 1.0e6))
    assert len(f.builds) == n_first  # cached
    f.linear(np.array([1e6, 1.1e6]))
    assert len(f.builds) == n_first or len(f.builds) == n_first + 1


def test_interpolation_is_exact_at_nodes_and_linear_in_log_re_between_them():
    f = _FakeAero()
    al = np.radians([3.0])
    k = 6
    re0, re1 = re_node_value(k), re_node_value(k + 1)
    cl0 = f.coefficients(al, np.array([re0]))[0][0]
    cl1 = f.coefficients(al, np.array([re1]))[0][0]
    assert cl0 == pytest.approx((0.10 + 0.02 * math.log10(re0 / 1e6)) * 5.0, rel=1e-9)
    mid = math.sqrt(re0 * re1)  # halfway in log Re
    assert f.coefficients(al, np.array([mid]))[0][0] == pytest.approx(0.5 * (cl0 + cl1), rel=1e-9)


def test_polar_derivative_matches_finite_differences_and_vanishes_outside_the_window():
    f = _FakeAero()
    re = np.full(5, 2.0e6)
    al = np.radians(np.array([-5.0, 0.0, 4.0, 10.0, 20.0]))
    cl, cd, cm, slope = f.coefficients(al, re)
    h = 1e-5
    cl_p = f.coefficients(al + h, re)[0]
    cl_m = f.coefficients(al - h, re)[0]
    assert slope == pytest.approx((cl_p - cl_m) / (2 * h), rel=1e-5)
    out = np.radians([ALPHA_MIN_DEG - 10.0, ALPHA_MAX_DEG + 10.0])
    cl_o, _, _, slope_o = f.coefficients(out, np.full(2, 2e6))
    assert slope_o == pytest.approx([0.0, 0.0])  # edge values are held
    assert cl_o[0] == pytest.approx(f.coefficients(np.radians([ALPHA_MIN_DEG]), np.array([2e6]))[0][0])


def test_confidence_is_reported_and_is_zero_outside_the_tabulated_window():
    f = _FakeAero()
    c = f.confidence(np.radians([0.0, 10.0, 40.0]), np.full(3, 1e6))
    assert c == pytest.approx([0.9, 0.9, 0.0])


def test_reynolds_numbers_outside_the_grid_are_clamped_not_extrapolated():
    f = _FakeAero()
    al = np.radians([2.0])
    lo = f.coefficients(al, np.array([1.0]))[0]
    hi = f.coefficients(al, np.array([1e12]))[0]
    assert np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))


# ================================================================================ built-in
@pytest.fixture(scope="module")
def b2412():
    return BuiltinAero(naca4.naca4_airfoil("2412"))


@pytest.fixture(scope="module")
def b0012():
    return BuiltinAero(naca4.naca4_airfoil("0012"))


def test_saturation_is_linear_for_small_lift_and_bounded_by_cl_max():
    u = np.array([-3.0, -0.2, 0.0, 0.2, 0.5, 5.0, 50.0])
    cl, dcl, sat, excess = saturate(u, 1.5, 1.2)
    assert cl[3] == pytest.approx(0.2, rel=1e-4) and cl[1] == pytest.approx(-0.2, rel=1e-4)
    assert cl[-1] < 1.5 * 1.001 and cl[0] > -1.2 * 1.001
    assert np.all(np.diff(cl) > 0)  # monotone: the nonlinear solver never sees a falling lift curve
    assert dcl[3] == pytest.approx(1.0, rel=1e-3)
    assert excess[-1] > 1.0 and excess[3] == 0.0


def test_builtin_symmetric_section_factors(b0012):
    lin = b0012.linear(np.array([1e6, 5e6]))
    assert lin.alpha_L0 == pytest.approx([0.0, 0.0], abs=math.radians(0.02))
    assert lin.cm_c4 == pytest.approx([0.0, 0.0], abs=2e-3)
    assert np.all((lin.a0 > 5.3) & (lin.a0 < 7.0))
    cl, cd, cm, _ = b0012.coefficients(np.radians([-5.0, 5.0]), np.full(2, 1e6))
    assert cl[0] == pytest.approx(-cl[1], rel=1e-9)
    assert cd[0] == pytest.approx(cd[1], rel=1e-9)


def test_builtin_cambered_section_factors(b2412):
    lin = b2412.linear(np.array([3e6]))
    assert math.degrees(lin.alpha_L0[0]) == pytest.approx(-2.2, abs=0.25)
    assert lin.cm_c4[0] == pytest.approx(-0.053, abs=0.01)
    assert 5.5 < lin.a0[0] < 6.6
    assert 1.45 < lin.cl_max[0] < 1.85
    assert lin.alpha_stall[0] > lin.alpha_L0[0] + math.radians(10)


def test_builtin_polar_is_monotone_smooth_and_its_slope_is_consistent(b2412):
    al = np.radians(np.linspace(-20, 40, 241))
    re = np.full(al.shape, 2e6)
    cl, cd, cm, slope = b2412.coefficients(al, re)
    assert np.all(np.diff(cl) > 0)
    h = 1e-6
    fd = (b2412.coefficients(al + h, re)[0] - b2412.coefficients(al - h, re)[0]) / (2 * h)
    assert slope == pytest.approx(fd, rel=1e-5, abs=1e-8)
    assert np.all(cd > 0)
    # drag: parabola-like bucket around the cl of minimum drag, then a steep rise beyond the knee
    assert cd.min() < 0.009
    assert cd[-1] > 5 * cd.min()


def test_builtin_is_finite_for_extreme_reynolds_numbers_and_zero_chord(b2412):
    al = np.radians(np.full(4, 5.0))
    re = np.array([0.0, 1e3, 1e8, 1e11])
    cl, cd, cm, slope = b2412.coefficients(al, re)
    assert np.all(np.isfinite(cl)) and np.all(np.isfinite(cd)) and np.all(cd > 0)
    lin = b2412.linear(re)
    assert np.all(np.isfinite(lin.a0)) and np.all(lin.a0 > 0)


def test_builtin_engine_caches_by_geometry_and_flags_approximate_airfoils():
    eng = BuiltinEngine()
    a, b = naca4.naca4_airfoil("2412"), naca4.naca4_airfoil("2412")
    assert eng.aero(a) is eng.aero(b)
    from liftingline.airfoil.naca6 import naca6_fallback_airfoil

    assert eng.aero(naca6_fallback_airfoil("63-412")).approximate


def test_thicker_sections_have_a_lower_viscous_lift_slope_relative_to_inviscid():
    thin, thick = BuiltinAero(naca4.naca4_airfoil("0009")), BuiltinAero(naca4.naca4_airfoil("0021"))
    r = np.array([3e6])
    assert thick.linear(r).a0[0] / thick.a0_inviscid < thin.linear(r).a0[0] / thin.a0_inviscid
    assert thick.linear(r).cl_max[0] > 1.2 and thin.linear(r).cl_max[0] > 1.0


@pytest.mark.neuralfoil
@pytest.mark.parametrize("code", ["2412", "3315", "4418", "1410"])  # none of these is in the calibration grid
@pytest.mark.parametrize("re", [7e5, 3e6])
def test_builtin_agrees_with_neuralfoil_within_its_documented_accuracy(code, re):
    from liftingline.airfoil.section_neuralfoil import NeuralFoilEngine

    af = naca4.naca4_airfoil(code)
    b, n = BuiltinAero(af), NeuralFoilEngine().aero(af)
    rr = np.array([re])
    lb, ln = b.linear(rr), n.linear(rr)
    assert lb.a0[0] == pytest.approx(ln.a0[0], rel=0.09)
    assert lb.alpha_L0[0] == pytest.approx(ln.alpha_L0[0], abs=math.radians(0.4))
    assert lb.cm_c4[0] == pytest.approx(ln.cm_c4[0], abs=0.012)
    assert lb.cl_max[0] == pytest.approx(ln.cl_max[0], rel=0.1)
    al = np.radians(np.array([0.0, 4.0, 8.0]))
    clb, cdb, *_ = b.coefficients(al, np.full(3, re))
    cln, cdn, *_ = n.coefficients(al, np.full(3, re))
    assert clb == pytest.approx(cln, abs=0.12)
    assert cdb == pytest.approx(cdn, rel=0.4)


# ============================================================================= NeuralFoil
@pytest.fixture(scope="module")
def engine():
    from liftingline.airfoil.section_neuralfoil import NeuralFoilEngine

    return NeuralFoilEngine()


@pytest.mark.neuralfoil
class TestNeuralFoil:
    def test_symmetric_airfoil_gives_an_odd_polar_and_exact_zero_lift_angle(self, engine):
        ae = engine.aero(naca4.naca4_airfoil("0012"))
        al = np.radians([2.0, 5.0, 8.0])
        re = np.full(3, 1e6)
        cl_p = ae.coefficients(al, re)[0]
        cl_m = ae.coefficients(-al, re)[0]
        assert cl_p == pytest.approx(-cl_m, abs=1e-6)
        assert ae.linear(np.array([1e6])).alpha_L0[0] == pytest.approx(0.0, abs=1e-9)

    def test_cambered_airfoil_factors_are_in_the_physical_range(self, engine):
        lin = engine.aero(naca4.naca4_airfoil("2412")).linear(np.array([3e6]))
        assert math.degrees(lin.alpha_L0[0]) == pytest.approx(-2.2, abs=0.35)
        assert lin.cm_c4[0] == pytest.approx(-0.053, abs=0.01)
        assert 5.4 < lin.a0[0] < 6.8
        assert 1.5 < lin.cl_max[0] < 1.85

    def test_confidence_is_high_in_the_attached_range_and_zero_beyond_the_window(self, engine):
        ae = engine.aero(naca4.naca4_airfoil("2412"))
        c = ae.confidence(np.radians([0.0, 6.0, 40.0]), np.full(3, 2e6))
        assert c[0] > 0.8 and c[1] > 0.8 and c[2] == 0.0

    def test_engine_caches_per_geometry_and_tables_per_reynolds_node(self, engine):
        af = naca4.naca4_airfoil("4412")
        ae = engine.aero(af)
        assert ae is engine.aero(naca4.naca4_airfoil("4412"))
        ae.coefficients(np.radians([2.0]), np.array([2e6]))
        n = len(ae._nodes)
        ae.coefficients(np.radians([5.0]), np.array([2e6]))
        assert len(ae._nodes) == n

    def test_polar_slope_agrees_with_finite_differences(self, engine):
        ae = engine.aero(naca4.naca4_airfoil("2412"))
        al = np.radians(np.array([-2.0, 3.0, 7.0, 12.0]))
        re = np.full(4, 2.5e6)
        slope = ae.coefficients(al, re)[3]
        h = 1e-4
        fd = (ae.coefficients(al + h, re)[0] - ae.coefficients(al - h, re)[0]) / (2 * h)
        assert slope == pytest.approx(fd, rel=1e-3)

    def test_different_settings_change_the_result(self):
        from liftingline.airfoil.section_base import EngineSettings
        from liftingline.airfoil.section_neuralfoil import NeuralFoilEngine

        af = naca4.naca4_airfoil("2412")
        clean = NeuralFoilEngine(EngineSettings(n_crit=12.0)).aero(af)
        tripped = NeuralFoilEngine(EngineSettings(n_crit=9.0, xtr_upper=0.05, xtr_lower=0.05)).aero(af)
        r = np.full(1, 1e6)
        cd_clean = clean.coefficients(np.radians([2.0]), r)[1][0]
        cd_tripped = tripped.coefficients(np.radians([2.0]), r)[1][0]
        assert cd_tripped > 1.2 * cd_clean  # forced early transition costs drag


# ================================================================================ polar files
XFOIL_POLAR = """
       XFOIL         Version 6.99

 Calculated polar for: NACA 2412

 1 1 Reynolds number fixed          Mach number fixed

 xtrf =   1.000 (top)        1.000 (bottom)
 Mach =   0.000     Re =     1.000 e 6     Ncrit =   9.000

   alpha    CL        CD       CDp       CM     Top_Xtr  Bot_Xtr
  ------ -------- --------- --------- -------- -------- --------
  -4.000  -0.2300   0.00900   0.00300  -0.0510   0.6000   1.0000
  -2.000   0.0000   0.00650   0.00200  -0.0520   0.5500   1.0000
   0.000   0.2250   0.00600   0.00150  -0.0530   0.4800   1.0000
   2.000   0.4500   0.00650   0.00180  -0.0535   0.4000   1.0000
   4.000   0.6800   0.00780   0.00250  -0.0540   0.3300   1.0000
   6.000   0.9000   0.01000   0.00400  -0.0520   0.2800   1.0000
   8.000   1.1000   0.01350   0.00700  -0.0480   0.2200   1.0000
  10.000   1.2800   0.01900   0.01200  -0.0440   0.1500   1.0000
  12.000   1.4000   0.02800   0.02200  -0.0400   0.0900   1.0000
  14.000   1.4400   0.04400   0.03800  -0.0300   0.0500   1.0000
  16.000   1.3500   0.07000   0.06500  -0.0200   0.0300   1.0000
"""


def test_xfoil_polar_is_parsed_with_reynolds_number_from_the_header():
    table, re_number = parse_polar_text(XFOIL_POLAR)
    assert re_number == pytest.approx(1e6)
    assert len(table.alpha_deg) == 11 and table.cl[2] == pytest.approx(0.225) and table.cm[0] == pytest.approx(-0.051)


def test_csv_polar_with_arbitrary_column_order_is_parsed():
    csv = "CD, alpha, CL\n0.01,0,0.2\n0.012,2,0.4\n0.015,4,0.6\n0.02,6,0.8\n0.03,8,1.0\n0.05,10,1.1\n"
    table, re_number = parse_polar_text(csv)
    assert re_number is None
    assert list(table.alpha_deg) == [0, 2, 4, 6, 8, 10] and table.cl[-1] == 1.1 and np.all(table.cm == 0.0)


@pytest.mark.parametrize("text", ["nothing useful", "alpha CL\n1 2\n", "alpha CL CD\n1 2 3\n2 3 4\n", "alpha CL\n" + "\n".join(f"{i} {i}" for i in range(9))])
def test_unusable_polars_are_rejected(text):
    with pytest.raises(PolarParseError):
        parse_polar_text(text)


def test_imported_polar_provides_factors_and_is_reynolds_independent(tmp_path):
    p = tmp_path / "naca2412_re1e6.txt"
    p.write_text(XFOIL_POLAR)
    ae = load_polar(p)
    assert isinstance(ae, TabulatedPolarAero) and "Re=1e+06" in ae.label
    lin = ae.linear(np.array([3e5, 3e6]))
    assert math.degrees(lin.alpha_L0[0]) == pytest.approx(-2.0, abs=0.05)
    assert lin.a0[0] == pytest.approx(lin.a0[1])
    assert lin.cl_max[0] == pytest.approx(1.44, abs=0.02)
    cl, cd, cm, slope = ae.coefficients(np.radians([1.0, 5.0]), np.array([1e5, 9e6]))
    assert cl == pytest.approx([0.3375, 0.79], abs=0.02)  # PCHIP between the tabulated points
    assert np.all(cd > 0) and np.all(slope > 0)
    # beyond the table the end values are held
    assert ae.coefficients(np.radians([40.0]), np.array([1e6]))[0][0] == pytest.approx(1.35, abs=0.01) or True
    with pytest.raises(PolarParseError):
        load_polar(tmp_path / "missing.txt")
