import math

import numpy as np
import pytest

from liftingline.core.wing import Section, Wing


def test_rectangular_reference_quantities():
    w = Wing.rectangular(10.0, 1.0)
    assert w.span == 10.0 and w.half_span == 5.0
    assert w.area == pytest.approx(10.0, rel=1e-12)
    assert w.aspect_ratio == pytest.approx(10.0)
    assert w.mac == pytest.approx(1.0)
    assert w.y_mac == pytest.approx(2.5)  # b/4
    assert w.taper_ratio == 1.0
    assert w.is_planar_unswept()


def test_tapered_reference_quantities_match_closed_forms():
    b, cr, lam = 10.0, 1.2, 0.5
    w = Wing.trapezoid(b, cr, lam * cr)
    assert w.area == pytest.approx(b * cr * (1 + lam) / 2, rel=1e-12)
    assert w.mac == pytest.approx(2 / 3 * cr * (1 + lam + lam**2) / (1 + lam), rel=1e-12)
    assert w.y_mac == pytest.approx(b / 6 * (1 + 2 * lam) / (1 + lam), rel=1e-12)
    assert w.taper_ratio == pytest.approx(lam)


def test_elliptic_reference_quantities_match_closed_forms():
    b, cr = 12.0, 1.5
    w = Wing.elliptic(b, cr)
    assert w.area == pytest.approx(math.pi * b * cr / 4, rel=1e-9)
    assert w.mac == pytest.approx(8 * cr / (3 * math.pi), rel=1e-9)
    assert w.y_mac == pytest.approx(2 * b / (3 * math.pi), rel=1e-9)
    assert w.chord(0.0) == pytest.approx(cr)
    assert w.chord(b / 2) == pytest.approx(0.0, abs=1e-12)
    # c/4 line stays straight
    assert np.ptp(w.x_c4(np.linspace(0, b / 2, 25))) == pytest.approx(0.0, abs=1e-12)


def test_chord_is_symmetric_and_piecewise_linear():
    w = Wing(sections=(Section(0, 2.0), Section(2.0, 1.0), Section(5.0, 0.4)))
    assert w.chord(-2.0) == pytest.approx(w.chord(2.0)) == pytest.approx(1.0)
    assert w.chord(3.5) == pytest.approx(0.7)
    assert w.area == pytest.approx(2 * (1.5 * 2.0 + 0.7 * 3.0), rel=1e-12)


def test_sweep_and_dihedral_builders():
    w = Wing.trapezoid(10.0, 1.0, 1.0, sweep_deg=30.0, sweep_at=0.25, dihedral_deg=5.0)
    assert w.sweep_c4_deg_by_segment()[0] == pytest.approx(30.0)
    assert w.dihedral_deg_by_segment()[0] == pytest.approx(5.0)
    assert not w.is_planar_unswept()
    # leading-edge sweep measured at x/c = 0 with a tapered wing: c/4 sweeps less
    t = Wing.trapezoid(10.0, 1.2, 0.6, sweep_deg=20.0, sweep_at=0.0)
    assert t.sections[1].x_le == pytest.approx(5.0 * math.tan(math.radians(20.0)))
    assert t.max_abs_sweep_c4_deg < 20.0
    # taper with a straight c/4 line is 'unswept' for classical theory
    assert Wing.trapezoid(10.0, 1.2, 0.6).is_planar_unswept()


def test_from_area_ar_taper_roundtrip():
    w = Wing.from_area_ar_taper(16.2, 7.5, 0.6)
    assert w.area == pytest.approx(16.2, rel=1e-12)
    assert w.aspect_ratio == pytest.approx(7.5, rel=1e-12)
    assert w.taper_ratio == pytest.approx(0.6)


def test_moment_reference_default_is_quarter_mac():
    w = Wing.rectangular(10.0, 2.0)
    assert w.moment_reference_x(0.25) == pytest.approx(0.5)
    assert w.moment_reference_x(0.0) == pytest.approx(0.0)


def test_airfoil_blend_weights():
    w = Wing(sections=(Section(0, 2.0, airfoil="A"), Section(2.0, 1.5, airfoil="B"), Section(6.0, 1.0, airfoil="C")))
    i0, i1, xi = w.airfoil_blend(np.array([-0.0, 1.0, 2.0, 4.0, 6.0, 7.0]))
    assert list(i0) == [0, 0, 1, 1, 1, 1]
    assert list(i1) == [1, 1, 2, 2, 2, 2]
    assert xi == pytest.approx([0.0, 0.5, 0.0, 0.5, 1.0, 1.0])


@pytest.mark.parametrize(
    "sections",
    [
        (Section(0, 1.0),),  # one section
        (Section(0.5, 1.0), Section(3.0, 1.0)),  # root not at y=0
        (Section(0, 1.0), Section(2.0, 1.0), Section(2.0, 1.0)),  # not increasing
        (Section(0, 1.0), Section(2.0, 0.0), Section(4.0, 0.5)),  # zero chord mid-span
        (Section(0, -1.0), Section(2.0, 1.0)),  # negative chord
    ],
)
def test_invalid_geometry_is_rejected(sections):
    with pytest.raises(ValueError):
        Wing(sections=sections)


def test_pointed_tip_is_allowed_and_serialisation_roundtrips():
    w = Wing.trapezoid(8.0, 2.0, 0.0, sweep_deg=10.0, dihedral_deg=3.0, twist_tip_deg=-2.0, incidence_deg=1.0, airfoil_root="NACA 2412", airfoil_tip="NACA 0009", name="Tapered")
    again = Wing.from_dict(w.to_dict())
    assert again == w
    assert again.area == pytest.approx(w.area)


def test_planform_outline_shapes():
    out = Wing.trapezoid(10.0, 1.0, 0.5).planform_outline(101)
    assert out["y"][0] == pytest.approx(-5.0) and out["y"][-1] == pytest.approx(5.0)
    assert np.all(out["x_te"] >= out["x_le"])
