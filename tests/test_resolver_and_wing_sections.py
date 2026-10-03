"""AirfoilResolver, WingSectionModel (blending, overrides, confidence) and engine-based pipeline runs."""

import math

import numpy as np
import pytest

from liftingline.airfoil import naca4, naca6
from liftingline.airfoil.dat_io import to_dat_text
from liftingline.airfoil.engines import BUILTIN, NEURALFOIL, available_engines, default_engine_kind, make_engine
from liftingline.airfoil.library import AirfoilLibrary
from liftingline.airfoil.resolver import AirfoilNotFound, AirfoilResolver
from liftingline.airfoil.section_base import EngineUnavailable
from liftingline.airfoil.section_builtin import BuiltinAero, BuiltinEngine
from liftingline.airfoil.section_model import OverriddenAero, SectionOverride, WingSectionModel
from liftingline.core.analysis import AnalysisRequest, run
from liftingline.core.flow import FlowCondition
from liftingline.core.section import LinearSection
from liftingline.core.wing import Section, Wing

DEG = math.pi / 180.0


# ====================================================================================== resolver
@pytest.fixture(scope="module")
def library(tmp_path_factory):
    d = tmp_path_factory.mktemp("lib")
    f6 = lambda code: naca6.naca6_fallback_airfoil(code, n_per_surface=40)  # noqa: E731
    (d / "naca63-412.dat").write_text(to_dat_text(f6("63-412").with_name("NACA 63-412")))
    (d / "naca651212.dat").write_text(to_dat_text(f6("65(1)-212").with_name("NACA 65(1)-212")))
    (d / "naca652212.dat").write_text(to_dat_text(f6("65(2)-212").with_name("NACA 65(2)-212")))
    (d / "e387.dat").write_text(to_dat_text(naca4.naca4_airfoil("4412", n_per_surface=40).with_name("E387")))
    (d / "bad.dat").write_text("not an airfoil\n")
    return AirfoilLibrary.scan(d)


def test_four_and_five_digit_codes_are_generated_exactly():
    r = AirfoilResolver()
    a = r.resolve("NACA 2412")
    assert a.airfoil.t_over_c == pytest.approx(0.12, abs=3e-4) and "exact" in a.source and not a.approximate
    assert r.resolve("2412") is not a and r.resolve("2412").airfoil.digest == a.airfoil.digest
    five = r.resolve("23012")
    assert "5-digit" in five.source and five.airfoil.max_camber > 0.01


def test_six_series_is_taken_from_the_library_when_present(library):
    r = AirfoilResolver(library).resolve("63-412")
    assert not r.approximate and "library file naca63-412.dat" in r.source and not r.note
    assert r.airfoil.name == "NACA 63-412"
    assert r.code is not None and r.code.series == 63


def test_missing_six_series_falls_back_to_a_flagged_approximation(library):
    r = AirfoilResolver(library).resolve("64A010")
    assert r.approximate and "APPROXIMATE" in r.note and "not found in the airfoil library" in r.note
    no_lib = AirfoilResolver().resolve("64A010")
    assert no_lib.approximate and "no airfoil library is set" in no_lib.note


def test_ambiguous_library_match_is_reported_in_the_note(library):
    r = AirfoilResolver(library).resolve("65-212")
    assert len(r.candidates) == 2 and "equally good" in r.note
    assert AirfoilResolver(library).resolve("65(2)-212").candidates[0].name == "naca652212"


def test_library_names_files_and_registered_airfoils(library, tmp_path):
    r = AirfoilResolver(library)
    assert "library file e387.dat" in r.resolve("e387").source
    p = tmp_path / "mine.dat"
    p.write_text(to_dat_text(naca4.naca4_airfoil("0009", n_per_surface=40).with_name("mine")))
    assert r.resolve(f"file:{p}").airfoil.t_over_c == pytest.approx(0.09, abs=3e-3)
    assert r.resolve(str(p)).airfoil.name == "mine"
    af = naca4.naca4_airfoil("0015")
    r.register("my wing section", af, "typed in")
    assert r.resolve("my wing section").airfoil is af
    assert r.resolve("  my wing section ").source == "typed in"


def test_unknown_or_unreadable_airfoils_raise_a_helpful_error(library, tmp_path):
    r = AirfoilResolver(library)
    with pytest.raises(AirfoilNotFound, match="Closest library entries"):
        r.resolve("e38")
    with pytest.raises(AirfoilNotFound):
        r.resolve("")
    with pytest.raises(AirfoilNotFound):
        r.resolve("totally unknown")
    with pytest.raises(AirfoilNotFound):
        r.resolve("bad")  # in the library but unreadable
    bad = tmp_path / "broken.dat"
    bad.write_text("x\n")
    with pytest.raises(AirfoilNotFound):
        r.resolve(f"file:{bad}")


def test_setting_a_library_clears_the_cache(library):
    r = AirfoilResolver()
    assert r.resolve("63-412").approximate
    r.set_library(library)
    assert not r.resolve("63-412").approximate


# ===================================================================================== engines
def test_engine_registry():
    avail = available_engines()
    assert avail[BUILTIN][0] is True
    assert default_engine_kind() in (NEURALFOIL, BUILTIN)
    assert make_engine(BUILTIN).name.startswith("Built-in")
    with pytest.raises(ValueError):
        make_engine("xfoil")
    if not avail[NEURALFOIL][0]:
        with pytest.raises(EngineUnavailable):
            make_engine(NEURALFOIL)


# ========================================================================= WingSectionModel
class _FakeAero:
    """A section with fixed numbers; optional confidence."""

    approximate = False

    def __init__(self, a0, aL0_deg, cm, cd, conf=None, label="fake"):
        self.a0, self.aL0, self.cm, self.cd, self.conf, self.label = a0, aL0_deg * DEG, cm, cd, conf, label

    def linear(self, re):
        from liftingline.core.section import LinearParams

        n = np.shape(re)
        return LinearParams(np.full(n, self.a0), np.full(n, self.aL0), np.full(n, self.cm), np.full(n, 1.5), np.full(n, 0.3))

    def coefficients(self, alpha, re):
        alpha = np.asarray(alpha, dtype=float)
        return self.a0 * (alpha - self.aL0), np.full(alpha.shape, self.cd), np.full(alpha.shape, self.cm), np.full(alpha.shape, self.a0)

    def confidence(self, alpha, re):
        return None if self.conf is None else np.full(np.shape(alpha), self.conf)


def test_airfoils_are_blended_linearly_between_sections_in_coefficient_space():
    wing = Wing(sections=(Section(0.0, 2.0), Section(2.0, 1.5), Section(6.0, 1.0)))
    A, B, C = _FakeAero(6.0, 0.0, 0.0, 0.01), _FakeAero(6.4, -2.0, -0.05, 0.02), _FakeAero(5.6, -4.0, -0.10, 0.03)
    m = WingSectionModel(wing, [A, B, C])
    y = np.array([0.0, 1.0, 2.0, 4.0, 6.0, -4.0])
    re = np.full(6, 1e6)
    lin = m.linear(y, re)
    assert lin.a0 == pytest.approx([6.0, 6.2, 6.4, 6.0, 5.6, 6.0])
    assert lin.alpha_L0 / DEG == pytest.approx([0.0, -1.0, -2.0, -3.0, -4.0, -3.0])
    co = m.evaluate(y, np.full(6, 0.05), re)
    assert co.cd == pytest.approx([0.01, 0.015, 0.02, 0.025, 0.03, 0.025])
    assert co.cm == pytest.approx([0.0, -0.025, -0.05, -0.075, -0.10, -0.075])
    assert co.cl[0] == pytest.approx(6.0 * 0.05)
    assert co.cl_alpha[1] == pytest.approx(6.2)


def test_the_section_count_must_match():
    wing = Wing.rectangular(10.0, 1.0)
    with pytest.raises(ValueError):
        WingSectionModel(wing, [_FakeAero(6, 0, 0, 0.01)])
    with pytest.raises(ValueError):
        WingSectionModel(wing, [_FakeAero(6, 0, 0, 0.01)] * 2, overrides=[None])


def test_overrides_replace_the_requested_factors_of_one_section_only():
    wing = Wing.rectangular(10.0, 1.0)
    base = BuiltinAero(naca4.naca4_airfoil("2412"))
    ov = SectionOverride(a0=5.0, alpha_L0=-0.03, cd0=0.012)
    m = WingSectionModel(wing, [base, base], overrides=[ov, None])
    re = np.full(2, 2e6)
    lin = m.linear(np.array([0.0, 5.0]), re)
    assert lin.a0 == pytest.approx([5.0, base.linear(re[:1]).a0[0]])
    assert lin.alpha_L0[0] == pytest.approx(-0.03) and lin.alpha_L0[1] == pytest.approx(base.alpha_L0)
    assert lin.cl_max[0] == pytest.approx(base.linear(re[:1]).cl_max[0])  # not overridden: base value kept
    # near zero lift the overridden polar is linear with the given slope; drag follows the override
    al = np.full(2, -0.03 + 0.04)
    co = m.evaluate(np.array([0.0, 5.0]), al, re)
    assert co.cl[0] == pytest.approx(5.0 * 0.04, rel=2e-3)
    assert co.cd[0] == pytest.approx(0.012 + 0.0076 * co.cl[0] ** 2, rel=1e-6)
    assert isinstance(m.aeros[0], OverriddenAero) and not isinstance(m.aeros[1], OverriddenAero)
    assert not SectionOverride().active and ov.active


def test_override_polar_saturates_at_the_given_cl_max():
    wing = Wing.rectangular(10.0, 1.0)
    base = BuiltinAero(naca4.naca4_airfoil("0012"))
    m = WingSectionModel(wing, [base, base], overrides=[SectionOverride(cl_max=1.0), None])
    cl = m.evaluate(np.array([0.0]), np.radians([40.0]), np.array([3e6])).cl[0]
    assert 0.9 < cl < 1.02


def test_confidence_is_the_minimum_over_the_blended_airfoils():
    wing = Wing(sections=(Section(0.0, 2.0), Section(5.0, 1.0)))
    m = WingSectionModel(wing, [_FakeAero(6, 0, 0, 0.01, conf=0.9), _FakeAero(6, 0, 0, 0.01, conf=0.4)])
    c = m.confidence(np.array([0.0, 2.5, 5.0]), np.zeros(3), np.full(3, 1e6))
    assert c == pytest.approx([0.9, 0.4, 0.4])
    assert WingSectionModel(wing, [_FakeAero(6, 0, 0, 0.01)] * 2).confidence(np.zeros(2), np.zeros(2), np.ones(2)) is None


def test_from_wing_resolves_airfoils_and_collects_notes(library):
    wing = Wing.trapezoid(10.0, 1.4, 0.8, airfoil_root="NACA 2412", airfoil_tip="64A010")
    resolver = AirfoilResolver(library)
    m = WingSectionModel.from_wing(wing, resolver, BuiltinEngine())
    assert m.label.startswith("Built-in") and m.approximate
    assert len(m.resolutions) == 2 and any("APPROXIMATE" in n for n in m.notes)
    ok = WingSectionModel.from_wing(wing.with_airfoils("NACA 2412"), resolver, BuiltinEngine())
    assert not ok.approximate and ok.notes == []


# ======================================================================== pipeline integration
def _req(wing, section, alpha, **kw):
    kw.setdefault("solver", "classical")
    return AnalysisRequest(wing, FlowCondition.from_isa(0.0, V=50.0, alpha_deg=alpha), section, **kw)


@pytest.fixture(scope="module")
def c172():
    return Wing.trapezoid(11.0, 1.63, 1.13, twist_tip_deg=-3.0, airfoil_root="NACA 2412", airfoil_tip="NACA 2412")


@pytest.fixture(scope="module")
def builtin_model(c172):
    return WingSectionModel.from_wing(c172, AirfoilResolver(), BuiltinEngine())


def test_engine_sections_default_to_the_nonlinear_solve_and_linear_sections_do_not(c172, builtin_model):
    assert run(_req(c172, builtin_model, 4.0)).info["nonlinear"] is True
    assert run(_req(c172, LinearSection(), 4.0)).info["nonlinear"] is False
    assert run(_req(c172, builtin_model, 4.0, nonlinear=False)).info["nonlinear"] is False


def test_linear_and_nonlinear_agree_below_stall_and_diverge_towards_it(c172, builtin_model):
    lin = run(_req(c172, builtin_model, 3.0, nonlinear=False)).totals
    non = run(_req(c172, builtin_model, 3.0, nonlinear=True)).totals
    assert non.CL == pytest.approx(lin.CL, rel=0.01)
    lin14 = run(_req(c172, builtin_model, 14.0, nonlinear=False)).totals
    non14 = run(_req(c172, builtin_model, 14.0, nonlinear=True)).totals
    assert non14.CL < lin14.CL  # the saturating section lift limits the wing


def test_linear_engine_solve_equals_a_linear_section_built_from_the_engine_factors():
    """Plumbing check: a rectangular wing has the same Re everywhere, so the engine model's linear solve must
    equal a LinearSection carrying the engine's (a0, alpha_L0) at that Reynolds number."""
    wing = Wing.rectangular(10.0, 1.2, airfoil_root="NACA 2412")
    model = WingSectionModel.from_wing(wing, AirfoilResolver(), BuiltinEngine())
    f = FlowCondition.from_isa(0.0, V=40.0, alpha_deg=3.0)
    lin = model.linear(np.array([0.0]), np.array([f.reynolds(1.2)]))
    ref = LinearSection(a0=float(lin.a0[0]), alpha_L0=float(lin.alpha_L0[0]))
    a = run(AnalysisRequest(wing, f, model, solver="classical", nonlinear=False, compressibility=False))
    b = run(AnalysisRequest(wing, f, ref, solver="classical", compressibility=False))
    assert a.totals.CL == pytest.approx(b.totals.CL, rel=1e-9)
    assert a.totals.CD_induced == pytest.approx(b.totals.CD_induced, rel=1e-9)


def test_reynolds_number_falls_towards_the_tip_and_profile_drag_is_positive(c172, builtin_model):
    r = run(_req(c172, builtin_model, 4.0))
    inner = np.abs(r.load.y) < 0.9 * c172.half_span
    right = inner & (r.load.y > 0)
    assert np.all(np.diff(r.load.re[right]) < 0)
    assert r.totals.CD_profile > 0 and 0.004 < r.totals.CD_profile < 0.012
    assert r.totals.CL == pytest.approx(0.40, abs=0.04)  # C172-class wing at 4 deg (CL_alpha ~ 0.085 per deg, washout -3 deg)


def test_wing_climbs_to_stall_without_failing_and_warns_near_cl_max(c172, builtin_model):
    cls = []
    for a in (0.0, 6.0, 12.0, 18.0, 24.0):
        r = run(_req(c172, builtin_model, a))
        assert r.solution.converged
        cls.append(r.totals.CL)
    assert np.all(np.diff(cls) > 0)
    assert 1.4 < cls[-1] < 1.9  # clean-wing CLmax range of this class
    assert any("stall" in w.lower() or "cl_max" in w for w in r.warnings)


def test_low_engine_confidence_produces_a_warning():
    wing = Wing.rectangular(10.0, 1.0)
    shaky = WingSectionModel(wing, [_FakeAero(6.0, -2, -0.05, 0.01, conf=0.2)] * 2)
    r = run(_req(wing, shaky, 4.0))
    assert r.info["section_confidence_min"] == pytest.approx(0.2)
    assert any("confidence" in w.lower() for w in r.warnings)
    fine = WingSectionModel(wing, [_FakeAero(6.0, -2, -0.05, 0.01, conf=0.95)] * 2)
    assert not any("confidence" in w.lower() for w in run(_req(wing, fine, 4.0)).warnings)


def test_approximate_geometry_note_reaches_the_user_warnings():
    wing = Wing.trapezoid(10.0, 1.4, 1.0, airfoil_root="63-412")
    model = WingSectionModel.from_wing(wing, AirfoilResolver(), BuiltinEngine())
    r = run(_req(wing, model, 3.0))
    assert any("APPROXIMATE" in w for w in r.warnings)


@pytest.mark.neuralfoil
def test_neuralfoil_and_builtin_agree_at_wing_level(c172):
    res = AirfoilResolver()
    nf = WingSectionModel.from_wing(c172, res, make_engine(NEURALFOIL))
    bi = WingSectionModel.from_wing(c172, res, make_engine(BUILTIN))
    for alpha in (2.0, 6.0):
        a, b = run(_req(c172, nf, alpha)).totals, run(_req(c172, bi, alpha)).totals
        assert a.CL == pytest.approx(b.CL, rel=0.03)
        assert a.CD == pytest.approx(b.CD, rel=0.10)
        assert a.Cm == pytest.approx(b.Cm, abs=0.012)
    r = run(_req(c172, nf, 6.0))
    assert r.info["section_confidence_min"] > 0.8 and r.info["nonlinear"] is True
