"""Project state, backend (request building, section description), report rows and presets - all GUI-free."""

import json
import math

import numpy as np
import pytest

from liftingline.airfoil import naca4
from liftingline.airfoil.dat_io import to_dat_text
from liftingline.core.analysis import run
from liftingline.core.wing import Wing
from liftingline.io import backend as backend_module
from liftingline.io.backend import Backend
from liftingline.io.presets import PRESETS
from liftingline.io.project import (
    ATMOSPHERE_MANUAL,
    DEFINE_AREA_AR_TAPER,
    DEFINE_SPAN_AR_TAPER,
    DEFINE_SPAN_AREA_TAPER,
    DEFINE_SPAN_CHORDS,
    MODE_ELLIPTIC,
    MODE_SECTIONS,
    SCHEMA_VERSION,
    SPEED_BY_MACH,
    FlowSpec,
    ProjectError,
    ProjectState,
    WingSpec,
    flow_from_spec,
    wing_from_spec,
)
from liftingline.io.report import fmt, result_to_csv, spanwise_table, totals_rows, totals_text


def builtin_state(**wing_kw):
    st = ProjectState()
    st.solver.engine = "builtin"
    for k, v in wing_kw.items():
        setattr(st.wing, k, v)
    st.wing.sync()
    return st


# ======================================================================================= planform
@pytest.mark.parametrize(
    "define_by, kw",
    [
        (DEFINE_SPAN_CHORDS, dict(span=12.0, root_chord=1.5, tip_chord=0.9)),
        (DEFINE_AREA_AR_TAPER, dict(area=14.4, aspect_ratio=10.0, taper_ratio=0.6)),
        (DEFINE_SPAN_AREA_TAPER, dict(span=12.0, area=14.4, taper_ratio=0.6)),
        (DEFINE_SPAN_AR_TAPER, dict(span=12.0, aspect_ratio=10.0, taper_ratio=0.6)),
    ],
)
def test_every_way_of_defining_a_trapezoid_gives_consistent_numbers(define_by, kw):
    spec = WingSpec(define_by=define_by, **kw)
    spec.sync()
    b, c_root, c_tip = spec.dimensions()
    S = 0.5 * b * (c_root + c_tip)
    assert (spec.area, spec.aspect_ratio, spec.taper_ratio) == pytest.approx((S, b * b / S, c_tip / c_root))
    wing = wing_from_spec(spec, ["", ""])
    assert wing.area == pytest.approx(S, rel=1e-12) and wing.span == pytest.approx(b)
    # all four descriptions of the same wing agree
    ref = WingSpec(span=12.0, root_chord=1.5, tip_chord=0.9)
    ref.sync()
    assert (b, c_root, c_tip) == pytest.approx(ref.dimensions())


def test_elliptic_planform_from_span_and_root_chord_or_from_area_and_aspect_ratio():
    a = WingSpec(mode=MODE_ELLIPTIC, span=10.0, root_chord=1.5)
    a.sync()
    assert a.area == pytest.approx(math.pi * 10.0 * 1.5 / 4.0) and a.aspect_ratio == pytest.approx(100.0 / a.area)
    b = WingSpec(mode=MODE_ELLIPTIC, define_by=DEFINE_AREA_AR_TAPER, area=a.area, aspect_ratio=a.aspect_ratio)
    b.sync()
    assert (b.span, b.root_chord) == pytest.approx((10.0, 1.5))
    assert wing_from_spec(a, ["", ""]).area == pytest.approx(a.area, rel=1e-6)


def test_the_table_of_a_trapezoid_rebuilds_the_same_wing_and_an_ellipse_closely():
    spec = WingSpec(span=11.0, root_chord=1.6, tip_chord=0.9, sweep_deg=12.0, dihedral_deg=2.0, twist_tip_deg=-2.5)
    rows = spec.to_table()
    table = WingSpec(mode=MODE_SECTIONS, sections=rows)
    w0, w1 = wing_from_spec(spec, ["", ""]), wing_from_spec(table, ["", ""])
    for y in (0.0, 1.3, 3.0, 5.5):
        assert (w1.chord(y), w1.x_le(y), w1.z_c4(y), w1.twist_deg(y)) == pytest.approx((w0.chord(y), w0.x_le(y), w0.z_c4(y), w0.twist_deg(y)))
    ell = WingSpec(mode=MODE_ELLIPTIC, span=10.0, root_chord=1.5)
    w_e, w_t = wing_from_spec(ell, ["", ""]), wing_from_spec(WingSpec(mode=MODE_SECTIONS, sections=ell.to_table()), ["", ""])
    assert w_t.area == pytest.approx(w_e.area, rel=5e-3)
    assert w_t.chord(0.0) == pytest.approx(1.5) and w_t.chord(5.0) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize(
    "kw, text",
    [
        (dict(span=-1.0), "span"),
        (dict(root_chord=0.0), "root chord"),
        (dict(tip_chord=-0.2), "tip chord"),
        (dict(sweep_deg=89.0), "sweep"),
    ],
)
def test_invalid_planforms_are_reported_in_words(kw, text):
    st = ProjectState()
    for k, v in kw.items():
        setattr(st.wing, k, v)
    problems = st.validate()
    assert problems and text in problems[0].lower()


def test_a_section_table_with_non_increasing_stations_is_rejected():
    st = ProjectState()
    st.wing.mode = MODE_SECTIONS
    st.wing.sections = [dict(y=0.0, chord=1.0), dict(y=3.0, chord=0.8), dict(y=2.0, chord=0.6)]
    st.airfoils.fit(3)
    assert any("increasing" in p for p in st.validate())


# ========================================================================================== flow
def test_flow_from_isa_by_speed_and_by_mach():
    by_v = flow_from_spec(FlowSpec(altitude_m=3000.0, V=70.0, alpha_deg=2.0))
    by_m = flow_from_spec(FlowSpec(altitude_m=3000.0, speed_mode=SPEED_BY_MACH, mach=by_v.mach, alpha_deg=2.0))
    assert by_m.V == pytest.approx(by_v.V) and by_v.rho == pytest.approx(0.9091, rel=1e-3)


def test_manual_air_properties_derive_the_speed_of_sound_and_viscosity():
    f = flow_from_spec(FlowSpec(atmosphere=ATMOSPHERE_MANUAL, rho=1.1, T=300.0))
    assert f.a == pytest.approx(347.2, rel=1e-3) and f.mu == pytest.approx(1.846e-5, rel=5e-3)
    assert flow_from_spec(FlowSpec(atmosphere=ATMOSPHERE_MANUAL, rho=1.1, T=300.0, mu=2e-5)).mu == 2e-5


def test_flow_errors_become_project_errors_and_supersonic_flow_is_refused():
    with pytest.raises(ProjectError):
        flow_from_spec(FlowSpec(altitude_m=60000.0))
    st = ProjectState()
    st.flow.speed_mode, st.flow.mach = SPEED_BY_MACH, 1.2
    assert any("subsonic" in p for p in st.validate())


# ===================================================================================== serialisation
def test_project_roundtrip_through_json_is_lossless(tmp_path):
    st = ProjectState()
    st.name, st.notes = "Test wing", "a note"
    st.wing.sweep_deg, st.wing.twist_tip_deg = 17.5, -2.0
    st.airfoils.keys[1] = "NACA 4412"
    st.airfoils.overrides[0] = {"a0": 5.7, "alpha_L0_deg": -2.2}
    st.flow.mu, st.flow.ground_effect = 1.8e-5, True
    path = tmp_path / "wing.json"
    st.save(path)
    assert json.loads(path.read_text())["schema_version"] == SCHEMA_VERSION
    assert ProjectState.load(path).to_dict() == st.to_dict()


def test_loading_tolerates_unknown_and_missing_keys_and_refuses_newer_files(tmp_path):
    data = ProjectState().to_dict()
    data["wing"]["future_option"] = 7
    data["flow"].pop("alpha_deg")
    del data["solver"]
    st = ProjectState.from_dict(data)
    assert st.flow.alpha_deg == FlowSpec().alpha_deg and st.solver.n_terms == 80
    data["schema_version"] = SCHEMA_VERSION + 1
    with pytest.raises(ProjectError, match="newer"):
        ProjectState.from_dict(data)
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(ProjectError):
        ProjectState.load(bad)
    with pytest.raises(ProjectError):
        ProjectState.load(tmp_path / "missing.json")


def test_airfoil_lists_follow_the_section_count():
    st = ProjectState()
    st.airfoils.keys = ["NACA 0012", "NACA 2412"]
    st.airfoils.fit(4)
    assert st.airfoils.keys == ["NACA 0012", "NACA 2412", "NACA 2412", "NACA 2412"]
    assert len(st.airfoils.overrides) == 4 and len(st.airfoils.polar_files) == 4
    st.airfoils.fit(2)
    assert st.airfoils.keys == ["NACA 0012", "NACA 2412"]


# ======================================================================================== backend
def test_backend_request_reproduces_a_hand_built_analysis():
    st = builtin_state()
    be = Backend()
    req = be.build_request(st)
    res = run(req)
    wing = Wing.trapezoid(11.0, 1.63, 1.13, airfoil_root="NACA 2412", airfoil_tip="NACA 2412")
    assert req.wing.area == pytest.approx(wing.area) and res.solver_used == "classical"
    from liftingline.airfoil.resolver import AirfoilResolver
    from liftingline.airfoil.section_builtin import BuiltinEngine
    from liftingline.airfoil.section_model import WingSectionModel
    from liftingline.core.analysis import AnalysisRequest

    ref = run(AnalysisRequest(wing, st.build_flow(), WingSectionModel.from_wing(wing, AirfoilResolver(), BuiltinEngine())))
    assert res.totals.CL == pytest.approx(ref.totals.CL, rel=1e-12) and res.totals.CD == pytest.approx(ref.totals.CD, rel=1e-12)


def test_solver_choices_ground_effect_and_reference_point_reach_the_request():
    st = builtin_state()
    st.solver.solver, st.solver.n_terms, st.solver.nonlinear = "numerical", 60, "off"
    st.flow.ground_effect, st.flow.ground_height_m, st.flow.extra_cd0 = True, 3.0, 0.004
    st.flow.ref_mode, st.flow.ref_x_abs_m, st.flow.ref_z_abs_m = "absolute", 0.7, 0.1
    req = Backend().build_request(st)
    assert (req.solver, req.n_terms, req.nonlinear, req.ground_height, req.extra_cd0) == ("numerical", 60, False, 3.0, 0.004)
    assert req.ref.x_abs == 0.7 and req.ref.z_abs == 0.1
    assert Backend().build_request(builtin_state()).ground_height is None


def test_neuralfoil_falls_back_to_the_builtin_engine_with_a_note(monkeypatch):
    monkeypatch.setattr(backend_module, "available_engines", lambda: {"neuralfoil": (False, "missing"), "builtin": (True, "")})
    st = ProjectState()
    st.solver.engine = "neuralfoil"
    kind, note = Backend.engine_kind(st.solver)
    assert kind == "builtin" and "not installed" in note
    res = run(Backend().build_request(st))
    assert any("not installed" in w for w in res.warnings)


def test_engines_are_cached_per_setting():
    be = Backend()
    st = builtin_state()
    e1, _ = be.engine(st.solver)
    st.solver.n_crit = 12.0  # irrelevant to the built-in engine: still the same instance
    assert be.engine(st.solver)[0] is e1


def test_airfoil_key_feedback_levels(tmp_path):
    be = Backend()
    ok, bad, approx = be.describe_key("NACA 2412"), be.describe_key("zz-not-an-airfoil"), be.describe_key("NACA 63-412")
    assert ok.level == "ok" and "12.0" in ok.text
    assert bad.level == "error" and "cannot find" in bad.text
    assert approx.level == "warning" and "APPROXIMATE" in approx.text
    assert be.describe_key("").level == "error"
    f = tmp_path / "myfoil.dat"
    f.write_text(to_dat_text(naca4.naca4_airfoil("4412")))
    assert be.describe_key(str(f)).level == "ok"


def test_unresolvable_airfoils_stop_the_request_with_the_section_number():
    st = builtin_state()
    st.airfoils.keys[1] = "nonsense"
    with pytest.raises(ProjectError, match="section 2"):
        Backend().build_request(st)


def test_overrides_and_imported_polars_reach_the_section_model(tmp_path):
    st = builtin_state()
    st.airfoils.overrides[0] = {"a0": 5.5, "cd0": 0.009, "alpha_L0_deg": -3.0}
    polar = tmp_path / "polar.txt"
    rows = ["alpha CL CD CM"] + [f"{a:.1f} {0.11 * (a + 2.0):.4f} {0.007 + 0.0001 * a * a:.5f} -0.05" for a in np.arange(-6, 18.5, 1.0)]
    polar.write_text("\n".join(rows))
    st.airfoils.polar_files[1] = str(polar)
    be = Backend()
    req = be.build_request(st)
    infos = be.describe_sections(req.section, req.wing, req.flow)
    assert infos[0].overridden == ("a0", "alpha_L0_deg", "cd0")
    assert infos[0].factors["a0"] == 5.5 and infos[0].factors["cd0"] == 0.009
    assert infos[0].factors["alpha_L0_deg"] == pytest.approx(-3.0)
    assert infos[0].engine_factors["a0"] != 5.5
    assert "imported polar" in infos[1].data_label.lower() or "polar" in infos[1].data_label.lower()
    assert infos[1].factors["a0"] == pytest.approx(0.11 * 180.0 / math.pi, rel=0.05)
    assert "imported polar" in req.section.label
    assert run(req).totals.CL > 0


def test_section_descriptions_give_factors_at_the_local_reynolds_number():
    be = Backend()
    req = be.build_request(builtin_state())
    root, tip = be.describe_sections(req.section, req.wing, req.flow)
    assert root.re == pytest.approx(req.flow.reynolds(1.63)) and tip.re == pytest.approx(req.flow.reynolds(1.13))
    assert -2.4 < root.factors["alpha_L0_deg"] < -1.9 and 5.5 < root.factors["a0"] < 6.6
    assert root.t_over_c == pytest.approx(0.12, abs=1e-3) and root.overridden == ()


def test_the_describer_copes_with_an_elliptic_tip_of_zero_chord():
    st = builtin_state(mode=MODE_ELLIPTIC)
    be = Backend()
    req = be.build_request(st)
    infos = be.describe_sections(req.section, req.wing, req.flow)
    assert infos[1].chord == 0.0 and infos[1].re > 0.0


# ========================================================================================= report
def test_number_formatting():
    assert fmt(0.0) == "0" and fmt(float("nan")) == "-" and fmt(float("inf")) == "inf"
    assert fmt(1234.5678) == "1235" and fmt(0.012346) == "0.01235"
    assert fmt(1.2345e-5) == "1.234e-05" and fmt(2.5e7) == "2.500e+07"


def test_result_rows_carry_units_and_round_off_is_not_shown():
    res = run(Backend().build_request(builtin_state()))
    rows = {r.key: r for r in totals_rows(res, "SI") if r.key}
    t = res.totals
    assert rows["L"].si_value == pytest.approx(t.L) and rows["L"].unit == "N" and rows["L"].coeff_value == pytest.approx(t.CL)
    assert rows["M_roll"].si_value == 0.0 and rows["M_yaw"].si_value == 0.0  # symmetric flow: exact zeros, not round-off
    assert "Y" not in rows  # no side-force row without sideslip
    imp = {r.key: r for r in totals_rows(res, "Imperial") if r.key}
    assert imp["L"].unit == "lbf" and float(imp["L"].value) == pytest.approx(t.L / 4.4482216, rel=2e-4)
    assert imp["S"].unit == "ft²" and imp["b"].unit == "ft"


def test_text_and_csv_exports_agree_with_the_totals():
    res = run(Backend().build_request(builtin_state()))
    text = totals_text(res)
    assert text.splitlines()[0].startswith("Quantity") and "Lift L" in text
    csv_text = result_to_csv(res, project_name="wing")
    lines = csv_text.splitlines()
    assert lines[0].startswith("# Lifting-line") and any(ln.startswith("L,") for ln in lines)
    lift = next(ln for ln in lines if ln.startswith("L,")).split(",")
    assert float(lift[1]) == pytest.approx(res.totals.L) and float(lift[2]) == pytest.approx(res.totals.CL)
    table = spanwise_table(res)
    header = next(i for i, ln in enumerate(lines) if ln.startswith("y [m]"))
    assert len(lines) - header - 1 == len(table["y_m"])
    assert float(lines[header + 1].split(",")[0]) == pytest.approx(table["y_m"][0])


# ======================================================================================== presets
@pytest.mark.parametrize("name", list(PRESETS))
def test_every_example_wing_runs_and_has_sensible_numbers(name):
    st = PRESETS[name]()
    st.solver.engine = "builtin"
    res = run(Backend().build_request(st))
    t = res.totals
    assert 0.05 < t.CL < 1.2 and 0.003 < t.CD < 0.05 and 5 < t.L_over_D < 80
    assert res.solver_used == ("numerical" if name in ("Cessna 172S (published wing)", "Light aircraft (Cessna 172-like)", "Swept wing, 30 degrees", "Sailplane wing, AR 22", "Wing in ground effect") else "classical")
    assert not st.validate()


def test_the_elliptic_example_has_unit_span_efficiency():
    st = PRESETS["Elliptic wing, AR 8"]()
    st.solver.engine = "builtin"
    t = run(Backend().build_request(st)).totals
    assert t.e == pytest.approx(1.0, abs=0.02)


# ========================================================================================= examples
def test_the_example_project_files_load_run_and_match_the_presets():
    from pathlib import Path

    folder = Path(__file__).resolve().parents[1] / "examples"
    files = sorted(folder.glob("*.json"))
    assert len(files) == len(PRESETS)
    for f in files:
        st = ProjectState.load(f)
        assert st.name in PRESETS and st.to_dict() == PRESETS[st.name]().to_dict(), f.name
        st.solver.engine = "builtin"
        assert run(Backend().build_request(st)).totals.CL > 0.0


# ============================================================================== engine fallback
def test_a_failing_section_engine_falls_back_to_the_builtin_one_with_a_note(monkeypatch):
    """E.g. NeuralFoil's dependencies fail to load on some machine: the analysis still runs, and says so."""
    real_make = backend_module.make_engine

    class Broken:
        name = "broken"

        def aero(self, airfoil):
            raise RuntimeError("casadi could not be loaded")

    monkeypatch.setattr(backend_module, "available_engines", lambda: {"neuralfoil": (True, ""), "builtin": (True, "")})
    monkeypatch.setattr(backend_module, "make_engine", lambda kind, settings=None: Broken() if kind == "neuralfoil" else real_make(kind, settings))
    st = ProjectState()
    st.solver.engine = "neuralfoil"
    be = Backend()
    req = be.build_request(st)
    res = run(req)
    assert "Built-in" in str(res.info["section_engine"]) and res.totals.CL > 0
    assert any("casadi could not be loaded" in w and "built-in engine was used instead" in w for w in res.warnings)
    # the built-in engine itself failing is a real error, not something to hide
    monkeypatch.setattr(backend_module, "make_engine", lambda kind, settings=None: Broken())
    st.solver.engine = "builtin"
    with pytest.raises(RuntimeError):
        Backend().build_request(st)
