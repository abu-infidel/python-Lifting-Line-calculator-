import math

import numpy as np
import pytest

from liftingline.airfoil import naca4, naca5, naca6
from liftingline.airfoil.dat_io import DatParseError, airfoil_from_text, parse_dat_text, to_dat_text
from liftingline.airfoil.designation import DesignationError, extract_codes, parse_naca, try_parse_naca
from liftingline.airfoil.geometry import Airfoil
from liftingline.airfoil.thin_airfoil import thin_airfoil, thin_airfoil_from_airfoil


# ============================================================ designation grammar
def test_four_digit_codes():
    c = parse_naca("NACA 2412")
    assert (c.family, c.m, c.p, c.thickness) == ("4", 0.02, 0.4, 0.12)
    assert parse_naca("naca0012").m == 0.0
    assert parse_naca("4415").p == 0.4
    for bad in ("2012", "24", "abcd", "", "12345678"):
        with pytest.raises(DesignationError):
            parse_naca(bad)


def test_five_digit_codes():
    c = parse_naca("NACA 23012")
    assert c.family == "5" and c.cl5 == pytest.approx(0.3) and c.p5 == 3 and not c.reflex and c.thickness == 0.12
    r = parse_naca("23112")
    assert r.reflex and r.p5 == 3
    assert parse_naca("24015").cl5 == pytest.approx(0.3)
    with pytest.raises(DesignationError):
        parse_naca("23912")  # position digit 9
    with pytest.raises(DesignationError):
        parse_naca("21112")  # no reflex line for position 1


@pytest.mark.parametrize(
    "text, series, family, sub, cli, t, a",
    [
        ("63-412", 63, "6", None, 0.4, 0.12, None),
        ("63412", 63, "6", None, 0.4, 0.12, None),
        ("NACA 63(2)-415", 63, "6", 2, 0.4, 0.15, None),
        ("632-415", 63, "6", 2, 0.4, 0.15, None),
        ("63_2-415", 63, "6", 2, 0.4, 0.15, None),
        ("63₂-415", 63, "6", 2, 0.4, 0.15, None),
        ("65(216)-415", 65, "6", 2, 0.4, 0.15, None),
        ("66,2-015", 66, "6", 2, 0.0, 0.15, None),
        ("64A010", 64, "6A", None, 0.0, 0.10, None),
        ("65A410", 65, "6A", None, 0.4, 0.10, None),
        ("63A-210", 63, "6A", None, 0.2, 0.10, None),
        ("63-215 a=0.6", 63, "6", None, 0.2, 0.15, 0.6),
        ("651212", 65, "6", 1, 0.2, 0.12, None),
        ("633418", 63, "6", 3, 0.4, 0.18, None),
        ("66-018", 66, "6", None, 0.0, 0.18, None),
    ],
)
def test_six_series_grammar(text, series, family, sub, cli, t, a):
    c = parse_naca(text)
    assert (c.series, c.family, c.sub) == (series, family, sub)
    assert c.cli == pytest.approx(cli) and c.thickness == pytest.approx(t)
    assert c.a == a
    assert c.is_6_series


def test_six_series_default_mean_line_parameter_and_keys():
    assert parse_naca("63-412").effective_a == 1.0
    assert parse_naca("64A410").effective_a == 0.8
    assert parse_naca("63-215 a=0.6").effective_a == 0.6
    assert parse_naca("63-412").key == parse_naca("63412").key == parse_naca("632-412").key
    assert parse_naca("63-412").key != parse_naca("64-412").key
    assert parse_naca("64A410").key != parse_naca("64-410").key


def test_descriptions_are_readable():
    assert "camber 2% at 40%" in parse_naca("2412").describe()
    assert "design cl 0.30" in parse_naca("23012").describe()
    assert "6A-series" in parse_naca("64A010").describe()
    with pytest.raises(DesignationError):
        parse_naca("6-3-412x")


def test_extract_codes_from_file_names_and_headers():
    assert extract_codes("naca63-412")[0].key == parse_naca("63-412").key
    assert extract_codes("naca633418")[0].sub == 3
    assert extract_codes("NACA 63(2)-415 AIRFOIL")[0].sub == 2
    assert extract_codes("naca64a010")[0].family == "6A"
    assert extract_codes("naca2412")[0].family == "4"
    assert extract_codes("63-412 AIRFOIL")[0].key == parse_naca("63-412").key
    assert extract_codes("e387") == [] and extract_codes("clarky") == []
    assert try_parse_naca("rubbish") is None


# ============================================================ generators / geometry
def test_naca4_geometry_is_exact():
    af = naca4.naca4_airfoil("NACA 2412")
    assert af.coords.shape == (201, 2)
    assert af.t_over_c == pytest.approx(0.12, abs=3e-4)
    assert af.x_tmax == pytest.approx(0.30, abs=0.02)
    assert af.max_camber == pytest.approx(0.02, abs=3e-4)
    assert af.x_camber == pytest.approx(0.40, abs=0.03)
    assert af.te_gap < 1e-9  # closed trailing edge
    assert af.coords[0, 1] > 0 > af.coords[-1, 1] or af.te_gap < 1e-9
    assert naca4.naca4_airfoil("0012").max_camber == pytest.approx(0.0, abs=1e-12)
    assert naca4.naca4_airfoil("0012", closed_te=False).te_gap == pytest.approx(0.0025, abs=2e-4)


def test_airfoil_orientation_and_split():
    af = naca4.naca4_airfoil("4412")
    # both surfaces start at the shared nose point (just ahead of the nominal LE for cambered sections)
    assert af.upper[0] == pytest.approx(af.lower[0]) and abs(af.upper[0, 0]) < 5e-4
    assert af.upper[-1, 0] == pytest.approx(1.0, abs=1e-9)
    assert np.all(af.y_upper(np.linspace(0.05, 0.95, 20)) > af.y_lower(np.linspace(0.05, 0.95, 20)))
    # counter-clockwise (Selig) ordering
    x, y = af.coords[:, 0], af.coords[:, 1]
    assert 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y) > 0


def _dyt(x, t):
    """d(half-thickness)/dx of the closed-TE NACA 4-digit thickness form."""
    x = np.asarray(x, dtype=float)
    return 5.0 * t * (0.14845 / np.sqrt(x) - 0.1260 - 0.7032 * x + 0.8529 * x**2 - 0.4144 * x**3)


def _first_order_midline(x, m=0.02, p=0.4, t=0.12):
    """Vertical mid-line of a NACA 4-digit section: true mean line + y_t y_t' dz/dx.

    Thickness is laid off *perpendicular* to the mean line, so the mid-point of the upper and
    lower surface at the same x is not exactly on the mean line; to first order it is displaced
    by y_t * y_t' * dz/dx (derivation: the surfaces sit at x -/+ y_t sin(theta)).
    """
    x = np.asarray(x, dtype=float)
    return naca4.camber_line(x, m, p) + naca4.half_thickness(x, t) * _dyt(x, t) * naca4.camber_slope(x, m, p)


def _midline_slope(x, m=0.02, p=0.4, t=0.12, h=1e-7):
    x = np.asarray(x, dtype=float)
    hi, lo = np.clip(x + h, 1e-12, 1.0), np.clip(x - h, 1e-12, 1.0)
    return (_first_order_midline(hi, m, p, t) - _first_order_midline(lo, m, p, t)) / (hi - lo)


def test_extracted_camber_matches_mean_line_plus_known_first_order_term():
    af = naca4.naca4_airfoil("2412", n_per_surface=150)
    x = np.linspace(0.03, 0.97, 60)
    # exact to ~2e-6 once the geometric (definition) term is accounted for ...
    assert af.camber(x) == pytest.approx(_first_order_midline(x), abs=1e-5)
    # ... and that term is the *whole* difference to the NACA mean line (up to ~1.2e-3 near the LE)
    assert np.max(np.abs(af.camber(x) - naca4.camber_line(x, 0.02, 0.4))) < 2e-3
    xs = np.linspace(0.1, 0.9, 41)
    assert af.camber_slope(xs) == pytest.approx(_midline_slope(xs), abs=5e-4)
    assert af.camber_slope(xs) == pytest.approx(naca4.camber_slope(xs, 0.02, 0.4), abs=1e-2)


def test_resample_keeps_shape_and_can_close_trailing_edge():
    af = naca4.naca4_airfoil("2412", closed_te=False)
    rs = af.resample(60, close_te=True)
    assert rs.coords.shape == (121, 2)
    assert rs.te_gap < 1e-12
    assert rs.t_over_c == pytest.approx(af.t_over_c, abs=2e-3)


def test_digest_is_stable_and_geometry_sensitive():
    a, b = naca4.naca4_airfoil("2412"), naca4.naca4_airfoil("2412")
    assert a.digest == b.digest != naca4.naca4_airfoil("2413").digest


# ============================================================ thin-airfoil anchors (plan)
@pytest.mark.parametrize(
    "code, aL0_deg, cm",
    [("2412", -2.077, -0.0531), ("4412", -4.154, -0.1062), ("2415", -2.077, -0.0531)],
)
def test_four_digit_thin_airfoil_values(code, aL0_deg, cm):
    c = parse_naca(code)
    r = thin_airfoil(lambda x: naca4.camber_slope(x, c.m, c.p))
    assert r.alpha_L0_deg == pytest.approx(aL0_deg, abs=2e-3)
    assert r.cm_c4 == pytest.approx(cm, abs=2e-4)
    # The same from *coordinates* (vertical mid-line). This differs from the NACA mean line by the
    # geometric term y_t y_t' dz/dx: about -0.04 deg for the 2412, -0.06 deg for the 2415 (it scales
    # with t^2 and camber) - a definition effect, not an error, and independent of point density.
    # Hence coordinate-based thin-airfoil numbers are a fallback; the panel method is the reference.
    rc = thin_airfoil_from_airfoil(naca4.naca4_airfoil(code, n_per_surface=200))
    assert rc.alpha_L0_deg == pytest.approx(aL0_deg, abs=0.1)
    assert rc.cm_c4 == pytest.approx(cm, abs=3e-3)
    # ... and it is reproduced by thin-airfoil theory applied to the first-order mid-line
    ref = thin_airfoil(lambda x: _midline_slope(x, c.m, c.p, c.thickness))
    assert rc.alpha_L0_deg == pytest.approx(ref.alpha_L0_deg, abs=0.01)
    assert abs(ref.alpha_L0_deg - aL0_deg) > 0.02 or c.m == 0.0  # the effect is real, not noise


def test_symmetric_airfoil_has_no_camber_effects():
    r = thin_airfoil_from_airfoil(naca4.naca4_airfoil("0012"))
    assert abs(r.alpha_L0_deg) < 1e-6 and abs(r.cm_c4) < 1e-9


@pytest.mark.parametrize("code, alpha_L0_deg", [("230", -1.094), ("240", -1.292), ("250", -1.483)])
def test_five_digit_standard_mean_lines_have_design_cl_03(code, alpha_L0_deg):
    c = parse_naca(f"{code}12")
    r = thin_airfoil(lambda x: naca5.camber_slope(x, c))
    assert r.cl_ideal == pytest.approx(0.30, abs=2e-3)
    assert r.alpha_L0_deg == pytest.approx(alpha_L0_deg, abs=0.01)


def test_five_digit_camber_lines_close_at_both_ends():
    for text in ("23012", "24015", "25112", "22112", "23112", "21012"):
        c = parse_naca(text)
        z = naca5.camber_line(np.array([0.0, 1.0]), c)
        assert z == pytest.approx([0.0, 0.0], abs=1e-12)
    # design cl scales with the first digit
    c4 = parse_naca("43012")
    assert thin_airfoil(lambda x: naca5.camber_slope(x, c4)).cl_ideal == pytest.approx(0.6, abs=4e-3)


@pytest.mark.parametrize(
    "a, alpha_i, alpha_L0, cm",
    [(1.0, 0.0, -9.119, -0.25), (0.8, 1.536, -7.579, -0.2019), (0.5, 3.035, -6.079, -0.1389)],
)
def test_six_series_mean_line_matches_verified_values(a, alpha_i, alpha_L0, cm):
    """c_li = 1: verified numerically during planning (z(0)=z(1)=0, design cl recovered)."""
    r = thin_airfoil(lambda x: naca6.mean_line_slope(x, 1.0, a), n=60000)
    assert r.cl_ideal == pytest.approx(1.0, abs=2e-3)
    assert r.alpha_ideal_deg == pytest.approx(alpha_i, abs=0.01)
    assert r.alpha_L0_deg == pytest.approx(alpha_L0, abs=0.02)
    assert r.cm_c4 == pytest.approx(cm, abs=1e-3)
    z = naca6.mean_line(np.array([0.0, 1.0]), 1.0, a)
    assert z == pytest.approx([0.0, 0.0], abs=1e-12)


def test_six_series_mean_line_slope_is_the_derivative_and_scales_with_cli():
    x = np.linspace(0.05, 0.95, 19)
    for a in (1.0, 0.8, 0.5):
        e = 1e-6
        numeric = (naca6.mean_line(x + e, 0.4, a) - naca6.mean_line(x - e, 0.4, a)) / (2 * e)
        assert naca6.mean_line_slope(x, 0.4, a) == pytest.approx(numeric, rel=1e-5, abs=1e-8)
    assert naca6.mean_line(x, 0.8, 0.8) == pytest.approx(2 * naca6.mean_line(x, 0.4, 0.8))


def test_six_series_approximate_fallback_is_flagged_and_has_the_right_camber():
    af = naca6.naca6_fallback_airfoil("63-412")
    assert af.approximate and af.notes
    assert af.t_over_c == pytest.approx(0.12, abs=3e-3)
    r = thin_airfoil_from_airfoil(naca6.naca6_fallback_airfoil("63-412", n_per_surface=250), n=3000)
    assert r.alpha_L0_deg == pytest.approx(-0.4 * 9.119, abs=0.1)  # exact mean line, c_li = 0.4, a = 1


# ============================================================ .dat reading / writing
def _lednicer_text(af, counts=True, blank_between=True, crlf=False):
    up, lo = af.upper, af.lower
    lines = ["NACA 2412 Lednicer"]
    if counts:
        lines.append(f"  {len(up)}.  {len(lo)}.")
        lines.append("")
    lines += [f"{x:.6f}  {y:.6f}" for x, y in up]
    if blank_between:
        lines.append("")
    lines += [f"{x:.6f}  {y:.6f}" for x, y in lo]
    return ("\r\n" if crlf else "\n").join(lines) + "\n"


def _same_shape(a: Airfoil, b: Airfoil, tol=2e-4):
    x = np.linspace(0.02, 0.98, 30)
    assert a.y_upper(x) == pytest.approx(b.y_upper(x), abs=tol)
    assert a.y_lower(x) == pytest.approx(b.y_lower(x), abs=tol)


@pytest.fixture(scope="module")
def ref_airfoil():
    return naca4.naca4_airfoil("2412", n_per_surface=60)


def test_selig_round_trip(ref_airfoil):
    back = airfoil_from_text(to_dat_text(ref_airfoil))
    assert back.name == ref_airfoil.name
    assert back.coords == pytest.approx(ref_airfoil.coords, abs=1e-6)


@pytest.mark.parametrize("counts, blank, crlf", [(True, True, False), (True, False, True), (False, True, False)])
def test_lednicer_variants(ref_airfoil, counts, blank, crlf):
    parsed = parse_dat_text(_lednicer_text(ref_airfoil, counts, blank, crlf))
    assert parsed.fmt == ("lednicer" if counts else "lednicer-nocount")
    _same_shape(airfoil_from_text(_lednicer_text(ref_airfoil, counts, blank, crlf)), ref_airfoil)


def test_messy_selig_file_is_read(ref_airfoil):
    pts = ref_airfoil.coords
    body = "\n".join(f"\t{x:.6E}, {y:.6E} ! pt" for x, y in pts)
    text = "My airfoil\nsecond header line\n\r\n" + body.replace("E", "D", 3)[:0] + body
    af = airfoil_from_text(text.replace("\n", "\r\n"))
    _same_shape(af, ref_airfoil)
    assert af.name == "My airfoil"


def test_percent_chord_rotated_scaled_and_reversed_input_is_normalised(ref_airfoil):
    ang = math.radians(7.0)
    rot = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]])
    raw = (ref_airfoil.coords @ rot.T) * 100.0 + np.array([3.0, -2.0])  # percent chord, rotated, shifted
    for pts in (raw, raw[::-1]):  # clockwise (lower surface first) as well
        af = Airfoil.from_raw("x", pts)
        assert af.coords[:, 0].min() == pytest.approx(0.0, abs=1e-9)
        assert af.coords[np.argmin(af.coords[:, 0])] == pytest.approx([0.0, 0.0], abs=1e-9)
        assert af.te_gap < 1e-6 and af.coords[0, 0] == pytest.approx(1.0, abs=1e-6)
        _same_shape(af, ref_airfoil, tol=5e-4)
    assert any("rotated" in n or "scaled" in n for n in Airfoil.from_raw("x", raw).notes)


def test_duplicate_leading_edge_point_is_dropped(ref_airfoil):
    pts = np.insert(ref_airfoil.coords, int(np.argmin(ref_airfoil.coords[:, 0])), [0.0, 0.0], axis=0)
    af = Airfoil.from_raw("dup", pts)
    assert len(af.coords) == len(ref_airfoil.coords)
    assert any("duplicate" in n for n in af.notes)


@pytest.mark.parametrize("text", ["", "just a title\n", "name\n1 2\n3 4\n"])
def test_unusable_files_raise_a_readable_error(text):
    with pytest.raises(DatParseError):
        airfoil_from_text(text)
