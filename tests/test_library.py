"""Library indexing / NACA-code resolution against synthetic fixtures that mimic UIUC naming."""

import pytest

from liftingline.airfoil import naca4, naca5, naca6
from liftingline.airfoil.dat_io import DatParseError, to_dat_text
from liftingline.airfoil.designation import parse_naca
from liftingline.airfoil.library import AirfoilLibrary, match_score


def _selig(airfoil, header):
    return to_dat_text(airfoil.with_name(header))


def _lednicer(airfoil, header):
    up, lo = airfoil.upper, airfoil.lower
    lines = [header, f"  {len(up)}.  {len(lo)}.", ""]
    lines += [f"{x:.6f}  {y:.6f}" for x, y in up] + [""] + [f"{x:.6f}  {y:.6f}" for x, y in lo]
    return "\n".join(lines) + "\n"


@pytest.fixture(scope="module")
def lib_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("uiuc_like")
    f6 = lambda code: naca6.naca6_fallback_airfoil(code, n_per_surface=40)  # noqa: E731
    files = {
        "naca2412.dat": _selig(naca4.naca4_airfoil("2412", n_per_surface=40), "NACA 2412 AIRFOIL"),
        "naca0012.dat": _selig(naca4.naca4_airfoil("0012", n_per_surface=40), "NACA 0012"),
        "naca23012.dat": _selig(naca5.naca5_airfoil("23012", n_per_surface=40), "NACA 23012"),
        "naca63-412.dat": _selig(f6("63-412"), "NACA 63-412"),
        "naca63-415.dat": _selig(f6("63-415"), "NACA 63-415"),
        "naca632415.dat": _selig(f6("63(2)-415"), "NACA 63(2)-415 AIRFOIL"),
        "naca64a010.dat": _selig(f6("64A010"), "NACA 64A010 AIRFOIL"),
        "naca651212.dat": _selig(f6("65(1)-212"), "NACA 65(1)-212"),
        "naca652212.dat": _selig(f6("65(2)-212"), "NACA 65(2)-212"),
        "e387.dat": _lednicer(naca4.naca4_airfoil("4412", n_per_surface=40), "E387 AIRFOIL"),
        "clarky.dat": _selig(naca4.naca4_airfoil("4412", n_per_surface=40), "CLARK Y").replace("\n", "\r\n").replace(" ", "\t"),
        "broken.dat": "garbage\nfoo bar\nnothing numeric here\n",
        "empty.dat": "",
    }
    for name, text in files.items():
        (d / name).write_text(text, encoding="utf-8")
    (d / "nested" / "deeper").mkdir(parents=True)
    (d / "nested" / "deeper" / "s1223.dat").write_text(_selig(naca4.naca4_airfoil("6412", n_per_surface=40), "S1223"))
    (d / "readme.md").write_text("not an airfoil")
    return d


@pytest.fixture(scope="module")
def lib(lib_dir):
    return AirfoilLibrary.scan(lib_dir)


def test_scan_indexes_recursively_and_ignores_other_files(lib):
    names = {e.name for e in lib}
    assert "s1223" in names and "naca63-412" in names
    assert "readme" not in names
    assert len(lib) == 14
    assert lib.root is not None


def test_scan_reports_progress_and_rejects_a_missing_folder(lib_dir, tmp_path):
    seen = []
    AirfoilLibrary.scan(lib_dir, progress=lambda i, n: seen.append((i, n)))
    assert seen and seen[-1][0] == seen[-1][1] == 14
    with pytest.raises(FileNotFoundError):
        AirfoilLibrary.scan(tmp_path / "nope")


def test_entries_carry_headers_and_recognised_codes(lib):
    e = lib.by_name("NACA632415.dat")
    assert e is not None and e.header.startswith("NACA 63(2)-415")
    assert any(c.is_6_series and c.sub == 2 for c in e.codes)
    assert lib.by_name("e387").codes == ()
    assert "CLARK Y" in lib.by_name("clarky").label
    assert lib.by_name("missing") is None


def test_search_is_case_insensitive_multi_word_and_naca_aware(lib):
    assert [e.name for e in lib.search("CLARK")] == ["clarky"]
    assert {e.name for e in lib.search("naca 2412")} >= {"naca2412"}
    assert lib.search("naca63-412")[0].name == "naca63-412"
    assert lib.search("63-412")[0].name == "naca63-412"  # NACA-aware: the dash form is resolved
    assert lib.search("airfoil e387")[0].name == "e387"
    assert lib.search("zzzz") == []
    assert len(lib.search("")) == len(lib)


@pytest.mark.parametrize(
    "designation, expected",
    [
        ("63-412", "naca63-412"),
        ("63412", "naca63-412"),
        ("NACA 63-412", "naca63-412"),
        ("632-412", "naca63-412"),  # subscript not in the file name: still the only candidate
        ("63(2)-415", "naca632415"),  # exact subscript match beats the plain 63-415
        ("63-415", "naca63-415"),  # no subscript requested: the plain entry ranks first
        ("64A010", "naca64a010"),
        ("64a010", "naca64a010"),
        ("2412", "naca2412"),
        ("23012", "naca23012"),
    ],
)
def test_resolve_by_canonical_key(lib, designation, expected):
    entry, ties = lib.resolve_best(parse_naca(designation))
    assert entry is not None and entry.name == expected
    assert len(ties) == 1


@pytest.mark.parametrize("designation", ["64-010", "63-612", "66-015", "4412", "63A-412"])
def test_unknown_designations_resolve_to_nothing(lib, designation):
    assert lib.resolve(parse_naca(designation)) == []
    assert lib.resolve_best(parse_naca(designation)) == (None, [])


def test_ambiguity_between_subscript_variants_is_reported(lib):
    entry, ties = lib.resolve_best(parse_naca("65-212"))
    assert {e.name for e in ties} == {"naca651212", "naca652212"}
    # asking for a specific subscript removes the ambiguity
    entry, ties = lib.resolve_best(parse_naca("65(2)-212"))
    assert entry.name == "naca652212" and len(ties) == 1


def test_match_score_ordering():
    t = parse_naca("63(2)-415")
    assert match_score(t, parse_naca("63(2)-415")) > match_score(t, parse_naca("63-415")) > match_score(t, parse_naca("63(1)-415"))
    assert match_score(t, parse_naca("64-415")) is None
    assert match_score(parse_naca("63-215 a=0.6"), parse_naca("63-215 a=0.8")) < match_score(
        parse_naca("63-215 a=0.6"), parse_naca("63-215")
    )


def test_load_parses_normalises_and_caches(lib):
    entry = lib.by_name("naca63-412")
    af = lib.load(entry)
    assert af.t_over_c == pytest.approx(0.12, abs=4e-3)
    assert af is lib.load(entry)  # cached
    assert af.source.endswith("naca63-412.dat")
    lednicer = lib.load(lib.by_name("e387"))
    assert any("lednicer" in n for n in lednicer.notes)
    assert lednicer.max_camber == pytest.approx(0.04, abs=2e-3)
    crlf_tab = lib.load(lib.by_name("clarky"))
    assert crlf_tab.t_over_c == pytest.approx(0.12, abs=4e-3)
    nested = lib.load(lib.by_name("s1223"))
    assert nested.max_camber == pytest.approx(0.06, abs=3e-3)


def test_loading_a_broken_file_raises_a_readable_error(lib):
    with pytest.raises(DatParseError):
        lib.load(lib.by_name("broken"))
    with pytest.raises(DatParseError):
        lib.load(lib.by_name("empty"))


def test_health_report_counts_formats_and_failures(lib):
    rep = lib.health_report()
    assert rep.total == 14 and rep.parsed_ok == 12
    assert {p.name for p, _ in rep.failures} == {"broken.dat", "empty.dat"}
    assert rep.formats["lednicer"] == 1 and rep.formats["selig"] == 11
    assert rep.with_naca_code >= 9
    text = rep.to_text()
    assert "failed               : 2" in text and "broken.dat" in text
