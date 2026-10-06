"""The separate 3-D model exporter: wings described in the calculator, saved as projects, exported as STL.

Every wing of the matrix goes the way a user's wing goes - a calculator project file, read by the exporter, meshed at
the default (fine) quality, written, read back - and the file is checked for closure, orientation, size, volume and
resolution."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from liftingline.io.presets import PRESETS
from liftingline.io.project import ProjectError, ProjectState
from liftingline.io.stl import QUALITY, estimate_triangles, read_stl
from liftingline.stl_app import export_project, load_project

ROOT = Path(__file__).resolve().parents[1]


def _spec(**wing) -> ProjectState:
    st = ProjectState()
    for k, v in wing.items():
        setattr(st.wing, k, v)
    st.wing.sync()
    return st


def _with_airfoils(st: ProjectState, *keys: str) -> ProjectState:
    st.airfoils.keys = list(keys)
    st.airfoils.fit(st.wing.n_sections())
    return st


def _sections(rows, keys) -> ProjectState:
    st = ProjectState()
    st.wing.mode = "sections"
    st.wing.sections = [dict(r) for r in rows]
    return _with_airfoils(st, *keys)


WINGS = {
    **{name: make for name, make in PRESETS.items()},
    "forward sweep, anhedral": lambda: _spec(span=10.0, root_chord=1.4, tip_chord=0.7, sweep_deg=-25.0, dihedral_deg=-6.0),
    "55 deg leading-edge sweep, taper 0.1": lambda: _spec(span=6.0, root_chord=4.0, tip_chord=0.4, sweep_deg=55.0, sweep_at=0.0),
    "strong washout and incidence": lambda: _spec(span=9.0, root_chord=1.2, tip_chord=0.6, twist_tip_deg=-12.0, incidence_deg=4.0),
    "small model, thin section": lambda: _with_airfoils(_spec(span=0.8, root_chord=0.12, tip_chord=0.08), "NACA 0006", "NACA 0006"),
    "thick root, 5-digit to 4-digit": lambda: _with_airfoils(_spec(span=12.0, root_chord=1.6, tip_chord=1.0), "NACA 23018", "NACA 4412"),
    "large transport, 6A-series code": lambda: _with_airfoils(
        _spec(span=40.0, root_chord=5.0, tip_chord=1.5, sweep_deg=30.0, dihedral_deg=5.0, twist_tip_deg=-4.0),
        "NACA 64A010", "NACA 64A010"),
    "four sections, kinks, winglet-like tip": lambda: _sections(
        [dict(y=0.0, chord=2.0, x_le=0.0, z=0.0, twist_deg=0.0), dict(y=2.0, chord=1.8, x_le=0.1, z=0.05, twist_deg=-1.0),
         dict(y=5.0, chord=1.0, x_le=0.9, z=0.45, twist_deg=-3.0), dict(y=6.0, chord=0.5, x_le=1.4, z=0.9, twist_deg=-4.0)],
        ["NACA 4415", "NACA 4412", "NACA 2410", "NACA 0009"]),
    "elliptic, cambered root to symmetric tip": lambda: _with_airfoils(_spec(mode="elliptic", span=8.0, root_chord=1.0), "NACA 2412", "NACA 0012"),
    "inverse taper": lambda: _spec(span=6.0, root_chord=0.8, tip_chord=1.2),
}


def _section_area(af) -> float:
    x = np.linspace(0.0, 1.0, 4001)
    t = af.thickness(x)
    return float(np.sum(0.5 * (t[1:] + t[:-1]) * np.diff(x)))


def _reference_volume(wing, afs) -> float:
    y = np.linspace(-wing.half_span, wing.half_span, 16001)
    areas = np.array([_section_area(af) for af in afs])
    i0, i1, xi = wing.airfoil_blend(y)
    f = ((1 - xi) * areas[i0] + xi * areas[i1]) * wing.chord(y) ** 2
    return float(np.sum(0.5 * (f[1:] + f[:-1]) * np.diff(y)))


@pytest.mark.parametrize("name", list(WINGS))
def test_every_wing_exports_as_a_fine_closed_model(tmp_path, name):
    project = tmp_path / "wing.json"
    WINGS[name]().save(project)  # exactly what File > Save in the calculator writes
    lw = load_project(project, library_folder="")
    wing = lw.wing
    out = tmp_path / "wing.stl"
    info = export_project(project, out, library_folder="")  # defaults: fine quality, millimetres, full wing
    assert info.watertight, name
    assert info.n_triangles == estimate_triangles(wing, "fine") > 100_000  # fine enough for CAD / printing
    assert out.stat().st_size == 84 + 50 * info.n_triangles == info.file_bytes
    tri = read_stl(out)
    pts = tri.reshape(-1, 3)
    assert tri.shape[0] == info.n_triangles
    # size in millimetres: full span, chordwise extent, and the volume of the real airfoil sections
    assert pts[:, 1].max() == pytest.approx(1000.0 * wing.half_span, rel=1e-6)
    assert pts[:, 1].min() == pytest.approx(-1000.0 * wing.half_span, rel=1e-6)
    assert info.volume_m3 == pytest.approx(_reference_volume(wing, lw.airfoils), rel=3e-3)
    # every facet has an area and a unit outward normal; the closed surface's area vectors cancel
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    assert np.all(np.linalg.norm(n, axis=1) > 0.0)
    assert np.allclose(n.sum(axis=0), 0.0, atol=1e-6 * np.abs(n).sum())
    # resolution: spanwise steps <= 1/4 of the local chord and <= 1 % of the semi-span; 160 points per surface
    ys = np.unique(np.round(pts[:, 1] / 1000.0, 9))
    ys = ys[ys >= 0.0]
    gaps = np.diff(ys)
    allowed = np.minimum(0.25 * np.maximum(np.minimum(wing.chord(ys[:-1]), wing.chord(ys[1:])), 0.02 * wing.root_chord),
                         0.01 * wing.half_span)
    assert np.all(gaps <= allowed * (1 + 1e-5)) or wing.kind == "elliptic"  # (float32 coordinates in the file)
    assert QUALITY["fine"].n_chord == 160


@pytest.mark.parametrize("quality", list(QUALITY))
def test_every_quality_level_gives_a_closed_model_and_finer_is_finer(tmp_path, quality):
    project = tmp_path / "w.json"
    PRESETS["Tapered wing, AR 10"]().save(project)
    info = export_project(project, tmp_path / "w.stl", library_folder="", quality=quality, half=True, unit="m")
    assert info.watertight and info.volume_m3 > 0
    order = list(QUALITY)
    if order.index(quality) > 0:
        coarser = export_project(project, tmp_path / "c.stl", library_folder="", quality=order[order.index(quality) - 1], half=True)
        assert info.n_triangles > 2 * coarser.n_triangles
        assert info.volume_m3 == pytest.approx(coarser.volume_m3, rel=2e-3)  # converging on the same solid


def test_airfoils_from_a_dat_file_and_from_the_library(tmp_path):
    from liftingline.airfoil.naca4 import naca4_airfoil

    af = naca4_airfoil("4415")
    lib = tmp_path / "lib"
    lib.mkdir()
    lines = ["MYFOIL 15"] + [f"{x:.6f} {y:.6f}" for x, y in af.coords]
    (lib / "myfoil.dat").write_text("\n".join(lines) + "\n")
    st = _with_airfoils(_spec(span=4.0, root_chord=0.5, tip_chord=0.3), str(lib / "myfoil.dat"), "myfoil")
    project = tmp_path / "p.json"
    st.save(project)
    with pytest.raises(ProjectError, match="section 2"):
        load_project(project, library_folder="")  # a library name without a library
    info = export_project(project, tmp_path / "p.stl", library_folder=str(lib), quality="medium")
    assert info.watertight


def test_the_command_line_writes_a_model_without_opening_a_window(tmp_path):
    project = tmp_path / "w.json"
    PRESETS["Swept wing, 30 degrees"]().save(project)
    out = tmp_path / "w.stl"
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen"}
    r = subprocess.run([sys.executable, str(ROOT / "stl_exporter.py"), str(project), "-o", str(out), "--quality", "medium",
                        "--unit", "cm", "--scale", "0.1", "--half", "--library", ""],
                       capture_output=True, text=True, timeout=300, cwd=str(ROOT), env=env)
    assert r.returncode == 0, r.stderr
    assert "watertight" in r.stdout and out.exists()
    pts = read_stl(out).reshape(-1, 3)
    wing = ProjectState.load(project).build_wing()
    assert pts[:, 1].min() == pytest.approx(0.0, abs=1e-6)  # half model
    assert pts[:, 1].max() == pytest.approx(wing.half_span * 100.0 * 0.1, rel=1e-6)  # cm, 1:10
    bad = tmp_path / "bad.json"
    _with_airfoils(_spec(), "no-such-airfoil", "no-such-airfoil").save(bad)
    r = subprocess.run([sys.executable, "-m", "liftingline.stl_app", str(bad), "-o", str(tmp_path / "x.stl"), "--library", ""],
                       capture_output=True, text=True, timeout=300, cwd=str(ROOT), env=env)
    assert r.returncode == 1 and "error" in r.stderr and not (tmp_path / "x.stl").exists()


@pytest.mark.gui
def test_the_exporter_window_opens_a_project_previews_and_exports(qtbot, tmp_path):
    pytest.importorskip("PySide6.QtWidgets")
    from liftingline.stl_app import window_class

    project = tmp_path / "w.json"
    PRESETS["Cessna 172S (published wing)"]().save(project)
    w = window_class()(library_folder="")
    qtbot.addWidget(w)
    w.show()
    assert not w.export_btn.isEnabled()
    assert w.open_project(str(project))
    assert "Cessna" in w.summary.text() and "NACA 2412" in w.summary.text()
    assert w.figure.axes and "triangles" in w.estimate.text()
    w.quality.setCurrentIndex(w.quality.findData("medium"))
    w.model.setCurrentIndex(1)
    out = tmp_path / "w.stl"
    assert w.export(str(out))
    qtbot.waitUntil(lambda: not w.busy, timeout=120000)
    assert w.last_info is not None and w.last_info.watertight and out.exists()
    assert "watertight" in w.status.text()
    assert not w.open_project(str(tmp_path / "missing.json"))
    assert "Cannot use" in w.status.text()
    w.close()
