"""STL export of the wing: a closed, outward-oriented surface at the right place, size and orientation."""

from __future__ import annotations

import math

import numpy as np
import pytest

from liftingline.airfoil.naca4 import naca4_airfoil as naca4
from liftingline.core.wing import Wing
from liftingline.io.backend import Backend
from liftingline.io.presets import PRESETS
from liftingline.io.stl import (
    StlError,
    export_wing_stl,
    is_watertight,
    read_stl,
    signed_volume,
    surface_area,
    wing_mesh,
)


@pytest.fixture(scope="module")
def backend():
    return Backend()


def _airfoils(backend, keys):
    return [backend.resolve(k).airfoil for k in keys]


def _section_area(af) -> float:
    x = np.linspace(0.0, 1.0, 4001)
    return float(np.trapezoid(af.thickness(x), x))


def _reference_volume(wing, afs, half=False) -> float:
    """Integral of (airfoil area x chord^2) over the span, with the airfoils blended as in the mesh."""
    y = np.linspace(0.0 if half else -wing.half_span, wing.half_span, 8001)
    areas = np.array([_section_area(af) for af in afs])
    i0, i1, xi = wing.airfoil_blend(y)
    return float(np.trapezoid(((1 - xi) * areas[i0] + xi * areas[i1]) * wing.chord(y) ** 2, y))


@pytest.mark.parametrize("name", list(PRESETS))
def test_every_example_wing_gives_a_closed_mesh_with_the_right_volume_and_span(backend, name):
    st = PRESETS[name]()
    wing = st.build_wing()
    afs = _airfoils(backend, st.airfoils.keys)
    V, F = wing_mesh(wing, afs)
    assert is_watertight(F)
    vol = signed_volume(V, F)
    assert vol > 0.0  # outward normals
    assert vol == pytest.approx(_reference_volume(wing, afs), rel=2e-3)
    assert V[:, 1].min() == pytest.approx(-wing.half_span) and V[:, 1].max() == pytest.approx(wing.half_span)
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    assert np.linalg.norm(np.cross(b - a, c - a), axis=1).min() > 0.0  # no degenerate triangles (elliptic tip too)
    # a wing's wetted area is a little more than twice its planform area
    assert 2.0 * wing.area < surface_area(V, F) < 2.2 * wing.area
    # the half model is closed at the root and holds half the volume
    Vh, Fh = wing_mesh(wing, afs, half=True)
    assert is_watertight(Fh) and Vh[:, 1].min() == 0.0
    # (not exactly half: on a twisted wing the quads are not planar and their diagonals do not mirror)
    assert signed_volume(Vh, Fh) == pytest.approx(0.5 * vol, rel=1e-4)


def test_twist_incidence_sweep_and_dihedral_put_the_sections_where_they_belong():
    af = naca4("0012")
    wing = Wing.trapezoid(10.0, 2.0, 1.0, sweep_deg=20.0, sweep_at=0.0, dihedral_deg=5.0, twist_tip_deg=-4.0,
                          incidence_deg=3.0)
    V, F = wing_mesh(wing, [af, af], n_chord=60, n_span=10)
    assert is_watertight(F) and signed_volume(V, F) > 0.0

    def section(y, verts):
        s = verts[np.isclose(verts[:, 1], y)]
        le, te = s[np.argmin(s[:, 0])], s[np.argmax(s[:, 0])]
        return le, te

    hs, c_tip = 5.0, 1.0
    le, te = section(hs, V)
    theta = math.radians(-4.0 + 3.0)  # tip twist + root incidence, nose-up positive
    assert le[2] - te[2] == pytest.approx(c_tip * math.sin(theta), abs=2e-3)  # a nose-up section has its LE higher
    assert np.hypot(te[0] - le[0], te[2] - le[2]) == pytest.approx(c_tip, abs=2e-3)
    x_c4 = hs * math.tan(math.radians(20.0)) + 0.25 * c_tip  # LE swept 20 deg
    z_c4 = hs * math.tan(math.radians(5.0))
    mid = le + 0.25 * (te - le)
    assert mid[0] == pytest.approx(x_c4, abs=2e-3) and mid[2] == pytest.approx(z_c4, abs=2e-3)
    le_r, te_r = section(0.0, V)
    assert le_r[2] - te_r[2] == pytest.approx(2.0 * math.sin(math.radians(3.0)), abs=3e-3)
    # without the incidence the root chord lies along x
    V0, _ = wing_mesh(wing, [af, af], n_chord=60, n_span=10, include_incidence=False)
    le0, te0 = section(0.0, V0)
    assert abs(le0[2] - te0[2]) < 1e-9
    # left and right are mirror images
    lel, tel = section(-hs, V)
    assert np.allclose(lel * [1, -1, 1], le) and np.allclose(tel * [1, -1, 1], te)


def test_blended_airfoils_and_the_upper_surface_faces_up(backend):
    wing = Wing.trapezoid(8.0, 1.5, 0.9)
    afs = _airfoils(backend, ["NACA 4415", "NACA 0009"])
    V, F = wing_mesh(wing, afs)
    assert is_watertight(F)
    assert signed_volume(V, F) == pytest.approx(_reference_volume(wing, afs), rel=2e-3)
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    n = np.cross(b - a, c - a)
    centre = (a + b + c) / 3.0
    top = centre[:, 2] > 0.05  # well above the chord line: the upper surface
    assert np.all(n[top, 2] > 0.0)
    assert np.allclose(n.sum(axis=0), 0.0, atol=1e-9)  # closed surface: the area vectors cancel


@pytest.mark.parametrize("ascii_format", [False, True])
def test_files_are_written_in_the_chosen_unit_and_scale_and_read_back(backend, tmp_path, ascii_format):
    st = PRESETS["Tapered wing, AR 10"]()
    wing = st.build_wing()
    afs = _airfoils(backend, st.airfoils.keys)
    path = tmp_path / "wing.stl"
    info = export_wing_stl(path, wing, afs, unit="mm", scale=0.05, n_chord=40, n_span=12, ascii=ascii_format)
    assert info.watertight and info.n_triangles > 0 and info.volume_m3 > 0
    tri = read_stl(path)
    assert tri.shape == (info.n_triangles, 3, 3)
    pts = tri.reshape(-1, 3)
    assert pts[:, 1].max() == pytest.approx(wing.half_span * 1000.0 * 0.05, rel=1e-5)  # mm, 1:20 model
    if not ascii_format:
        assert path.read_bytes()[:80].startswith(b"Lifting-line wing calculator")
        assert path.stat().st_size == 84 + 50 * info.n_triangles
    else:
        assert path.read_text().startswith("solid ")
    path_m = tmp_path / "wing_m.stl"
    export_wing_stl(path_m, wing, afs, unit="m", n_chord=40, n_span=12, ascii=ascii_format)
    assert np.allclose(read_stl(path_m) * 1000.0 * 0.05, tri, rtol=1e-5, atol=1e-6)


def test_bad_options_are_refused(backend, tmp_path):
    wing = Wing.rectangular(6.0, 1.0)
    afs = [naca4("2412"), naca4("2412")]
    with pytest.raises(StlError, match="unit"):
        export_wing_stl(tmp_path / "a.stl", wing, afs, unit="furlong")
    with pytest.raises(StlError, match="scale"):
        export_wing_stl(tmp_path / "a.stl", wing, afs, scale=0.0)
    with pytest.raises(StlError, match="resolution"):
        wing_mesh(wing, afs, n_chord=4)
    with pytest.raises(StlError, match="one airfoil"):
        wing_mesh(wing, afs[:1])
