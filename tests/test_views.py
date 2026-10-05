"""Front and side views of the Planform tab: built from the real sections, consistent with the 3-D model."""

from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from liftingline.airfoil.naca4 import naca4_airfoil  # noqa: E402
from liftingline.core.sections3d import front_silhouette, section_outline, section_shapes  # noqa: E402
from liftingline.core.wing import Wing  # noqa: E402
from liftingline.gui.plots import plot_planform  # noqa: E402
from liftingline.gui.theme import LIGHT  # noqa: E402
from liftingline.io.stl import wing_mesh  # noqa: E402

THICK, THIN = naca4_airfoil("4418"), naca4_airfoil("0009")


def _wing(**kw) -> Wing:
    args = dict(sweep_deg=0.0, dihedral_deg=0.0, twist_tip_deg=0.0, incidence_deg=0.0)
    args.update(kw)
    return Wing.trapezoid(10.0, 1.8, 0.9, airfoil_root="NACA 4418", airfoil_tip="NACA 0009", **args)


def test_front_view_shows_section_thickness_twist_and_dihedral():
    flat = _wing()
    lo, hi = front_silhouette(flat, [THICK, THIN], np.array([0.0, 5.0]))
    assert hi[0] - lo[0] == pytest.approx(1.8 * THICK.t_over_c, rel=0.03)  # thick root
    assert hi[1] - lo[1] == pytest.approx(0.9 * THIN.t_over_c, rel=0.03)  # thin tip airfoil, smaller chord
    twisted = _wing(twist_tip_deg=-8.0)
    lo_t, hi_t = front_silhouette(twisted, [THICK, THIN], np.array([5.0]))
    assert hi_t[0] - lo_t[0] > max(hi[1] - lo[1], 0.9 * math.sin(math.radians(8.0)))  # a twisted section stands taller
    dihedral = _wing(dihedral_deg=6.0)
    lo_d, hi_d = front_silhouette(dihedral, [THICK, THIN], np.array([5.0]))
    assert 0.5 * (lo_d[0] + hi_d[0]) - 0.5 * (lo[1] + hi[1]) == pytest.approx(5.0 * math.tan(math.radians(6.0)), abs=0.01)


def test_side_view_sections_move_with_sweep_and_turn_with_twist():
    _f, shapes = section_shapes([THICK, THIN], 60)
    plain = _wing()
    swept = _wing(sweep_deg=30.0, twist_tip_deg=-5.0, incidence_deg=2.0)
    x0, z0 = section_outline(plain, shapes, 5.0)
    x1, z1 = section_outline(swept, shapes, 5.0)
    assert x1.min() - x0.min() == pytest.approx(5.0 * math.tan(math.radians(30.0)), abs=0.03)  # tip moved aft by sweep
    le, te = int(np.argmin(x1)), int(np.argmax(x1))
    assert math.degrees(math.atan2(z1[le] - z1[te], x1[te] - x1[le])) == pytest.approx(-3.0, abs=0.3)  # -5 twist + 2 incidence
    root_x, root_z = section_outline(swept, shapes, 0.0)
    assert np.ptp(root_z) > 2.0 * np.ptp(z0) * 0.9  # the root is the thick 4418, the tip the thin 0009


@pytest.mark.parametrize("kw", [dict(), dict(sweep_deg=35.0, dihedral_deg=7.0, twist_tip_deg=-6.0, incidence_deg=3.0)])
def test_views_and_3d_model_agree_station_by_station(kw):
    wing = _wing(**kw)
    V, F = wing_mesh(wing, [THICK, THIN], quality="medium")
    for y in (0.0, 2.5, 5.0, -3.0):
        at = V[np.isclose(V[:, 1], y)]
        if not len(at):
            continue
        lo, hi = front_silhouette(wing, [THICK, THIN], np.array([y]), n_chord=80)
        assert at[:, 2].min() == pytest.approx(lo[0], abs=2e-4) and at[:, 2].max() == pytest.approx(hi[0], abs=2e-4)


def test_planform_tab_draws_front_and_side_views_with_the_tip_airfoil():
    wing = _wing(sweep_deg=30.0, twist_tip_deg=-6.0)
    fig = Figure(figsize=(8, 8), layout="constrained")
    plot_planform(fig, None, wing, "SI", LIGHT, None, [THICK, THIN])
    titles = [a.get_title() for a in fig.axes]
    assert any(t.startswith("Front view: sections") for t in titles)
    side = [a for a in fig.axes if a.get_title().startswith("Side view")][0]
    labels = [t.get_text() for t in side.get_legend().get_texts()]
    assert any("root: NACA 4418" in s for s in labels) and any("tip: NACA 0009" in s and "-6.0" in s for s in labels)
    # without airfoils (an unknown key) the views fall back to the chord lines instead of failing
    fig2 = Figure(figsize=(8, 8), layout="constrained")
    plot_planform(fig2, None, wing, "SI", LIGHT, None, None)
    assert any("airfoils not available" in a.get_title() for a in fig2.axes)


def test_narrow_canvases_get_the_compact_layout_with_the_same_content():
    wing = _wing(sweep_deg=20.0, twist_tip_deg=-4.0)
    fig = Figure(figsize=(4.5, 6.5), dpi=100, layout="constrained")  # a 1280 x 720 laptop's plot area
    plot_planform(fig, None, wing, "SI", LIGHT, None, [THICK, THIN])
    titles = [a.get_title() for a in fig.axes]
    assert "Front view (heights enlarged)" in titles and "Side view (to scale)" in titles
    side = [a for a in fig.axes if a.get_title() == "Side view (to scale)"][0]
    labels = [t.get_text() for t in side.get_legend().get_texts()]
    assert any(s.startswith("tip: 0009, -4.0") for s in labels)
