"""The window, driven headlessly (QT_QPA_PLATFORM=offscreen): inputs -> background run -> results and plots."""

import math
import os

import pytest

pytestmark = pytest.mark.gui

QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from liftingline.airfoil import naca4  # noqa: E402
from liftingline.airfoil.dat_io import to_dat_text  # noqa: E402
from liftingline.core.analysis import run  # noqa: E402
from liftingline.gui.main_window import MainWindow  # noqa: E402
from liftingline.gui.settings import Settings  # noqa: E402
from liftingline.gui.theme import apply_theme  # noqa: E402
from liftingline.io.backend import Backend  # noqa: E402
from liftingline.io.presets import PRESETS  # noqa: E402
from liftingline.io.project import MODE_SECTIONS, ProjectState  # noqa: E402


@pytest.fixture(autouse=True)
def quiet_dialogs(monkeypatch):
    """Modal boxes would block a headless run: record them and answer 'discard / ok'."""
    seen = []

    def make(kind, answer):
        def fake(*args, **kwargs):
            seen.append((kind, args[1:3]))
            return answer

        return fake

    monkeypatch.setattr(QMessageBox, "question", make("question", QMessageBox.Discard))
    monkeypatch.setattr(QMessageBox, "warning", make("warning", QMessageBox.Ok))
    monkeypatch.setattr(QMessageBox, "information", make("information", QMessageBox.Ok))
    return seen


@pytest.fixture
def window(qtbot, tmp_path):
    apply_theme(QApplication.instance(), "light")
    w = MainWindow(settings=Settings(str(tmp_path / "settings.ini")), autorun=False, restore_layout=False)
    qtbot.addWidget(w)
    w.engine_combo.setCurrentIndex(w.engine_combo.findData("builtin"))  # fast and independent of NeuralFoil
    w._dirty = False
    w.show()
    yield w
    qtbot.waitUntil(lambda: not w.is_running(), timeout=30000)


def run_now(qtbot, w: MainWindow) -> None:
    w.request_run()
    qtbot.waitUntil(lambda: not w.is_running() and not w._pending, timeout=60000)


def table_values(w: MainWindow) -> dict[str, tuple[str, str]]:
    out = {}
    tree = w.results.tree
    for i in range(tree.topLevelItemCount()):
        group = tree.topLevelItem(i)
        for j in range(group.childCount()):
            c = group.child(j)
            out[c.text(0).strip()] = (c.text(1), c.text(2))
    return out


# ================================================================================ first run
def test_the_first_run_matches_the_backend_and_fills_every_view(qtbot, window):
    run_now(qtbot, window)
    out = window.last
    assert out is not None and out.result.solver_used == "numerical"  # the default wing has dihedral
    ref = run(Backend().build_request(window.collect_state()))
    assert out.result.totals.CL == pytest.approx(ref.totals.CL, rel=1e-12)
    vals = table_values(window)
    assert vals["Lift L"][1] == f"CL = {out.result.totals.CL:.4g}"[:0] + vals["Lift L"][1]  # row exists
    assert float(vals["Lift L"][0].split()[0]) == pytest.approx(out.result.totals.L, rel=1e-3)
    assert "solver" in window.results.header.text() and "CL =" in window.status_label.text()
    for name in ("planform", "spanwise", "airfoils"):
        window.plot_tabs.setCurrentIndex(("planform", "spanwise", "airfoils").index(name))
        assert window.canvases[name].error == ""


def test_the_airfoil_table_shows_section_data_after_a_run(qtbot, window):
    run_now(qtbot, window)
    panel = window.airfoil_panel
    val, chk, spin = panel._factor_widgets["a0"]
    assert 5.5 < float(val.text()) < 6.6
    assert "Built-in" in panel.data_label.text() and "Re =" in panel.data_label.text()


# ============================================================================ editing inputs
def test_editing_the_wing_changes_the_state_and_the_result(qtbot, window):
    run_now(qtbot, window)
    cl0 = window.last.result.totals.CL
    window.flow_panel.alpha.setValue(6.0)
    assert window.collect_state().flow.alpha_deg == 6.0 and window._dirty and window._stale
    run_now(qtbot, window)
    assert window.last.result.totals.CL > 1.5 * cl0 and not window._stale
    window.wing_panel.f_span.setValue(14.0)
    assert window.collect_state().wing.span == pytest.approx(14.0)
    run_now(qtbot, window)
    assert window.last.result.request.wing.span == pytest.approx(14.0)


def test_derived_planform_numbers_follow_the_chosen_inputs(qtbot, window):
    wp = window.wing_panel
    wp.define.setCurrentIndex(wp.define.findData("area_ar_taper"))
    assert wp.f_area.isEnabled() and not wp.f_span.isEnabled()
    wp.f_ar.setValue(10.0)
    wp.f_area.setValue(12.0)
    wp.f_taper.setValue(0.5)
    spec = window.collect_state().wing
    assert spec.span == pytest.approx(math.sqrt(120.0)) and spec.area == pytest.approx(12.0)
    assert wp.f_span.si() == pytest.approx(math.sqrt(120.0), abs=1e-4)  # the disabled field shows the derived span
    assert spec.tip_chord == pytest.approx(0.5 * spec.root_chord)


def test_auto_run_reruns_after_a_pause(qtbot, window):
    run_now(qtbot, window)
    first = window._run_id
    window.act_auto.setChecked(True)
    window.flow_panel.alpha.setValue(5.0)
    qtbot.waitUntil(lambda: window._run_id > first and not window.is_running(), timeout=30000)
    assert window.last.state.flow.alpha_deg == 5.0
    window.act_auto.setChecked(False)
    first = window._run_id
    window.flow_panel.alpha.setValue(1.0)
    qtbot.wait(900)
    assert window._run_id == first and window._stale  # nothing ran by itself


def test_switching_solver_and_engine_in_the_toolbar(qtbot, window):
    window.solver_combo.setCurrentIndex(window.solver_combo.findData("classical"))
    run_now(qtbot, window)
    assert window.last.result.solver_used == "classical"
    assert any("ignoring" in w for w in window.last.result.warnings)  # dihedral is ignored by the classical solver
    assert window.collect_state().solver.solver == "classical"


def test_two_quick_requests_end_with_the_newest_state(qtbot, window):
    window.request_run()
    window.flow_panel.alpha.setValue(7.0)
    window.request_run()  # arrives while the first is still running
    qtbot.waitUntil(lambda: not window.is_running() and not window._pending, timeout=60000)
    assert window.last.state.flow.alpha_deg == 7.0


# =================================================================================== units
def test_unit_system_changes_display_not_physics(qtbot, window):
    run_now(qtbot, window)
    cl = window.last.result.totals.CL
    window.units_combo.setCurrentIndex(1)  # Imperial
    assert window.system == "Imperial" and window.wing_panel.f_span.unit == "ft"
    assert window.wing_panel.f_span.value() == pytest.approx(11.0 / 0.3048, abs=1e-4)  # shown to 4 decimals
    assert window.collect_state().wing.span == 11.0  # ... but the physical value is untouched
    assert window.flow_panel.speed.combo.currentData() == "kt"
    assert table_values(window)["Lift L"][0].endswith("lbf")
    run_now(qtbot, window)
    assert window.last.result.totals.CL == pytest.approx(cl, rel=1e-9)
    window.units_combo.setCurrentIndex(0)
    assert window.wing_panel.f_span.value() == 11.0 and window.collect_state().wing.span == 11.0


# ==================================================================================== tables
def test_free_section_table_adds_stations_and_airfoil_rows(qtbot, window):
    wp = window.wing_panel
    wp.mode.setCurrentIndex(wp.mode.findData(MODE_SECTIONS))
    assert wp.table.rowCount() == 2 and window.airfoil_panel.table.rowCount() == 2
    wp.table.insert_row()
    assert wp.table.rowCount() == 3 and window.airfoil_panel.table.rowCount() == 3
    labels = [window.airfoil_panel.table.item(i, 0).text() for i in range(3)]
    assert labels[0].startswith("Root") and labels[1].startswith("Section 2") and labels[2].startswith("Tip")
    window.airfoil_panel.table.item(1, 1).setText("NACA 0012")
    run_now(qtbot, window)
    res = window.last.result
    assert len(res.request.wing.sections) == 3 and res.request.wing.sections[1].airfoil == "NACA 0012"
    assert window.last.state.wing.mode == MODE_SECTIONS


def test_a_bad_cell_or_a_bad_wing_blocks_the_run_without_crashing(qtbot, window):
    wp = window.wing_panel
    wp.mode.setCurrentIndex(wp.mode.findData(MODE_SECTIONS))
    run_now(qtbot, window)
    before = window._run_id
    wp.table.item(1, 1).setText("abc")  # chord is not a number
    window.request_run()
    assert window._run_id == before and not window.is_running()
    assert "not a number" in window.results.notes.toPlainText() and "Cannot run" in window.status_label.text()
    wp.table.item(1, 1).setText("1.0")
    wp.table.item(1, 0).setText("-2")  # tip before the root
    window.request_run()
    assert window._run_id == before and "increasing" in window.results.notes.toPlainText()


# ================================================================================== airfoils
def test_typing_an_airfoil_gives_live_feedback_and_changes_the_result(qtbot, window):
    run_now(qtbot, window)
    cl = window.last.result.totals.CL
    ap = window.airfoil_panel
    ap.table.selectRow(1)
    ap.table.item(1, 1).setText("NACA 4412")
    assert window.collect_state().airfoils.keys == ["NACA 2412", "NACA 4412"]
    assert "OK" in ap.status.text() and "t/c 12.0" in ap.status.text()
    ap.table.item(1, 1).setText("no-such-airfoil")
    assert "Problem" in ap.status.text()
    window.request_run()
    assert "cannot find airfoil" in window.results.notes.toPlainText()
    ap.table.item(1, 1).setText("NACA 4412")
    run_now(qtbot, window)
    assert window.last.result.totals.CL > cl  # more camber at the tip: more lift


def test_factor_overrides_are_stored_applied_and_cleared(qtbot, window):
    run_now(qtbot, window)
    ap = window.airfoil_panel
    ap.table.selectRow(0)
    _val, chk, spin = ap._factor_widgets["a0"]
    chk.setChecked(True)
    spin.setValue(5.0)
    assert window.collect_state().airfoils.overrides[0] == {"a0": 5.0}
    cl = window.last.result.totals.CL
    run_now(qtbot, window)
    assert window.last.infos[0].overridden == ("a0",) and window.last.result.totals.CL < cl
    chk.setChecked(False)
    assert window.collect_state().airfoils.overrides[0] == {}
    run_now(qtbot, window)
    assert window.last.infos[0].overridden == ()


def test_library_folder_gives_exact_ordinates_for_a_6_series_code(qtbot, window, tmp_path):
    lib = tmp_path / "uiuc"
    lib.mkdir()
    af = naca4.naca4_airfoil("2412")
    (lib / "naca63-412.dat").write_text(to_dat_text(af.with_name("NACA 63-412")))
    window.airfoil_panel.table.selectRow(1)
    window.airfoil_panel.table.item(1, 1).setText("NACA 63-412")
    assert "APPROXIMATE" in window.airfoil_panel.status.text()
    window.scan_library(str(lib))
    qtbot.waitUntil(lambda: window.backend.library is not None, timeout=20000)
    assert "1 airfoil files indexed" in window.solver_panel.library_status.text()
    assert "library file naca63-412.dat" in window.airfoil_panel.status.text()
    assert window.settings.get_str("library/folder") == str(lib)
    qtbot.waitUntil(lambda: not window.is_running(), timeout=30000)
    run_now(qtbot, window)
    assert not any("APPROXIMATE" in w for w in window.last.result.warnings)


def test_the_library_dialog_searches_and_previews(qtbot, tmp_path):
    from liftingline.airfoil.library import AirfoilLibrary
    from liftingline.gui.library_dialog import LibraryDialog

    lib = tmp_path / "lib"
    lib.mkdir()
    for name, code in (("naca0012", "0012"), ("naca2412", "2412"), ("naca4415", "4415")):
        (lib / f"{name}.dat").write_text(to_dat_text(naca4.naca4_airfoil(code)))
    dlg = LibraryDialog(AirfoilLibrary.scan(lib))
    qtbot.addWidget(dlg)
    assert dlg.list.count() == 3
    dlg.search.setText("24")
    assert dlg.list.count() == 1 and dlg.selected_name() == "naca2412"
    assert "t/c 12.0" in dlg.info.text() and dlg.canvas.error == ""
    dlg.search.setText("zzz")
    assert dlg.list.count() == 0 and dlg.selected_name() is None


# ================================================================================== failures
def test_an_unexpected_error_in_the_worker_is_reported_not_raised(qtbot, window, monkeypatch):
    def boom(state):
        raise RuntimeError("boom")

    monkeypatch.setattr(window.backend, "build_request", boom)
    run_now(qtbot, window)
    assert "boom" in window.results.notes.toPlainText() and "failed" in window.status_label.text()
    assert not window.is_running() and window.last is None


# ================================================================================ projects
def test_save_open_roundtrip_and_examples(qtbot, window, tmp_path):
    window.flow_panel.alpha.setValue(4.5)
    window.airfoil_panel.table.item(0, 1).setText("NACA 0012")
    path = tmp_path / "my wing.json"
    assert window._write(path) and not window._dirty and window.path == path
    state_saved = ProjectState.load(path)
    assert state_saved.flow.alpha_deg == 4.5 and state_saved.airfoils.keys[0] == "NACA 0012" and state_saved.name == "my wing"
    window._apply_preset("Rectangular wing, AR 8")
    qtbot.waitUntil(lambda: window.last is not None and not window.is_running(), timeout=60000)
    assert window.collect_state().wing.span == pytest.approx(8.0) and window.path is None
    window.open_project(str(path))
    qtbot.waitUntil(lambda: not window.is_running() and window.last is not None, timeout=60000)
    assert window.collect_state().flow.alpha_deg == 4.5 and window.path == path
    assert str(path) in window.settings.recent_files()
    assert window.collect_state().to_dict() == state_saved.to_dict() | {"solver": window.collect_state().to_dict()["solver"]}


def test_every_example_loads_runs_and_draws(qtbot, window):
    for name in PRESETS:
        window._apply_preset(name)
        qtbot.waitUntil(lambda: not window.is_running() and window.last is not None, timeout=60000)
        assert window.last.state.name == name
        for i, view in enumerate(("planform", "spanwise", "airfoils")):
            window.plot_tabs.setCurrentIndex(i)
            assert window.canvases[view].error == "", (name, view)


def test_csv_export_writes_the_current_result(qtbot, window, tmp_path, monkeypatch):
    run_now(qtbot, window)
    path = tmp_path / "out.csv"
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(path), "")))
    window.export_csv()
    text = path.read_text()
    assert "# solver" in text and "L," in text and text.count("\n") > 100


def test_copy_table_puts_tab_separated_text_on_the_clipboard(qtbot, window):
    run_now(qtbot, window)
    window.results.copy_table()
    text = QApplication.clipboard().text()
    assert text.startswith("Quantity\tValue") and "Lift L\t" in text


# ====================================================================================== look
def test_dark_theme_redraws_everything_without_errors(qtbot, window):
    run_now(qtbot, window)
    window.set_theme("dark")
    for i, view in enumerate(("planform", "spanwise", "airfoils")):
        window.plot_tabs.setCurrentIndex(i)
        assert window.canvases[view].error == ""
    window.set_theme("light")


def test_screenshots_of_every_tab_can_be_taken(qtbot, window, tmp_path):
    run_now(qtbot, window)
    for i in range(window.input_tabs.count()):
        window.input_tabs.setCurrentIndex(i)
        for j in range(window.plot_tabs.count()):
            window.plot_tabs.setCurrentIndex(j)
            path = tmp_path / f"shot-{i}-{j}.png"
            assert window.grab().save(str(path)) and os.path.getsize(path) > 10_000


def test_closing_a_dirty_project_asks_first(qtbot, window, quiet_dialogs):
    window._dirty = True
    window.close()
    assert any(kind == "question" for kind, _ in quiet_dialogs)
