"""Controls, polars/stall, trim, ideal twist and compare - the tools beyond a single analysis."""

import os

import pytest

pytestmark = pytest.mark.gui

QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtWidgets import QApplication, QInputDialog, QMessageBox  # noqa: E402

from liftingline.gui.dialogs import MODE_ALPHA_FOR_CL, MODE_ALPHA_FOR_LIFT, MODE_SPEED_FOR_LIFT, OptimalTwistDialog, TrimDialog  # noqa: E402
from liftingline.gui.main_window import MainWindow  # noqa: E402
from liftingline.gui.settings import Settings  # noqa: E402
from liftingline.gui.theme import apply_theme  # noqa: E402
from liftingline.io.backend import Backend  # noqa: E402
from liftingline.io.presets import PRESETS  # noqa: E402
from liftingline.io.project import MODE_SECTIONS, ProjectState  # noqa: E402


@pytest.fixture(autouse=True)
def quiet_dialogs(monkeypatch):
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
    w.engine_combo.setCurrentIndex(w.engine_combo.findData("builtin"))
    w._dirty = False
    w.show()
    yield w
    qtbot.waitUntil(lambda: not w.is_running() and not w.is_sweeping(), timeout=60000)


def run_now(qtbot, w):
    w.request_run()
    qtbot.waitUntil(lambda: not w.is_running() and not w._pending, timeout=60000)


def load_preset(qtbot, w, name):
    w.apply_state(PRESETS[name]())
    w.engine_combo.setCurrentIndex(w.engine_combo.findData("builtin"))
    run_now(qtbot, w)


# =========================================================================== control surfaces
def test_an_aileron_added_in_the_table_rolls_the_wing(qtbot, window):
    load_preset(qtbot, window, "Rectangular wing, AR 8")
    cp = window.controls_panel
    cp.add_surface("aileron")
    assert cp.table.rowCount() == 1 and window.collect_state().controls[0]["kind"] == "aileron"
    run_now(qtbot, window)
    assert abs(window.last.result.totals.C_roll) < 1e-12  # zero deflection: no effect
    cp.table.item(0, 5).setText("5")  # deflection, degrees
    assert window.collect_state().controls[0]["deflection_deg"] == 5.0
    run_now(qtbot, window)
    t = window.last.result.totals
    assert t.C_roll < -0.01 and t.C_yaw > 0.0  # right wing up, adverse yaw
    assert window.last.result.request.controls is not None


def test_flaps_raise_the_lift_and_pitch_the_nose_down(qtbot, window):
    load_preset(qtbot, window, "Rectangular wing, AR 8")
    base = window.last.result.totals
    cp = window.controls_panel
    cp.add_surface("flap")
    cp.table.item(0, 5).setText("10")
    run_now(qtbot, window)
    t = window.last.result.totals
    assert t.CL > base.CL + 0.1 and t.Cm < base.Cm and abs(t.C_roll) < 1e-12
    assert "&tau;" in cp.info.text() or "tau" in cp.info.text().lower()


def test_invalid_control_entries_block_the_run_and_are_explained(qtbot, window):
    load_preset(qtbot, window, "Rectangular wing, AR 8")
    cp = window.controls_panel
    cp.add_surface("aileron")
    before = window._run_id
    cp.table.item(0, 2).setText("90")  # inboard end beyond the outboard end (95 %)? no: 90 < 95 is fine
    cp.table.item(0, 3).setText("80")
    window.request_run()
    assert window._run_id == before and "span range" in window.results.notes.toPlainText()
    cp.table.item(0, 3).setText("abc")
    window.request_run()
    assert "not a number" in cp.problem.text() or "not a number" in window.results.notes.toPlainText()
    cp.remove_selected()
    assert cp.table.rowCount() == 0
    run_now(qtbot, window)
    assert window.last is not None


def test_control_surfaces_survive_a_project_roundtrip(qtbot, window, tmp_path):
    cp = window.controls_panel
    cp.add_surface("flap")
    cp.add_surface("aileron")
    cp.table.item(1, 5).setText("-3")
    path = tmp_path / "ctl.json"
    assert window._write(path)
    loaded = ProjectState.load(path)
    assert [c["kind"] for c in loaded.controls] == ["flap", "aileron"] and loaded.controls[1]["deflection_deg"] == -3.0
    window.apply_state(PRESETS["Rectangular wing, AR 8"]())
    assert window.controls_panel.table.rowCount() == 0
    window.open_project(str(path))
    assert window.controls_panel.table.rowCount() == 2 and window.collect_state().controls == loaded.controls


# ====================================================================================== polars
def finish_sweep(qtbot, window):
    qtbot.waitUntil(lambda: not window.is_sweeping(), timeout=60000)


def test_an_alpha_sweep_fills_the_polars_tab_and_reports_the_stall(qtbot, window):
    load_preset(qtbot, window, "Rectangular wing, AR 8")
    pv = window.polars
    pv.variable.setCurrentIndex(pv.variable.findData("alpha"))
    pv.rng["angle"][0].set_si(-2.0)
    pv.rng["angle"][1].set_si(26.0)
    pv.points.setValue(15)
    pv.run_btn.click()
    assert window.is_sweeping() and pv.cancel_btn.isEnabled()
    finish_sweep(qtbot, window)
    sw = pv.sweep
    assert sw is not None and len(sw.points) == 15 and all(p.ok for p in sw.points)
    assert pv.stall is not None and pv.stall.reached and "First stall" in pv.summary.text()
    assert pv.export_btn.isEnabled() and pv.canvas.error == ""
    window.plot_tabs.setCurrentWidget(pv)
    window.input_tabs.setCurrentIndex(0)
    window.flow_panel.alpha.setValue(2.0)  # inputs changed: the summary says so
    assert "inputs changed" in pv.summary.text()


def test_other_sweeps_run_and_the_deflection_sweep_needs_a_control_surface(qtbot, window):
    load_preset(qtbot, window, "Rectangular wing, AR 8")
    pv = window.polars
    pv.variable.setCurrentIndex(pv.variable.findData("deflection"))
    assert not pv.run_btn.isEnabled()  # no control surface yet
    window.controls_panel.add_surface("aileron")
    assert pv.control.count() == 1 and pv.run_btn.isEnabled()
    pv.points.setValue(5)
    pv.rng["angle"][0].set_si(-10.0)
    pv.rng["angle"][1].set_si(10.0)
    pv.run_btn.click()
    finish_sweep(qtbot, window)
    roll = pv.sweep.column("C_roll")
    assert roll[0] > 0 > roll[-1] and abs(roll[2]) < 1e-12 and pv.canvas.error == ""
    for variable in ("speed", "altitude", "beta"):
        pv.variable.setCurrentIndex(pv.variable.findData(variable))
        pv.points.setValue(4)
        pv.run_btn.click()
        finish_sweep(qtbot, window)
        assert pv.sweep.variable == variable and pv.canvas.error == "" and all(p.ok for p in pv.sweep.points), variable


def test_a_sweep_can_be_cancelled(qtbot, window):
    load_preset(qtbot, window, "Cessna 172S (published wing)")
    pv = window.polars
    pv.variable.setCurrentIndex(pv.variable.findData("alpha"))
    pv.points.setValue(120)
    pv.run_btn.click()
    pv.cancel_btn.click()
    finish_sweep(qtbot, window)
    assert len(pv.sweep.points) < 120 and any("cancelled" in n for n in pv.sweep.notes)


def test_sweep_csv_export(qtbot, window, tmp_path, monkeypatch):
    load_preset(qtbot, window, "Rectangular wing, AR 8")
    pv = window.polars
    pv.points.setValue(4)
    pv.run_btn.click()
    finish_sweep(qtbot, window)
    path = tmp_path / "sweep.csv"
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(path), "")))
    pv.export_btn.click()
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 5 and lines[0].startswith("value,CL")


# ======================================================================================== trim
def test_trim_dialog_finds_the_angle_and_applies_it(qtbot, window):
    load_preset(qtbot, window, "Rectangular wing, AR 8")
    state = window.collect_state()
    dlg = TrimDialog(lambda: window.backend.build_request(state), "SI", 0.5, 8000.0)
    qtbot.addWidget(dlg)
    dlg.mode.setCurrentIndex(dlg.mode.findData(MODE_ALPHA_FOR_CL))
    dlg.cl.setValue(0.8)
    got = []
    dlg.apply_requested.connect(lambda v, x: got.append((v, x)))
    dlg.solve_btn.click()
    qtbot.waitUntil(lambda: dlg.apply_btn.isEnabled(), timeout=60000)
    assert "angle of attack" in dlg.result.text() and "CL = 0.8000" in dlg.result.text()
    dlg.apply_btn.click()
    assert got and got[0][0] == "alpha_deg" and 6.0 < got[0][1] < 12.0
    window._apply_trim(*got[0])
    assert window.collect_state().flow.alpha_deg == pytest.approx(got[0][1])
    qtbot.waitUntil(lambda: not window.is_running() and window.last is not None, timeout=60000)
    assert window.last.result.totals.CL == pytest.approx(0.8, abs=1e-6)


def test_trim_dialog_for_lift_and_speed_and_for_unreachable_targets(qtbot, window):
    load_preset(qtbot, window, "Rectangular wing, AR 8")
    state = window.collect_state()
    dlg = TrimDialog(lambda: window.backend.build_request(state), "SI", 0.5, 8000.0)
    qtbot.addWidget(dlg)
    dlg.mode.setCurrentIndex(dlg.mode.findData(MODE_SPEED_FOR_LIFT))
    assert dlg.lift.isVisibleTo(dlg) and not dlg.cl.isVisibleTo(dlg)
    dlg.lift.set_si(20000.0)
    dlg.solve_btn.click()
    qtbot.waitUntil(lambda: dlg.apply_btn.isEnabled(), timeout=60000)
    assert "airspeed" in dlg.result.text() and "L = 20000" in dlg.result.text()
    dlg.mode.setCurrentIndex(dlg.mode.findData(MODE_ALPHA_FOR_LIFT))
    dlg.lift.set_si(1e8)  # far more than the wing can carry
    dlg.solve_btn.click()
    qtbot.waitUntil(lambda: dlg.solve_btn.isEnabled() and dlg.result.text() != "", timeout=60000)
    assert "not reached" in dlg.result.text() and not dlg.apply_btn.isEnabled()


# =================================================================================== ideal twist
def test_ideal_twist_turns_the_wing_into_a_table_with_an_elliptic_load(qtbot, window):
    load_preset(qtbot, window, "Tapered wing, AR 10")
    window.solver_combo.setCurrentIndex(window.solver_combo.findData("classical"))
    state = window.collect_state()
    dlg = OptimalTwistDialog(lambda: window.backend.build_request(state), 0.6, "SI")
    qtbot.addWidget(dlg)
    dlg.stations.setValue(21)
    dlg.compute_btn.click()
    qtbot.waitUntil(lambda: dlg.apply_btn.isEnabled(), timeout=60000)
    assert "Root angle of attack" in dlg.result.text() and dlg.canvas.error == ""
    got = []
    dlg.apply_requested.connect(got.append)
    dlg.apply_btn.click()
    window._apply_ideal_twist(got[0])
    qtbot.waitUntil(lambda: not window.is_running() and window.last is not None, timeout=60000)
    st = window.collect_state()
    n = len(got[0].rows)
    assert n == 21 and st.wing.mode == MODE_SECTIONS and len(st.wing.sections) == n and window.airfoil_panel.table.rowCount() == n
    assert st.airfoils.keys[0] == "NACA 4415" and st.airfoils.keys[-1] == "NACA 4412"  # nearest original section
    t = window.last.result.totals
    assert t.CL == pytest.approx(0.6, rel=0.03) and t.e > 0.99


# ======================================================================================= compare
def test_pinned_results_are_compared_in_a_table_and_overlaid(qtbot, window, monkeypatch):
    load_preset(qtbot, window, "Rectangular wing, AR 8")
    names = iter(["alpha 5", "alpha 8", "alpha 2"])
    monkeypatch.setattr(QInputDialog, "getText", staticmethod(lambda *a, **k: (next(names), True)))
    window.pin_result()
    assert window.plot_tabs.currentWidget() is window.compare
    window.flow_panel.alpha.setValue(8.0)
    run_now(qtbot, window)
    window.pin_result()
    cv = window.compare
    assert [s.name for s in cv.snapshots] == ["alpha 5", "alpha 8"]
    assert cv.table.columnCount() == 4 and cv.table.horizontalHeaderItem(2).text() == "alpha 8"
    rows = {cv.table.item(i, 0).text(): cv.table.item(i, 3).text() for i in range(cv.table.rowCount())}
    assert rows["Lift L (CL)"].startswith("+") and cv.canvas.error == ""  # lift grew with the angle of attack
    assert cv.table.item(0, 1).text() != cv.table.item(0, 2).text() and "N" in cv.table.item(0, 1).toolTip()
    window.flow_panel.alpha.setValue(2.0)
    run_now(qtbot, window)
    window.pin_result()
    cv.table.selectColumn(2)
    cv.remove_selected()
    assert [s.name for s in cv.snapshots] == ["alpha 5", "alpha 2"]
    cv.clear()
    assert cv.snapshots == [] and cv.table.columnCount() == 0


def test_pinning_without_a_result_or_beyond_the_limit_is_politely_refused(qtbot, window, quiet_dialogs, monkeypatch):
    window.pin_result()
    assert any(kind == "information" for kind, _ in quiet_dialogs)
    load_preset(qtbot, window, "Rectangular wing, AR 8")
    monkeypatch.setattr(QInputDialog, "getText", staticmethod(lambda *a, **k: ("x", True)))
    for _ in range(7):
        window.pin_result()
    assert len(window.compare.snapshots) == 6


def test_screenshots_of_the_new_tabs(qtbot, window, tmp_path):
    load_preset(qtbot, window, "Rectangular wing, AR 8")
    window.controls_panel.add_surface("aileron")
    window.polars.points.setValue(9)
    window.polars.run_btn.click()
    finish_sweep(qtbot, window)
    window.input_tabs.setCurrentWidget(window.input_tabs.widget(2))
    for j in range(window.plot_tabs.count()):
        window.plot_tabs.setCurrentIndex(j)
        path = tmp_path / f"extra-{j}.png"
        assert window.grab().save(str(path)) and os.path.getsize(path) > 10_000


# ============================================================================== report and self-test
@pytest.mark.parametrize("name", ["Rectangular wing, AR 8", "Swept wing, 30 degrees", "Cessna 172S (published wing)"])
def test_the_plotted_lift_loading_integrates_to_the_wing_lift(name):
    """For swept wings 'c cl' does not (cl is normal to the quarter-chord line); lift per span must, for every wing."""
    import numpy as np

    from liftingline.core.analysis import run
    from liftingline.gui.plots import lift_loading

    st = PRESETS[name]()
    st.solver.engine = "builtin"
    res = run(Backend().build_request(st))
    y, lp, ell = lift_loading(res)
    order = np.argsort(res.load.y)
    assert float(np.sum(lp * res.load.width[order])) == pytest.approx(res.totals.S * res.totals.CL, rel=1e-9)
    assert float(np.sum(ell * res.load.width[order])) == pytest.approx(res.totals.S * res.totals.CL, rel=5e-3)


def test_pdf_report_has_all_pages_and_embeds_the_sweep(qtbot, window, tmp_path, monkeypatch):
    load_preset(qtbot, window, "Swept wing, 30 degrees")
    path = tmp_path / "report.pdf"
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(path), "")))
    window.export_pdf_report()
    data = path.read_bytes()
    assert data.startswith(b"%PDF") and len(data) > 30_000
    assert "4 pages" in window.results.log_box.toPlainText()
    window.polars.points.setValue(5)
    window.polars.run_btn.click()
    finish_sweep(qtbot, window)
    window.export_pdf_report()
    assert "5 pages" in window.results.log_box.toPlainText()


def test_pdf_report_needs_a_result(qtbot, window, quiet_dialogs):
    window.export_pdf_report()
    assert any(kind == "information" for kind, _ in quiet_dialogs)


def test_the_self_test_runs_from_the_window_and_passes(qtbot, window):
    dlg = window.run_self_test(modal=False)
    qtbot.addWidget(dlg)
    qtbot.waitUntil(lambda: "checks passed" in dlg.status.text() or "failed" in dlg.status.text(), timeout=120000)
    assert dlg.status.text() == "All checks passed" and "ALL CHECKS PASSED" in dlg.text.toPlainText()
    dlg.close()


# ================================================================================ help and drops
def test_dropped_files_open_projects_and_assign_airfoils(qtbot, window, tmp_path):
    from liftingline.airfoil import naca4
    from liftingline.airfoil.dat_io import to_dat_text

    dat = tmp_path / "myfoil.dat"
    dat.write_text(to_dat_text(naca4.naca4_airfoil("4412")))
    window.airfoil_panel.table.selectRow(1)
    window.open_dropped_file(str(dat))
    assert window.collect_state().airfoils.keys[1] == str(dat)
    assert "assigned" in window.results.log_box.toPlainText()
    proj = tmp_path / "wing.json"
    PRESETS["Rectangular wing, AR 8"]().save(proj)
    window.open_dropped_file(str(proj))
    qtbot.waitUntil(lambda: not window.is_running() and window.last is not None, timeout=60000)
    assert window.collect_state().wing.span == pytest.approx(8.0)
    window.open_dropped_file(str(tmp_path / "notes.pdf"))
    assert "ignored dropped file" in window.results.log_box.toPlainText()


def test_the_help_texts_show(qtbot, window, quiet_dialogs):
    window.show_limits()
    window.show_conventions()
    assert sum(1 for kind, _ in quiet_dialogs if kind == "information") == 2
