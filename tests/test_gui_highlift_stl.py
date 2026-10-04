"""The High lift tab (flap / slat toggles, stall analysis) and the STL export, driven through the window."""

import pytest

pytestmark = pytest.mark.gui

QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from liftingline.gui.main_window import MainWindow  # noqa: E402
from liftingline.gui.settings import Settings  # noqa: E402
from liftingline.gui.stl_dialog import StlExportDialog  # noqa: E402
from liftingline.gui.theme import apply_theme  # noqa: E402
from liftingline.io.presets import PRESETS  # noqa: E402
from liftingline.io.project import ProjectState  # noqa: E402
from liftingline.io.stl import read_stl  # noqa: E402


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
    w.pool.waitForDone(60000)


def run_now(qtbot, w):
    w.request_run()
    qtbot.waitUntil(lambda: not w.is_running() and not w._pending, timeout=60000)


def load_preset(qtbot, w, name="Rectangular wing, AR 8"):
    w.apply_state(PRESETS[name]())
    w.engine_combo.setCurrentIndex(w.engine_combo.findData("builtin"))
    run_now(qtbot, w)


# ================================================================================ high lift
def test_the_high_lift_tab_is_there_and_starts_retracted(window):
    titles = [window.input_tabs.tabText(i) for i in range(window.input_tabs.count())]
    assert "High lift" in titles
    hp = window.highlift_panel
    assert not hp.flap_box.isChecked() and not hp.slat_box.isChecked()
    assert not window.collect_state().high_lift.active
    assert "1.30" in hp.flap_info.text() or "+1.30" in hp.flap_info.text()  # single-slotted, landing: +1.30


def test_toggling_flaps_and_slats_changes_the_result_and_untoggling_restores_it_exactly(qtbot, window):
    load_preset(qtbot, window)
    clean = window.last.result.totals
    hp = window.highlift_panel
    hp.flap_box.setChecked(True)
    assert window.collect_state().high_lift.flaps_on
    run_now(qtbot, window)
    r = window.last.result
    assert r.totals.CL > clean.CL + 0.3 and r.totals.Cm < clean.Cm
    assert "landing setting" in window.results.header.text()
    assert any("High-lift devices deployed" in w for w in r.warnings)
    hp.slat_box.setChecked(True)
    run_now(qtbot, window)
    assert "slat" in window.last.result.info["high_lift"]
    hp.flap_box.setChecked(False)
    hp.slat_box.setChecked(False)
    run_now(qtbot, window)
    again = window.last.result.totals
    assert again.CL == clean.CL and again.CD == clean.CD and again.Cm == clean.Cm
    assert "high_lift" not in window.last.result.info


def test_device_type_sets_its_typical_chord_extension_and_phase_changes_the_numbers(qtbot, window):
    hp = window.highlift_panel
    hp.flap_box.setChecked(True)
    hp.flap_type.setCurrentIndex(hp.flap_type.findData("fowler"))
    assert hp.flap_ext.isEnabled() and hp.flap_ext.si() == pytest.approx(1.25)
    assert "+1.62" in hp.flap_info.text() or "+1.63" in hp.flap_info.text()  # 1.3 x 1.25
    hp.flap_type.setCurrentIndex(hp.flap_type.findData("split"))
    assert not hp.flap_ext.isEnabled() and hp.flap_ext.si() == 1.0
    hp.phase.setCurrentIndex(hp.phase.findData("takeoff"))
    assert window.collect_state().high_lift.phase == "takeoff"
    assert "-10" in hp.flap_info.text() and "+0.63" in hp.flap_info.text()  # 0.9 x 0.7


def test_invalid_device_spans_block_the_run_with_a_message(qtbot, window):
    load_preset(qtbot, window)
    hp = window.highlift_panel
    hp.flap_box.setChecked(True)
    hp.flap_in.set_si(70.0)
    hp.flap_out.set_si(40.0)
    hp._edited()
    assert "Invalid high-lift devices" in hp.problem.text()
    window.request_run()
    assert "High lift" in window.status_label.text() or "high-lift" in window.status_label.text().lower()


def test_stall_analysis_reports_clean_and_deployed_maximum_lift(qtbot, window):
    load_preset(qtbot, window)
    hp = window.highlift_panel
    hp.flap_box.setChecked(True)
    hp.slat_box.setChecked(True)
    hp.weight.set_si(12000.0)
    dlg = window.high_lift_report(modal=False)
    assert dlg is not None
    qtbot.waitUntil(lambda: dlg.status.text() in ("Finished", "Failed", "Stopped"), timeout=120000)
    assert dlg.status.text() == "Finished", dlg.text.toPlainText()
    text = dlg.text.toPlainText()
    assert "Clean wing" in text and "Devices out" in text and "Handbook estimate" in text and "stall speed" in text
    assert "CL_max clean" in hp.summary.text() and "stall speed" in hp.summary.text()
    dlg.close()


def test_stall_analysis_needs_a_deployed_device(qtbot, window, quiet_dialogs):
    assert window.high_lift_report(modal=False) is None
    assert any("Deploy" in str(args) for kind, args in quiet_dialogs)


def test_high_lift_settings_survive_save_and_open(qtbot, window, tmp_path):
    hp = window.highlift_panel
    hp.flap_box.setChecked(True)
    hp.flap_type.setCurrentIndex(hp.flap_type.findData("double_slotted"))
    hp.weight.set_si(8000.0)
    path = tmp_path / "hl.json"
    assert window._write(path)
    saved = ProjectState.load(path).high_lift
    assert saved.flaps_on and saved.flap_type == "double_slotted" and saved.weight_N == 8000.0
    window.apply_state(PRESETS["Rectangular wing, AR 8"]())
    assert not hp.flap_box.isChecked()
    window.open_project(str(path))
    qtbot.waitUntil(lambda: not window.is_running() and not window._pending, timeout=60000)
    assert hp.flap_box.isChecked() and hp.flap_type.currentData() == "double_slotted"
    assert window.collect_state().high_lift == saved


# ===================================================================================== STL
def test_the_wing_is_exported_as_a_closed_stl_model(qtbot, window, tmp_path):
    load_preset(qtbot, window, "Light aircraft (Cessna 172-like)")
    path = tmp_path / "wing.stl"
    info = window.export_stl(str(path), dict(unit="mm", scale=1.0, half=False, n_chord=40, n_span=20,
                                             include_incidence=True, ascii=False))
    assert info is not None and info.watertight and path.exists()
    pts = read_stl(path).reshape(-1, 3)
    span = window.collect_state().build_wing().span
    assert pts[:, 1].max() - pts[:, 1].min() == pytest.approx(span * 1000.0, rel=1e-6)
    assert "wing.stl" in window.results.log_box.toPlainText()


def test_stl_dialog_options_and_size_preview(qtbot):
    dlg = StlExportDialog(10.0)
    qtbot.addWidget(dlg)
    assert dlg.options()["unit"] == "mm" and dlg.options()["half"] is False
    assert "10000 mm" in dlg.size.text()
    dlg.scale.set_si(0.05)
    dlg.model.setCurrentIndex(1)
    dlg.unit.setCurrentIndex(dlg.unit.findData("cm"))
    o = dlg.options()
    assert o["half"] is True and o["scale"] == 0.05 and o["unit"] == "cm"
    assert "25 cm" in dlg.size.text()


def test_stl_export_refuses_a_broken_wing(qtbot, window, quiet_dialogs, tmp_path):
    st = window.collect_state()
    st.airfoils.keys = ["no-such-airfoil.dat", "no-such-airfoil.dat"]
    window.apply_state(st)
    assert window.export_stl(str(tmp_path / "x.stl"), dict(unit="mm")) is None
    assert not (tmp_path / "x.stl").exists()
    assert quiet_dialogs
