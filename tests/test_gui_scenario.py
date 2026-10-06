"""The Scenario workspace (forces along a start-to-end airspeed run, playback, export), driven through the window."""

import csv
import json
import math

import numpy as np
import pytest

pytestmark = pytest.mark.gui

QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from liftingline.core.scenario import RAMP_PER_SPEED, ScenarioSpec, run_scenario  # noqa: E402
from liftingline.gui.main_window import MainWindow  # noqa: E402
from liftingline.gui.settings import Settings  # noqa: E402
from liftingline.gui.theme import apply_theme  # noqa: E402
from liftingline.io.backend import Backend  # noqa: E402
from liftingline.io.presets import PRESETS  # noqa: E402
from liftingline.io.project import ProjectState  # noqa: E402

MPH = 0.44704
PLOT_TABS = ["Planform", "Spanwise loads", "Airfoils", "Polars and stall", "Compare"]


@pytest.fixture(autouse=True)
def quiet_dialogs(monkeypatch):
    for name in ("question", "warning", "information"):
        monkeypatch.setattr(QMessageBox, name, lambda *a, **k: QMessageBox.Discard)


def make_window(qtbot, tmp_path, name="settings.ini"):
    apply_theme(QApplication.instance(), "light")
    w = MainWindow(settings=Settings(str(tmp_path / name)), autorun=False, restore_layout=False)
    qtbot.addWidget(w)
    w.engine_combo.setCurrentIndex(w.engine_combo.findData("builtin"))
    w._dirty = False
    return w


@pytest.fixture
def window(qtbot, tmp_path):
    w = make_window(qtbot, tmp_path)
    w.apply_state(PRESETS["Rectangular wing, AR 8"]())
    w.engine_combo.setCurrentIndex(w.engine_combo.findData("builtin"))
    w.resize(1280, 720)
    w.show()
    yield w
    qtbot.waitUntil(lambda: not w.is_running() and not w.is_scenario_running(), timeout=60000)
    w.pool.waitForDone(60000)


SMALL = ScenarioSpec(v_start=0.0, v_end=40.0, v_rotate=20.0, alpha_ground_deg=0.0, ramp_target_deg=8.0, mass_kg=500.0,
                     time_model=True, thrust_static_N=1500.0, thrust_end_N=1100.0, rolling_friction=0.02, points=15,
                     speed_unit="m/s")


def run_scenario_now(qtbot, w, spec=SMALL):
    w.show_scenario(True)
    w.scenario.load_spec(spec)
    w.scenario.run_btn.click()
    qtbot.waitUntil(lambda: not w.is_scenario_running(), timeout=120000)
    assert w.scenario.result is not None, w.scenario.problem.text()
    return w.scenario.result


# ================================================================================ the workspace
def test_scenario_is_an_extra_workspace_and_leaves_the_calculator_unchanged(window):
    w = window
    tabs = [w.plot_tabs.tabText(i) for i in range(w.plot_tabs.count())]
    assert tabs == PLOT_TABS  # no tab added to the calculator's plots
    assert w.central_stack.currentWidget() is w.plot_tabs and not w.act_scenario.isChecked()
    w.act_scenario.trigger()
    assert w.central_stack.currentWidget() is w.scenario and w.act_scenario.isChecked()
    w.scenario.back_btn.click()
    assert w.central_stack.currentWidget() is w.plot_tabs and not w.act_scenario.isChecked()
    w.show_scenario(True)
    assert w.act_scenario.isChecked()
    assert w.minimumSizeHint().width() <= 1280  # still fits a 16:9 laptop screen


def test_scenario_edits_do_not_rerun_the_main_analysis(qtbot, window):
    w = window
    w.request_run()
    qtbot.waitUntil(lambda: not w.is_running() and not w._pending, timeout=60000)
    before = w.results.tree.topLevelItemCount(), w.results.header.text(), w.status_label.text()
    w.show_scenario(True)
    w.scenario.mass.setValue(1234.0)  # as typed (kg)
    w.scenario.time_box.setChecked(True)
    assert w._dirty and not w.is_running() and not w._pending
    assert (w.results.tree.topLevelItemCount(), w.results.header.text(), w.status_label.text()) == before
    assert w.collect_state().scenario.mass_kg == pytest.approx(1234.0)


def test_invalid_scenario_inputs_disable_the_run(window):
    sv = window.scenario
    sv.load_spec(ScenarioSpec(v_start=0.0, v_end=30.0, v_rotate=40.0))
    assert not sv.run_btn.isEnabled() and "rotation" in sv.problem.text()
    sv.v_rotate.setValue(20.0)  # as typed: 20 mph
    assert sv.run_btn.isEnabled() and sv.problem.text() == ""


def test_speed_unit_switch_keeps_the_values(window):
    sv = window.scenario
    sv.load_spec(ScenarioSpec(v_end=90 * MPH, v_rotate=60 * MPH, ramp_mode=RAMP_PER_SPEED, ramp_rate_deg_per_mps=0.5,
                              speed_unit="mph"))
    before = sv.spec()
    sv.unit.setCurrentIndex(sv.unit.findData("km/h"))
    after = sv.spec()
    assert after.speed_unit == "km/h" and sv.v_end.value() == pytest.approx(90 * 1.609344, abs=0.006)
    assert after.v_end == pytest.approx(before.v_end, rel=1e-9) and after.v_rotate == pytest.approx(before.v_rotate)
    assert after.ramp_rate_deg_per_mps == pytest.approx(0.5, rel=1e-3)
    assert "km/h" in sv.rate_speed_label.text()


# ================================================================================ running
def test_a_run_from_the_window_matches_the_core_and_fills_the_playback(qtbot, window):
    w = window
    res = run_scenario_now(qtbot, w)
    sv = w.scenario
    direct = run_scenario(Backend().build_request(w.collect_state()), SMALL)
    assert np.allclose(res.data["L_N"], direct.data["L_N"], rtol=1e-12)
    assert res.liftoff is not None and res.liftoff.V == pytest.approx(direct.liftoff.V, rel=1e-12)
    assert sv.pages.currentIndex() == 1 and sv.summary.text().startswith("Lift-off at")
    assert sv.export_csv_btn.isEnabled() and sv.liftoff_btn.isEnabled()
    assert "scenario run: lift-off" in w.results.log_box.toPlainText()
    # the calculator itself still shows its own plots when coming back
    w.show_scenario(False)
    assert w.central_stack.currentWidget() is w.plot_tabs


def test_playback_controls(qtbot, window):
    w = window
    res = run_scenario_now(qtbot, w)
    sv = w.scenario
    t_end = float(res.data["t_s"][-1])
    assert sv.readouts["Time"].text() == "0.00 s" and sv.readouts["Status"].text() == "on the ground"
    sv.play_btn.click()
    assert sv._playing and sv.play_btn.text() == "Pause" and sv._timer.isActive()
    sv.play_btn.click()
    assert not sv._playing and sv.play_btn.text() == "Play"
    sv.speed.setCurrentIndex(sv.speed.findData(2.0))
    sv.seek(0.0)
    sv.advance(1.5)  # 1.5 s of real time at 2x = 3 s of the run
    assert sv._pos == pytest.approx(3.0)
    assert sv.readouts["Time"].text() == "3.00 s"
    c = sv.current()
    assert c["V_mps"] == pytest.approx(np.interp(3.0, res.data["t_s"], res.data["V_mps"]))
    sv.play()
    sv.advance(1e6)  # past the end: stops there
    assert sv._pos == pytest.approx(t_end) and not sv._playing and sv.slider.value() == 1000
    sv.slider.setValue(500)  # scrubbing
    assert sv._pos == pytest.approx(t_end / 2)
    sv.liftoff_btn.click()
    lof = res.liftoff
    assert sv._pos == pytest.approx(lof.t) and sv.readouts["Status"].text().startswith("lifted off")
    assert abs(sv.current()["L_minus_W_N"]) < 0.02 * lof_weight(res)
    sv.to_start_btn.click()
    assert sv._pos == 0.0
    sv.play()
    qtbot.waitUntil(lambda: sv._pos > 0.0, timeout=5000)  # the timer really moves it
    w.show_scenario(False)
    assert not sv._playing  # never animating out of sight


def lof_weight(res):
    return float(res.data["W_N"][0])


def test_playback_without_the_time_model_runs_over_airspeed(qtbot, window):
    w = window
    spec = ScenarioSpec(v_start=5.0, v_end=45.0, v_rotate=20.0, ramp_target_deg=8.0, mass_kg=500.0, points=11,
                        speed_unit="m/s")
    res = run_scenario_now(qtbot, w, spec)
    sv = w.scenario
    assert not res.has_time and sv.readouts["Time"].text() == "(no time model)"
    assert sv.playback_rate() == pytest.approx(40.0 / 10.0)  # the whole run takes ten seconds at 1x
    sv.seek(5.0)
    sv.advance(1.0)
    assert sv._pos == pytest.approx(9.0) and sv.readouts["Airspeed"].text() == "9.00 m/s"


def test_exports_from_the_window(qtbot, window, tmp_path):
    w = window
    res = run_scenario_now(qtbot, w)
    sv = w.scenario
    assert sv.export(str(tmp_path / "run.csv"), "csv") and sv.export(str(tmp_path / "run.txt"), "txt")
    lines = (tmp_path / "run.csv").read_text().splitlines()
    rows = list(csv.DictReader(ln for ln in lines if not ln.startswith("#") and not ln.startswith('"#')))
    assert len(rows) == res.n and float(rows[-1]["V_mps"]) == pytest.approx(40.0)
    assert "lift-off" in (tmp_path / "run.txt").read_text()
    assert sv.summary.text() == "Exported run.txt"


def test_scenario_settings_are_saved_with_the_project(qtbot, window, tmp_path):
    w = window
    w.scenario.load_spec(SMALL)
    path = tmp_path / "proj.json"
    w.collect_state().save(path)
    data = json.loads(path.read_text())
    assert data["scenario"]["mass_kg"] == 500.0
    w2 = make_window(qtbot, tmp_path, "other.ini")
    w2.apply_state(ProjectState.load(path))
    assert w2.scenario.spec() == SMALL
    assert math.isclose(w2.collect_state().scenario.v_rotate, 20.0)
