"""The Scenario tab: forces along a run from a start to an end airspeed, with an animated, scrubbable playback.

Two pages: *Setup* (the schedule, the aircraft mass, the optional takeoff-run model) and *Results and playback* (graphs
with a moving cursor, the numbers at the cursor, play / pause, playback speed, a slider to move through the run, and
export as CSV or text). The scenario uses the wing, airfoils, atmosphere, high-lift devices and extra drag of the
other tabs; it never changes them, and nothing in the rest of the window depends on it.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import QSignalBlocker, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..core import units
from ..core.scenario import RAMP_PER_SECOND, RAMP_PER_SPEED, RAMP_TARGET, ScenarioResult, ScenarioSpec, to_csv, to_text
from ..io.project import ProjectState
from .plots import make_canvas, style_axes
from .theme import Theme
from .widgets import GuardedCombo, GuardedSpinBox, UnitSpin, muted, set_role

PLAYBACK_SPEEDS = (0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0)
#: without a time model the timeline runs over airspeed; at 1x the whole run plays in this many seconds
SPEED_TIMELINE_SECONDS = 10.0
FRAME_MS = 40


def _form(box: QWidget) -> QFormLayout:
    """A form whose labels move above the fields when the page is narrow (16:9 laptops)."""
    f = QFormLayout(box)
    f.setRowWrapPolicy(QFormLayout.WrapLongRows)
    f.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
    return f


def _speed_spin(unit: str) -> UnitSpin:
    return UnitSpin("speed", unit=unit, decimals=2, minimum=0.0, maximum=400.0, step=0.5)


class ScenarioView(QWidget):
    run_requested = Signal()
    cancel_requested = Signal()
    edited = Signal()
    back_requested = Signal()

    def __init__(self, system: str = "SI", parent=None) -> None:
        super().__init__(parent)
        self.system = system
        self.result: ScenarioResult | None = None
        self._busy = False
        self._playing = False
        self._pos = 0.0  # playback position on the timeline (s of run time, or m/s of airspeed)
        self._last_tick = 0.0
        self._artists: dict = {}
        self._timer = QTimer(self)
        self._timer.setInterval(FRAME_MS)
        self._timer.timeout.connect(self._tick)
        self._rate_unit = "mph"
        self._build()
        self.load_spec(ScenarioSpec())

    # ===================================================================== layout
    def _build(self) -> None:
        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        head = QHBoxLayout()
        title = QLabel("<b>Scenario</b> - the wing's forces along a run from a start to an end airspeed")
        title.setWordWrap(True)
        head.addWidget(title, 1)
        self.back_btn = QPushButton("Back to the calculator")
        self.back_btn.clicked.connect(self.back_requested.emit)
        head.addWidget(self.back_btn)
        lay.addLayout(head)
        self.pages = QTabWidget()
        lay.addWidget(self.pages)
        self.pages.addTab(self._build_setup(), "Setup")
        self.pages.addTab(self._build_playback(), "Results and playback")

    def _build_setup(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        body = QWidget()
        v = QVBoxLayout(body)

        box = QGroupBox("Speed range")
        f = _form(box)
        self.unit = GuardedCombo()
        for u in units.units_of("speed"):
            self.unit.addItem(u, u)
        self.v_start, self.v_end = _speed_spin("mph"), _speed_spin("mph")
        self.points = GuardedSpinBox()
        self.points.setRange(5, 400)
        self.points.setToolTip("Analysis points along the run (the lift-off point is found exactly in between)")
        f.addRow("Speed unit", self.unit)
        f.addRow("Start airspeed", self.v_start)
        f.addRow("End airspeed", self.v_end)
        f.addRow("Points", self.points)
        v.addWidget(box)

        box = QGroupBox("Angle of attack")
        f = _form(box)
        self.alpha_ground = UnitSpin(None, unit="deg", decimals=2, minimum=-15.0, maximum=25.0, step=0.5,
                                     tooltip="Angle of attack of the fuselage reference line before rotation (the wing's root "
                                             "incidence is added, as on the Flow tab)")
        self.v_rotate = _speed_spin("mph")
        self.v_rotate.setToolTip("From this airspeed on the angle of attack rises linearly")
        self.ramp = GuardedCombo()
        self.ramp.addItem("Target angle at end speed", RAMP_TARGET)
        self.ramp.addItem("Degrees per speed unit", RAMP_PER_SPEED)
        self.ramp.addItem("Degrees per second", RAMP_PER_SECOND)
        self.ramp.setToolTip("How the angle of attack rises after rotation (degrees per second needs the time model)")
        self.target = UnitSpin(None, unit="deg", decimals=2, minimum=-15.0, maximum=35.0, step=0.5)
        self.rate_speed = QDoubleSpinBox()
        self.rate_speed.setDecimals(4)
        self.rate_speed.setRange(-10.0, 10.0)
        self.rate_speed.setSingleStep(0.05)
        self.rate_speed.setKeyboardTracking(False)
        self.rate_time = UnitSpin(None, unit="deg/s", decimals=2, minimum=-20.0, maximum=20.0, step=0.5)
        f.addRow("Before rotation", self.alpha_ground)
        f.addRow("Rotation airspeed", self.v_rotate)
        f.addRow("Rise after rotation", self.ramp)
        f.addRow("Target angle", self.target)
        self.rate_speed_label = QLabel("Rate (deg per mph)")
        f.addRow(self.rate_speed_label, self.rate_speed)
        f.addRow("Rate (deg per s)", self.rate_time)
        v.addWidget(box)

        box = QGroupBox("Aircraft")
        f = _form(box)
        self.mass = UnitSpin("mass", self.system, decimals=1, minimum=0.001, maximum=1e7, step=10.0,
                             tooltip="Aircraft mass: lift-off when the wing's lift equals the weight m g")
        self.weight_label = muted()
        f.addRow("Mass", self.mass)
        f.addRow(self.weight_label)
        v.addWidget(box)

        self.time_box = QGroupBox("Compute time and distance (takeoff-run model)")
        self.time_box.setCheckable(True)
        f = _form(self.time_box)
        self.thrust0 = UnitSpin("force", self.system, decimals=1, minimum=0.0, maximum=1e8, step=50.0,
                                tooltip="Thrust at standstill")
        self.thrust1 = UnitSpin("force", self.system, decimals=1, minimum=0.0, maximum=1e8, step=50.0,
                                tooltip="Thrust at the end airspeed (linear in between)")
        self.mu = QDoubleSpinBox()
        self.mu.setDecimals(3)
        self.mu.setRange(0.0, 0.5)
        self.mu.setSingleStep(0.005)
        self.mu.setKeyboardTracking(False)
        self.mu.setToolTip("Rolling friction: about 0.02 on a paved runway, 0.04-0.05 on short grass, 0.10 on long grass")
        f.addRow("Static thrust", self.thrust0)
        f.addRow("Thrust at end speed", self.thrust1)
        f.addRow("Rolling friction", self.mu)
        f.addRow(muted("m dV/dt = thrust - drag - friction x (weight - lift) on a level runway without wind; "
                       "the climb after lift-off is not modelled."))
        v.addWidget(self.time_box)

        self.ground_box = QGroupBox("Ground effect")
        self.ground_box.setCheckable(True)
        f = _form(self.ground_box)
        self.height = UnitSpin("length", self.system, decimals=3, minimum=0.01, maximum=1e4, step=0.1,
                               tooltip="Height of the wing's quarter-chord line above the runway")
        f.addRow("Wing height", self.height)
        v.addWidget(self.ground_box)

        v.addWidget(muted(
            "The run uses the wing, airfoils, altitude, high-lift devices and extra drag of the other tabs; their speed "
            "and angle of attack are replaced by this schedule. Nothing here changes the main analysis."
        ))
        v.addStretch(1)
        scroll.setWidget(body)
        outer.addWidget(scroll, 1)

        row = QHBoxLayout()
        self.run_btn = QPushButton("Run scenario")
        self.run_btn.clicked.connect(self.run_requested.emit)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel_requested.emit)
        self.bar = QProgressBar()
        self.bar.setMaximumWidth(160)
        self.bar.setTextVisible(False)
        self.bar.hide()
        row.addWidget(self.run_btn)
        row.addWidget(self.cancel_btn)
        row.addWidget(self.bar)
        row.addStretch(1)
        outer.addLayout(row)
        self.problem = QLabel()
        self.problem.setWordWrap(True)
        set_role(self.problem, "error")
        outer.addWidget(self.problem)

        for w in (self.v_start, self.v_end, self.v_rotate, self.alpha_ground, self.target, self.rate_time, self.mass,
                  self.thrust0, self.thrust1, self.height):
            w.valueChanged.connect(self._edited)
        for w in (self.rate_speed, self.mu):
            w.valueChanged.connect(self._edited)
        self.points.valueChanged.connect(self._edited)
        self.ramp.currentIndexChanged.connect(self._edited)
        self.time_box.toggled.connect(self._edited)
        self.ground_box.toggled.connect(self._edited)
        self.unit.currentIndexChanged.connect(self._unit_changed)
        return page

    def _build_playback(self) -> QWidget:
        page = QWidget()
        v = QVBoxLayout(page)
        v.setContentsMargins(0, 0, 0, 0)
        self.canvas = make_canvas(toolbar=False)
        v.addWidget(self.canvas, 1)

        row = QHBoxLayout()
        self.to_start_btn = QPushButton("|<")
        self.to_start_btn.setToolTip("Back to the start")
        self.play_btn = QPushButton("Play")
        self.to_end_btn = QPushButton(">|")
        self.to_end_btn.setToolTip("To the end")
        self.liftoff_btn = QPushButton("Lift-off")
        self.liftoff_btn.setToolTip("Jump to the lift-off point")
        self.speed = GuardedCombo()
        for s in PLAYBACK_SPEEDS:
            self.speed.addItem(f"{s:g}x", s)
        self.speed.setCurrentIndex(PLAYBACK_SPEEDS.index(1.0))
        self.speed.setToolTip("Playback speed")
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 1000)
        self.slider.setToolTip("Drag to move through the run")
        for w in (self.to_start_btn, self.play_btn, self.to_end_btn, self.liftoff_btn, self.speed):
            row.addWidget(w)
        row.addWidget(self.slider, 1)
        v.addLayout(row)

        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        self.readouts: dict[str, QLabel] = {}
        names = ("Time", "Airspeed", "Angle of attack", "Lift", "Weight", "Lift - weight", "Drag", "Thrust",
                 "Acceleration", "Distance", "cl / cl,max", "Status")
        short = {"Angle of attack": "AoA", "Lift - weight": "L - W", "Acceleration": "Accel.", "cl / cl,max": "cl/cl,max"}
        for i, name in enumerate(names):
            r, c = divmod(i, 3)
            lab = muted(short.get(name, name))
            lab.setToolTip(name)
            val = QLabel("-")
            val.setTextInteractionFlags(Qt.TextSelectableByMouse)
            grid.addWidget(lab, r, 2 * c)
            grid.addWidget(val, r, 2 * c + 1)
            self.readouts[name] = val
        v.addLayout(grid)

        row = QHBoxLayout()
        self.export_csv_btn = QPushButton("Export CSV...")
        self.export_txt_btn = QPushButton("Export text...")
        for b in (self.export_csv_btn, self.export_txt_btn):
            b.setEnabled(False)
            row.addWidget(b)
        self.summary = muted()
        self.summary.setTextInteractionFlags(Qt.TextSelectableByMouse)
        row.addWidget(self.summary, 1)
        v.addLayout(row)

        self.to_start_btn.clicked.connect(lambda: self.seek(0.0))
        self.to_end_btn.clicked.connect(lambda: self.seek(self._timeline_end()))
        self.play_btn.clicked.connect(self.toggle_play)
        self.liftoff_btn.clicked.connect(self.seek_liftoff)
        self.slider.valueChanged.connect(self._slider_moved)
        self.export_csv_btn.clicked.connect(lambda: self.export(kind="csv"))
        self.export_txt_btn.clicked.connect(lambda: self.export(kind="txt"))
        self._set_playback_enabled(False)
        return page

    # ===================================================================== state
    def speed_unit(self) -> str:
        return self.unit.currentData() or "mph"

    def spec(self) -> ScenarioSpec:
        u = self.speed_unit()
        per_unit = float(units.to_si("speed", 1.0, u))  # m/s per display unit
        return ScenarioSpec(
            v_start=self.v_start.si(), v_end=self.v_end.si(), v_rotate=self.v_rotate.si(),
            alpha_ground_deg=self.alpha_ground.si(), ramp_mode=self.ramp.currentData(),
            ramp_rate_deg_s=self.rate_time.si(), ramp_rate_deg_per_mps=self.rate_speed.value() / per_unit,
            ramp_target_deg=self.target.si(), mass_kg=self.mass.si(), time_model=self.time_box.isChecked(),
            thrust_static_N=self.thrust0.si(), thrust_end_N=self.thrust1.si(), rolling_friction=self.mu.value(),
            ground_effect=self.ground_box.isChecked(), ground_height_m=self.height.si(), points=self.points.value(),
            speed_unit=u,
        )

    def load_spec(self, s: ScenarioSpec) -> None:
        self._busy = True
        try:
            with QSignalBlocker(self.unit):
                i = self.unit.findData(s.speed_unit)
                self.unit.setCurrentIndex(i if i >= 0 else self.unit.findData("mph"))
            u = self.speed_unit()
            for w in (self.v_start, self.v_end, self.v_rotate):
                w.set_unit(u)
            self.v_start.set_si(s.v_start)
            self.v_end.set_si(s.v_end)
            self.v_rotate.set_si(s.v_rotate)
            self.alpha_ground.set_si(s.alpha_ground_deg)
            with QSignalBlocker(self.ramp):
                self.ramp.setCurrentIndex(max(self.ramp.findData(s.ramp_mode), 0))
            self.target.set_si(s.ramp_target_deg)
            self.rate_time.set_si(s.ramp_rate_deg_s)
            with QSignalBlocker(self.rate_speed):
                self.rate_speed.setValue(s.ramp_rate_deg_per_mps * float(units.to_si("speed", 1.0, u)))
            self.mass.set_si(s.mass_kg)
            with QSignalBlocker(self.time_box), QSignalBlocker(self.ground_box):
                self.time_box.setChecked(bool(s.time_model))
                self.ground_box.setChecked(bool(s.ground_effect))
            self.thrust0.set_si(s.thrust_static_N)
            self.thrust1.set_si(s.thrust_end_N)
            with QSignalBlocker(self.mu), QSignalBlocker(self.points):
                self.mu.setValue(s.rolling_friction)
                self.points.setValue(int(s.points))
            self.height.set_si(s.ground_height_m)
        finally:
            self._busy = False
        self._refresh()

    def load(self, state: ProjectState) -> None:
        self.load_spec(state.scenario)
        self.clear()

    def store(self, state: ProjectState) -> None:
        state.scenario = self.spec()

    def problems(self) -> list[str]:
        return [f"Scenario: {m}" for m in self.spec().problems()]

    def set_system(self, system: str) -> None:
        self.system = system
        for w in (self.mass, self.thrust0, self.thrust1, self.height):
            w.set_system(system)
        self._refresh()
        if self.result is not None:
            self._draw()

    # ===================================================================== slots
    def _unit_changed(self, _i: int) -> None:
        old_rate = self.spec_rate_si
        u = self.speed_unit()
        for w in (self.v_start, self.v_end, self.v_rotate):
            w.set_unit(u)
        with QSignalBlocker(self.rate_speed):
            self.rate_speed.setValue(old_rate * float(units.to_si("speed", 1.0, u)))
        self._edited()
        if self.result is not None:
            self._draw()

    @property
    def spec_rate_si(self) -> float:
        """The per-speed rate in deg per (m/s), from the field and the unit it was entered in."""
        return self.rate_speed.value() / float(units.to_si("speed", 1.0, self._rate_unit))

    def _edited(self, *_a) -> None:
        if self._busy:
            return
        self._refresh()
        self.edited.emit()

    def _refresh(self) -> None:
        s = self.spec()
        self._rate_unit = self.speed_unit()
        mode = s.ramp_mode
        self.target.setEnabled(mode == RAMP_TARGET)
        self.rate_speed.setEnabled(mode == RAMP_PER_SPEED)
        self.rate_time.setEnabled(mode == RAMP_PER_SECOND)
        self.rate_speed_label.setText(f"Rate (deg per {self.speed_unit()})")
        uF = units.display_unit(self.system, "force")
        self.weight_label.setText(f"weight m g = {units.from_si('force', s.weight_N, uF):.1f} {uF}")
        msgs = s.problems()
        self.problem.setText("Cannot run: " + "; ".join(msgs) if msgs else "")
        self.run_btn.setEnabled(not msgs and not self.bar.isVisible())

    # ===================================================================== running
    def set_running(self, on: bool) -> None:
        self.bar.setVisible(on)
        self.cancel_btn.setEnabled(on)
        self.run_btn.setEnabled(not on and not self.spec().problems())
        if on:
            self.bar.setRange(0, 0)
            self.pause()

    def set_progress(self, done: int, total: int) -> None:
        self.bar.setRange(0, max(total, 1))
        self.bar.setValue(done)

    def mark_stale(self) -> None:
        if self.result is not None and "(inputs changed" not in self.summary.text():
            self.summary.setText(self.summary.text() + "  (inputs changed since this run)")

    def show_error(self, message: str) -> None:
        self.set_running(False)
        self.problem.setText("The scenario failed: " + message)

    def clear(self) -> None:
        self.pause()
        self.result = None
        self.summary.setText("")
        self._set_playback_enabled(False)
        fig = self.canvas.figure
        fig.clear()
        self.canvas.canvas.draw_idle()

    def show_result(self, res: ScenarioResult) -> None:
        self.set_running(False)
        self.result = res
        u = self.speed_unit()
        lof = res.liftoff
        if lof is not None:
            when = f" after {lof.t:.2f} s and {self._len(lof.x)}" if res.has_time else ""
            text = f"Lift-off at {units.from_si('speed', lof.V, u):.2f} {u}{when}, angle of attack {lof.alpha_deg:.2f} deg."
        else:
            text = res.warnings[0] if res.warnings else "No lift-off."
        self.summary.setText(text)
        self.problem.setText("")
        self._set_playback_enabled(True)
        self._draw()
        self.seek(0.0)
        self.pages.setCurrentIndex(1)

    def _len(self, x: float) -> str:
        uL = units.display_unit(self.system, "length")
        return f"{units.from_si('length', x, uL):.1f} {uL}"

    # ===================================================================== playback
    def _timeline(self) -> np.ndarray:
        d = self.result.data
        return d["t_s"] if self.result.has_time else d["V_mps"]

    def _timeline_end(self) -> float:
        return float(self._timeline()[-1]) if self.result is not None else 0.0

    def hideEvent(self, event) -> None:  # noqa: N802 - Qt naming
        self.pause()  # never keep animating out of sight
        super().hideEvent(event)

    def _set_playback_enabled(self, on: bool) -> None:
        for w in (self.to_start_btn, self.play_btn, self.to_end_btn, self.liftoff_btn, self.slider, self.speed,
                  self.export_csv_btn, self.export_txt_btn):
            w.setEnabled(on)
        self.liftoff_btn.setEnabled(on and self.result is not None and self.result.liftoff is not None)

    def toggle_play(self) -> None:
        if self._playing:
            self.pause()
        else:
            self.play()

    def play(self) -> None:
        if self.result is None:
            return
        tl = self._timeline()
        if self._pos >= float(tl[-1]) - 1e-12:
            self._pos = float(tl[0])
        self._playing = True
        self._last_tick = time.perf_counter()
        self.play_btn.setText("Pause")
        self._timer.start()

    def pause(self) -> None:
        self._playing = False
        self._timer.stop()
        self.play_btn.setText("Play")

    def playback_rate(self) -> float:
        """Timeline units per second of real time at the chosen playback speed."""
        factor = float(self.speed.currentData() or 1.0)
        if self.result is None:
            return factor
        if self.result.has_time:
            return factor
        tl = self._timeline()
        return factor * float(tl[-1] - tl[0]) / SPEED_TIMELINE_SECONDS

    def advance(self, seconds: float) -> None:
        """Move the playback on by ``seconds`` of real time (the timer calls this; tests can too)."""
        if self.result is None:
            return
        tl = self._timeline()
        pos = self._pos + seconds * self.playback_rate()
        if pos >= float(tl[-1]):
            pos = float(tl[-1])
            self.pause()
        self.seek(pos)

    def _tick(self) -> None:
        now = time.perf_counter()
        dt, self._last_tick = now - self._last_tick, now
        self.advance(dt)

    def seek(self, pos: float) -> None:
        if self.result is None:
            return
        tl = self._timeline()
        self._pos = float(min(max(pos, float(tl[0])), float(tl[-1])))
        span = float(tl[-1] - tl[0]) or 1.0
        with QSignalBlocker(self.slider):
            self.slider.setValue(int(round(1000 * (self._pos - float(tl[0])) / span)))
        self._update_cursor()

    def seek_liftoff(self) -> None:
        lof = self.result.liftoff if self.result is not None else None
        if lof is not None:
            self.seek(lof.t if self.result.has_time else lof.V)

    def _slider_moved(self, value: int) -> None:
        if self.result is None:
            return
        tl = self._timeline()
        self._pos = float(tl[0]) + value / 1000.0 * float(tl[-1] - tl[0])
        self._update_cursor()

    def current(self) -> dict[str, float]:
        """Every column of the result at the playback position (interpolated between points)."""
        tl = self._timeline()
        d = self.result.data
        out = {}
        for k, col in d.items():
            ok = np.isfinite(col)
            out[k] = float(np.interp(self._pos, tl[ok], col[ok])) if ok.sum() >= 2 else math.nan
        return out

    def _update_cursor(self) -> None:
        if self.result is None:
            return
        c = self.current()
        self._fill_readouts(c)
        x = self._pos if self.result.has_time else self._sp(self._pos)
        for line in self._artists.get("cursors", ()):
            line.set_xdata([x, x])
        for key, marker in self._artists.get("markers", {}).items():
            marker.set_data([x], [self._artists["yconv"][key](c)])
        self.canvas.canvas.draw_idle()

    def _sp(self, v: float) -> float:
        return float(units.from_si("speed", v, self.speed_unit()))

    def _fill_readouts(self, c: dict[str, float]) -> None:
        uF = units.display_unit(self.system, "force")
        F = lambda v: "-" if not math.isfinite(v) else f"{units.from_si('force', v, uF):,.1f} {uF}"  # noqa: E731
        u = self.speed_unit()
        r = self.readouts
        r["Time"].setText(f"{c['t_s']:.2f} s" if self.result.has_time else "(no time model)")
        r["Airspeed"].setText(f"{self._sp(c['V_mps']):.2f} {u}")
        r["Angle of attack"].setText(f"{c['alpha_deg']:.2f} deg")
        r["Lift"].setText(F(c["L_N"]))
        r["Weight"].setText(F(c["W_N"]))
        r["Lift - weight"].setText(F(c["L_minus_W_N"]))
        r["Drag"].setText(F(c["D_N"]))
        r["Thrust"].setText(F(c["T_N"]))
        r["Acceleration"].setText("-" if not math.isfinite(c["a_mps2"]) else f"{c['a_mps2']:.3f} m/s²")
        r["Distance"].setText("-" if not math.isfinite(c["x_m"]) else self._len(c["x_m"]))
        r["cl / cl,max"].setText("-" if not math.isfinite(c["cl_ratio"]) else f"{c['cl_ratio']:.2f}")
        lof = self.result.liftoff
        tl_lof = (lof.t if self.result.has_time else lof.V) if lof is not None else math.inf
        status = "lifted off" if self._pos >= tl_lof - 1e-9 else "on the ground"
        if c["cl_ratio"] >= 0.98:
            status += ", STALL"
        r["Status"].setText(status)
        set_role(r["Status"], "error" if "STALL" in status else ("ok" if status.startswith("lifted") else ""))

    # ===================================================================== drawing
    def _draw(self) -> None:
        self.canvas.draw_with(lambda fig, th: self._paint(fig, th))
        self._update_cursor()

    def _paint(self, fig, th: Theme) -> None:
        res = self.result
        d = res.data
        fig.clear()
        fig.set_facecolor(th.bg)
        u = self.speed_unit()
        uF = units.display_unit(self.system, "force")
        Fc = lambda v: units.from_si("force", v, uF)  # noqa: E731
        has_t = res.has_time
        x = d["t_s"] if has_t else np.array([self._sp(v) for v in d["V_mps"]])
        xlabel = "time [s]" if has_t else f"airspeed [{u}]"
        axes = fig.subplots(3, 1, sharex=True)
        for a in axes:
            style_axes(a, th)
        a0, a1, a2 = axes
        a0.plot(x, Fc(d["L_N"]), color=th.series[0], lw=1.6, label="lift")
        a0.plot(x, Fc(d["W_N"]), "--", color=th.bad, lw=1.2, label="weight")
        a0.plot(x, Fc(d["D_N"]), color=th.series[3], lw=1.3, label="drag")
        if has_t:
            a0.plot(x, Fc(d["T_N"]), color=th.series[2], lw=1.2, label="thrust")
        a0.set_ylabel(f"force [{uF}]", fontsize=8)
        a0.set_title("Forces", fontsize=9, loc="right")
        # the legend sits above the panel, so the event labels inside it never cover it
        leg = a0.legend(fontsize=7, frameon=False, ncol=4, loc="lower left", bbox_to_anchor=(0.0, 1.0),
                        borderaxespad=0.1, handlelength=1.6, columnspacing=1.0)
        for t in leg.get_texts():
            t.set_color(th.muted)
        a1.plot(x, d["alpha_deg"], color=th.series[1], lw=1.5)
        st = d["stalled"] > 0.5
        if st.any():
            a1.plot(x[st], d["alpha_deg"][st], "o", ms=3, color=th.bad, label="section stalled")
        a1.set_ylabel("alpha [deg]", fontsize=8)
        a1.set_title("Angle of attack (fuselage reference)", fontsize=9)
        if has_t:
            a2.plot(x, [self._sp(v) for v in d["V_mps"]], color=th.series[0], lw=1.5)
            a2.set_ylabel(f"airspeed [{u}]", fontsize=8)
            a2b = a2.twinx()
            uL = units.display_unit(self.system, "length")
            a2b.plot(x, units.from_si("length", d["x_m"], uL), color=th.series[3], lw=1.2)
            a2b.set_ylabel(f"distance [{uL}]", fontsize=8, color=th.series[3])
            a2b.tick_params(colors=th.muted, labelsize=7)
            a2.set_title("Airspeed and distance", fontsize=9)
        else:
            a2.plot(x, d["L_over_W"], color=th.series[0], lw=1.5)
            a2.axhline(1.0, ls="--", color=th.bad, lw=1.0)
            a2.set_ylabel("lift / weight", fontsize=8)
            a2.set_title("Lift / weight (1 = lift-off)", fontsize=9)
        a2.set_xlabel(xlabel, fontsize=8)
        for e in res.events:
            if e.name in ("rotation", "lift-off", "stall", "thrust limit") and math.isfinite(e.t if has_t else e.V):
                xe = e.t if has_t else self._sp(e.V)
                for a in axes:
                    a.axvline(xe, ls=":", lw=0.9, color=th.muted)
                a0.annotate(e.name, (xe, 1.0), xycoords=("data", "axes fraction"), fontsize=7, color=th.muted,
                            rotation=90, va="top", ha="right")
        # the playback cursor, drawn where the playback stands (a redraw after a resize keeps the position)
        c = self.current()
        xc = self._pos if has_t else self._sp(self._pos)
        cursors = [a.axvline(xc, color=th.accent, lw=1.2) for a in axes]
        yconv = {"lift": lambda c: Fc(c["L_N"]), "alpha": lambda c: c["alpha_deg"]}
        axes_of = {"lift": a0, "alpha": a1}
        if has_t:
            yconv["speed"], axes_of["speed"] = (lambda c: self._sp(c["V_mps"])), a2
        else:
            yconv["ratio"], axes_of["ratio"] = (lambda c: c["L_over_W"]), a2
        colors = {"lift": th.series[0], "alpha": th.series[1], "speed": th.series[0], "ratio": th.series[0]}
        markers = {k: axes_of[k].plot([xc], [yconv[k](c)], "o", color=colors[k], ms=6)[0] for k in yconv}
        self._artists = {"cursors": cursors, "markers": markers, "yconv": yconv}

    # ===================================================================== export
    def export(self, path: str | None = None, kind: str = "csv") -> bool:
        if self.result is None:
            return False
        if not path:
            filt = "CSV files (*.csv)" if kind == "csv" else "Text files (*.txt)"
            path, _ = QFileDialog.getSaveFileName(self, "Export the scenario run", f"scenario.{kind}", filt)
            if not path:
                return False
            if not path.lower().endswith(f".{kind}"):
                path += f".{kind}"
        text = to_csv(self.result, self.speed_unit()) if kind == "csv" else to_text(self.result, self.speed_unit())
        try:
            Path(path).write_text(text, encoding="utf-8")
        except OSError as exc:
            self.summary.setText(f"Could not write {path}: {exc}")
            return False
        self.summary.setText(f"Exported {Path(path).name}")
        return True
