"""The Polars tab: parameter sweeps (angle of attack, speed, altitude, sideslip, control deflection) and stall."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..core.sweeps import ALPHA, ALTITUDE, BETA, DEFLECTION, SPEED, StallInfo, SweepResult
from .plots import make_canvas, plot_sweep
from .widgets import GuardedCombo, GuardedSpinBox, UnitSpin, muted, set_role

#: variable -> (label, default start, default stop) in native units (deg, m/s, m)
_VARIABLES = (
    (ALPHA, "Angle of attack", -4.0, 20.0),
    (SPEED, "Airspeed", 20.0, 100.0),
    (ALTITUDE, "Altitude", 0.0, 6000.0),
    (BETA, "Sideslip angle", -10.0, 10.0),
    (DEFLECTION, "Control deflection", -15.0, 15.0),
)


class PolarsView(QWidget):
    run_requested = Signal(str, float, float, int, str, str)  # variable, start, stop, points, hold, control
    cancel_requested = Signal()

    def __init__(self, system: str = "SI", parent=None) -> None:
        super().__init__(parent)
        self._system = system
        self.sweep: SweepResult | None = None
        self.stall: StallInfo | None = None
        self._current: dict[str, float] = {}
        self._build()
        self._variable_changed()
        self.clear()

    # ===================================================================== layout
    def _build(self) -> None:
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)

        box = QGroupBox("Sweep")
        v = QVBoxLayout(box)
        row = QHBoxLayout()
        row.addWidget(QLabel("Vary"))
        self.variable = GuardedCombo()
        for key, label, _a, _b in _VARIABLES:
            self.variable.addItem(label, key)
        self.variable.currentIndexChanged.connect(self._variable_changed)
        row.addWidget(self.variable)
        row.addStretch(1)
        v.addLayout(row)
        row = QHBoxLayout()  # the range on its own line keeps the tab narrow enough for small screens
        self.from_label, self.to_label = QLabel("from"), QLabel("to")
        row.addWidget(self.from_label)
        # one pair of range fields per kind of variable; only the matching pair is shown
        self.rng = {
            "angle": (UnitSpin(None, unit="deg", decimals=2, minimum=-90.0, maximum=90.0, step=1.0), UnitSpin(None, unit="deg", decimals=2, minimum=-90.0, maximum=90.0, step=1.0)),
            "speed": (UnitSpin("speed", self._system, decimals=1, minimum=1.0, maximum=1000.0, step=5.0), UnitSpin("speed", self._system, decimals=1, minimum=1.0, maximum=1000.0, step=5.0)),
            "altitude": (UnitSpin("length", self._system, decimals=0, minimum=-2000.0, maximum=47000.0, step=500.0), UnitSpin("length", self._system, decimals=0, minimum=-2000.0, maximum=47000.0, step=500.0)),
        }
        for lo, hi in self.rng.values():
            row.addWidget(lo)
        row.addWidget(self.to_label)
        for lo, hi in self.rng.values():
            row.addWidget(hi)
        row.addWidget(QLabel("points"))
        self.points = GuardedSpinBox()
        self.points.setRange(3, 200)
        self.points.setValue(25)
        row.addWidget(self.points)
        row.addStretch(1)
        v.addLayout(row)

        row = QHBoxLayout()
        self.hold_label = QLabel("keep constant")
        self.hold = GuardedCombo()
        self.hold.addItem("airspeed", "speed")
        self.hold.addItem("Mach number", "mach")
        self.control_label = QLabel("surface")
        self.control = GuardedCombo()
        for w in (self.hold_label, self.hold, self.control_label, self.control):
            row.addWidget(w)
        self.run_btn = QPushButton("Run sweep")
        self.run_btn.setToolTip("Run the analysis at every point of the sweep (uses the current inputs)")
        self.run_btn.clicked.connect(self._run)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel_requested)
        self.export_btn = QPushButton("Export CSV...")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self._export)
        self.bar = QProgressBar()
        self.bar.setMaximumWidth(160)
        self.bar.setTextVisible(False)
        self.bar.hide()
        for w in (self.run_btn, self.cancel_btn, self.bar, self.export_btn):
            row.addWidget(w)
        row.addStretch(1)
        v.addLayout(row)
        lay.addWidget(box)

        self.canvas = make_canvas()
        lay.addWidget(self.canvas, 1)
        self.summary = muted("")
        lay.addWidget(self.summary)

    # ======================================================================= state
    def set_system(self, system: str) -> None:
        self._system = system
        for lo, hi in self.rng.values():
            lo.set_system(system)
            hi.set_system(system)
        self.redraw()

    def set_controls(self, names: list[str]) -> None:
        keep = self.control.currentData()
        self.control.clear()
        for n in names:
            self.control.addItem(n, n)
        i = self.control.findData(keep)
        if i >= 0:
            self.control.setCurrentIndex(i)
        self._variable_changed()

    def set_current(self, current: dict[str, float]) -> None:
        """Operating point of the inputs (shown as a dotted line in the plots)."""
        self._current = dict(current)

    def _kind(self) -> str:
        v = self.variable.currentData()
        return {SPEED: "speed", ALTITUDE: "altitude"}.get(v, "angle")

    def _variable_changed(self, *_a) -> None:
        var = self.variable.currentData()
        kind = self._kind()
        defaults = {k: (a, b) for k, _l, a, b in _VARIABLES}
        for name, (lo, hi) in self.rng.items():
            show = name == kind
            lo.setVisible(show)
            hi.setVisible(show)
            if show and var in defaults:
                a, b = defaults[var]
                cur = self._current.get(var)
                if var == SPEED and cur:
                    a, b = 0.5 * cur, 2.0 * cur
                lo.set_si(a)
                hi.set_si(b)
        self.hold_label.setVisible(var == ALTITUDE)
        self.hold.setVisible(var == ALTITUDE)
        has_ctrl = self.control.count() > 0
        self.control_label.setVisible(var == DEFLECTION)
        self.control.setVisible(var == DEFLECTION)
        self.run_btn.setEnabled(var != DEFLECTION or has_ctrl)
        self.run_btn.setToolTip(
            "Run the analysis at every point of the sweep (uses the current inputs)"
            if self.run_btn.isEnabled()
            else "Define a control surface on the Controls tab first"
        )

    # ===================================================================== actions
    def _run(self) -> None:
        var = self.variable.currentData()
        lo, hi = self.rng[self._kind()]
        control = self.control.currentData() or "" if var == DEFLECTION else ""
        self.run_requested.emit(var, lo.si(), hi.si(), self.points.value(), self.hold.currentData(), control)

    def set_running(self, on: bool) -> None:
        self.run_btn.setEnabled(not on)
        self.cancel_btn.setEnabled(on)
        self.bar.setVisible(on)
        if on:
            self.bar.setRange(0, 0)

    def set_progress(self, done: int, total: int) -> None:
        self.bar.setRange(0, max(total, 1))
        self.bar.setValue(done)

    def _export(self) -> None:
        if self.sweep is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export sweep", "sweep.csv", "CSV files (*.csv)")
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(self.sweep.to_csv())

    # ======================================================================== view
    def clear(self) -> None:
        self.sweep, self.stall = None, None
        self.export_btn.setEnabled(False)
        self.summary.setText("Choose what to vary and press Run sweep. Angle-of-attack sweeps also report the first stall.")
        set_role(self.summary, "muted")
        self.canvas.draw_with(None, "No sweep yet.")

    def show_result(self, sweep: SweepResult, stall: StallInfo | None) -> None:
        self.sweep, self.stall = sweep, stall
        self.export_btn.setEnabled(True)
        self.redraw()
        bad = [p for p in sweep.points if not p.ok]
        lines = []
        if stall is not None:
            lines.append(self._stall_text(stall))
        nonconv = sum(1 for p in sweep.points if p.ok and not p.converged)
        if bad:
            lines.append(f"{len(bad)} point(s) failed: {bad[0].error}")
        if nonconv:
            lines.append(f"{nonconv} point(s) did not fully converge (hollow markers): the wing is beyond stall there.")
        lines += sweep.notes
        self.summary.setText("  |  ".join(lines) if lines else f"{len(sweep.points)} points, solver: {sweep.solver}, section data: {sweep.engine}")
        set_role(self.summary, "warning" if (bad or nonconv) else "")

    @staticmethod
    def _stall_text(s: StallInfo) -> str:
        if not s.available:
            return "Stall: " + s.note
        if not s.reached:
            return "Stall: " + s.note
        peak = f"CL,max = {s.CL_max:.3f} at {s.alpha_CL_max:.1f} deg" + ("" if s.CL_max_is_peak else " (end of the sweep: not a peak)")
        return (
            f"First stall at {s.alpha_first_stall:.1f} deg angle of attack near |y| = {s.y_first_stall:.2f} m (CL = {s.CL_first_stall:.3f}); {peak}"
        )

    def redraw(self) -> None:
        if self.sweep is None:
            return
        sweep, stall, system = self.sweep, self.stall, self._system
        current = self._current.get(sweep.variable)
        self.canvas.draw_with(lambda fig, th: plot_sweep(fig, sweep, th, system, current, stall))

    def mark_stale(self) -> None:
        if self.sweep is not None:
            self.summary.setText(self.summary.text().split("  |  inputs changed")[0] + "  |  inputs changed since this sweep")
            set_role(self.summary, "warning")
