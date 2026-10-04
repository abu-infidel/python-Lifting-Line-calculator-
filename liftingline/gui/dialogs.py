"""Dialogs that run a small computation in the background: trim (find alpha or speed) and ideal twist."""

from __future__ import annotations

from typing import Callable

import numpy as np
from PySide6.QtCore import QThreadPool, Signal
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..core import units
from ..core.analysis import AnalysisRequest
from ..core.optimal import OptimalTwist, optimal_twist
from ..core.trim import TrimError, TrimResult, trim_alpha, trim_speed
from .plots import make_canvas, style_axes
from .widgets import GuardedCombo, GuardedSpinBox, UnitSpin, muted, set_role
from .workers import FuncTask

MODE_ALPHA_FOR_CL = "alpha_cl"
MODE_ALPHA_FOR_LIFT = "alpha_lift"
MODE_SPEED_FOR_LIFT = "speed_lift"


class _BackgroundDialog(QDialog):
    """A dialog with a Solve button whose work runs in the global thread pool."""

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(460)
        self._task: FuncTask | None = None

    def _launch(self, fn: Callable[[], object], done: Callable[[object], None], failed: Callable[[str], None]) -> None:
        task = FuncTask(lambda _progress: fn())
        task.signals.finished.connect(done)
        task.signals.failed.connect(failed)
        self._task = task  # keep the signal object alive
        QThreadPool.globalInstance().start(task)


class TrimDialog(_BackgroundDialog):
    """Find the angle of attack for a target lift coefficient or lift, or the airspeed for a given weight."""

    apply_requested = Signal(str, float)  # "alpha_deg" (degrees) or "speed" (m/s)

    def __init__(
        self,
        build_request: Callable[[], AnalysisRequest],
        system: str,
        current_cl: float,
        current_lift: float,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__("Trim", parent)
        self._build_request = build_request
        self._result: TrimResult | None = None
        lay = QVBoxLayout(self)
        form = QFormLayout()
        self.mode = GuardedCombo()
        self.mode.addItem("Angle of attack for a lift coefficient", MODE_ALPHA_FOR_CL)
        self.mode.addItem("Angle of attack for a given lift (weight, level flight)", MODE_ALPHA_FOR_LIFT)
        self.mode.addItem("Airspeed for a given lift (weight) at the current angle of attack", MODE_SPEED_FOR_LIFT)
        self.mode.currentIndexChanged.connect(self._mode_changed)
        form.addRow("Find", self.mode)
        self.cl = UnitSpin(None, unit="", decimals=4, minimum=-3.0, maximum=4.0, step=0.05, tooltip="Target wing lift coefficient")
        self.cl.set_si(current_cl if np.isfinite(current_cl) and current_cl > 0 else 0.5)
        self.lift = UnitSpin("force", system, decimals=2, minimum=0.001, maximum=1e9, step=100.0, tooltip="Lift to be carried (the weight in level flight)")
        self.lift.set_si(current_lift if current_lift > 0 else 10000.0)
        form.addRow("Target CL", self.cl)
        form.addRow("Lift / weight", self.lift)
        self.form = form
        lay.addLayout(form)
        self.solve_btn = QPushButton("Solve")
        self.solve_btn.clicked.connect(self._solve)
        lay.addWidget(self.solve_btn)
        self.bar = QProgressBar()
        self.bar.setRange(0, 0)
        self.bar.hide()
        lay.addWidget(self.bar)
        self.result = QLabel()
        self.result.setWordWrap(True)
        lay.addWidget(self.result)
        self.note = muted("Every evaluation is a full analysis with the current inputs, including the section data.")
        lay.addWidget(self.note)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        self.apply_btn = buttons.addButton("Apply to the inputs", QDialogButtonBox.AcceptRole)
        self.apply_btn.setEnabled(False)
        buttons.accepted.connect(self._apply)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)
        self._mode_changed()

    def _mode_changed(self, *_a) -> None:
        m = self.mode.currentData()
        self.form.setRowVisible(self.cl, m == MODE_ALPHA_FOR_CL)
        self.form.setRowVisible(self.lift, m != MODE_ALPHA_FOR_CL)
        self.apply_btn.setEnabled(False)
        self._result = None

    def _solve(self) -> None:
        mode, cl, lift = self.mode.currentData(), self.cl.si(), self.lift.si()
        self.solve_btn.setEnabled(False)
        self.apply_btn.setEnabled(False)
        self.bar.show()
        self.result.setText("")

        def work():
            req = self._build_request()
            if mode == MODE_ALPHA_FOR_CL:
                return trim_alpha(req, CL=cl)
            if mode == MODE_ALPHA_FOR_LIFT:
                return trim_alpha(req, lift=lift)
            return trim_speed(req, lift)

        self._launch(work, self._done, self._failed)

    def _done(self, res: TrimResult) -> None:
        self._result = res
        self.solve_btn.setEnabled(True)
        self.bar.hide()
        t = res.totals
        what = f"angle of attack {res.value:.3f} deg" if res.variable == "alpha_deg" else f"airspeed {res.value:.2f} m/s ({units.from_si('speed', res.value, 'kt'):.1f} kt)"
        self.result.setText(
            f"<b>{what}</b> for {res.target}<br>CL = {t.CL:.4f}, CD = {t.CD:.5f}, L/D = {t.L_over_D:.2f}, "
            f"L = {t.L:.5g} N, Cm = {t.Cm:.4f} ({res.evaluations} analyses)"
            + "".join(f"<br><i>note: {w}</i>" for w in res.warnings[:3])
        )
        set_role(self.result, "")
        self.apply_btn.setEnabled(True)

    def _failed(self, message: str) -> None:
        self.solve_btn.setEnabled(True)
        self.bar.hide()
        self.result.setText(message.split(": ", 1)[-1] if message.startswith(TrimError.__name__) else message)
        set_role(self.result, "error")

    def _apply(self) -> None:
        if self._result is not None:
            self.apply_requested.emit(self._result.variable, self._result.value)
            self.accept()


class OptimalTwistDialog(_BackgroundDialog):
    """Compute the twist that gives an elliptic load at a target CL, preview it and apply it as a section table."""

    apply_requested = Signal(object)  # OptimalTwist

    def __init__(self, build_request: Callable[[], AnalysisRequest], current_cl: float, system: str, parent: QWidget | None = None) -> None:
        super().__init__("Ideal twist for an elliptic load", parent)
        self._build_request = build_request
        self._system = system
        self._result: OptimalTwist | None = None
        lay = QVBoxLayout(self)
        lay.addWidget(muted(
            "Finds the twist distribution that makes this planform and these airfoils carry an elliptic lift distribution "
            "(zero induced-drag penalty, span efficiency 1) at the chosen lift coefficient. The wing becomes a table of "
            "sections; the angle of attack is set for the target lift."
        ))
        form = QFormLayout()
        self.cl = UnitSpin(None, unit="", decimals=3, minimum=-1.0, maximum=3.0, step=0.05, tooltip="Design lift coefficient")
        self.cl.set_si(current_cl if np.isfinite(current_cl) and current_cl > 0.05 else 0.5)
        self.stations = GuardedSpinBox()
        self.stations.setRange(5, 41)
        self.stations.setValue(15)
        self.stations.setToolTip("Sections of the resulting table, dense towards the tip; more give a closer fit")
        form.addRow("Design CL", self.cl)
        form.addRow("Stations", self.stations)
        lay.addLayout(form)
        self.compute_btn = QPushButton("Compute")
        self.compute_btn.clicked.connect(self._compute)
        lay.addWidget(self.compute_btn)
        self.bar = QProgressBar()
        self.bar.setRange(0, 0)
        self.bar.hide()
        lay.addWidget(self.bar)
        self.canvas = make_canvas(toolbar=False)
        self.canvas.setMinimumHeight(190)
        self.canvas.draw_with(None, "Press Compute.")
        lay.addWidget(self.canvas)
        self.result = QLabel()
        self.result.setWordWrap(True)
        lay.addWidget(self.result)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        self.apply_btn = buttons.addButton("Apply to the wing", QDialogButtonBox.AcceptRole)
        self.apply_btn.setEnabled(False)
        buttons.accepted.connect(self._apply)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)

    def _compute(self) -> None:
        cl, n = self.cl.si(), self.stations.value()
        self.compute_btn.setEnabled(False)
        self.apply_btn.setEnabled(False)
        self.bar.show()
        self._launch(lambda: optimal_twist(self._build_request(), cl, n), self._done, self._failed)

    def _done(self, ot: OptimalTwist) -> None:
        self._result = ot
        self.compute_btn.setEnabled(True)
        self.bar.hide()
        unit = units.display_unit(self._system, "length")
        y = [units.from_si("length", r["y"], unit) for r in ot.rows]
        tw = [r["twist_deg"] for r in ot.rows]

        def draw(fig, theme):
            fig.clear()
            fig.set_facecolor(theme.bg)
            ax = fig.add_subplot(111)
            style_axes(ax, theme)
            ax.plot(y, tw, "o-", color=theme.series[0], lw=1.6, markersize=3.5)
            ax.set_xlabel(f"y [{unit}]", fontsize=8)
            ax.set_ylabel("twist [deg]", fontsize=8)
            ax.set_title("Twist relative to the root (nose-up positive)", fontsize=9)

        self.canvas.draw_with(draw)
        p = ot.predicted
        text = (
            f"Root angle of attack <b>{ot.alpha_root_deg:.3f} deg</b>, twist from 0 at the root to {tw[-1]:.2f} deg at the tip "
            f"(largest {ot.max_abs_twist_deg:.2f} deg), {len(ot.rows)} sections."
        )
        if p:
            text += f"<br>Predicted: CL = {p['CL']:.4f}, span efficiency e = {p['e']:.4f}."
        text += "".join(f"<br><i>{n}</i>" for n in ot.notes)
        self.result.setText(text)
        set_role(self.result, "")
        self.apply_btn.setEnabled(True)

    def _failed(self, message: str) -> None:
        self.compute_btn.setEnabled(True)
        self.bar.hide()
        self.result.setText(message)
        set_role(self.result, "error")

    def _apply(self) -> None:
        if self._result is not None:
            self.apply_requested.emit(self._result)
            self.accept()
