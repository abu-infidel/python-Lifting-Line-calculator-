"""The High lift tab: deployable trailing-edge flaps and leading-edge slats for takeoff and landing.

Both devices are off by default; with both off the analysis is exactly the clean wing. The published section
increments the program uses are shown next to the inputs (see :mod:`liftingline.core.highlift`).
"""

from __future__ import annotations

from PySide6.QtCore import QSignalBlocker, Qt, Signal
from PySide6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..core.highlift import (
    DATA_SOURCE,
    FLAP_TYPES,
    LANDING,
    PHASE_LABELS,
    PHASES,
    SLAT_TYPES,
    HighLiftError,
    HighLiftSystem,
)
from ..io.project import HighLiftSpec, ProjectState
from .widgets import GuardedCombo, UnitSpin, muted, set_role


def _percent(tip: str) -> UnitSpin:
    return UnitSpin(None, unit="%", decimals=1, minimum=0.0, maximum=100.0, step=5.0, tooltip=tip)


class HighLiftPanel(QWidget):
    changed = Signal()
    report_requested = Signal()

    def __init__(self, system: str = "SI", parent=None) -> None:
        super().__init__(parent)
        self._busy = False
        self._build(system)
        self.load_spec(HighLiftSpec())

    # ===================================================================== layout
    def _build(self, system: str) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)

        phase_box = QGroupBox("Flight phase")
        f = QFormLayout(phase_box)
        self.phase = GuardedCombo()
        for p in PHASES:
            self.phase.addItem(f"{PHASE_LABELS[p]} setting", p)
        self.phase.setToolTip(
            "Takeoff: flaps at a moderate deflection (about 70 % of the landing gain in maximum lift). "
            "Landing: flaps fully down."
        )
        f.addRow("Devices set for", self.phase)
        root.addWidget(phase_box)

        # ------------------------------------------------------------------ flaps
        self.flap_box = QGroupBox("Deploy trailing-edge flaps")
        self.flap_box.setCheckable(True)
        self.flap_box.setToolTip("Toggle the flaps: off means retracted (the clean wing)")
        f = QFormLayout(self.flap_box)
        self.flap_type = GuardedCombo()
        for t in FLAP_TYPES.values():
            self.flap_type.addItem(t.label, t.key)
            self.flap_type.setItemData(self.flap_type.count() - 1, t.description, Qt.ToolTipRole)
        self.flap_in = _percent("Inboard end of the flaps, per cent of the semi-span")
        self.flap_out = _percent("Outboard end of the flaps, per cent of the semi-span")
        self.flap_chord = UnitSpin(None, unit="%", decimals=1, minimum=10.0, maximum=45.0, step=1.0,
                                   tooltip="Flap chord, per cent of the local wing chord (flap drag and pitching moment)")
        self.flap_ext = UnitSpin(None, decimals=2, minimum=1.0, maximum=1.5, step=0.05,
                                 tooltip="Chord with the flap extended / clean chord (Fowler-type flaps move aft)")
        f.addRow("Type", self.flap_type)
        f.addRow("From", self.flap_in)
        f.addRow("To", self.flap_out)
        f.addRow("Flap chord", self.flap_chord)
        f.addRow("Chord extension c'/c", self.flap_ext)
        self.flap_info = muted()
        self.flap_info.setTextFormat(Qt.RichText)
        f.addRow(self.flap_info)
        root.addWidget(self.flap_box)

        # ------------------------------------------------------------------ slats
        self.slat_box = QGroupBox("Deploy leading-edge slats")
        self.slat_box.setCheckable(True)
        self.slat_box.setToolTip("Toggle the leading-edge devices: off means retracted (the clean wing)")
        f = QFormLayout(self.slat_box)
        self.slat_type = GuardedCombo()
        for t in SLAT_TYPES.values():
            self.slat_type.addItem(t.label, t.key)
            self.slat_type.setItemData(self.slat_type.count() - 1, t.description, Qt.ToolTipRole)
        self.slat_in = _percent("Inboard end of the slats, per cent of the semi-span")
        self.slat_out = _percent("Outboard end of the slats, per cent of the semi-span")
        self.slat_ext = UnitSpin(None, decimals=2, minimum=1.0, maximum=1.5, step=0.05,
                                 tooltip="Chord with the slat extended / clean chord")
        f.addRow("Type", self.slat_type)
        f.addRow("From", self.slat_in)
        f.addRow("To", self.slat_out)
        f.addRow("Chord extension c'/c", self.slat_ext)
        self.slat_info = muted()
        self.slat_info.setTextFormat(Qt.RichText)
        f.addRow(self.slat_info)
        root.addWidget(self.slat_box)

        # ------------------------------------------------------------------ stall analysis
        box = QGroupBox("Takeoff and landing stall")
        v = QVBoxLayout(box)
        form = QFormLayout()
        self.weight = UnitSpin("force", system, decimals=1, minimum=0.0, maximum=1e8, step=100.0,
                               tooltip="Aircraft weight for the stall and reference speeds (0: speeds not computed)")
        form.addRow("Aircraft weight", self.weight)
        v.addLayout(form)
        row = QHBoxLayout()
        self.report_button = QPushButton("Analyse stall: clean vs devices out")
        self.report_button.setToolTip(
            "Find the maximum lift of the clean wing and of the wing with the devices deployed (angle-of-attack sweeps "
            "at the Flow tab's speed and altitude), the stall speeds and the handbook comparison"
        )
        self.report_button.clicked.connect(self.report_requested.emit)
        row.addWidget(self.report_button)
        row.addStretch(1)
        v.addLayout(row)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        self.summary.setTextInteractionFlags(Qt.TextSelectableByMouse)
        v.addWidget(self.summary)
        root.addWidget(box)

        root.addWidget(muted(
            "The devices change only the sections they cover, using published handbook increments: maximum lift, "
            "zero-lift angle and flap drag (" + DATA_SOURCE + "). With both boxes unticked the wing is analysed exactly "
            "as without this tab. High-lift devices belong to low-speed takeoff and landing; a warning appears above "
            "Mach 0.35."
        ))
        self.problem = QLabel()
        self.problem.setWordWrap(True)
        set_role(self.problem, "error")
        root.addWidget(self.problem)
        root.addStretch(1)

        for w in (self.flap_in, self.flap_out, self.flap_chord, self.flap_ext, self.slat_in, self.slat_out,
                  self.slat_ext, self.weight):
            w.valueChanged.connect(self._edited)
        self.phase.currentIndexChanged.connect(self._edited)
        self.flap_box.toggled.connect(self._edited)
        self.slat_box.toggled.connect(self._edited)
        self.flap_type.currentIndexChanged.connect(lambda _i: self._type_changed(self.flap_type, self.flap_ext, FLAP_TYPES))
        self.slat_type.currentIndexChanged.connect(lambda _i: self._type_changed(self.slat_type, self.slat_ext, SLAT_TYPES))

    # ======================================================================= state
    def spec(self) -> HighLiftSpec:
        return HighLiftSpec(
            phase=self.phase.currentData() or LANDING,
            flaps_on=self.flap_box.isChecked(),
            flap_type=self.flap_type.currentData(),
            flap_y_in=self.flap_in.si() / 100.0,
            flap_y_out=self.flap_out.si() / 100.0,
            flap_chord_fraction=self.flap_chord.si() / 100.0,
            flap_chord_extension=self.flap_ext.si(),
            slats_on=self.slat_box.isChecked(),
            slat_type=self.slat_type.currentData(),
            slat_y_in=self.slat_in.si() / 100.0,
            slat_y_out=self.slat_out.si() / 100.0,
            slat_chord_extension=self.slat_ext.si(),
            weight_N=self.weight.si(),
        )

    def load_spec(self, h: HighLiftSpec) -> None:
        self._busy = True
        try:
            for w in (self.phase, self.flap_type, self.slat_type):
                w.blockSignals(True)
            self.phase.setCurrentIndex(max(self.phase.findData(h.phase), 0))
            self.flap_type.setCurrentIndex(max(self.flap_type.findData(h.flap_type), 0))
            self.slat_type.setCurrentIndex(max(self.slat_type.findData(h.slat_type), 0))
            for w in (self.phase, self.flap_type, self.slat_type):
                w.blockSignals(False)
            with QSignalBlocker(self.flap_box), QSignalBlocker(self.slat_box):
                self.flap_box.setChecked(bool(h.flaps_on))
                self.slat_box.setChecked(bool(h.slats_on))
            self.flap_in.set_si(100.0 * h.flap_y_in)
            self.flap_out.set_si(100.0 * h.flap_y_out)
            self.flap_chord.set_si(100.0 * h.flap_chord_fraction)
            self.flap_ext.set_si(h.flap_chord_extension)
            self.slat_in.set_si(100.0 * h.slat_y_in)
            self.slat_out.set_si(100.0 * h.slat_y_out)
            self.slat_ext.set_si(h.slat_chord_extension)
            self.weight.set_si(h.weight_N)
        finally:
            self._busy = False
        self._refresh()

    def load(self, state: ProjectState) -> None:
        self.load_spec(state.high_lift)
        self.summary.setText("")

    def store(self, state: ProjectState) -> None:
        state.high_lift = self.spec()

    def problems(self) -> list[str]:
        return [f"High lift: {m}" for m in self.spec().problems()]

    def set_system(self, system: str) -> None:
        self.weight.set_system(system)

    def show_summary(self, text: str) -> None:
        self.summary.setText(text)

    # ======================================================================= slots
    def _type_changed(self, combo: GuardedCombo, ext: UnitSpin, table: dict) -> None:
        if self._busy:
            return
        t = table[combo.currentData()]
        ext.set_si(t.default_extension)  # a typical chord extension for the new type (1 when it does not extend)
        self._edited()

    def _edited(self, *_args) -> None:
        if self._busy:
            return
        self._refresh()
        self.changed.emit()

    def _refresh(self) -> None:
        h = self.spec()
        ft, st = FLAP_TYPES[h.flap_type], SLAT_TYPES[h.slat_type]
        self.flap_ext.setEnabled(ft.times_extension)
        self.slat_ext.setEnabled(st.times_extension)
        msgs = h.problems() if h.active else []
        self.problem.setText("Invalid high-lift devices: " + "; ".join(msgs) if msgs else "")
        try:  # describe what each device does, even while it is retracted
            flaps = HighLiftSystem.from_spec(h.phase, dict(type=h.flap_type, y_in=0.0, y_out=1.0,
                                                           chord_fraction=min(max(h.flap_chord_fraction, 0.1), 0.45),
                                                           chord_extension=min(max(h.flap_chord_extension, 1.0), 1.5)), None)
            slats = HighLiftSystem.from_spec(h.phase, None, dict(type=h.slat_type, y_in=0.0, y_out=1.0,
                                                                 chord_extension=min(max(h.slat_chord_extension, 1.0), 1.5)))
        except HighLiftError:
            self.flap_info.setText("")
            self.slat_info.setText("")
            return
        phase = PHASE_LABELS[h.phase].lower()
        self.flap_info.setText(
            f"Published section increments ({phase}): maximum lift <b>{flaps.flap_dcl_max():+.2f}</b> "
            f"(table value {ft.table_text()}{', x 0.7 for takeoff' if h.phase == 'takeoff' else ''}), zero-lift angle "
            f"<b>{flaps.flap_dalpha_L0_deg():+.0f}&deg;</b>, typical deflection {flaps.flap_deflection_deg():.0f}&deg;, "
            f"profile drag {flaps.flap_dcd():+.4f}, moment {flaps.flap_dcm():+.3f}."
        )
        self.slat_info.setText(
            f"Published section increment: maximum lift <b>{slats.slat_dcl_max():+.2f}</b> (table value "
            f"{st.table_text()}); the lift curve is extended to a higher stall angle, the lift below the stall is unchanged."
        )
