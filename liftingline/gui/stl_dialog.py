"""Options for exporting the wing as a 3-D model (STL)."""

from __future__ import annotations

from PySide6.QtWidgets import QCheckBox, QDialog, QDialogButtonBox, QFormLayout, QLabel, QVBoxLayout

from ..io.stl import DEFAULT_UNIT, UNITS
from .widgets import GuardedCombo, GuardedSpinBox, UnitSpin, muted

_UNIT_LABELS = {"mm": "millimetres (mm)", "cm": "centimetres (cm)", "m": "metres (m)", "in": "inches (in)", "ft": "feet (ft)"}


class StlExportDialog(QDialog):
    """Units, scale and resolution of the STL export."""

    def __init__(self, span_m: float, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export wing as 3-D model (STL)")
        self.span_m = span_m
        lay = QVBoxLayout(self)
        form = QFormLayout()
        self.unit = GuardedCombo()
        for u in UNITS:
            self.unit.addItem(_UNIT_LABELS.get(u, u), u)
        self.unit.setCurrentIndex(self.unit.findData(DEFAULT_UNIT))
        self.unit.setToolTip("STL files carry no units: the coordinates are written in this unit (CAD and slicers expect mm)")
        self.scale = UnitSpin(None, decimals=4, minimum=1e-4, maximum=1e4, step=0.01,
                              tooltip="Model scale: 1 = full size, 0.05 = a 1:20 model")
        self.scale.set_si(1.0)
        self.model = GuardedCombo()
        self.model.addItem("Full wing (both halves)", False)
        self.model.addItem("Right half, closed at the root (half-span model)", True)
        self.n_chord = GuardedSpinBox()
        self.n_chord.setRange(20, 300)
        self.n_chord.setValue(80)
        self.n_chord.setToolTip("Points along each airfoil surface (clustered at the leading and trailing edges)")
        self.n_span = GuardedSpinBox()
        self.n_span.setRange(10, 300)
        self.n_span.setValue(40)
        self.n_span.setToolTip("Spanwise intervals per half-wing (every wing section is always a station)")
        self.incidence = QCheckBox("Include the root incidence (model in body axes)")
        self.incidence.setChecked(True)
        self.ascii = QCheckBox("ASCII instead of binary STL (larger file)")
        form.addRow("Units", self.unit)
        form.addRow("Model scale", self.scale)
        form.addRow("Model", self.model)
        form.addRow("Chordwise points", self.n_chord)
        form.addRow("Spanwise intervals", self.n_span)
        form.addRow(self.incidence)
        form.addRow(self.ascii)
        lay.addLayout(form)
        self.size = QLabel()
        lay.addWidget(self.size)
        lay.addWidget(muted(
            "The surface is built from the real airfoil coordinates with twist, sweep and dihedral, closed at the tips "
            "(and at the root of a half model): a watertight mesh with outward normals. Axes: x aft, y to the right wing, "
            "z up. The trailing edge is closed to a sharp edge."
        ))
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Export...")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)
        self.unit.currentIndexChanged.connect(self._refresh)
        self.scale.valueChanged.connect(self._refresh)
        self.model.currentIndexChanged.connect(self._refresh)
        self._refresh()

    def options(self) -> dict:
        return dict(
            unit=self.unit.currentData(), scale=self.scale.si(), half=bool(self.model.currentData()),
            n_chord=self.n_chord.value(), n_span=self.n_span.value(), include_incidence=self.incidence.isChecked(),
            ascii=self.ascii.isChecked(),
        )

    def _refresh(self, *_a) -> None:
        o = self.options()
        span = self.span_m * (0.5 if o["half"] else 1.0) * o["scale"] / UNITS[o["unit"]]
        self.size.setText(f"Model span in the file: <b>{span:.6g} {o['unit']}</b>")
