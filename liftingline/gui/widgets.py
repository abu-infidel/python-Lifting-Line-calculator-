"""Small reusable widgets: unit-aware numeric fields, scroll-safe combo boxes and spin boxes."""

from __future__ import annotations

from PySide6.QtCore import QSignalBlocker, Qt, Signal
from PySide6.QtWidgets import QComboBox, QDoubleSpinBox, QHBoxLayout, QLabel, QSpinBox, QWidget

from ..core import units


class GuardedCombo(QComboBox):
    """A combo box that ignores the mouse wheel unless it has keyboard focus (so scrolling a panel is safe)."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFocusPolicy(Qt.StrongFocus)

    def wheelEvent(self, event) -> None:
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()

    def set_index_silently(self, index: int) -> None:
        with QSignalBlocker(self):
            self.setCurrentIndex(index)

    def set_data_silently(self, data) -> None:
        i = self.findData(data)
        if i >= 0:
            self.set_index_silently(i)


class GuardedSpinBox(QSpinBox):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setKeyboardTracking(False)
        self.setAlignment(Qt.AlignRight)

    def wheelEvent(self, event) -> None:
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()

    def set_silently(self, value: int) -> None:
        with QSignalBlocker(self):
            self.setValue(int(value))


class UnitSpin(QDoubleSpinBox):
    """A numeric field for one physical quantity: it shows the value in a display unit, the program reads SI.

    ``kind`` is a quantity kind of :mod:`liftingline.core.units` (``"length"``, ``"speed"`` ...) or ``None`` for
    plain numbers (angles in degrees, coefficients). ``minimum``, ``maximum`` and ``step`` are given in SI. Edits
    are announced with ``valueChanged`` when editing finishes (keyboard tracking is off); ``set_si`` never
    emits.
    """

    def __init__(
        self,
        kind: str | None = None,
        system: str = "SI",
        *,
        unit: str | None = None,
        decimals: int = 4,
        minimum: float = -1e9,
        maximum: float = 1e9,
        step: float = 0.1,
        tooltip: str = "",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.kind = kind
        self._follow_system = unit is None and kind is not None
        self._unit = unit if unit is not None else (units.display_unit(system, kind) if kind else "")
        self._min_si, self._max_si, self._step_si = minimum, maximum, step
        self.setDecimals(decimals)
        self.setKeyboardTracking(False)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setAccelerated(True)
        self.setAlignment(Qt.AlignRight)
        if tooltip:
            self.setToolTip(tooltip)
        self._apply_unit()
        # The exact SI value is kept next to the displayed one, which is rounded to the field's decimals:
        # switching units or re-reading a field must not slowly degrade what the user entered.
        self._si_exact = self._hide(self.value())
        self.valueChanged.connect(self._edited)  # connected first, so it runs before any outside listener

    def _edited(self, _value: float) -> None:
        self._si_exact = self._hide(self.value())

    # ---------------------------------------------------------------------- conversion
    def _show(self, x: float) -> float:
        return float(units.from_si(self.kind, x, self._unit)) if self.kind else float(x)

    def _hide(self, x: float) -> float:
        return float(units.to_si(self.kind, x, self._unit)) if self.kind else float(x)

    def _apply_unit(self) -> None:
        with QSignalBlocker(self):
            self.setRange(self._show(self._min_si), self._show(self._max_si))
            self.setSingleStep(abs(self._show(self._step_si) - self._show(0.0)) or 0.1)
            self.setSuffix(f" {self._unit}" if self._unit else "")

    # ------------------------------------------------------------------------- API
    @property
    def unit(self) -> str:
        return self._unit

    def si(self) -> float:
        return self._si_exact

    def set_si(self, x: float) -> None:
        self._si_exact = float(x)
        with QSignalBlocker(self):
            self.setValue(self._show(x))

    def set_unit(self, unit: str) -> None:
        """Change the display unit, keeping the physical value (exactly)."""
        if unit == self._unit:
            return
        x = self._si_exact
        self._unit = unit
        self._apply_unit()
        self.set_si(x)

    def set_system(self, system: str) -> None:
        if self._follow_system and self.kind:
            self.set_unit(units.display_unit(system, self.kind))

    def wheelEvent(self, event) -> None:
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()


class SpeedField(QWidget):
    """Airspeed with its own unit selector (m/s, km/h, kt, mph, ft/s)."""

    valueChanged = Signal()

    def __init__(self, system: str = "SI", parent=None) -> None:
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.spin = UnitSpin("speed", unit="m/s", decimals=2, minimum=0.01, maximum=1000.0, step=1.0)
        self.combo = GuardedCombo()
        for u in units.units_of("speed"):
            self.combo.addItem(u, u)
        self.set_system(system)
        lay.addWidget(self.spin, 1)
        lay.addWidget(self.combo)
        self.spin.valueChanged.connect(lambda _v: self.valueChanged.emit())
        self.combo.currentIndexChanged.connect(self._unit_changed)

    def _unit_changed(self, _i: int) -> None:
        self.spin.set_unit(self.combo.currentData())
        self.spin.setSuffix("")  # the combo shows the unit
        self.valueChanged.emit()

    def set_system(self, system: str) -> None:
        unit = "m/s" if system == "SI" else "kt"
        x = self.spin.si()
        with QSignalBlocker(self.combo):
            self.combo.setCurrentIndex(self.combo.findData(unit))
        self.spin.set_unit(unit)
        self.spin.setSuffix("")
        self.spin.set_si(x)

    def si(self) -> float:
        return self.spin.si()

    def set_si(self, v: float) -> None:
        self.spin.set_si(v)
        self.spin.setSuffix("")

    def setEnabled(self, on: bool) -> None:  # noqa: N802 - Qt naming
        super().setEnabled(on)


def set_role(label: QLabel, role: str) -> None:
    """Style a label through the theme's ``role`` property ("muted", "error", "ok", "warning" or "")."""
    label.setProperty("role", role)
    label.style().unpolish(label)
    label.style().polish(label)


def muted(text: str = "", wrap: bool = True) -> QLabel:
    lab = QLabel(text)
    lab.setWordWrap(wrap)
    set_role(lab, "muted")
    return lab
