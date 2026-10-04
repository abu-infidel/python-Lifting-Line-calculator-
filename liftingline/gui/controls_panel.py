"""The Controls tab: flaps and ailerons as a table (span range, chord fraction, deflection, effectiveness)."""

from __future__ import annotations

import copy
import math

from PySide6.QtCore import QSignalBlocker, Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..core.controls import (
    DEFAULT_EFFECTIVENESS,
    KIND_AILERON,
    KIND_FLAP,
    ControlSurface,
    flap_effectiveness,
    flap_moment_slope,
    validate_rows,
)
from ..io.project import ProjectState
from .widgets import GuardedCombo, muted, set_role

#: key, header, scale between the stored value and the displayed one, tooltip
_COLUMNS = (
    ("name", "Name", None, "Name of the surface (used to pick it in a deflection sweep)"),
    ("kind", "Type", None, "Flap: both wings deflect alike. Aileron: the right and left surfaces move oppositely"),
    ("y_in", "In %", 100.0, "Inboard end, per cent of the semi-span"),
    ("y_out", "Out %", 100.0, "Outboard end, per cent of the semi-span"),
    ("chord_fraction", "Chord %", 100.0, "Chord of the control surface, per cent of the local wing chord"),
    ("deflection_deg", "Defl. deg", 1.0, "Deflection in degrees, trailing edge down positive (right surface for an aileron)"),
    ("effectiveness", "Eff.", 1.0, "Factor on the thin-airfoil effectiveness for viscous effects (0.85 typical)"),
)


class ControlsPanel(QWidget):
    changed = Signal()
    deflections_changed = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._busy = False
        self._build()

    # ===================================================================== layout
    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        box = QGroupBox("Control surfaces")
        v = QVBoxLayout(box)
        self.table = QTableWidget(0, len(_COLUMNS))
        self.table.setHorizontalHeaderLabels([h for _, h, _, _ in _COLUMNS])
        for c, (_k, _h, _s, tip) in enumerate(_COLUMNS):
            self.table.horizontalHeaderItem(c).setToolTip(tip)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setMinimumHeight(150)
        self.table.itemChanged.connect(self._item_changed)
        self.table.currentCellChanged.connect(lambda *_a: self._refresh_info())
        v.addWidget(self.table)
        row = QHBoxLayout()
        for text, tip, slot in (
            ("Add flap", "A plain flap: both wings deflect alike", lambda: self.add_surface(KIND_FLAP)),
            ("Add aileron", "An aileron: the right and left surfaces move oppositely", lambda: self.add_surface(KIND_AILERON)),
            ("Remove", "Remove the selected surface", self.remove_selected),
        ):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            row.addWidget(b)
        row.addStretch(1)
        v.addLayout(row)
        self.info = QLabel()
        self.info.setWordWrap(True)
        self.info.setTextInteractionFlags(Qt.TextSelectableByMouse)
        v.addWidget(self.info)
        v.addWidget(muted(
            "Deflection is positive trailing edge down; for an aileron it is the right-hand surface (the left one moves the "
            "other way). The surface changes the section's lift and moment as thin-airfoil flap theory predicts, times the "
            "effectiveness factor (0.85 allows for boundary-layer effects). Maximum lift and flap drag are not modelled."
        ))
        root.addWidget(box)
        self.problem = QLabel()
        self.problem.setWordWrap(True)
        set_role(self.problem, "error")
        root.addWidget(self.problem)
        root.addStretch(1)

    # ======================================================================= state
    def load(self, state: ProjectState) -> None:
        self._set_rows(copy.deepcopy(state.controls))

    def store(self, state: ProjectState) -> None:
        state.controls = self.rows()

    def _set_rows(self, rows: list[dict]) -> None:
        self._busy = True
        try:
            self.table.setRowCount(len(rows))
            for r, d in enumerate(rows):
                for c, (key, _head, scale, _tip) in enumerate(_COLUMNS):
                    if key == "kind":
                        combo = GuardedCombo()
                        combo.addItem("Flap", KIND_FLAP)
                        combo.addItem("Aileron", KIND_AILERON)
                        combo.set_data_silently(d.get("kind", KIND_AILERON))
                        combo.currentIndexChanged.connect(lambda _i: self._edited())
                        self.table.setCellWidget(r, c, combo)
                        continue
                    value = d.get(key, "" if key == "name" else 0.0)
                    text = str(value) if key == "name" else f"{float(value) * scale:.5g}"
                    item = QTableWidgetItem(text)
                    if key != "name":
                        item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    self.table.setItem(r, c, item)
        finally:
            self._busy = False
        self._refresh_info()

    def rows(self) -> list[dict]:
        out = []
        for r in range(self.table.rowCount()):
            d: dict = {}
            for c, (key, _head, scale, _tip) in enumerate(_COLUMNS):
                if key == "kind":
                    combo = self.table.cellWidget(r, c)
                    d[key] = combo.currentData() if combo is not None else KIND_AILERON
                    continue
                text = self.table.item(r, c).text().strip() if self.table.item(r, c) else ""
                if key == "name":
                    d[key] = text or f"Surface {r + 1}"
                    continue
                try:
                    d[key] = float(text.replace(",", ".")) / scale
                except ValueError:
                    d[key] = math.nan
            out.append(d)
        return out

    def problems(self) -> list[str]:
        out = []
        for i, d in enumerate(self.rows()):
            for key, head, _, _tip in _COLUMNS:
                if key not in ("name", "kind") and not math.isfinite(d.get(key, math.nan)):
                    out.append(f"control surface {i + 1}: '{head}' is not a number")
        if out:
            return out
        return validate_rows(self.rows())

    def has_deflection(self) -> bool:
        return any(abs(d.get("deflection_deg", 0.0)) > 0.0 for d in self.rows())

    # ======================================================================= rows
    def add_surface(self, kind: str) -> None:
        rows = self.rows()
        base = "Flap" if kind == KIND_FLAP else "Aileron"
        names = {d["name"] for d in rows}
        name, k = base, 2
        while name in names:
            name, k = f"{base} {k}", k + 1
        proto = (
            ControlSurface(name, KIND_FLAP, 0.1, 0.6, 0.3, 0.0, DEFAULT_EFFECTIVENESS)
            if kind == KIND_FLAP
            else ControlSurface(name, KIND_AILERON, 0.6, 0.95, 0.25, 0.0, DEFAULT_EFFECTIVENESS)
        )
        rows.append(proto.to_dict())
        self._set_rows(rows)
        self.table.selectRow(len(rows) - 1)
        self.changed.emit()

    def remove_selected(self) -> None:
        r = self.table.currentRow()
        rows = self.rows()
        if 0 <= r < len(rows):
            del rows[r]
            self._set_rows(rows)
            self.changed.emit()

    # ===================================================================== slots
    def _item_changed(self, item: QTableWidgetItem) -> None:
        if self._busy:
            return
        ok = True
        if item.column() > 1:
            try:
                float(item.text().replace(",", "."))
            except ValueError:
                ok = False
        with QSignalBlocker(self.table):
            item.setBackground(QColor(0, 0, 0, 0) if ok else QColor("#fca5a5"))
        self._edited()

    def _edited(self) -> None:
        if self._busy:
            return
        self._refresh_info()
        self.changed.emit()

    def _refresh_info(self) -> None:
        if self._busy:
            return
        msgs = self.problems()
        self.problem.setText("Invalid control surfaces: " + "; ".join(msgs) if msgs else "")
        r = self.table.currentRow()
        rows = self.rows()
        if not 0 <= r < len(rows) or msgs:
            self.info.setText("")
            return
        d = rows[r]
        cf, eff = d["chord_fraction"], d["effectiveness"]
        tau = flap_effectiveness(cf)
        self.info.setText(
            f"<b>{d['name']}</b>: thin-airfoil effectiveness &tau; = {tau:.3f}, with the factor {eff:.2f}: "
            f"{eff * tau:.3f} (1&deg; of deflection acts like {eff * tau:.2f}&deg; of angle of attack); "
            f"moment change {eff * flap_moment_slope(cf):.3f} per radian."
        )
