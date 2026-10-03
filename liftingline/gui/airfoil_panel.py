"""The Airfoils tab: one airfoil per wing section (NACA code, library name or .dat file), live feedback,
imported polars and per-section factor overrides."""

from __future__ import annotations

import copy
from pathlib import Path

from PySide6.QtCore import QSignalBlocker, Qt, Signal
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QCompleter,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QStyledItemDelegate,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..io.backend import Backend, SectionInfo
from ..io.project import AirfoilSpec, ProjectState
from .theme import current_theme
from .widgets import UnitSpin, muted

#: (override key, label, tooltip, decimals, minimum, maximum, step, default)
FACTOR_ROWS = (
    ("a0", "Lift slope a0 [1/rad]", "dcl/dalpha of the section in the linear range (2 pi = 6.283 for a thin flat plate)", 3, 0.5, 20.0, 0.1, 6.0),
    ("alpha_L0_deg", "Zero-lift angle [deg]", "Angle of attack at zero lift (negative for positive camber)", 3, -30.0, 30.0, 0.1, -1.0),
    ("cm_c4", "Moment cm,c/4", "Pitching-moment coefficient about the quarter chord at zero lift (nose-up positive)", 4, -1.0, 1.0, 0.005, -0.03),
    ("cl_max", "Max lift cl,max", "Maximum section lift coefficient", 3, 0.2, 4.0, 0.05, 1.5),
    ("cd0", "Profile drag cd0", "Constant part of the parabolic drag polar cd = cd0 + 0.0076 cl^2", 5, 0.0005, 0.5, 0.0005, 0.008),
)


class _KeyDelegate(QStyledItemDelegate):
    """Line editor for the airfoil key with auto-completion from the library's names."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.names: list[str] = []

    def createEditor(self, parent, option, index):  # noqa: N802 - Qt naming
        editor = QLineEdit(parent)
        if self.names:
            comp = QCompleter(self.names, editor)
            comp.setCaseSensitivity(Qt.CaseInsensitive)
            comp.setFilterMode(Qt.MatchContains)
            editor.setCompleter(comp)
        return editor


class AirfoilPanel(QWidget):
    changed = Signal()
    selection_changed = Signal(int)
    library_pick_requested = Signal(int)

    def __init__(self, backend: Backend, parent=None) -> None:
        super().__init__(parent)
        self.backend = backend
        self._spec = AirfoilSpec()
        self._labels: list[str] = ["Root", "Tip"]
        self._infos: list[SectionInfo] | None = None
        self._busy = False
        self._build()
        self._rebuild_rows()

    # ======================================================================= layout
    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)

        box = QGroupBox("Airfoil of each section")
        v = QVBoxLayout(box)
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Section", "Airfoil (code, name or file)"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setMinimumHeight(110)
        self.delegate = _KeyDelegate(self.table)
        self.table.setItemDelegateForColumn(1, self.delegate)
        self.table.itemChanged.connect(self._item_changed)
        self.table.currentCellChanged.connect(self._current_changed)
        v.addWidget(self.table)

        row = QHBoxLayout()
        for text, tip, slot in (
            ("Library...", "Search the airfoil library folder", self._pick_library),
            ("Open .dat...", "Use an airfoil coordinate file", self._pick_file),
            ("Import polar...", "Use measured or XFOIL polar data instead of the section engine", self._pick_polar),
            ("Clear polar", "Go back to the section engine's data", self._clear_polar),
        ):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            row.addWidget(b)
        row.addStretch(1)
        v.addLayout(row)
        self.hint = muted("Type a code such as NACA 2412, 23012, 63-412 or 64A010, a library file name such as e387, or "
                          "use the buttons. 6-series sections need the UIUC .dat files (Library folder on the Solver tab).")
        v.addWidget(self.hint)
        root.addWidget(box)

        self.detail = QGroupBox("Selected section")
        g = QVBoxLayout(self.detail)
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        g.addWidget(self.status)
        self.data_label = muted("")
        g.addWidget(self.data_label)
        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.addWidget(QLabel("<b>Factor</b>"), 0, 0)
        grid.addWidget(QLabel("<b>Section data</b>"), 0, 1)
        grid.addWidget(QLabel("<b>Override</b>"), 0, 2, 1, 2)
        self._factor_widgets: dict[str, tuple[QLabel, QCheckBox, UnitSpin]] = {}
        for r, (key, label, tip, dec, lo, hi, step, default) in enumerate(FACTOR_ROWS, start=1):
            name = QLabel(label)
            name.setToolTip(tip)
            val = QLabel("-")
            val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            chk = QCheckBox()
            chk.setToolTip("Replace this factor, e.g. by wind-tunnel data")
            spin = UnitSpin(None, unit="", decimals=dec, minimum=lo, maximum=hi, step=step)
            spin.set_si(default)
            spin.setEnabled(False)
            chk.toggled.connect(lambda on, k=key: self._override_toggled(k, on))
            spin.valueChanged.connect(lambda _v, k=key: self._override_edited(k))
            grid.addWidget(name, r, 0)
            grid.addWidget(val, r, 1)
            grid.addWidget(chk, r, 2)
            grid.addWidget(spin, r, 3)
            self._factor_widgets[key] = (val, chk, spin)
        g.addLayout(grid)
        g.addWidget(muted("Factors are evaluated at the section's own Reynolds number after each run. An override replaces "
                          "the factor with a smooth parametric polar built on it."))
        root.addWidget(self.detail)
        root.addStretch(1)

    # ======================================================================== state
    def load(self, state: ProjectState) -> None:
        self._spec = copy.deepcopy(state.airfoils)
        self._infos = None
        self._rebuild_rows()

    def store(self, state: ProjectState) -> None:
        state.airfoils = copy.deepcopy(self._spec)
        state.airfoils.fit(len(self._labels))

    def spec(self) -> AirfoilSpec:
        return self._spec

    def problems(self) -> list[str]:
        out = []
        for i, key in enumerate(self._spec.keys):
            if self.backend.describe_key(key).level == "error":
                out.append(f"airfoil of section {i + 1}: {self.backend.describe_key(key).text}")
        return out

    def set_stations(self, labels: list[str]) -> None:
        """The wing's sections changed: adapt the rows, keeping what the user entered."""
        if labels == self._labels and len(self._spec.keys) == len(labels):
            return
        self._labels = list(labels)
        self._spec.fit(len(labels))
        self._rebuild_rows()

    def set_completions(self, names: list[str]) -> None:
        self.delegate.names = names

    def show_infos(self, infos: list[SectionInfo] | None) -> None:
        """Section data of the latest analysis (engine values of the factors, data source)."""
        self._infos = infos
        self._refresh_detail()

    def selected(self) -> int:
        r = self.table.currentRow()
        return r if 0 <= r < len(self._spec.keys) else 0

    def set_key(self, index: int, key: str) -> None:
        if 0 <= index < len(self._spec.keys):
            self._spec.keys[index] = key
            self._spec.polar_files[index] = ""
            self._busy = True
            try:
                self.table.item(index, 1).setText(key)
            finally:
                self._busy = False
            self._refresh_status(index)
            self._refresh_detail()
            self.changed.emit()

    def refresh_status(self) -> None:
        for i in range(len(self._spec.keys)):
            self._refresh_status(i)
        self._refresh_detail()

    # ======================================================================== rows
    def _rebuild_rows(self) -> None:
        self._busy = True
        try:
            n = len(self._labels)
            self._spec.fit(n)
            self.table.setRowCount(n)
            for i, label in enumerate(self._labels):
                head = QTableWidgetItem(label)
                head.setFlags(head.flags() & ~Qt.ItemIsEditable)
                self.table.setItem(i, 0, head)
                self.table.setItem(i, 1, QTableWidgetItem(self._spec.keys[i]))
            self.table.setMaximumHeight(60 + 28 * max(n, 2))
        finally:
            self._busy = False
        for i in range(n):
            self._refresh_status(i)
        if self.table.currentRow() < 0 and n:
            self.table.selectRow(0)
        self._refresh_detail()

    def _refresh_status(self, i: int) -> None:
        st = self.backend.describe_key(self._spec.keys[i])
        item = self.table.item(i, 1)
        if item is None:
            return
        th = current_theme()
        color = {"ok": th.fg, "warning": th.warn, "error": th.bad}[st.level]
        item.setForeground(QBrush(QColor(color)))
        item.setToolTip(st.text)

    # ====================================================================== details
    def _refresh_detail(self) -> None:
        i = self.selected()
        if not self._spec.keys:
            return
        st = self.backend.describe_key(self._spec.keys[i])
        th = current_theme()
        color = {"ok": th.good, "warning": th.warn, "error": th.bad}[st.level]
        mark = {"ok": "OK", "warning": "Check", "error": "Problem"}[st.level]
        self.status.setText(f"<span style='color:{color}'><b>{mark}</b></span> &nbsp;{_escape(st.text)}")
        polar = self._spec.polar_files[i]
        info = self._infos[i] if self._infos and i < len(self._infos) else None
        if polar:
            self.data_label.setText(f"Section data: imported polar {Path(polar).name}")
        elif info is not None:
            self.data_label.setText(f"Section data: {info.data_label} at Re = {info.re:.3g}")
        else:
            self.data_label.setText("Section data are evaluated when the analysis runs.")
        ov = self._spec.overrides[i]
        for key, (val, chk, spin) in self._factor_widgets.items():
            engine = info.engine_factors.get(key) if info else None
            val.setText("-" if engine is None else f"{engine:.4g}")
            with QSignalBlocker(chk), QSignalBlocker(spin):
                chk.setChecked(key in ov)
                spin.setEnabled(key in ov)
                if key in ov:
                    spin.set_si(ov[key])
                elif engine is not None:
                    spin.set_si(engine)

    # ======================================================================= slots
    def _item_changed(self, item: QTableWidgetItem) -> None:
        if self._busy or item.column() != 1:
            return
        i = item.row()
        self._spec.keys[i] = item.text().strip()
        self._spec.polar_files[i] = ""
        self._refresh_status(i)
        self._refresh_detail()
        self.changed.emit()

    def _current_changed(self, row: int, _c: int, _pr: int, _pc: int) -> None:
        if self._busy or row < 0:
            return
        self._refresh_detail()
        self.selection_changed.emit(row)

    def _override_toggled(self, key: str, on: bool) -> None:
        i = self.selected()
        ov = self._spec.overrides[i]
        _val, _chk, spin = self._factor_widgets[key]
        spin.setEnabled(on)
        if on:
            ov[key] = spin.si()
        else:
            ov.pop(key, None)
        self.changed.emit()

    def _override_edited(self, key: str) -> None:
        i = self.selected()
        _val, chk, spin = self._factor_widgets[key]
        if chk.isChecked():
            self._spec.overrides[i][key] = spin.si()
            self.changed.emit()

    def _pick_library(self) -> None:
        self.library_pick_requested.emit(self.selected())

    def _pick_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open airfoil coordinate file", "", "Airfoil files (*.dat *.txt);;All files (*)")
        if path:
            self.set_key(self.selected(), path)

    def _pick_polar(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Import polar", "", "Polar files (*.txt *.csv *.dat *.pol);;All files (*)")
        if path:
            i = self.selected()
            self._spec.polar_files[i] = path
            self._refresh_detail()
            self.changed.emit()

    def _clear_polar(self) -> None:
        i = self.selected()
        if self._spec.polar_files[i]:
            self._spec.polar_files[i] = ""
            self._refresh_detail()
            self.changed.emit()


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
