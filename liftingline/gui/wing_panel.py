"""The Wing tab: planform (trapezoid, elliptic or a free table of sections), sweep, dihedral, twist."""

from __future__ import annotations

import copy
import math

from PySide6.QtCore import QSignalBlocker, Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFormLayout,
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

from ..core import units
from ..io.presets import PRESETS
from ..io.project import (
    DEFINE_AREA_AR_TAPER,
    DEFINE_SPAN_AR_TAPER,
    DEFINE_SPAN_AREA_TAPER,
    DEFINE_SPAN_CHORDS,
    MODE_ELLIPTIC,
    MODE_SECTIONS,
    MODE_TRAPEZOID,
    ProjectError,
    ProjectState,
    WingSpec,
    wing_from_spec,
)
from .widgets import GuardedCombo, UnitSpin, muted, set_role

_TABLE_COLUMNS = (
    ("y", "length", "y"),
    ("chord", "length", "chord"),
    ("x_le", "length", "x LE"),
    ("z", "length", "z c/4"),
    ("twist_deg", None, "twist"),
)


class SectionTable(QTableWidget):
    """Editable table of the wing's sections (right half): y, chord, leading-edge x, c/4 height, twist."""

    changed = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(0, len(_TABLE_COLUMNS), parent)
        self._unit = "m"
        self._busy = False
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setAlternatingRowColors(True)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.verticalHeader().setDefaultSectionSize(24)
        self._update_headers()
        self.itemChanged.connect(self._item_changed)
        self.setMinimumHeight(170)

    # -------------------------------------------------------------------------- headers
    def _update_headers(self) -> None:
        heads = [f"{name}\n[{self._unit}]" if kind else f"{name}\n[deg]" for _, kind, name in _TABLE_COLUMNS]
        self.setHorizontalHeaderLabels(heads)

    def _update_row_labels(self) -> None:
        n = self.rowCount()
        labels = [("1 (root)" if i == 0 else f"{i + 1} (tip)" if i == n - 1 else str(i + 1)) for i in range(n)]
        self.setVerticalHeaderLabels(labels)

    # -------------------------------------------------------------------------- content
    def _display(self, kind: str | None, value_si: float) -> str:
        v = units.from_si(kind, value_si, self._unit) if kind else value_si
        return f"{v:.5g}"

    def set_rows(self, rows: list[dict[str, float]], length_unit: str) -> None:
        self._busy = True
        try:
            self._unit = length_unit
            self.setRowCount(len(rows))
            for r, row in enumerate(rows):
                for c, (key, kind, _) in enumerate(_TABLE_COLUMNS):
                    item = QTableWidgetItem(self._display(kind, float(row.get(key, 0.0))))
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    if r == 0 and key == "y":  # the root is always at y = 0
                        item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                    self.setItem(r, c, item)
            self._update_headers()
            self._update_row_labels()
        finally:
            self._busy = False

    def _cell(self, r: int, c: int) -> float:
        item = self.item(r, c)
        try:
            return float(item.text().replace(",", ".")) if item is not None else math.nan
        except ValueError:
            return math.nan

    def rows_si(self) -> list[dict[str, float]]:
        out = []
        for r in range(self.rowCount()):
            row = {}
            for c, (key, kind, _) in enumerate(_TABLE_COLUMNS):
                v = self._cell(r, c)
                row[key] = float(units.to_si(kind, v, self._unit)) if kind and math.isfinite(v) else v
            out.append(row)
        return out

    def problems(self) -> list[str]:
        msgs = []
        for r in range(self.rowCount()):
            for c, (_, _, name) in enumerate(_TABLE_COLUMNS):
                if not math.isfinite(self._cell(r, c)):
                    msgs.append(f"section table, row {r + 1}: '{name}' is not a number")
        return msgs

    def set_length_unit(self, unit: str) -> None:
        if unit == self._unit:
            return
        rows = self.rows_si()
        self.set_rows(rows, unit)

    # ---------------------------------------------------------------------- row actions
    def add_row(self) -> None:
        rows = self.rows_si()
        last = dict(rows[-1])
        step = rows[-1]["y"] - rows[-2]["y"] if len(rows) > 1 else 1.0
        last["y"] = rows[-1]["y"] + (step if step > 0 else 1.0)
        rows.append(last)
        self.set_rows(rows, self._unit)
        self.changed.emit()

    def insert_row(self) -> None:
        rows = self.rows_si()
        r = max(self.currentRow(), 0)
        if r >= len(rows) - 1:
            r = len(rows) - 2
        a, b = rows[r], rows[r + 1]
        rows.insert(r + 1, {k: 0.5 * (a[k] + b[k]) for k in a})
        self.set_rows(rows, self._unit)
        self.changed.emit()

    def remove_row(self) -> None:
        rows = self.rows_si()
        r = self.currentRow()
        if len(rows) <= 2 or r <= 0:  # keep a root and a tip; the root cannot be removed
            return
        del rows[r]
        self.set_rows(rows, self._unit)
        self.changed.emit()

    # -------------------------------------------------------------------------- editing
    def _item_changed(self, item: QTableWidgetItem) -> None:
        if self._busy:
            return
        ok = math.isfinite(self._cell(item.row(), item.column()))
        with QSignalBlocker(self):
            item.setBackground(QColor(0, 0, 0, 0) if ok else QColor("#fca5a5"))
            item.setToolTip("" if ok else "not a number")
        self.changed.emit()


class WingPanel(QWidget):
    """Planform and geometry editor. Holds a :class:`WingSpec`; ``load``/``store`` exchange it with the project."""

    changed = Signal()
    sections_changed = Signal()
    preset_requested = Signal(str)

    _DEFINE_LABELS = {
        MODE_TRAPEZOID: (
            (DEFINE_SPAN_CHORDS, "Span, root chord, tip chord"),
            (DEFINE_AREA_AR_TAPER, "Area, aspect ratio, taper ratio"),
            (DEFINE_SPAN_AREA_TAPER, "Span, area, taper ratio"),
            (DEFINE_SPAN_AR_TAPER, "Span, aspect ratio, taper ratio"),
        ),
        MODE_ELLIPTIC: (
            (DEFINE_SPAN_CHORDS, "Span and root chord"),
            (DEFINE_AREA_AR_TAPER, "Area and aspect ratio"),
        ),
    }

    def __init__(self, system: str = "SI", parent=None) -> None:
        super().__init__(parent)
        self._system = system
        self._spec = WingSpec()
        self._busy = False
        self._build()
        self._refresh_all()

    # ======================================================================== layout
    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)

        row = QHBoxLayout()
        row.addWidget(QLabel("Start from"))
        self.preset = GuardedCombo()
        self.preset.addItem("example wing...", None)
        for name in PRESETS:
            self.preset.addItem(name, name)
        self.preset.setToolTip("Replace the whole project (wing, airfoils, flow) by a ready-made example")
        self.preset.activated.connect(self._preset_chosen)
        row.addWidget(self.preset, 1)
        root.addLayout(row)

        # ---- planform
        self.plan_box = QGroupBox("Planform")
        form = self.plan_form = QFormLayout(self.plan_box)
        self.mode = GuardedCombo()
        self.mode.addItem("Straight tapered (trapezoid)", MODE_TRAPEZOID)
        self.mode.addItem("Elliptic", MODE_ELLIPTIC)
        self.mode.addItem("Free table of sections", MODE_SECTIONS)
        self.mode.setToolTip("Straight tapered wing, exact elliptic planform, or any piecewise-linear planform")
        self.mode.currentIndexChanged.connect(self._mode_changed)
        form.addRow("Planform", self.mode)
        self.define = GuardedCombo()
        self.define.setToolTip("Which three numbers you enter; the rest are calculated")
        self.define.currentIndexChanged.connect(self._define_changed)
        form.addRow("Define by", self.define)

        def spin(kind, **kw):
            s = UnitSpin(kind, self._system, **kw)
            s.valueChanged.connect(self._field_edited)
            return s

        self.f_span = spin("length", decimals=4, minimum=1e-3, maximum=1e4, step=0.1, tooltip="Projected span b (tip to tip)")
        self.f_root = spin("length", decimals=4, minimum=1e-4, maximum=1e3, step=0.05, tooltip="Chord at the root (plane of symmetry)")
        self.f_tip = spin("length", decimals=4, minimum=0.0, maximum=1e3, step=0.05, tooltip="Chord at the tip (0 gives a pointed tip)")
        self.f_area = spin("area", decimals=4, minimum=1e-4, maximum=1e5, step=0.5, tooltip="Projected planform area S")
        self.f_ar = spin(None, unit="", decimals=3, minimum=0.2, maximum=100.0, step=0.5, tooltip="Aspect ratio b^2/S")
        self.f_taper = spin(None, unit="", decimals=3, minimum=0.0, maximum=10.0, step=0.05, tooltip="Tip chord / root chord")
        self._rows = {}
        for key, label, w in (
            ("span", "Span b", self.f_span),
            ("root", "Root chord", self.f_root),
            ("tip", "Tip chord", self.f_tip),
            ("area", "Area S", self.f_area),
            ("ar", "Aspect ratio", self.f_ar),
            ("taper", "Taper ratio", self.f_taper),
        ):
            form.addRow(label, w)
            self._rows[key] = (form.rowCount() - 1, w)
        root.addWidget(self.plan_box)

        # ---- sweep, dihedral, twist
        self.geo_box = QGroupBox("Sweep, dihedral and twist")
        gform = QFormLayout(self.geo_box)
        self.f_sweep = spin(None, unit="deg", decimals=3, minimum=-80.0, maximum=80.0, step=1.0, tooltip="Sweep of the line at the chord fraction below (positive = swept back)")
        self.f_sweep_at = spin(None, unit="% chord", decimals=1, minimum=0.0, maximum=100.0, step=5.0, tooltip="0 = leading edge, 25 = quarter chord, 100 = trailing edge")
        self.f_dihedral = spin(None, unit="deg", decimals=3, minimum=-60.0, maximum=60.0, step=0.5, tooltip="Dihedral of the quarter-chord line (positive = tips up)")
        self.f_twist = spin(None, unit="deg", decimals=3, minimum=-30.0, maximum=30.0, step=0.5, tooltip="Tip twist relative to the root, nose-up positive (washout is negative)")
        self.f_incidence = spin(None, unit="deg", decimals=3, minimum=-20.0, maximum=20.0, step=0.5, tooltip="Root incidence: added to the flow angle of attack")
        gform.addRow("Sweep", self.f_sweep)
        gform.addRow("   measured at", self.f_sweep_at)
        gform.addRow("Dihedral", self.f_dihedral)
        gform.addRow("Tip twist (washout -)", self.f_twist)
        gform.addRow("Root incidence", self.f_incidence)
        self.geo_hint = muted("Sweep, dihedral and twist come from the section table.")
        gform.addRow(self.geo_hint)
        root.addWidget(self.geo_box)

        # ---- section table
        self.table_box = QGroupBox("Sections (right half-wing)")
        tl = QVBoxLayout(self.table_box)
        self.table = SectionTable()
        self.table.changed.connect(self._table_edited)
        tl.addWidget(self.table)
        btns = QHBoxLayout()
        for text, tip, slot in (
            ("Add", "Append a section after the tip", self.table.add_row),
            ("Insert", "Insert a section after the selected row", self.table.insert_row),
            ("Remove", "Remove the selected section", self.table.remove_row),
        ):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            btns.addWidget(b)
        btns.addStretch(1)
        tl.addLayout(btns)
        tl.addWidget(muted("y is measured from the plane of symmetry; x LE is the leading-edge position behind the root "
                           "leading edge; z is the height of the quarter-chord point; twist is nose-up positive."))
        root.addWidget(self.table_box)

        # ---- derived
        self.derived = QLabel()
        self.derived.setWordWrap(True)
        self.derived.setTextInteractionFlags(Qt.TextSelectableByMouse)
        root.addWidget(self.derived)
        self.problem = QLabel()
        self.problem.setWordWrap(True)
        set_role(self.problem, "error")
        root.addWidget(self.problem)
        root.addStretch(1)

    # ======================================================================= state
    def load(self, state: ProjectState) -> None:
        self._spec = copy.deepcopy(state.wing)
        self._spec.sync()
        self._refresh_all()

    def store(self, state: ProjectState) -> None:
        self._read_fields()
        state.wing = copy.deepcopy(self._spec)

    def spec(self) -> WingSpec:
        self._read_fields()
        return self._spec

    def set_system(self, system: str) -> None:
        self._system = system
        for w in (self.f_span, self.f_root, self.f_tip, self.f_area):
            w.set_system(system)
        self.table.set_length_unit(units.display_unit(system, "length"))
        self._refresh_derived()

    def problems(self) -> list[str]:
        return self.table.problems() if self._spec.mode == MODE_SECTIONS else []

    # ======================================================================= helpers
    def _chosen(self) -> set[str]:
        """The planform fields that are inputs for the current ``define_by`` (the others are results)."""
        s = self._spec
        if s.mode == MODE_ELLIPTIC:
            return {"span", "root"} if s.define_by == DEFINE_SPAN_CHORDS else {"area", "ar"}
        return {
            DEFINE_SPAN_CHORDS: {"span", "root", "tip"},
            DEFINE_AREA_AR_TAPER: {"area", "ar", "taper"},
            DEFINE_SPAN_AREA_TAPER: {"span", "area", "taper"},
            DEFINE_SPAN_AR_TAPER: {"span", "ar", "taper"},
        }.get(s.define_by, {"span", "root", "tip"})

    def _read_fields(self) -> None:
        """Copy the widgets into the spec. Only the input fields are read: the displayed results (rounded to the
        field's decimals) must not leak back into the numbers the analysis uses."""
        s = self._spec
        if s.mode != MODE_SECTIONS:
            chosen = self._chosen()
            for key, widget, attr in (
                ("span", self.f_span, "span"),
                ("root", self.f_root, "root_chord"),
                ("tip", self.f_tip, "tip_chord"),
                ("area", self.f_area, "area"),
                ("ar", self.f_ar, "aspect_ratio"),
                ("taper", self.f_taper, "taper_ratio"),
            ):
                if key in chosen:
                    setattr(s, attr, widget.si())
            s.sync()
        else:
            s.sections = self.table.rows_si()
        s.sweep_deg = self.f_sweep.si()
        s.sweep_at = self.f_sweep_at.si() / 100.0
        s.dihedral_deg = self.f_dihedral.si()
        s.twist_tip_deg = self.f_twist.si()
        s.incidence_deg = self.f_incidence.si()

    def _refresh_all(self) -> None:
        """Push the spec into every widget without emitting change signals."""
        self._busy = True
        try:
            s = self._spec
            self.mode.set_data_silently(s.mode)
            self._fill_define_combo()
            self._push_fields()
            self.f_sweep.set_si(s.sweep_deg)
            self.f_sweep_at.set_si(100.0 * s.sweep_at)
            self.f_dihedral.set_si(s.dihedral_deg)
            self.f_twist.set_si(s.twist_tip_deg)
            self.f_incidence.set_si(s.incidence_deg)
            if s.mode == MODE_SECTIONS:
                self.table.set_rows(s.sections, units.display_unit(self._system, "length"))
            self._apply_mode_visibility()
        finally:
            self._busy = False
        self._refresh_derived()

    def _push_fields(self) -> None:
        s = self._spec
        for w, v in (
            (self.f_span, s.span),
            (self.f_root, s.root_chord),
            (self.f_tip, s.tip_chord),
            (self.f_area, s.area),
            (self.f_ar, s.aspect_ratio),
            (self.f_taper, s.taper_ratio),
        ):
            w.set_si(v)

    def _fill_define_combo(self) -> None:
        s = self._spec
        options = self._DEFINE_LABELS.get(s.mode, ())
        with QSignalBlocker(self.define):
            self.define.clear()
            for key, label in options:
                self.define.addItem(label, key)
            i = self.define.findData(s.define_by)
            if i < 0:
                i = 0
                if options:
                    s.define_by = options[0][0]
            self.define.setCurrentIndex(i)

    def _apply_mode_visibility(self) -> None:
        s = self._spec
        form = self.plan_form
        simple = s.mode != MODE_SECTIONS
        elliptic = s.mode == MODE_ELLIPTIC
        self.define.setEnabled(simple)
        form.setRowVisible(self.define, simple)
        for key, (row, _w) in self._rows.items():
            visible = simple and not (elliptic and key in ("tip", "taper"))
            form.setRowVisible(row, visible)
        chosen = self._chosen()  # which fields are inputs and which are results
        for key, (_row, w) in self._rows.items():
            w.setEnabled(key in chosen)
        self.table_box.setVisible(s.mode == MODE_SECTIONS)
        self.geo_box.setVisible(True)
        free = s.mode == MODE_TRAPEZOID
        self.f_sweep.setEnabled(free)
        self.f_sweep_at.setEnabled(free)
        self.f_dihedral.setEnabled(free)
        self.f_twist.setEnabled(s.mode != MODE_SECTIONS)
        self.geo_hint.setVisible(s.mode == MODE_SECTIONS)
        if elliptic:
            self.geo_hint.setText("An elliptic planform has a straight quarter-chord line; only twist and incidence apply.")
            self.geo_hint.setVisible(True)
        else:
            self.geo_hint.setText("Sweep, dihedral and twist come from the section table.")

    def _refresh_derived(self) -> None:
        keys = [""] * self._spec.n_sections()
        try:
            wing = wing_from_spec(self._spec, keys)
        except ProjectError as exc:
            self.derived.setText("")
            self.problem.setText(f"Invalid wing: {exc}")
            return
        self.problem.setText("")
        uL, uA = units.display_unit(self._system, "length"), units.display_unit(self._system, "area")
        sweep = wing.sweep_c4_deg_by_segment()
        dih = wing.dihedral_deg_by_segment()
        self.derived.setText(
            f"<b>Area</b> {units.from_si('area', wing.area, uA):.4g} {uA} &nbsp; <b>AR</b> {wing.aspect_ratio:.3g} &nbsp; "
            f"<b>taper</b> {wing.taper_ratio:.3g}<br>"
            f"<b>MAC</b> {units.from_si('length', wing.mac, uL):.4g} {uL} at y = {units.from_si('length', wing.y_mac, uL):.3g} {uL} &nbsp; "
            f"<b>c/4 sweep</b> {sweep[0]:.1f}&deg;" + (f" ... {sweep[-1]:.1f}&deg;" if abs(sweep[-1] - sweep[0]) > 0.05 else "")
            + f" &nbsp; <b>dihedral</b> {dih[0]:.1f}&deg;" + (f" ... {dih[-1]:.1f}&deg;" if abs(dih[-1] - dih[0]) > 0.05 else "")
        )

    # ------------------------------------------------------------- station labels
    def station_labels(self) -> list[str]:
        """Names of the wing's sections for the airfoil table."""
        s = self._spec
        uL = units.display_unit(self._system, "length")
        if s.mode != MODE_SECTIONS:
            try:
                half = 0.5 * s.dimensions()[0]
            except ProjectError:
                half = 0.5 * s.span
            return ["Root (y = 0)", f"Tip (y = {units.from_si('length', half, uL):.3g} {uL})"]
        rows = s.sections
        n = len(rows)
        out = []
        for i, r in enumerate(rows):
            y = units.from_si("length", r.get("y", 0.0), uL)
            tag = "Root" if i == 0 else "Tip" if i == n - 1 else f"Section {i + 1}"
            out.append(f"{tag} (y = {y:.3g} {uL})")
        return out

    # ========================================================================== slots
    def _preset_chosen(self, index: int) -> None:
        name = self.preset.itemData(index)
        if name:
            self.preset_requested.emit(name)
        with QSignalBlocker(self.preset):
            self.preset.setCurrentIndex(0)

    def _mode_changed(self, _i: int) -> None:
        if self._busy:
            return
        self._read_fields()
        s = self._spec
        new = self.mode.currentData()
        if new == s.mode:
            return
        old = s.mode
        if new == MODE_SECTIONS and old != MODE_SECTIONS:
            s.sections = s.to_table()  # start from the planform that is on screen
        s.mode = new
        if new != MODE_SECTIONS:
            s.define_by = DEFINE_SPAN_CHORDS if s.define_by not in dict(self._DEFINE_LABELS[new]) else s.define_by
            s.sync()
        self._refresh_all()
        self.sections_changed.emit()
        self.changed.emit()

    def _define_changed(self, _i: int) -> None:
        if self._busy:
            return
        self._read_fields()
        self._spec.define_by = self.define.currentData()
        self._spec.sync()
        self._busy = True
        try:
            self._push_fields()
            self._apply_mode_visibility()
        finally:
            self._busy = False
        self._refresh_derived()
        self.changed.emit()

    def _field_edited(self, *_args) -> None:
        if self._busy:
            return
        self._read_fields()
        if self._spec.mode != MODE_SECTIONS:
            self._busy = True
            try:
                self._push_fields()
            finally:
                self._busy = False
        self._refresh_derived()
        self.sections_changed.emit()
        self.changed.emit()

    def _table_edited(self) -> None:
        if self._busy:
            return
        self._read_fields()
        self._refresh_derived()
        self.sections_changed.emit()
        self.changed.emit()

