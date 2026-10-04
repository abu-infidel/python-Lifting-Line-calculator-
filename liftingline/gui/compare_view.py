"""The Compare tab: pin results and look at their totals and spanwise loads side by side."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QHeaderView,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..core.analysis import AnalysisResult
from ..io.report import totals_rows
from .plots import make_canvas, plot_compare
from .widgets import muted

MAX_SNAPSHOTS = 6


@dataclass
class Snapshot:
    name: str
    result: AnalysisResult


class CompareView(QWidget):
    changed = Signal()

    def __init__(self, system: str = "SI", parent=None) -> None:
        super().__init__(parent)
        self._system = system
        self.snapshots: list[Snapshot] = []
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        row = QHBoxLayout()
        self.remove_btn = QPushButton("Remove selected")
        self.remove_btn.clicked.connect(self.remove_selected)
        self.clear_btn = QPushButton("Clear all")
        self.clear_btn.clicked.connect(self.clear)
        row.addWidget(self.remove_btn)
        row.addWidget(self.clear_btn)
        row.addWidget(muted("Pin the current result with Tools > Pin result for comparison (Ctrl+P), change the inputs, pin again."), 1)
        lay.addLayout(row)
        split = QSplitter(Qt.Vertical)
        self.table = QTableWidget(0, 0)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.setSelectionBehavior(QTableWidget.SelectColumns)
        split.addWidget(self.table)
        self.canvas = make_canvas()
        split.addWidget(self.canvas)
        split.setSizes([260, 380])
        lay.addWidget(split, 1)
        self.redraw()

    # ---------------------------------------------------------------------- content
    def pin(self, name: str, result: AnalysisResult) -> bool:
        if len(self.snapshots) >= MAX_SNAPSHOTS:
            return False
        self.snapshots.append(Snapshot(name, result))
        self.redraw()
        self.changed.emit()
        return True

    def clear(self) -> None:
        self.snapshots.clear()
        self.redraw()
        self.changed.emit()

    def remove_selected(self) -> None:
        cols = sorted({i.column() for i in self.table.selectedIndexes()}, reverse=True)
        for c in cols:
            if 1 <= c <= len(self.snapshots):
                del self.snapshots[c - 1]
        self.redraw()
        self.changed.emit()

    def set_system(self, system: str) -> None:
        self._system = system
        self.redraw()

    def redraw(self) -> None:
        self._fill_table()
        snaps, system = self.snapshots, self._system
        self.canvas.draw_with(lambda fig, th: plot_compare(fig, snaps, th, system))
        self.remove_btn.setEnabled(bool(snaps))
        self.clear_btn.setEnabled(bool(snaps))

    def _fill_table(self) -> None:
        snaps = self.snapshots
        t = self.table
        t.clear()
        if not snaps:
            t.setRowCount(0)
            t.setColumnCount(0)
            return
        per = [totals_rows(s.result, self._system) for s in snaps]
        wanted = [r for r in per[0] if r.group in ("Forces and moments", "Efficiency", "Stall")]
        extra = len(snaps) > 1
        t.setColumnCount(1 + len(snaps) + (1 if extra else 0))
        heads = ["Quantity"] + [s.name for s in snaps] + (["change last vs first"] if extra else [])
        t.setHorizontalHeaderLabels(heads)
        t.setRowCount(len(wanted))
        for i, row in enumerate(wanted):
            label = row.label.strip()
            t.setItem(i, 0, QTableWidgetItem(f"{label} ({row.coeff_name})" if row.coeff_name else label))
            values = []
            for j, rows in enumerate(per):
                match = next((r for r in rows if r.label.strip() == label), None)
                if match is None:
                    text, tip = "", ""
                elif match.coeff:  # coefficients compare best; the dimensional value goes to the tooltip
                    text, tip = match.coeff, f"{match.value} {match.unit}".strip()
                else:
                    text, tip = f"{match.value} {match.unit}".strip(), ""
                item = QTableWidgetItem(text)
                item.setToolTip(tip)
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                t.setItem(i, 1 + j, item)
                if match is None:
                    values.append(None)
                else:
                    values.append(match.coeff_value if match.coeff_value is not None else match.si_value)
            if extra:
                a, b = values[0], values[-1]
                delta = "" if (a is None or b is None or a == 0) else f"{100.0 * (b - a) / abs(a):+.1f} %"
                item = QTableWidgetItem(delta)
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                t.setItem(i, 1 + len(snaps), item)
