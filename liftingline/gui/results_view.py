"""The results panel: totals table, validity warnings and the run log."""

from __future__ import annotations

import time

from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor, QFont
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTextBrowser,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..core.analysis import AnalysisResult
from ..io.report import totals_rows, totals_text
from .theme import current_theme


def _html_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class ResultsView(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._result: AnalysisResult | None = None
        self._system = "SI"
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)

        self.header = QLabel("No analysis yet")
        self.header.setWordWrap(True)
        self.header.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lay.addWidget(self.header)

        split = QSplitter(Qt.Vertical)
        lay.addWidget(split, 1)

        self.tree = QTreeWidget()
        self.tree.setColumnCount(3)
        self.tree.setHeaderLabels(["Quantity", "Value", "Coefficient"])
        self.tree.setRootIsDecorated(False)
        self.tree.setAlternatingRowColors(True)
        self.tree.setUniformRowHeights(True)
        self.tree.header().setStretchLastSection(True)
        self.tree.setColumnWidth(0, 160)
        self.tree.setColumnWidth(1, 118)
        split.addWidget(self.tree)

        self.notes = QTextBrowser()
        self.notes.setOpenExternalLinks(False)
        self.notes.setMinimumHeight(60)
        split.addWidget(self.notes)

        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumBlockCount(500)
        self.log_box.setPlaceholderText("Run log")
        split.addWidget(self.log_box)
        split.setSizes([360, 130, 90])

        row = QHBoxLayout()
        self.copy_btn = QPushButton("Copy table")
        self.copy_btn.setToolTip("Copy the results as tab-separated text (pastes into a spreadsheet)")
        self.copy_btn.clicked.connect(self.copy_table)
        self.copy_btn.setEnabled(False)
        row.addWidget(self.copy_btn)
        row.addStretch(1)
        lay.addLayout(row)

    # ----------------------------------------------------------------------- content
    def set_system(self, system: str) -> None:
        self._system = system
        if self._result is not None:
            self.show_result(self._result)

    def show_result(self, result: AnalysisResult) -> None:
        self._result = result
        th = current_theme()
        info = result.info
        reason = str(info.get("solver_reason", "")).split(": ", 1)[-1]
        bits = [f"<b>{result.solver_used}</b> solver" + (f" ({_html_escape(reason)})" if reason else "")]
        bits.append(f"section data: <b>{_html_escape(str(info.get('section_engine', '')))}</b>")
        if info.get("nonlinear"):
            bits.append("nonlinear section polars")
        if info.get("high_lift"):
            bits.append(f"<b>{_html_escape(str(info['high_lift']))}</b>")
        bits.append(f"{1000 * float(info.get('solve_time_s', 0.0)):.0f} ms")
        self.header.setText(" &middot; ".join(bits))

        self.tree.clear()
        bold = QFont(self.tree.font())
        bold.setBold(True)
        group_item: QTreeWidgetItem | None = None
        current = None
        for r in totals_rows(result, self._system):
            if r.group != current:
                current = r.group
                group_item = QTreeWidgetItem(self.tree, [r.group])
                group_item.setFirstColumnSpanned(True)
                group_item.setFont(0, bold)
                group_item.setForeground(0, QBrush(QColor(th.accent)))
                group_item.setFlags(Qt.ItemIsEnabled)
            value = f"{r.value} {r.unit}".strip()
            coeff = f"{r.coeff_name} = {r.coeff}" if r.coeff_name and r.coeff else r.coeff
            item = QTreeWidgetItem(group_item, [r.label, value, coeff])
            item.setTextAlignment(1, Qt.AlignRight | Qt.AlignVCenter)
            if r.tip:
                for c in range(3):
                    item.setToolTip(c, r.tip)
        self.tree.expandAll()
        self.copy_btn.setEnabled(True)
        self._show_notes(result.warnings, [])

    def _show_notes(self, warnings: list[str], problems: list[str]) -> None:
        th = current_theme()
        parts = []
        for msg in problems:
            parts.append(f"<p style='color:{th.bad}'><b>Cannot run:</b> {_html_escape(msg)}</p>")
        if warnings:
            parts.append(f"<p style='color:{th.warn}'><b>Validity notes</b></p><ul>")
            parts += [f"<li style='color:{th.warn}'>{_html_escape(w)}</li>" for w in warnings]
            parts.append("</ul>")
        elif not problems:
            parts.append(f"<p style='color:{th.good}'>No validity warnings: this case is inside the range of the theory.</p>")
        self.notes.setHtml("".join(parts))

    def show_problems(self, problems: list[str]) -> None:
        """Inputs that cannot be analysed (the previous result stays visible)."""
        self._show_notes([], problems)

    def show_error(self, text: str) -> None:
        th = current_theme()
        self.notes.setHtml(f"<p style='color:{th.bad}'><b>Analysis failed</b></p><p>{_html_escape(text)}</p>")

    def clear(self) -> None:
        self._result = None
        self.tree.clear()
        self.notes.clear()
        self.header.setText("No analysis yet")
        self.copy_btn.setEnabled(False)

    def log(self, text: str) -> None:
        self.log_box.appendPlainText(f"{time.strftime('%H:%M:%S')}  {text}")

    def copy_table(self) -> None:
        if self._result is not None:
            QApplication.clipboard().setText(totals_text(self._result, self._system))
            self.log("results copied to the clipboard")
