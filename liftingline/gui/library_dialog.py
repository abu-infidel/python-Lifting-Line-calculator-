"""Dialogs: the airfoil-library browser and a generic text-report dialog with a progress bar."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QProgressBar,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..airfoil.library import AirfoilLibrary, LibraryEntry
from .plots import make_canvas, plot_airfoils
from .widgets import muted


class LibraryDialog(QDialog):
    """Search a library by name or NACA designation and preview the airfoils."""

    MAX_ROWS = 300

    def __init__(self, library: AirfoilLibrary, parent: QWidget | None = None, query: str = "") -> None:
        super().__init__(parent)
        self.library = library
        self.setWindowTitle("Airfoil library")
        self.resize(820, 520)
        self._entries: list[LibraryEntry] = []

        lay = QVBoxLayout(self)
        self.search = QLineEdit(query)
        self.search.setPlaceholderText("search by name or designation, e.g. e387, s1223, 63-412, 64a010")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._refresh)
        lay.addWidget(self.search)

        split = QSplitter(Qt.Horizontal)
        self.list = QListWidget()
        self.list.currentRowChanged.connect(self._preview)
        self.list.itemDoubleClicked.connect(lambda _i: self.accept())
        split.addWidget(self.list)
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        self.canvas = make_canvas(toolbar=False)
        rl.addWidget(self.canvas, 1)
        self.info = QLabel()
        self.info.setWordWrap(True)
        self.info.setTextInteractionFlags(Qt.TextSelectableByMouse)
        rl.addWidget(self.info)
        split.addWidget(right)
        split.setSizes([300, 520])
        lay.addWidget(split, 1)

        self.count = muted("")
        lay.addWidget(self.count)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Use this airfoil")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)
        self._refresh()

    def _refresh(self) -> None:
        found = self.library.search(self.search.text(), self.MAX_ROWS)
        self._entries = found
        self.list.clear()
        for e in found:
            self.list.addItem(QListWidgetItem(e.label))
        total = len(self.library)
        shown = len(found)
        self.count.setText(f"{shown} shown of {total} airfoils" + (f" (first {self.MAX_ROWS})" if shown >= self.MAX_ROWS else ""))
        if found:
            self.list.setCurrentRow(0)
        else:
            self.canvas.draw_with(None, "No matching airfoil")
            self.info.setText("")

    def _preview(self, row: int) -> None:
        if not 0 <= row < len(self._entries):
            return
        entry = self._entries[row]
        try:
            af = self.library.load(entry)
        except Exception as exc:
            self.canvas.draw_with(None, f"Cannot read this file:\n{exc}")
            self.info.setText(str(entry.path))
            return
        self.canvas.draw_with(lambda fig, th: plot_airfoils(fig, [(entry.name, af)], 0, th))
        notes = ("<br>" + "; ".join(af.notes)) if af.notes else ""
        self.info.setText(
            f"<b>{entry.label}</b><br>t/c {100 * af.t_over_c:.1f} %, camber {100 * abs(af.max_camber):.2f} % "
            f"at {100 * af.x_camber:.0f} % chord<br><span style='color:gray'>{entry.path}</span>{notes}"
        )

    def selected_name(self) -> str | None:
        row = self.list.currentRow()
        return self._entries[row].name if 0 <= row < len(self._entries) else None


class ReportDialog(QDialog):
    """Shows the text result of a long task, with a progress bar while it runs."""

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(640, 460)
        lay = QVBoxLayout(self)
        self.status = QLabel("Working...")
        lay.addWidget(self.status)
        self.bar = QProgressBar()
        self.bar.setRange(0, 0)
        lay.addWidget(self.bar)
        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setLineWrapMode(QPlainTextEdit.NoWrap)
        font = self.text.font()
        font.setFamily("monospace")
        self.text.setFont(font)
        lay.addWidget(self.text, 1)
        row = QHBoxLayout()
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        row.addWidget(buttons)
        lay.addLayout(row)

    def set_progress(self, done: int, total: int) -> None:
        self.bar.setRange(0, max(total, 1))
        self.bar.setValue(done)
        self.status.setText(f"{done} of {total}")

    def finish(self, text: str, status: str = "Done") -> None:
        self.bar.setRange(0, 1)
        self.bar.setValue(1)
        self.status.setText(status)
        self.text.setPlainText(text)
