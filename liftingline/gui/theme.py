"""Look and feel: Fusion style, light and dark palettes, and the matching colour set for the plots."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication


@dataclass(frozen=True)
class Theme:
    name: str  # "light" | "dark"
    bg: str  # plot / editor background
    panel: str  # window and panel background
    fg: str
    muted: str
    grid: str
    accent: str
    accent2: str
    good: str
    warn: str
    bad: str
    series: tuple[str, ...]  # line colours for plots

    @property
    def dark(self) -> bool:
        return self.name == "dark"


LIGHT = Theme(
    name="light",
    bg="#ffffff",
    panel="#f3f4f6",
    fg="#1f2933",
    muted="#6b7280",
    grid="#d9dde3",
    accent="#2563eb",
    accent2="#d97706",
    good="#15803d",
    warn="#b45309",
    bad="#b91c1c",
    series=("#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed", "#0891b2"),
)

DARK = Theme(
    name="dark",
    bg="#1b1f27",
    panel="#232833",
    fg="#e5e7eb",
    muted="#9ca3af",
    grid="#3a4150",
    accent="#60a5fa",
    accent2="#fbbf24",
    good="#4ade80",
    warn="#fbbf24",
    bad="#f87171",
    series=("#60a5fa", "#f87171", "#34d399", "#fbbf24", "#a78bfa", "#22d3ee"),
)

_current: Theme = LIGHT


def current_theme() -> Theme:
    return _current


def _palette(t: Theme) -> QPalette:
    p = QPalette()
    c = QColor
    p.setColor(QPalette.Window, c(t.panel))
    p.setColor(QPalette.WindowText, c(t.fg))
    p.setColor(QPalette.Base, c(t.bg))
    p.setColor(QPalette.AlternateBase, c(t.panel))
    p.setColor(QPalette.Text, c(t.fg))
    p.setColor(QPalette.Button, c(t.panel))
    p.setColor(QPalette.ButtonText, c(t.fg))
    p.setColor(QPalette.BrightText, c("#ffffff"))
    p.setColor(QPalette.ToolTipBase, c(t.bg))
    p.setColor(QPalette.ToolTipText, c(t.fg))
    p.setColor(QPalette.PlaceholderText, c(t.muted))
    p.setColor(QPalette.Highlight, c(t.accent))
    p.setColor(QPalette.HighlightedText, c("#ffffff" if not t.dark else "#0b1220"))
    p.setColor(QPalette.Link, c(t.accent))
    for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
        p.setColor(QPalette.Disabled, role, c(t.muted))
    return p


def _stylesheet(t: Theme) -> str:
    return f"""
    QToolTip {{ background: {t.bg}; color: {t.fg}; border: 1px solid {t.grid}; padding: 4px; }}
    QGroupBox {{ font-weight: 600; border: 1px solid {t.grid}; border-radius: 6px; margin-top: 12px; padding: 10px 6px 6px 6px; }}
    QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; }}
    QTabBar::tab {{ padding: 6px 12px; }}
    QHeaderView::section {{ background: {t.panel}; color: {t.fg}; padding: 4px 6px; border: none; border-bottom: 1px solid {t.grid}; }}
    QTableWidget, QTreeWidget, QListWidget, QTextBrowser, QPlainTextEdit {{ border: 1px solid {t.grid}; border-radius: 4px; }}
    QPushButton {{ padding: 4px 10px; }}
    QLabel[role="muted"] {{ color: {t.muted}; }}
    QLabel[role="error"] {{ color: {t.bad}; }}
    QLabel[role="ok"] {{ color: {t.good}; }}
    QLabel[role="warning"] {{ color: {t.warn}; }}
    """


def apply_theme(app: QApplication, mode: str = "system") -> Theme:
    """Apply Fusion with a light or dark palette (``mode`` is ``"system"``, ``"light"`` or ``"dark"``)."""
    global _current
    if mode == "system":
        try:
            dark = app.styleHints().colorScheme() == Qt.ColorScheme.Dark
        except AttributeError:  # Qt older than 6.5
            dark = False
    else:
        dark = mode == "dark"
    theme = DARK if dark else LIGHT
    app.setStyle("Fusion")
    app.setPalette(_palette(theme))
    app.setStyleSheet(_stylesheet(theme))
    _current = theme
    return theme
