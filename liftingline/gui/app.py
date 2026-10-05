"""Application entry point: ``python main.py`` calls :func:`run_gui`."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from .main_window import APP_TITLE, MainWindow
from .settings import APP, ORG, Settings
from .theme import apply_theme


def run_gui(argv: list[str] | None = None) -> int:
    app = QApplication.instance() or QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName(APP)
    app.setOrganizationName(ORG)
    app.setApplicationDisplayName(APP_TITLE)
    settings = Settings()
    apply_theme(app, settings.get_str("ui/theme", "system"))
    window = MainWindow(settings=settings)
    if window.prefer_maximized:  # small (or highly scaled) screens: use all of it, nothing off-screen
        window.showMaximized()
    else:
        window.show()
    return app.exec()
