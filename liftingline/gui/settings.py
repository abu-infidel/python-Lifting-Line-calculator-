"""Persistent user settings (library folder, units, theme, window layout, recent files)."""

from __future__ import annotations

from PySide6.QtCore import QByteArray, QSettings

ORG, APP = "LiftingLineCalculator", "WingCalculator"
MAX_RECENT = 8


class Settings:
    """A thin typed wrapper around :class:`QSettings`. Pass ``path`` (an ``.ini`` file) to keep tests isolated."""

    def __init__(self, path: str | None = None) -> None:
        self._q = QSettings(path, QSettings.IniFormat) if path else QSettings(ORG, APP)

    def get(self, key: str, default=None):
        return self._q.value(key, default)

    def set(self, key: str, value) -> None:
        self._q.setValue(key, value)

    # -------------------------------------------------------------------- typed access
    def get_str(self, key: str, default: str = "") -> str:
        v = self._q.value(key, default)
        return default if v is None else str(v)

    def get_bool(self, key: str, default: bool = False) -> bool:
        v = self._q.value(key, default)
        if isinstance(v, str):
            return v.lower() in ("1", "true", "yes", "on")
        return bool(v)

    def get_bytes(self, key: str) -> QByteArray | None:
        v = self._q.value(key)
        return v if isinstance(v, QByteArray) and not v.isEmpty() else None

    # ---------------------------------------------------------------------- recents
    def recent_files(self) -> list[str]:
        v = self._q.value("files/recent", [])
        if isinstance(v, str):
            v = [v]
        return [str(p) for p in (v or [])]

    def add_recent(self, path: str) -> None:
        items = [p for p in self.recent_files() if p != path]
        items.insert(0, path)
        self._q.setValue("files/recent", items[:MAX_RECENT])

    def sync(self) -> None:
        self._q.sync()
