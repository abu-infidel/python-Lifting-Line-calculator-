"""Turn an airfoil *key* (what the user typed or picked) into geometry.

Keys are tried in this order:

1. airfoils registered under that key (e.g. a ``.dat`` the user loaded in the GUI),
2. ``file:<path>`` or an existing file path,
3. a file in the library by name (``naca63-412``, ``e387`` ...),
4. a NACA designation: 4- and 5-digit codes are generated exactly; 6/6A-series codes are looked up in
   the library by canonical key (exact ordinates) and only if absent fall back to an *approximate* section,
   which is flagged on the result.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .dat_io import DatParseError, load_dat
from .designation import DesignationError, NacaCode, parse_naca
from .geometry import Airfoil
from .library import AirfoilLibrary, LibraryEntry
from .naca4 import naca4_airfoil
from .naca5 import naca5_airfoil
from .naca6 import naca6_fallback_airfoil


class AirfoilNotFound(ValueError):
    """The key is not a NACA code, a file or a library entry."""


@dataclass(frozen=True)
class Resolution:
    key: str
    airfoil: Airfoil
    source: str  # human-readable origin
    approximate: bool = False
    candidates: tuple[LibraryEntry, ...] = ()  # equally good library matches (when ambiguous)
    note: str = ""  # something the user should know (approximation, ambiguity ...)
    code: NacaCode | None = None


class AirfoilResolver:
    def __init__(self, library: AirfoilLibrary | None = None) -> None:
        self.library = library
        self._registered: dict[str, Resolution] = {}
        self._cache: dict[str, Resolution] = {}

    def set_library(self, library: AirfoilLibrary | None) -> None:
        self.library = library
        self._cache.clear()

    def register(self, key: str, airfoil: Airfoil, source: str = "loaded by the user") -> None:
        self._registered[key] = Resolution(key, airfoil, source)
        self._cache.pop(key, None)

    # ------------------------------------------------------------------------------
    def resolve(self, key: str) -> Resolution:
        key = key.strip()
        if not key:
            raise AirfoilNotFound("no airfoil given")
        if key in self._registered:
            return self._registered[key]
        if key not in self._cache:
            self._cache[key] = self._resolve(key)
        return self._cache[key]

    def _resolve(self, key: str) -> Resolution:
        path = Path(key[5:]) if key.lower().startswith("file:") else Path(key)
        if (key.lower().startswith("file:") or os.sep in key or key.lower().endswith(".dat")) and path.is_file():
            return self._from_file(key, path)
        if self.library is not None:
            entry = self.library.by_name(key)
            if entry is not None:
                return self._from_entry(key, entry)
        try:
            code = parse_naca(key)
        except DesignationError:
            code = None
        if code is not None:
            return self._from_code(key, code)
        hint = ""
        if self.library is not None:
            close = [e.name for e in self.library.search(key, 5)]
            if close:
                hint = f" Closest library entries: {', '.join(close)}."
        raise AirfoilNotFound(
            f"cannot find airfoil {key!r}: it is not a NACA designation, not a file and not in the library.{hint}"
        )

    # ------------------------------------------------------------------- sources
    def _from_file(self, key: str, path: Path) -> Resolution:
        try:
            return Resolution(key, load_dat(path), f"file {path.name}")
        except DatParseError as exc:
            raise AirfoilNotFound(f"cannot read airfoil file {path}: {exc}") from exc

    def _from_entry(self, key: str, entry: LibraryEntry) -> Resolution:
        assert self.library is not None
        try:
            return Resolution(key, self.library.load(entry), f"library file {entry.path.name}")
        except DatParseError as exc:
            raise AirfoilNotFound(f"cannot read library file {entry.path}: {exc}") from exc

    def _from_code(self, key: str, code: NacaCode) -> Resolution:
        if code.family == "4":
            return Resolution(key, naca4_airfoil(code), "analytic NACA 4-digit (exact)", code=code)
        if code.family == "5":
            return Resolution(key, naca5_airfoil(code), "analytic NACA 5-digit (exact)", code=code)
        # 6 / 6A series: exact ordinates only come from the library
        if self.library is not None:
            entry, ties = self.library.resolve_best(code)
            if entry is not None:
                note = ""
                if len(ties) > 1:
                    note = (
                        f"{len(ties)} equally good library files match {code.name} "
                        f"({', '.join(e.path.name for e in ties)}); using {entry.path.name}"
                    )
                try:
                    af = self.library.load(entry)
                except DatParseError as exc:
                    raise AirfoilNotFound(f"cannot read library file {entry.path}: {exc}") from exc
                return Resolution(key, af.with_name(code.name), f"library file {entry.path.name}", False, tuple(ties), note, code)
        af = naca6_fallback_airfoil(code)
        where = "was not found in the airfoil library" if self.library is not None else "needs the airfoil library, and none has been chosen"
        return Resolution(
            key,
            af,
            "approximate 6-series (exact mean line, 4-digit thickness)",
            True,
            (),
            f"{code.name} {where}: using an APPROXIMATE section. Lift and moment follow the exact mean line, "
            "but drag is unreliable; select the airfoil's .dat file for exact results.",
            code,
        )
