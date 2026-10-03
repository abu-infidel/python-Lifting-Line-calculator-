"""A folder of airfoil ``.dat`` files (e.g. the UIUC coordinate database) as a searchable library.

* :meth:`AirfoilLibrary.scan` indexes a folder recursively, reading only the first line of each
  file, so even ~1600 files index in well under a second.
* :meth:`AirfoilLibrary.resolve` finds the file for a NACA designation by comparing *canonical
  keys* (series, A-flag, design lift coefficient, thickness) derived from every file name and header
  line, not by comparing strings, so ``63-412``, ``63412`` and ``632-412`` all find ``naca63412.dat``
  whatever the library's naming habits are.
* :meth:`AirfoilLibrary.health_report` parses every file once and summarises formats, repairs and
  failures, so problems with a particular library are visible instead of silent.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from .dat_io import DatParseError, _decode, airfoil_from_text, parse_dat_text, read_header
from .designation import NacaCode, extract_codes, try_parse_naca
from .geometry import Airfoil, GeometryError

_EXTENSIONS = {".dat", ".txt"}

Progress = Callable[[int, int], None]


@dataclass(frozen=True)
class LibraryEntry:
    name: str  # file name without extension
    path: Path
    header: str  # first non-empty line (usually the airfoil name)
    codes: tuple[NacaCode, ...] = ()  # NACA designations recognised in the name / header

    @property
    def label(self) -> str:
        """Display text: the file name, plus the header when it adds information."""
        h = " ".join(self.header.split())
        return self.name if not h or h.lower() == self.name.lower() else f"{self.name}  -  {h}"


def match_score(target: NacaCode, candidate: NacaCode) -> float | None:
    """How well ``candidate`` matches ``target`` (higher is better); ``None`` if it is a different airfoil.

    The designation key (series, A-flag, design cl, thickness) must agree. The optional low-drag-range
    subscript and the mean-line parameter ``a`` only rank the candidates: matching values score
    higher, a missing value on one side is neutral and conflicting values rank last.
    """
    if target.key != candidate.key:
        return None
    score = 0.0
    if target.sub is not None and candidate.sub is not None:
        score += 2.0 if target.sub == candidate.sub else -1.0
    elif target.sub is None and candidate.sub is None:
        score += 1.0
    if target.a is not None and candidate.a is not None:
        score += 1.0 if abs(target.a - candidate.a) < 1e-6 else -1.0
    return score


@dataclass
class HealthReport:
    total: int = 0
    parsed_ok: int = 0
    failures: list[tuple[Path, str]] = field(default_factory=list)
    formats: dict[str, int] = field(default_factory=dict)
    repaired: int = 0  # files that needed normalisation (rotation, scaling, reordering ...)
    with_naca_code: int = 0

    def to_text(self, max_failures: int = 50) -> str:
        lines = [
            f"files indexed        : {self.total}",
            f"parsed successfully  : {self.parsed_ok}",
            f"failed               : {len(self.failures)}",
            "coordinate formats   : " + ", ".join(f"{k} {v}" for k, v in sorted(self.formats.items())),
            f"normalised/repaired  : {self.repaired}",
            f"NACA designation seen: {self.with_naca_code}",
        ]
        if self.failures:
            lines.append("failures:")
            for path, msg in self.failures[:max_failures]:
                lines.append(f"  {path.name}: {msg}")
            if len(self.failures) > max_failures:
                lines.append(f"  ... and {len(self.failures) - max_failures} more")
        return "\n".join(lines)


class AirfoilLibrary:
    def __init__(self, entries: Iterable[LibraryEntry] = (), root: Path | None = None) -> None:
        self.entries: list[LibraryEntry] = sorted(entries, key=lambda e: (e.name.lower(), str(e.path)))
        self.root = root
        self._cache: dict[tuple[Path, float], Airfoil] = {}

    # -------------------------------------------------------------------- scanning
    @classmethod
    def scan(cls, folder: str | Path, *, progress: Progress | None = None) -> "AirfoilLibrary":
        root = Path(folder)
        if not root.is_dir():
            raise FileNotFoundError(f"airfoil library folder not found: {root}")
        paths = sorted(
            Path(dirpath, fn)
            for dirpath, _, files in os.walk(root)
            for fn in files
            if Path(fn).suffix.lower() in _EXTENSIONS
        )
        entries = []
        for i, path in enumerate(paths):
            header = read_header(path)
            codes: list[NacaCode] = []
            for text in (path.stem, header):
                for c in extract_codes(text):
                    if c not in codes:
                        codes.append(c)
            entries.append(LibraryEntry(path.stem, path, header, tuple(codes)))
            if progress is not None and (i % 50 == 0 or i == len(paths) - 1):
                progress(i + 1, len(paths))
        return cls(entries, root)

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries)

    # ----------------------------------------------------------------------- lookup
    def by_name(self, name: str) -> LibraryEntry | None:
        key = name.strip().lower()
        if key.endswith(tuple(_EXTENSIONS)):
            key = key.rsplit(".", 1)[0]
        for e in self.entries:
            if e.name.lower() == key:
                return e
        return None

    def search(self, query: str, limit: int = 200) -> list[LibraryEntry]:
        """Entries whose name/header contain every word of ``query`` (case-insensitive).

        If the query is a NACA designation, the entries it resolves to are listed first.
        """
        q = query.strip().lower()
        if not q:
            return self.entries[:limit]
        results: list[LibraryEntry] = []
        code = try_parse_naca(query)
        if code is not None:
            results.extend(self.resolve(code))
        tokens = q.split()
        scored = []
        for e in self.entries:
            hay = f"{e.name} {e.header}".lower()
            if all(t in hay for t in tokens):
                name = e.name.lower()
                rank = 0 if name == q else 1 if name.startswith(q) else 2
                scored.append((rank, name, e))
        for _, _, e in sorted(scored, key=lambda s: (s[0], s[1])):
            if e not in results:
                results.append(e)
        return results[:limit]

    def resolve(self, code: NacaCode) -> list[LibraryEntry]:
        """Library entries for a NACA designation, best match first (empty if none)."""
        ranked: list[tuple[float, str, LibraryEntry]] = []
        for e in self.entries:
            scores = [s for c in e.codes if (s := match_score(code, c)) is not None]
            if scores:
                ranked.append((max(scores), e.name.lower(), e))
        ranked.sort(key=lambda r: (-r[0], r[1]))
        return [e for _, _, e in ranked]

    def resolve_best(self, code: NacaCode) -> tuple[LibraryEntry | None, list[LibraryEntry]]:
        """(chosen entry, ties). ``ties`` has more than one element when the choice is ambiguous
        (several equally good files) and the user should pick."""
        ranked: list[tuple[float, LibraryEntry]] = []
        for e in self.entries:
            scores = [s for c in e.codes if (s := match_score(code, c)) is not None]
            if scores:
                ranked.append((max(scores), e))
        if not ranked:
            return None, []
        top = max(s for s, _ in ranked)
        ties = [e for s, e in ranked if s == top]
        return ties[0], ties

    # ------------------------------------------------------------------------ loading
    def load(self, entry: LibraryEntry) -> Airfoil:
        """Parse and normalise a library file (cached; re-read if the file changes)."""
        try:
            mtime = entry.path.stat().st_mtime
        except OSError as exc:
            raise DatParseError(f"cannot read {entry.path}: {exc}") from exc
        key = (entry.path, mtime)
        if key not in self._cache:
            text = _decode(entry.path.read_bytes())
            self._cache[key] = airfoil_from_text(text, fallback_name=entry.name, source=str(entry.path))
        return self._cache[key]

    # ------------------------------------------------------------------------- health
    def health_report(self, progress: Progress | None = None) -> HealthReport:
        rep = HealthReport(total=len(self.entries))
        for i, e in enumerate(self.entries):
            try:
                text = _decode(e.path.read_bytes())
                parsed = parse_dat_text(text, e.name)
                af = Airfoil.from_raw(parsed.name or e.name, parsed.xy)
                rep.parsed_ok += 1
                rep.formats[parsed.fmt] = rep.formats.get(parsed.fmt, 0) + 1
                if af.notes or parsed.warnings:
                    rep.repaired += 1
                if e.codes:
                    rep.with_naca_code += 1
            except (DatParseError, GeometryError, ValueError, OSError) as exc:
                rep.failures.append((e.path, str(exc)))
            if progress is not None and (i % 25 == 0 or i == len(self.entries) - 1):
                progress(i + 1, len(self.entries))
        return rep
