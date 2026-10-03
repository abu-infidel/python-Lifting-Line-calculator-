"""Import a measured or XFOIL polar as the section data of an airfoil.

Reads XFOIL ``polar`` output files and simple CSV/whitespace tables with an ``alpha`` column (degrees)
and ``CL`` / ``CD`` (optionally ``CM``) columns, in any order and with any header names that contain
those words. The polar is taken as Reynolds-number independent (it is for the Re it was made at);
outside its angle-of-attack range the end values are held. PCHIP interpolation keeps sparse data
from wiggling.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator

from .section_base import PolarTable, TabulatedAirfoilAero

_NUM = re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)([eEdD][-+]?\d+)?$")


class PolarParseError(ValueError):
    """The file does not contain a usable polar."""


def _split(line: str) -> list[str]:
    return [t for t in re.split(r"[,\s;|]+", line.strip()) if t]


def _is_number(token: str) -> bool:
    return bool(_NUM.match(token))


def parse_polar_text(text: str) -> tuple[PolarTable, float | None]:
    """Parse polar text; returns the table and the Reynolds number found in the header (if any)."""
    lines = text.splitlines()
    re_number = None
    for line in lines[:40]:
        m = re.search(r"Re\s*=?\s*([\d.]+)\s*e\s*([+-]?\d+)", line, re.I)
        if m:
            re_number = float(m.group(1)) * 10.0 ** int(m.group(2))
            break
    header_idx, header = None, None
    for i, line in enumerate(lines):
        toks = [t.lower() for t in _split(line)]
        if "alpha" in toks and "cl" in toks:
            header_idx, header = i, toks
            break
    if header is None:
        raise PolarParseError("no header line with 'alpha' and 'CL' columns found")
    try:
        col_a, col_cl = header.index("alpha"), header.index("cl")
    except ValueError as exc:  # pragma: no cover - guarded above
        raise PolarParseError(str(exc)) from exc
    col_cd = header.index("cd") if "cd" in header else None
    col_cm = header.index("cm") if "cm" in header else None
    if col_cd is None:
        raise PolarParseError("the polar needs a CD column")
    rows = []
    for line in lines[header_idx + 1 :]:
        toks = _split(line)
        if len(toks) <= max(col_a, col_cl, col_cd) or not all(_is_number(toks[c]) for c in (col_a, col_cl, col_cd)):
            continue
        cm = float(toks[col_cm].replace("D", "E")) if col_cm is not None and len(toks) > col_cm and _is_number(toks[col_cm]) else 0.0
        rows.append((float(toks[col_a]), float(toks[col_cl]), float(toks[col_cd].replace("D", "E")), cm))
    if len(rows) < 5:
        raise PolarParseError(f"only {len(rows)} data rows found; need at least 5")
    data = np.array(sorted(rows))
    keep = np.r_[True, np.diff(data[:, 0]) > 1e-9]
    data = data[keep]
    return PolarTable(alpha_deg=data[:, 0], cl=data[:, 1], cd=np.maximum(data[:, 2], 1e-5), cm=data[:, 3]), re_number


class TabulatedPolarAero(TabulatedAirfoilAero):
    """Section data from an imported polar (Reynolds-number independent)."""

    _interp = staticmethod(PchipInterpolator)

    def __init__(self, table: PolarTable, label: str = "imported polar", re_number: float | None = None) -> None:
        super().__init__()
        self._table = table
        self.label = label
        self.re_number = re_number
        self.approximate = False

    def _build_table(self, re_node: float) -> PolarTable:
        return self._table


def load_polar(path: str | Path) -> TabulatedPolarAero:
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise PolarParseError(f"cannot read {p}: {exc}") from exc
    table, re_number = parse_polar_text(text)
    extra = f", Re={re_number:.2g}" if re_number else ""
    return TabulatedPolarAero(table, label=f"polar {p.name}{extra}", re_number=re_number)
