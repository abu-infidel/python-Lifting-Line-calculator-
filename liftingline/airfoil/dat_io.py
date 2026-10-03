"""Reading and writing airfoil ``.dat`` coordinate files (UIUC / Selig database formats).

Formats handled (auto-detected):

* **Selig**  - a name line, then one list of points from the upper-surface trailing edge
  around the leading edge to the lower-surface trailing edge.
* **Lednicer** - a name line, a count line such as ``17. 17.``, then the upper surface and the
  lower surface, both from leading edge to trailing edge (blank lines between blocks).
* **two blocks without counts** - like Lednicer, but without the count line (detected from the
  x coordinate jumping back from the trailing edge to the leading edge).

Tolerated quirks: several header lines, CRLF line endings, tabs/commas/semicolons, Fortran ``D``
exponents, trailing text after the numbers, non-UTF-8 bytes, percent-of-chord units.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .geometry import Airfoil, GeometryError

_SPLIT = re.compile(r"[,\s;]+")


class DatParseError(ValueError):
    """The file does not contain usable airfoil coordinates."""


@dataclass
class ParsedDat:
    name: str
    xy: np.ndarray  # raw points in Selig order (not yet normalised)
    fmt: str  # "selig" | "lednicer" | "lednicer-nocount"
    warnings: list[str] = field(default_factory=list)


def _to_float(token: str) -> float | None:
    t = token.replace("D", "E").replace("d", "e")
    try:
        return float(t)
    except ValueError:
        return None


def _data_pair(line: str) -> tuple[float, float] | None:
    """(x, y) when the line starts with two numbers, else None."""
    tokens = [t for t in _SPLIT.split(line.strip()) if t]
    if len(tokens) < 2:
        return None
    a, b = _to_float(tokens[0]), _to_float(tokens[1])
    if a is None or b is None or not (np.isfinite(a) and np.isfinite(b)):
        return None
    return a, b


def _is_count_line(pair: tuple[float, float]) -> bool:
    a, b = pair
    return a >= 3 and b >= 3 and abs(a - round(a)) < 1e-9 and abs(b - round(b)) < 1e-9


def _decode(raw: bytes) -> str:
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def parse_dat_text(text: str, fallback_name: str = "") -> ParsedDat:
    """Parse the text of a ``.dat`` file (see module docstring for the supported layouts)."""
    warnings: list[str] = []
    header: list[str] = []
    entries: list[tuple[float, float] | None] = []  # None marks a blank/ignored line inside the data
    started = False
    for line in text.splitlines():
        stripped = line.strip()
        pair = _data_pair(stripped) if stripped else None
        if not started:
            if pair is not None:
                started = True
                entries.append(pair)
            elif stripped:
                header.append(stripped)
            continue
        if pair is not None:
            entries.append(pair)
        elif not stripped:
            entries.append(None)
        else:
            warnings.append(f"ignored non-numeric line: {stripped[:40]!r}")
    name = header[0] if header else fallback_name

    nums = [e for e in entries if e is not None]
    if not nums:
        raise DatParseError("no numeric coordinate lines found")

    # --- Lednicer with a count line -------------------------------------------------
    if _is_count_line(nums[0]):
        n_up, n_lo = int(round(nums[0][0])), int(round(nums[0][1]))
        body = nums[1:]
        if abs(len(body) - (n_up + n_lo)) <= 2 and len(body) >= n_up:
            upper = np.array(body[:n_up], dtype=float)
            lower = np.array(body[n_up : n_up + n_lo], dtype=float)
            if len(body) != n_up + n_lo:
                warnings.append(f"point count {len(body)} differs from header {n_up}+{n_lo}")
            return ParsedDat(name, _lednicer_to_selig(upper, lower), "lednicer", warnings)
        warnings.append("first line looks like a point count but does not match the data; treated as coordinates")

    pts = np.array(nums, dtype=float)

    # --- two blocks both running leading edge -> trailing edge (no count line) -------
    span = float(np.ptp(pts[:, 0])) or 1.0
    jumps = np.where(np.diff(pts[:, 0]) < -0.5 * span)[0]
    if len(jumps) == 1:
        k = int(jumps[0]) + 1
        first, second = pts[:k], pts[k:]
        if len(first) >= 4 and len(second) >= 4 and first[0, 0] < first[-1, 0] and second[0, 0] < second[-1, 0]:
            blocks = sorted((first, second), key=lambda b: -float(np.mean(b[:, 1])))
            return ParsedDat(name, _lednicer_to_selig(blocks[0], blocks[1]), "lednicer-nocount", warnings)

    return ParsedDat(name, pts, "selig", warnings)


def _lednicer_to_selig(upper: np.ndarray, lower: np.ndarray) -> np.ndarray:
    """Upper and lower surface (both LE -> TE) to one Selig-ordered list."""
    if len(lower) and len(upper) and np.hypot(*(upper[0] - lower[0])) < 1e-9:
        lower = lower[1:]  # shared leading-edge point
    return np.vstack([upper[::-1], lower])


def airfoil_from_text(text: str, fallback_name: str = "airfoil", source: str = "") -> Airfoil:
    parsed = parse_dat_text(text, fallback_name)
    notes = list(parsed.warnings)
    if parsed.fmt != "selig":
        notes.insert(0, f"read as {parsed.fmt} format")
    try:
        return Airfoil.from_raw(parsed.name or fallback_name, parsed.xy, source=source, notes=tuple(notes))
    except GeometryError as exc:
        raise DatParseError(str(exc)) from exc


def load_dat(path: str | Path) -> Airfoil:
    """Read an airfoil ``.dat`` file; raises :class:`DatParseError` with a readable message."""
    p = Path(path)
    try:
        text = _decode(p.read_bytes())
    except OSError as exc:
        raise DatParseError(f"cannot read {p}: {exc}") from exc
    return airfoil_from_text(text, fallback_name=p.stem, source=str(p))


def read_header(path: str | Path, max_bytes: int = 512) -> str:
    """First non-empty line of a file (cheap; used when indexing a large library)."""
    try:
        with open(path, "rb") as fh:
            chunk = fh.read(max_bytes)
    except OSError:
        return ""
    for line in _decode(chunk).splitlines():
        if line.strip():
            return line.strip()
    return ""


def to_dat_text(airfoil: Airfoil, precision: int = 6) -> str:
    """Selig-format text for an airfoil."""
    lines = [airfoil.name or "airfoil"]
    fmt = f"{{:.{precision}f}} {{:.{precision}f}}"
    lines += [fmt.format(x, y) for x, y in airfoil.coords]
    return "\n".join(lines) + "\n"


def save_dat(airfoil: Airfoil, path: str | Path) -> None:
    Path(path).write_text(to_dat_text(airfoil), encoding="utf-8")
