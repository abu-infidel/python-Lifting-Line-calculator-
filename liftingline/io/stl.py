"""Wing surface meshes and STL files (GUI-free; used by the separate 3-D model exporter app).

The surface is built from the real airfoil coordinates of every wing section: at each spanwise station the
section's airfoil (blended linearly between neighbouring sections, as in the analysis) is scaled to the local chord,
turned nose-up by the local twist (plus the root incidence, optionally) about its quarter-chord point, and placed on
the quarter-chord line with its sweep and dihedral. Consecutive stations are joined by triangles; the tips (and the
root of a half model) are closed with caps, so the mesh is **watertight** with **outward normals** - ready for CAD,
3-D printing, rendering or CFD meshing.

Resolution: chordwise points are cosine-clustered at the leading and trailing edges; spanwise stations include every
wing section and are spaced no wider than a fraction of the *local* chord and of the semi-span, so narrow tips and
kinks are as finely resolved as the root. :data:`QUALITY` holds the presets (``fine`` is the default).

Frame: x aft (root leading edge at 0), y towards the right wing, z up - a right-handed frame, so the model is not
mirrored. STL files carry no units; the coordinates are written in the unit chosen (millimetres by default, the
convention of most CAD and slicer programs), times an optional model scale.

The trailing edge is closed (a thin trailing-edge gap is blended away over the rear of the airfoil), so every section
is a single closed loop. A tip with zero chord (an elliptic planform) ends in a point.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from ..airfoil.geometry import Airfoil, cosine_spacing
from ..core.sections3d import SectionShapeError, blended_shape, closed_section, place
from ..core.wing import KIND_ELLIPTIC, Wing

#: length units of the exported coordinates, metres per unit
UNITS = {"mm": 1e-3, "cm": 1e-2, "m": 1.0, "in": 0.0254, "ft": 0.3048}
DEFAULT_UNIT = "mm"


@dataclass(frozen=True)
class MeshQuality:
    """Mesh resolution: chordwise intervals per surface, largest spanwise step (fractions of chord and semi-span)."""

    label: str
    n_chord: int
    step_chord: float  # spanwise step <= this fraction of the local chord ...
    step_semispan: float  # ... and <= this fraction of the semi-span


QUALITY: dict[str, MeshQuality] = {
    "coarse": MeshQuality("Coarse (quick look)", 40, 1.0, 0.05),
    "medium": MeshQuality("Medium", 80, 0.5, 0.02),
    "fine": MeshQuality("Fine (recommended)", 160, 0.25, 0.01),
    "very_fine": MeshQuality("Very fine (CFD / large prints)", 300, 0.1, 0.005),
}
DEFAULT_QUALITY = "fine"
MAX_STATIONS_PER_HALF = 2000


class StlError(ValueError):
    """The wing cannot be turned into a mesh (the message is meant for the user)."""


@dataclass(frozen=True)
class MeshInfo:
    """What was written (geometry in metres, before the unit and the model scale are applied)."""

    n_vertices: int
    n_triangles: int
    volume_m3: float
    area_m2: float
    bounds_m: tuple[tuple[float, float, float], tuple[float, float, float]]
    watertight: bool
    file_bytes: int = 0


# =============================================================================== mesh building
def spanwise_stations(wing: Wing, step_chord: float, step_semispan: float) -> np.ndarray:
    """Right-half stations from root to tip: every wing section, with steps no wider than ``step_chord`` x local
    chord and ``step_semispan`` x semi-span."""
    hs = wing.half_span
    if wing.kind == KIND_ELLIPTIC:  # y = hs sin(phi): dy = hs cos(phi) dphi follows the chord c0 cos(phi)
        dphi = min(step_chord * wing.root_chord / hs, step_semispan)
        n = min(max(int(math.ceil(0.5 * math.pi / dphi)), 8), MAX_STATIONS_PER_HALF)
        return hs * np.sin(0.5 * math.pi * np.arange(n + 1) / n)
    ys = np.array([s.y for s in wing.sections], dtype=float)
    floor = 0.02 * wing.root_chord  # a (nearly) pointed tip still gets a finite step
    out = [ys[0]]
    for y0, y1 in zip(ys[:-1], ys[1:]):
        c_min = max(min(float(wing.chord(y0)), float(wing.chord(y1))), floor)
        step = min(step_chord * c_min, step_semispan * hs)
        m = max(2, int(math.ceil((y1 - y0) / step)))
        out += list(np.linspace(y0, y1, m + 1)[1:])
    if len(out) - 1 > MAX_STATIONS_PER_HALF:
        raise StlError("the requested resolution needs too many spanwise stations; choose a coarser quality")
    return np.array(out)


def _section_shape(af: Airfoil, frac: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(x, z_upper, z_lower) of an airfoil at chord fractions ``frac``, closed trailing edge (shared with the views)."""
    try:
        return closed_section(af, frac)
    except SectionShapeError as exc:
        raise StlError(str(exc)) from exc


def _ring2d(x: np.ndarray, zu: np.ndarray, zl: np.ndarray) -> np.ndarray:
    """Closed loop of 2n points (n = chordwise intervals): TE, upper surface to the nose, lower surface back."""
    n = x.size - 1
    up = np.column_stack([x[::-1], zu[::-1]])[:n]  # TE, ..., the point just behind the nose
    nose = np.array([[x[0], zu[0]]])
    lo = np.column_stack([x[1:n], zl[1:n]])  # behind the nose, ..., the point just ahead of the TE
    return np.vstack([up, nose, lo])


def wing_mesh(
    wing: Wing,
    airfoils: Sequence[Airfoil],
    *,
    quality: str | MeshQuality = DEFAULT_QUALITY,
    half: bool = False,
    include_incidence: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Vertices (V, 3) in metres and triangles (F, 3) of the wing surface.

    ``airfoils`` has one entry per wing section. ``quality`` is a key of :data:`QUALITY` or a :class:`MeshQuality`.
    ``half`` builds the right half only, closed at the root (a half-span model)."""
    q = QUALITY[quality] if isinstance(quality, str) else quality
    if isinstance(quality, str) and quality not in QUALITY:
        raise StlError(f"unknown mesh quality {quality!r}")
    if len(airfoils) != len(wing.sections):
        raise StlError("need one airfoil per wing section")
    if not 8 <= q.n_chord <= 600 or not (q.step_chord > 0 and q.step_semispan > 0):
        raise StlError("the mesh resolution must be 8-600 chordwise intervals with positive spanwise steps")
    n = q.n_chord
    frac = cosine_spacing(n)  # n + 1 points from the nose (0) to the trailing edge (1)
    shapes = [_section_shape(af, frac) for af in airfoils]

    ys_right = spanwise_stations(wing, q.step_chord, q.step_semispan)
    ys = ys_right if half else np.concatenate([-ys_right[:0:-1], ys_right])
    chord, x_c4, z_c4 = wing.chord(ys), wing.x_c4(ys), wing.z_c4(ys)

    m = 2 * n  # points per ring
    point = chord <= 1e-9 * wing.root_chord  # zero-chord stations (elliptic tips) are single points
    if np.any(point[1:] & point[:-1]):
        raise StlError("two consecutive stations have zero chord")
    verts, start = [], np.zeros(len(ys), dtype=np.int64)
    count = 0
    for j, y in enumerate(ys):
        start[j] = count
        if point[j]:
            verts.append(np.array([[x_c4[j], y, z_c4[j]]]))
            count += 1
            continue
        loop = _ring2d(*blended_shape(wing, shapes, float(y)))
        X, Z = place(wing, float(y), loop[:, 0], loop[:, 1], include_incidence)  # the same placement as the views
        verts.append(np.column_stack([X, np.full(m, y), Z]))
        count += m
    V = np.vstack(verts)

    k = np.arange(m)
    k1 = (k + 1) % m
    side = []
    for j in range(len(ys) - 1):
        r0, r1 = start[j] + k, start[j + 1] + k
        if point[j + 1]:
            side.append(np.column_stack([r0, start[j] + k1, np.full(m, start[j + 1])]))
        elif point[j]:
            side.append(np.column_stack([np.full(m, start[j]), start[j + 1] + k1, r1]))
        else:
            r0k1, r1k1 = start[j] + k1, start[j + 1] + k1
            side.append(np.column_stack([r0, r0k1, r1k1]))
            side.append(np.column_stack([r0, r1k1, r1]))
    F_side = np.vstack(side)

    def cap(s0: int, outward_y: float) -> np.ndarray:
        up = np.array([s0 + n - i for i in range(n + 1)])  # up[i]: upper surface at chord index i (0 nose, n TE)
        lo = np.array([s0 + n + i if i < n else s0 for i in range(n + 1)])
        i = np.arange(1, n - 1)
        tris = np.vstack([
            [[up[0], up[1], lo[1]]],
            np.column_stack([up[i], up[i + 1], lo[i + 1]]),
            np.column_stack([up[i], lo[i + 1], lo[i]]),
            [[up[n - 1], up[n], lo[n - 1]]],
        ]).astype(np.int64)
        nrm = np.cross(V[tris[:, 1]] - V[tris[:, 0]], V[tris[:, 2]] - V[tris[:, 0]])
        return tris[:, ::-1] if float(np.sum(nrm[:, 1])) * outward_y < 0.0 else tris

    caps = []
    if not point[0]:
        caps.append(cap(int(start[0]), -1.0))
    if not point[-1]:
        caps.append(cap(int(start[-1]), 1.0))
    # Orient the side surface outwards: on the upper surface around mid-chord the outward normal points up.
    mid_ids = np.array([start[j] + n // 2 for j in range(len(ys)) if not point[j]])
    tri = F_side[np.isin(F_side[:, 0], mid_ids)]
    nz = np.cross(V[tri[:, 1]] - V[tri[:, 0]], V[tri[:, 2]] - V[tri[:, 0]])[:, 2]
    if float(np.sum(nz)) < 0.0:
        F_side = F_side[:, ::-1]
    return V, np.vstack([F_side] + caps)


def estimate_triangles(wing: Wing, quality: str = DEFAULT_QUALITY, half: bool = False) -> int:
    """Number of triangles :func:`wing_mesh` will produce (for a size preview, without building the mesh)."""
    q = QUALITY[quality]
    ys = spanwise_stations(wing, q.step_chord, q.step_semispan)
    point = wing.chord(ys) <= 1e-9 * wing.root_chord
    m = 2 * q.n_chord
    per_half = sum(m if (point[j] or point[j + 1]) else 2 * m for j in range(len(ys) - 1))
    cap = 2 * q.n_chord - 2
    caps = (0 if point[-1] else cap) * (1 if half else 2) + (cap if half and not point[0] else 0)
    return per_half * (1 if half else 2) + caps


# =============================================================================== mesh checks
def signed_volume(V: np.ndarray, F: np.ndarray) -> float:
    """Volume enclosed by a closed triangle mesh (positive for outward normals)."""
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def surface_area(V: np.ndarray, F: np.ndarray) -> float:
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    return float(0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum())


def is_watertight(F: np.ndarray) -> bool:
    """Every edge is shared by exactly two triangles that traverse it in opposite directions (closed, oriented)."""
    F = np.asarray(F, dtype=np.int64)
    e = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    if np.any(e[:, 0] == e[:, 1]):
        return False
    base = int(F.max()) + 1
    fwd = e[:, 0] * base + e[:, 1]
    if np.unique(fwd).size != fwd.size:  # an edge traversed twice in the same direction
        return False
    rev = e[:, 1] * base + e[:, 0]
    return bool(np.all(np.isin(rev, fwd)))


def triangle_areas(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)


# =============================================================================== file output
_RECORD = np.dtype([("n", "<f4", (3,)), ("v", "<f4", (3, 3)), ("attr", "<u2")])


def write_stl(path: str | Path, V: np.ndarray, F: np.ndarray, *, factor: float = 1.0, name: str = "wing",
              ascii: bool = False) -> int:
    """Write triangles ``F`` of vertices ``V`` (scaled by ``factor``) as binary (default) or ASCII STL; returns bytes."""
    P = np.asarray(V, dtype=float) * factor
    tri = P[F]
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    length = np.linalg.norm(nrm, axis=1, keepdims=True)
    nrm = np.divide(nrm, length, out=np.zeros_like(nrm), where=length > 0.0)
    path = Path(path)
    if ascii:
        safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name) or "wing"
        rows = np.concatenate([nrm[:, None, :], tri], axis=1).reshape(-1, 12)
        body = "".join(
            "  facet normal {:.6e} {:.6e} {:.6e}\n    outer loop\n"
            "      vertex {:.7e} {:.7e} {:.7e}\n      vertex {:.7e} {:.7e} {:.7e}\n      vertex {:.7e} {:.7e} {:.7e}\n"
            "    endloop\n  endfacet\n".format(*r) for r in rows.tolist()
        )
        path.write_text(f"solid {safe}\n{body}endsolid {safe}\n", encoding="ascii")
        return path.stat().st_size
    rec = np.zeros(len(F), dtype=_RECORD)
    rec["n"], rec["v"] = nrm, tri
    header = f"Lifting-line wing calculator - {name}".encode("ascii", "replace")[:80].ljust(80, b" ")
    with path.open("wb") as fh:
        fh.write(header)
        fh.write(struct.pack("<I", len(F)))
        fh.write(rec.tobytes())
    return path.stat().st_size


def read_stl(path: str | Path) -> np.ndarray:
    """Triangles (F, 3, 3) of a binary or ASCII STL file (used to check exports)."""
    data = Path(path).read_bytes()
    if len(data) >= 84:
        n = struct.unpack("<I", data[80:84])[0]
        if len(data) == 84 + 50 * n:
            return np.frombuffer(data[84:], dtype=_RECORD, count=n)["v"].astype(float)
    verts = [line.split()[1:4] for line in data.decode("ascii").splitlines() if line.strip().startswith("vertex")]
    return np.array(verts, dtype=float).reshape(-1, 3, 3)


def export_wing_stl(
    path: str | Path,
    wing: Wing,
    airfoils: Sequence[Airfoil],
    *,
    quality: str | MeshQuality = DEFAULT_QUALITY,
    unit: str = DEFAULT_UNIT,
    scale: float = 1.0,
    half: bool = False,
    include_incidence: bool = True,
    ascii: bool = False,
) -> MeshInfo:
    """Build the wing surface and write it as STL in ``unit`` (``mm``, ``cm``, ``m``, ``in``, ``ft``) times ``scale``."""
    if unit not in UNITS:
        raise StlError(f"unknown unit {unit!r}; choose one of {', '.join(UNITS)}")
    if not (scale > 0.0) or not math.isfinite(scale):
        raise StlError("the model scale must be a positive number")
    V, F = wing_mesh(wing, airfoils, quality=quality, half=half, include_incidence=include_incidence)
    size = write_stl(path, V, F, factor=scale / UNITS[unit], name=wing.name, ascii=ascii)
    lo, hi = V.min(axis=0), V.max(axis=0)
    return MeshInfo(
        n_vertices=len(V), n_triangles=len(F), volume_m3=signed_volume(V, F), area_m2=surface_area(V, F),
        bounds_m=(tuple(float(v) for v in lo), tuple(float(v) for v in hi)), watertight=is_watertight(F),
        file_bytes=size,
    )


__all__ = ["UNITS", "DEFAULT_UNIT", "QUALITY", "DEFAULT_QUALITY", "MeshQuality", "MeshInfo", "StlError",
           "export_wing_stl", "wing_mesh", "write_stl", "read_stl", "spanwise_stations", "estimate_triangles",
           "signed_volume", "surface_area", "is_watertight", "triangle_areas"]
