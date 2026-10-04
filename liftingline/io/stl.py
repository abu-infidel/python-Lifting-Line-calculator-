"""Export the wing as a 3-D surface model in STL format (GUI-free).

The surface is built from the real airfoil coordinates of every wing section: at each spanwise station the
section's airfoil (blended linearly between neighbouring sections, as in the analysis) is scaled to the local chord,
turned nose-up by the local twist (plus the root incidence, optionally) about its quarter-chord point, and placed on
the quarter-chord line with its sweep and dihedral. Consecutive stations are joined by triangles; the tips (and the
root of a half model) are closed with caps, so the mesh is **watertight** with **outward normals** - ready for CAD,
3-D printing, rendering or CFD meshing.

Frame: x aft (root leading edge at 0), y towards the right wing, z up - a right-handed frame, so the model is not
mirrored. STL files carry no units; the coordinates are written in the unit chosen (millimetres by default, the
convention of most CAD and slicer programs), times an optional model scale.

The trailing edge is closed (a thin trailing-edge gap is blended away over the rear of the airfoil), so every
section is a single closed loop. A tip with zero chord (an elliptic planform) ends in a point.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from ..airfoil.geometry import Airfoil, cosine_spacing
from ..core.wing import KIND_ELLIPTIC, Wing

#: length units of the exported coordinates, metres per unit
UNITS = {"mm": 1e-3, "cm": 1e-2, "m": 1.0, "in": 0.0254, "ft": 0.3048}
DEFAULT_UNIT = "mm"


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


# =============================================================================== mesh building
def _stations(wing: Wing, n_span: int) -> np.ndarray:
    """Right-half stations from root to tip: every section plus subdivisions (denser towards an elliptic tip)."""
    hs = wing.half_span
    if wing.kind == KIND_ELLIPTIC:
        k = np.arange(n_span + 1)
        return hs * np.sin(0.5 * math.pi * k / n_span)
    ys = np.array([s.y for s in wing.sections], dtype=float)
    out = [ys[0]]
    for y0, y1 in zip(ys[:-1], ys[1:]):
        m = max(2, int(math.ceil(n_span * (y1 - y0) / hs)))
        out += list(np.linspace(y0, y1, m + 1)[1:])
    return np.array(out)


def _section_shape(af: Airfoil, frac: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(x, z_upper, z_lower) of an airfoil at chord fractions ``frac`` (0 nose, 1 trailing edge), closed TE."""
    x0 = float(af.coords[:, 0].min())
    x = x0 + (1.0 - x0) * frac
    zu, zl = np.asarray(af.y_upper(x), dtype=float), np.asarray(af.y_lower(x), dtype=float)
    nose = 0.5 * (zu[0] + zl[0])
    zu[0] = zl[0] = nose
    mid = 0.5 * (zu[-1] + zl[-1])
    ramp = frac**2  # blend a trailing-edge gap away over the rear of the section
    zu = zu - (zu[-1] - mid) * ramp
    zl = zl - (zl[-1] - mid) * ramp
    zu[-1] = zl[-1] = mid
    return x, zu, zl


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
    n_chord: int = 80,
    n_span: int = 40,
    half: bool = False,
    include_incidence: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Vertices (V, 3) in metres and triangles (F, 3) of the wing surface.

    ``airfoils`` has one entry per wing section. ``n_chord`` is the number of chordwise intervals per surface,
    ``n_span`` the approximate number of spanwise intervals per half-wing. ``half`` builds the right half only, closed
    at the root (a half-span model)."""
    if len(airfoils) != len(wing.sections):
        raise StlError("need one airfoil per wing section")
    if not 8 <= n_chord <= 400 or not 4 <= n_span <= 400:
        raise StlError("the mesh resolution must be 8-400 chordwise and 4-400 spanwise intervals")
    frac = cosine_spacing(n_chord)  # n_chord + 1 points from the nose (0) to the trailing edge (1)
    shapes = [_section_shape(af, frac) for af in airfoils]

    ys_right = _stations(wing, n_span)
    ys = ys_right if half else np.concatenate([-ys_right[:0:-1], ys_right])
    i0, i1, xi = wing.airfoil_blend(ys)
    chord, x_c4, z_c4 = wing.chord(ys), wing.x_c4(ys), wing.z_c4(ys)
    theta = np.radians(wing.twist_deg(ys) + (wing.incidence_deg if include_incidence else 0.0))

    root_chord = wing.root_chord
    rings: list[np.ndarray | None] = []  # 3-D loops, or None for a station that is a single point (zero chord)
    points: list[np.ndarray] = []
    for j, y in enumerate(ys):
        c = float(chord[j])
        if c <= 1e-9 * root_chord:
            rings.append(None)
            points.append(np.array([x_c4[j], y, z_c4[j]]))
            continue
        a, b, w = int(i0[j]), int(i1[j]), float(xi[j])
        xa = (1.0 - w) * shapes[a][0] + w * shapes[b][0]
        zu = (1.0 - w) * shapes[a][1] + w * shapes[b][1]
        zl = (1.0 - w) * shapes[a][2] + w * shapes[b][2]
        loop = _ring2d(xa, zu, zl)
        dx, dz = (loop[:, 0] - 0.25) * c, loop[:, 1] * c
        ct, st = math.cos(theta[j]), math.sin(theta[j])  # nose-up rotation about the quarter chord
        X = x_c4[j] + dx * ct + dz * st
        Z = z_c4[j] - dx * st + dz * ct
        rings.append(np.column_stack([X, np.full_like(X, y), Z]))
        points.append(np.empty(0))

    m = 2 * n_chord  # points per ring
    verts: list[np.ndarray] = []
    index: list[np.ndarray | int] = []
    count = 0
    for ring, pt in zip(rings, points):
        if ring is None:
            verts.append(pt[None, :])
            index.append(count)
            count += 1
        else:
            verts.append(ring)
            index.append(np.arange(count, count + m))
            count += m
    V = np.vstack(verts)

    side: list[tuple[int, int, int]] = []
    for j in range(len(rings) - 1):
        r0, r1 = index[j], index[j + 1]
        if isinstance(r0, int) and isinstance(r1, int):
            raise StlError("two consecutive stations have zero chord")
        for k in range(m):
            k1 = (k + 1) % m
            if isinstance(r1, int):  # the next station is a point (zero-chord tip)
                side.append((r0[k], r0[k1], r1))
            elif isinstance(r0, int):
                side.append((r0, r1[k1], r1[k]))
            else:
                side.append((r0[k], r0[k1], r1[k1]))
                side.append((r0[k], r1[k1], r1[k]))
    F_side = np.array(side, dtype=np.int64)

    def cap(ring_idx: np.ndarray, outward_y: float) -> np.ndarray:
        n = n_chord
        up = [int(ring_idx[n - i]) for i in range(n + 1)]  # up[i]: upper surface at chord index i (up[0] nose, up[n] TE)
        lo = [int(ring_idx[n + i]) if i < n else int(ring_idx[0]) for i in range(n + 1)]
        tris = [(up[0], up[1], lo[1])]
        for i in range(1, n - 1):
            tris += [(up[i], up[i + 1], lo[i + 1]), (up[i], lo[i + 1], lo[i])]
        tris.append((up[n - 1], up[n], lo[n - 1]))
        tris = np.array(tris, dtype=np.int64)
        nrm = np.cross(V[tris[:, 1]] - V[tris[:, 0]], V[tris[:, 2]] - V[tris[:, 0]])
        if float(np.sum(nrm[:, 1])) * outward_y < 0.0:
            tris = tris[:, ::-1]
        return tris

    caps = []
    if not isinstance(index[0], int):
        caps.append(cap(index[0], -1.0))
    if not isinstance(index[-1], int):
        caps.append(cap(index[-1], 1.0))
    # Orient the side surface outwards: on the upper surface, around mid-chord, the outward normal points up.
    k_mid = n_chord // 2  # ring index of the upper surface at mid-chord
    mid_ids = np.array([int(r[k_mid]) for r in index if not isinstance(r, int)])
    tri = F_side[np.isin(F_side[:, 0], mid_ids)]
    nz = np.cross(V[tri[:, 1]] - V[tri[:, 0]], V[tri[:, 2]] - V[tri[:, 0]])[:, 2]
    if float(np.sum(nz)) < 0.0:
        F_side = F_side[:, ::-1]
    F = np.vstack([F_side] + caps) if caps else F_side
    return V, F


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
    e = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    if np.any(e[:, 0] == e[:, 1]):
        return False
    directed = {tuple(x) for x in e.tolist()}
    if len(directed) != len(e):  # an edge traversed twice in the same direction: inconsistent orientation
        return False
    return all((b, a) in directed for a, b in directed)


# =============================================================================== file output
def write_stl(path: str | Path, V: np.ndarray, F: np.ndarray, *, factor: float = 1.0, name: str = "wing",
              ascii: bool = False) -> None:
    """Write triangles ``F`` of vertices ``V`` (scaled by ``factor``) as a binary (default) or ASCII STL file."""
    P = np.asarray(V, dtype=float) * factor
    tri = P[F]
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    length = np.linalg.norm(nrm, axis=1, keepdims=True)
    nrm = np.divide(nrm, length, out=np.zeros_like(nrm), where=length > 0.0)
    path = Path(path)
    if ascii:
        safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name) or "wing"
        lines = [f"solid {safe}"]
        for n, t in zip(nrm, tri):
            lines.append(f"  facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}")
            lines.append("    outer loop")
            for v in t:
                lines.append(f"      vertex {v[0]:.6e} {v[1]:.6e} {v[2]:.6e}")
            lines.append("    endloop")
            lines.append("  endfacet")
        lines.append(f"endsolid {safe}")
        path.write_text("\n".join(lines) + "\n", encoding="ascii")
        return
    rec = np.zeros(len(F), dtype=[("n", "<f4", (3,)), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    rec["n"], rec["v"] = nrm, tri
    header = f"Lifting-line wing calculator - {name}".encode("ascii", "replace")[:80].ljust(80, b" ")
    with path.open("wb") as fh:
        fh.write(header)
        fh.write(struct.pack("<I", len(F)))
        fh.write(rec.tobytes())


def read_stl(path: str | Path) -> np.ndarray:
    """Triangles (F, 3, 3) of a binary or ASCII STL file (used to check exports)."""
    data = Path(path).read_bytes()
    if len(data) >= 84:
        n = struct.unpack("<I", data[80:84])[0]
        if len(data) == 84 + 50 * n:
            rec = np.frombuffer(data[84:], dtype=[("n", "<f4", (3,)), ("v", "<f4", (3, 3)), ("attr", "<u2")], count=n)
            return rec["v"].astype(float)
    verts = [list(map(float, line.split()[1:4])) for line in data.decode("ascii").splitlines() if line.strip().startswith("vertex")]
    return np.array(verts, dtype=float).reshape(-1, 3, 3)


def export_wing_stl(
    path: str | Path,
    wing: Wing,
    airfoils: Sequence[Airfoil],
    *,
    unit: str = DEFAULT_UNIT,
    scale: float = 1.0,
    n_chord: int = 80,
    n_span: int = 40,
    half: bool = False,
    include_incidence: bool = True,
    ascii: bool = False,
) -> MeshInfo:
    """Build the wing surface and write it as STL in ``unit`` (``mm``, ``cm``, ``m``, ``in``, ``ft``) times ``scale``."""
    if unit not in UNITS:
        raise StlError(f"unknown unit {unit!r}; choose one of {', '.join(UNITS)}")
    if not (scale > 0.0) or not math.isfinite(scale):
        raise StlError("the model scale must be a positive number")
    V, F = wing_mesh(wing, airfoils, n_chord=n_chord, n_span=n_span, half=half, include_incidence=include_incidence)
    write_stl(path, V, F, factor=scale / UNITS[unit], name=wing.name, ascii=ascii)
    lo, hi = V.min(axis=0), V.max(axis=0)
    return MeshInfo(
        n_vertices=len(V), n_triangles=len(F), volume_m3=signed_volume(V, F), area_m2=surface_area(V, F),
        bounds_m=(tuple(float(v) for v in lo), tuple(float(v) for v in hi)), watertight=is_watertight(F),
    )


__all__ = ["UNITS", "DEFAULT_UNIT", "MeshInfo", "StlError", "export_wing_stl", "wing_mesh", "write_stl", "read_stl",
           "signed_volume", "surface_area", "is_watertight"]
