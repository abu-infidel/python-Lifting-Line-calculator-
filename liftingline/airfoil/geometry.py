"""Airfoil geometry: normalised coordinates, surface interpolation, camber and thickness.

Coordinates are kept in **Selig order** (upper-surface trailing edge -> leading edge ->
lower-surface trailing edge, i.e. counter-clockwise), scaled to unit chord with the
leading edge at the origin and the trailing-edge mid-point on the x axis.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
from scipy.interpolate import CubicSpline


def cosine_spacing(n: int) -> np.ndarray:
    """``n + 1`` stations from 0 to 1 clustered at both ends, x = (1 - cos(beta))/2."""
    beta = np.linspace(0.0, math.pi, n + 1)
    return 0.5 * (1.0 - np.cos(beta))


class GeometryError(ValueError):
    """The coordinates cannot describe an airfoil."""


def _signed_area(xy: np.ndarray) -> float:
    x, y = xy[:, 0], xy[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def normalize_coordinates(xy: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Bring raw airfoil coordinates to the canonical form; returns (coords, notes)."""
    pts = np.asarray(xy, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise GeometryError("airfoil coordinates must be an (n, 2) array")
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    notes: list[str] = []
    if len(pts) > 1:  # drop repeated consecutive points (e.g. a duplicated leading-edge point)
        keep = np.r_[True, np.hypot(*np.diff(pts, axis=0).T) > 1e-12]
        if not keep.all():
            notes.append(f"removed {int((~keep).sum())} duplicate point(s)")
        pts = pts[keep]
    if len(pts) < 8:
        raise GeometryError(f"too few points ({len(pts)}) to describe an airfoil")

    te = 0.5 * (pts[0] + pts[-1])
    i_le = int(np.argmax(np.hypot(*(pts - te).T)))
    le = pts[i_le]
    chord_vec = te - le
    chord = float(np.hypot(*chord_vec))
    if chord <= 0.0:
        raise GeometryError("degenerate airfoil (zero chord)")
    ang = math.atan2(chord_vec[1], chord_vec[0])
    c, s = math.cos(-ang), math.sin(-ang)
    rot = np.array([[c, -s], [s, c]])
    out = ((pts - le) @ rot.T) / chord
    if abs(math.degrees(ang)) > 0.05:
        notes.append(f"rotated by {math.degrees(-ang):+.2f} deg so the chord line lies along x")
    if abs(chord - 1.0) > 5e-3:
        notes.append(f"scaled from chord length {chord:.4g} to 1")

    i_le = int(np.argmin(out[:, 0]))
    upper_mean = float(np.mean(out[: i_le + 1, 1])) if i_le > 0 else 0.0
    lower_mean = float(np.mean(out[i_le:, 1])) if i_le < len(out) - 1 else 0.0
    if upper_mean < lower_mean - 1e-9:  # lower surface listed first
        out = out[::-1].copy()
        notes.append("reversed point order (lower surface came first)")
    if _signed_area(out) < 0.0:
        out = out[::-1].copy()
        notes.append("reversed point order (clockwise input)")
    out = out - out[int(np.argmin(out[:, 0]))]  # nose (min-x point) exactly at the origin
    return out, notes


@dataclass(frozen=True, eq=False)
class Airfoil:
    name: str
    coords: np.ndarray  # (n, 2), Selig order, unit chord, LE at the origin
    source: str = ""
    approximate: bool = False  # geometry is an approximation (flagged in the UI)
    notes: tuple[str, ...] = field(default_factory=tuple)

    # ---------------------------------------------------------------- construction
    @classmethod
    def from_raw(
        cls, name: str, xy: np.ndarray, *, source: str = "", approximate: bool = False, notes: tuple[str, ...] = ()
    ) -> "Airfoil":
        coords, extra = normalize_coordinates(xy)
        return cls(name=name, coords=coords, source=source, approximate=approximate, notes=tuple(notes) + tuple(extra))

    # ------------------------------------------------------------------- identity
    @cached_property
    def digest(self) -> str:
        """Stable hash of the geometry (used as a cache key)."""
        return hashlib.sha1(np.round(self.coords, 7).tobytes()).hexdigest()[:16]

    # ------------------------------------------------------------------- surfaces
    @cached_property
    def _split(self) -> tuple[np.ndarray, np.ndarray]:
        i_le = int(np.argmin(self.coords[:, 0]))
        upper = self.coords[: i_le + 1][::-1]
        lower = self.coords[i_le:]
        return upper, lower

    @property
    def upper(self) -> np.ndarray:
        """Upper surface, leading edge -> trailing edge, shape (n, 2)."""
        return self._split[0]

    @property
    def lower(self) -> np.ndarray:
        """Lower surface, leading edge -> trailing edge, shape (n, 2)."""
        return self._split[1]

    @cached_property
    def _x0(self) -> float:
        """x of the nose (leftmost point). 0 for normalised files; slightly negative for analytic
        cambered NACA sections, whose nominal leading edge lies just behind the nose."""
        return float(self.coords[:, 0].min())

    @staticmethod
    def _spline(surface: np.ndarray, x0: float) -> CubicSpline:
        order = np.argsort(surface[:, 0], kind="stable")
        x, y = surface[order, 0] - x0, surface[order, 1]
        keep = np.r_[True, np.diff(x) > 1e-10]
        x, y = np.clip(x[keep], 0.0, None), y[keep]
        return CubicSpline(np.sqrt(x), y, bc_type="natural")  # s = sqrt(x - x0) removes the LE singularity

    @cached_property
    def _splines(self) -> tuple[CubicSpline, CubicSpline]:
        return self._spline(self.upper, self._x0), self._spline(self.lower, self._x0)

    def _s(self, x) -> np.ndarray:
        return np.sqrt(np.clip(np.asarray(x, dtype=float) - self._x0, 0.0, None))

    def y_upper(self, x) -> np.ndarray:
        return self._splines[0](self._s(x))

    def y_lower(self, x) -> np.ndarray:
        return self._splines[1](self._s(x))

    def camber(self, x) -> np.ndarray:
        return 0.5 * (self.y_upper(x) + self.y_lower(x))

    def thickness(self, x) -> np.ndarray:
        return self.y_upper(x) - self.y_lower(x)

    def camber_slope(self, x) -> np.ndarray:
        """dz/dx of the mean camber line."""
        s = np.maximum(self._s(x), 1e-6)
        d = 0.5 * (self._splines[0](s, 1) + self._splines[1](s, 1))
        return d / (2.0 * s)

    # ----------------------------------------------------------------- properties
    @cached_property
    def _scan(self) -> dict[str, float]:
        x = np.linspace(0.0, 1.0, 2001)
        t, z = self.thickness(x), self.camber(x)
        i_t, i_z = int(np.argmax(t)), int(np.argmax(np.abs(z)))
        return dict(t=float(t[i_t]), x_t=float(x[i_t]), z=float(z[i_z]), x_z=float(x[i_z]))

    @property
    def t_over_c(self) -> float:
        """Maximum thickness / chord."""
        return self._scan["t"]

    @property
    def x_tmax(self) -> float:
        return self._scan["x_t"]

    @property
    def max_camber(self) -> float:
        """Signed maximum camber / chord."""
        return self._scan["z"]

    @property
    def x_camber(self) -> float:
        return self._scan["x_z"]

    @property
    def te_gap(self) -> float:
        return float(np.hypot(*(self.coords[0] - self.coords[-1])))

    # ------------------------------------------------------------------ resampling
    def resample(self, n_per_surface: int = 80, *, close_te: bool = False) -> "Airfoil":
        """Cosine-spaced copy with ``2 n + 1`` points (upper and lower share the nose point);
        optionally closes a small trailing-edge gap."""
        x0 = self._x0
        x = x0 + (1.0 - x0) * cosine_spacing(n_per_surface)
        yu, yl = self.y_upper(x), self.y_lower(x)
        if close_te:
            mid = 0.5 * (yu[-1] + yl[-1])
            ramp = ((x - x0) / (1.0 - x0)) ** 2  # blend the gap away smoothly towards the trailing edge
            yu = yu - (yu[-1] - mid) * ramp
            yl = yl - (yl[-1] - mid) * ramp
        upper = np.column_stack([x[::-1], yu[::-1]])
        lower = np.column_stack([x[1:], yl[1:]])
        coords = np.vstack([upper, lower])
        return Airfoil(self.name, coords, self.source, self.approximate, self.notes)

    def with_name(self, name: str, source: str | None = None, **flags) -> "Airfoil":
        return Airfoil(
            name,
            self.coords,
            self.source if source is None else source,
            flags.get("approximate", self.approximate),
            flags.get("notes", self.notes),
        )


def build_from_camber_thickness(
    name: str,
    thickness_half,
    camber,
    camber_slope,
    *,
    n_per_surface: int = 100,
    source: str = "",
    approximate: bool = False,
    notes: tuple[str, ...] = (),
) -> Airfoil:
    """Airfoil from a mean line and a half-thickness distribution (NACA construction).

    The half-thickness ``thickness_half(x)`` is laid off perpendicular to the mean line
    ``camber(x)``, with ``camber_slope(x)`` its derivative. Points are cosine-spaced.
    """
    x = cosine_spacing(n_per_surface)
    yt = np.asarray(thickness_half(x), dtype=float)
    z = np.asarray(camber(x), dtype=float)
    theta = np.arctan(np.asarray(camber_slope(x), dtype=float))
    xu, yu = x - yt * np.sin(theta), z + yt * np.cos(theta)
    xl, yl = x + yt * np.sin(theta), z - yt * np.cos(theta)
    upper = np.column_stack([xu[::-1], yu[::-1]])  # TE -> LE
    lower = np.column_stack([xl[1:], yl[1:]])  # LE (shared) excluded -> TE
    coords = np.vstack([upper, lower])
    return Airfoil(name=name, coords=coords, source=source, approximate=approximate, notes=tuple(notes))
