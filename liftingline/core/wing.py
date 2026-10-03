"""Wing geometry.

A wing is described by its **right half** as a table of spanwise *sections* and is
mirrored about the root plane. Geometry frame ("internal frame", SI metres):

* ``x`` aft (leading edge of the root section at ``x = 0``), ``y`` to starboard,
  ``z`` up;
* the quarter-chord (c/4) point of each section carries the lifting line.

Between sections everything varies linearly, except for ``kind="elliptic"`` where
the chord follows an exact ellipse (c/4 line straight) so analytic benchmarks can be
reproduced to machine precision.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from functools import cached_property

import numpy as np

KIND_SECTIONS = "sections"
KIND_ELLIPTIC = "elliptic"


@dataclass(frozen=True)
class Section:
    """One spanwise station of the right half-wing."""

    y: float  # lateral position of the section, m (first section must be 0)
    chord: float  # m
    x_le: float = 0.0  # aft position of the leading edge, m
    z: float = 0.0  # height of the c/4 point above the root c/4 point, m
    twist_deg: float = 0.0  # geometric twist about c/4, nose-up positive
    airfoil: str = ""  # key into the airfoil registry

    def to_dict(self) -> dict:
        return dict(
            y=self.y, chord=self.chord, x_le=self.x_le, z=self.z, twist_deg=self.twist_deg, airfoil=self.airfoil
        )

    @classmethod
    def from_dict(cls, d: dict) -> "Section":
        return cls(
            y=float(d["y"]),
            chord=float(d["chord"]),
            x_le=float(d.get("x_le", 0.0)),
            z=float(d.get("z", 0.0)),
            twist_deg=float(d.get("twist_deg", 0.0)),
            airfoil=str(d.get("airfoil", "")),
        )


@dataclass(frozen=True)
class Wing:
    sections: tuple[Section, ...]
    incidence_deg: float = 0.0  # root incidence added to the flow angle of attack
    kind: str = KIND_SECTIONS
    name: str = "Wing"

    def __post_init__(self) -> None:
        problems = self.validate()
        if problems:
            raise ValueError("; ".join(problems))

    # ---------------------------------------------------------------- validation
    def validate(self) -> list[str]:
        """Return a list of geometry problems (empty if the wing is usable)."""
        s = self.sections
        problems: list[str] = []
        if len(s) < 2:
            return ["a wing needs at least a root and a tip section"]
        if self.kind not in (KIND_SECTIONS, KIND_ELLIPTIC):
            problems.append(f"unknown planform kind {self.kind!r}")
        if abs(s[0].y) > 1e-12:
            problems.append("the first section must be at y = 0 (the root)")
        ys = [sec.y for sec in s]
        if any(b <= a for a, b in zip(ys, ys[1:])):
            problems.append("section y positions must be strictly increasing")
        if s[-1].y <= 0.0:
            problems.append("the semi-span must be positive")
        for i, sec in enumerate(s):
            last = i == len(s) - 1
            if sec.chord < 0.0 or (sec.chord == 0.0 and not last):
                problems.append(f"section {i + 1}: chord must be positive (only the tip may be 0)")
        if s[0].chord <= 0.0:
            problems.append("root chord must be positive")
        if self.kind == KIND_ELLIPTIC and len(s) != 2:
            problems.append("an elliptic planform is defined by exactly a root and a tip section")
        return problems

    # ------------------------------------------------------------ cached tables
    @cached_property
    def _ys(self) -> np.ndarray:
        return np.array([s.y for s in self.sections], dtype=float)

    @cached_property
    def _tables(self) -> dict[str, np.ndarray]:
        s = self.sections
        return {
            "chord": np.array([x.chord for x in s], dtype=float),
            "x_le": np.array([x.x_le for x in s], dtype=float),
            "z": np.array([x.z for x in s], dtype=float),
            "twist": np.array([x.twist_deg for x in s], dtype=float),
        }

    def _interp(self, key: str, y) -> np.ndarray:
        ay = np.abs(np.asarray(y, dtype=float))
        return np.interp(ay, self._ys, self._tables[key])

    # ------------------------------------------------------------ planform data
    @property
    def half_span(self) -> float:
        return float(self.sections[-1].y)

    @property
    def span(self) -> float:
        """Projected span b (m)."""
        return 2.0 * self.half_span

    @property
    def root_chord(self) -> float:
        return float(self.sections[0].chord)

    @property
    def tip_chord(self) -> float:
        return 0.0 if self.kind == KIND_ELLIPTIC else float(self.sections[-1].chord)

    def chord(self, y) -> np.ndarray:
        if self.kind == KIND_ELLIPTIC:
            eta = np.abs(np.asarray(y, dtype=float)) / self.half_span
            return self.root_chord * np.sqrt(np.clip(1.0 - eta**2, 0.0, None))
        return self._interp("chord", y)

    def x_le(self, y) -> np.ndarray:
        if self.kind == KIND_ELLIPTIC:  # straight quarter-chord line
            x0 = self.sections[0].x_le
            return x0 + 0.25 * (self.root_chord - self.chord(y))
        return self._interp("x_le", y)

    def x_c4(self, y) -> np.ndarray:
        return self.x_le(y) + 0.25 * self.chord(y)

    def z_c4(self, y) -> np.ndarray:
        return self._interp("z", y)

    def twist_deg(self, y) -> np.ndarray:
        return self._interp("twist", y)

    def airfoil_blend(self, y) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Airfoil interpolation at stations ``y``: (inboard idx, outboard idx, outboard weight)."""
        ay = np.clip(np.abs(np.asarray(y, dtype=float)), 0.0, self.half_span)
        idx = np.clip(np.searchsorted(self._ys, ay, side="right") - 1, 0, len(self._ys) - 2)
        y0, y1 = self._ys[idx], self._ys[idx + 1]
        xi = np.clip((ay - y0) / (y1 - y0), 0.0, 1.0)
        return idx, idx + 1, xi

    # ------------------------------------------------------- reference quantities
    def _integrate_half(self, f) -> float:
        """Integral of ``f(y)`` over the right half-span (exact for piecewise cubics)."""
        if self.kind == KIND_ELLIPTIC:  # y = hs sin(phi) removes the tip sqrt-singularity
            n = 2000
            phi = np.linspace(0.0, 0.5 * math.pi, n + 1)
            y = self.half_span * np.sin(phi)
            vals = f(y) * self.half_span * np.cos(phi)
            h = 0.5 * math.pi / n
            return float(h / 3.0 * (vals[0] + vals[-1] + 4.0 * vals[1:-1:2].sum() + 2.0 * vals[2:-1:2].sum()))
        total = 0.0
        for y0, y1 in zip(self._ys[:-1], self._ys[1:]):
            ym = 0.5 * (y0 + y1)
            total += (y1 - y0) / 6.0 * (float(f(y0)) + 4.0 * float(f(ym)) + float(f(y1)))
        return total

    @cached_property
    def area(self) -> float:
        """Projected planform area S (m^2)."""
        return 2.0 * self._integrate_half(self.chord)

    @property
    def aspect_ratio(self) -> float:
        return self.span**2 / self.area

    @property
    def taper_ratio(self) -> float:
        return self.tip_chord / self.root_chord

    @cached_property
    def mac(self) -> float:
        """Mean aerodynamic chord (m)."""
        return 2.0 / self.area * self._integrate_half(lambda y: self.chord(y) ** 2)

    @cached_property
    def y_mac(self) -> float:
        """Spanwise position of the MAC (m)."""
        return 2.0 / self.area * self._integrate_half(lambda y: self.chord(y) * y)

    @cached_property
    def x_le_mac(self) -> float:
        """Aft position of the leading edge of the MAC (m)."""
        return 2.0 / self.area * self._integrate_half(lambda y: self.chord(y) * self.x_le(y))

    def moment_reference_x(self, x_frac_mac: float = 0.25) -> float:
        """x of the point at ``x_frac_mac`` of the MAC measured from the MAC leading edge."""
        return self.x_le_mac + x_frac_mac * self.mac

    # --------------------------------------------------------- sweep & dihedral
    def sweep_c4_deg_by_segment(self) -> np.ndarray:
        if self.kind == KIND_ELLIPTIC:
            return np.zeros(1)
        x4 = self._tables["x_le"] + 0.25 * self._tables["chord"]
        return np.degrees(np.arctan2(np.diff(x4), np.diff(self._ys)))

    def dihedral_deg_by_segment(self) -> np.ndarray:
        return np.degrees(np.arctan2(np.diff(self._tables["z"]), np.diff(self._ys)))

    @property
    def max_abs_sweep_c4_deg(self) -> float:
        return float(np.max(np.abs(self.sweep_c4_deg_by_segment())))

    @property
    def max_abs_dihedral_deg(self) -> float:
        return float(np.max(np.abs(self.dihedral_deg_by_segment())))

    def is_planar_unswept(self, sweep_tol_deg: float = 2.0, dihedral_tol_deg: float = 0.5) -> bool:
        """True when classical (straight lifting line) theory applies to the geometry."""
        return self.max_abs_sweep_c4_deg <= sweep_tol_deg and self.max_abs_dihedral_deg <= dihedral_tol_deg

    # -------------------------------------------------------------- plot helper
    def planform_outline(self, n: int = 241) -> dict[str, np.ndarray]:
        """Leading/trailing-edge polylines of the full wing for drawing (x aft, y starboard)."""
        theta = np.linspace(0.0, math.pi, n)
        y = -self.half_span * np.cos(theta)
        return {"y": y, "x_le": self.x_le(y), "x_te": self.x_le(y) + self.chord(y), "x_c4": self.x_c4(y)}

    # ----------------------------------------------------------------- builders
    @classmethod
    def trapezoid(
        cls,
        span: float,
        root_chord: float,
        tip_chord: float,
        *,
        sweep_deg: float = 0.0,
        sweep_at: float = 0.25,
        dihedral_deg: float = 0.0,
        twist_tip_deg: float = 0.0,
        incidence_deg: float = 0.0,
        airfoil_root: str = "",
        airfoil_tip: str | None = None,
        name: str = "Wing",
    ) -> "Wing":
        """Straight-tapered wing. ``sweep_deg`` is measured along the ``sweep_at`` chord line
        (0 = leading edge, 0.25 = quarter chord, 1 = trailing edge); dihedral is positive up."""
        hs = 0.5 * span
        x_tip = hs * math.tan(math.radians(sweep_deg)) + sweep_at * (root_chord - tip_chord)
        z_tip = hs * math.tan(math.radians(dihedral_deg))
        return cls(
            sections=(
                Section(0.0, root_chord, 0.0, 0.0, 0.0, airfoil_root),
                Section(hs, tip_chord, x_tip, z_tip, twist_tip_deg, airfoil_root if airfoil_tip is None else airfoil_tip),
            ),
            incidence_deg=incidence_deg,
            name=name,
        )

    @classmethod
    def rectangular(cls, span: float, chord: float, **kw) -> "Wing":
        return cls.trapezoid(span, chord, chord, **kw)

    @classmethod
    def elliptic(
        cls,
        span: float,
        root_chord: float,
        *,
        twist_tip_deg: float = 0.0,
        incidence_deg: float = 0.0,
        airfoil_root: str = "",
        airfoil_tip: str | None = None,
        name: str = "Elliptic wing",
    ) -> "Wing":
        hs = 0.5 * span
        return cls(
            sections=(
                Section(0.0, root_chord, 0.0, 0.0, 0.0, airfoil_root),
                Section(hs, 0.0, 0.25 * root_chord, 0.0, twist_tip_deg, airfoil_root if airfoil_tip is None else airfoil_tip),
            ),
            incidence_deg=incidence_deg,
            kind=KIND_ELLIPTIC,
            name=name,
        )

    @classmethod
    def from_area_ar_taper(
        cls, area: float, aspect_ratio: float, taper_ratio: float, **kw
    ) -> "Wing":
        """Trapezoidal wing from planform area, aspect ratio and taper ratio."""
        span = math.sqrt(aspect_ratio * area)
        root = 2.0 * area / (span * (1.0 + taper_ratio))
        return cls.trapezoid(span, root, taper_ratio * root, **kw)

    def with_airfoils(self, *airfoils: str) -> "Wing":
        """Copy of the wing with the given airfoil keys assigned to the sections in order
        (a single key is applied to all sections)."""
        if len(airfoils) == 1:
            airfoils = airfoils * len(self.sections)
        if len(airfoils) != len(self.sections):
            raise ValueError("give one airfoil per section, or a single airfoil for all")
        return replace(self, sections=tuple(replace(s, airfoil=a) for s, a in zip(self.sections, airfoils)))

    # ------------------------------------------------------------ serialisation
    def to_dict(self) -> dict:
        return dict(
            name=self.name,
            kind=self.kind,
            incidence_deg=self.incidence_deg,
            sections=[s.to_dict() for s in self.sections],
        )

    @classmethod
    def from_dict(cls, d: dict) -> "Wing":
        return cls(
            sections=tuple(Section.from_dict(s) for s in d["sections"]),
            incidence_deg=float(d.get("incidence_deg", 0.0)),
            kind=str(d.get("kind", KIND_SECTIONS)),
            name=str(d.get("name", "Wing")),
        )
