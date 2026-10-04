"""Control surfaces (plain flaps and ailerons) as equivalent changes of the section angle of attack.

Thin-airfoil theory for a plain flap of chord fraction ``cf`` hinged at ``x_h = 1 - cf`` (``theta_h = arccos(2 cf - 1)``)
gives, for a deflection ``delta`` (trailing edge down positive),

    d(alpha_equivalent) = tau * delta,        tau = 1 - (theta_h - sin theta_h) / pi,
    d(cm_c/4)           = -(1/2) sin(theta_h) (1 - cos theta_h) * delta.

Real flaps are less effective than the inviscid value (boundary-layer thickening, gaps, separation at large
deflections), so every surface carries a ``effectiveness`` factor (default 0.85) on both terms. A
:class:`ControlSet` implements the :class:`~liftingline.core.solver_classical.ControlEffects` protocol used by both
lifting-line solvers: flaps act on both wings alike, ailerons act oppositely (``delta`` is the deflection of the
*right* aileron; the left one moves the other way).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Iterable

import numpy as np

KIND_FLAP = "flap"
KIND_AILERON = "aileron"
KINDS = (KIND_FLAP, KIND_AILERON)

DEFAULT_EFFECTIVENESS = 0.85
MAX_DEFLECTION_DEG = 45.0


class ControlError(ValueError):
    """A control surface description that cannot be used."""


def hinge_angle(chord_fraction: float) -> float:
    """``theta_h = arccos(2 cf - 1)`` for a flap of chord fraction ``cf`` (0 < cf < 1)."""
    return math.acos(2.0 * chord_fraction - 1.0)


def flap_effectiveness(chord_fraction: float) -> float:
    """Thin-airfoil flap effectiveness ``tau = d(alpha_L0)/d(delta)`` (inviscid, 0 < tau < 1)."""
    th = hinge_angle(chord_fraction)
    return 1.0 - (th - math.sin(th)) / math.pi


def flap_moment_slope(chord_fraction: float) -> float:
    """Thin-airfoil ``d(cm_c/4)/d(delta)`` per radian (negative: a flap-down deflection pitches the nose down)."""
    th = hinge_angle(chord_fraction)
    return -0.5 * math.sin(th) * (1.0 - math.cos(th))


@dataclass(frozen=True)
class ControlSurface:
    """One flap or aileron: a span range, a chord fraction and a deflection."""

    name: str = "Aileron"
    kind: str = KIND_AILERON
    y_in: float = 0.6  # inboard end, fraction of the semi-span
    y_out: float = 0.95  # outboard end, fraction of the semi-span
    chord_fraction: float = 0.25  # flap chord / local wing chord
    deflection_deg: float = 0.0  # trailing edge down positive (the right-hand aileron for ailerons)
    effectiveness: float = DEFAULT_EFFECTIVENESS  # viscous correction factor on the thin-airfoil value

    def problems(self) -> list[str]:
        out = []
        if self.kind not in KINDS:
            out.append(f"unknown control surface type {self.kind!r}")
        if not (0.0 <= self.y_in < self.y_out <= 1.0):
            out.append("the span range must satisfy 0 <= inboard < outboard <= 100 % of the semi-span")
        if not (0.05 <= self.chord_fraction <= 0.6):
            out.append("the control chord must be between 5 % and 60 % of the local chord")
        if abs(self.deflection_deg) > MAX_DEFLECTION_DEG:
            out.append(f"the deflection must be within +-{MAX_DEFLECTION_DEG:.0f} degrees")
        if not (0.2 <= self.effectiveness <= 1.0):
            out.append("the effectiveness factor must be between 0.2 and 1")
        return out

    @property
    def tau(self) -> float:
        return self.effectiveness * flap_effectiveness(self.chord_fraction)

    def to_dict(self) -> dict:
        return dict(
            name=self.name, kind=self.kind, y_in=self.y_in, y_out=self.y_out, chord_fraction=self.chord_fraction,
            deflection_deg=self.deflection_deg, effectiveness=self.effectiveness,
        )

    @classmethod
    def from_dict(cls, d: dict) -> "ControlSurface":
        known = {k: d[k] for k in ("name", "kind", "y_in", "y_out", "chord_fraction", "deflection_deg", "effectiveness") if k in d}
        for key in ("y_in", "y_out", "chord_fraction", "deflection_deg", "effectiveness"):
            if key in known:
                known[key] = float(known[key])
        return cls(**known)


@dataclass(frozen=True)
class ControlSet:
    """The control surfaces of a wing; implements the lifting-line ``ControlEffects`` protocol."""

    surfaces: tuple[ControlSurface, ...] = field(default_factory=tuple)
    half_span: float = 1.0

    def __post_init__(self) -> None:
        problems = [f"{s.name}: {p}" for s in self.surfaces for p in s.problems()]
        if problems:
            raise ControlError("; ".join(problems))
        if not self.half_span > 0.0:
            raise ControlError("the semi-span must be positive")

    @classmethod
    def from_dicts(cls, rows: Iterable[dict], half_span: float) -> "ControlSet":
        try:
            return cls(tuple(ControlSurface.from_dict(r) for r in rows), half_span)
        except (TypeError, ValueError) as exc:
            if isinstance(exc, ControlError):
                raise
            raise ControlError(f"invalid control surface: {exc}") from exc

    @property
    def active(self) -> bool:
        return any(s.deflection_deg != 0.0 for s in self.surfaces)

    def _mask(self, s: ControlSurface, y: np.ndarray) -> np.ndarray:
        eta = np.abs(y) / self.half_span
        return (eta >= s.y_in) & (eta <= s.y_out)

    def _sign(self, s: ControlSurface, y: np.ndarray) -> np.ndarray:
        return np.ones_like(y) if s.kind == KIND_FLAP else np.sign(y)

    def d_alpha(self, y: np.ndarray) -> np.ndarray:
        """Equivalent change of the section angle of attack, rad (trailing-edge-down positive)."""
        y = np.asarray(y, dtype=float)
        out = np.zeros_like(y)
        for s in self.surfaces:
            out += self._mask(s, y) * self._sign(s, y) * s.tau * math.radians(s.deflection_deg)
        return out

    def d_cm(self, y: np.ndarray) -> np.ndarray:
        """Change of the c/4 pitching-moment coefficient of the sections."""
        y = np.asarray(y, dtype=float)
        out = np.zeros_like(y)
        for s in self.surfaces:
            slope = s.effectiveness * flap_moment_slope(s.chord_fraction)
            out += self._mask(s, y) * self._sign(s, y) * slope * math.radians(s.deflection_deg)
        return out

    # ----------------------------------------------------------------------------- variants
    def with_deflection(self, name: str, deflection_deg: float) -> "ControlSet":
        """Copy with the surface called ``name`` deflected (all surfaces of that name if several)."""
        if not any(s.name == name for s in self.surfaces):
            raise ControlError(f"no control surface called {name!r}")
        return replace(self, surfaces=tuple(replace(s, deflection_deg=deflection_deg) if s.name == name else s for s in self.surfaces))


def validate_rows(rows: Iterable[dict]) -> list[str]:
    """Readable problems of a table of control-surface dicts (empty if it can be used)."""
    msgs = []
    names = []
    for i, r in enumerate(rows):
        try:
            s = ControlSurface.from_dict(r)
        except (TypeError, ValueError) as exc:
            msgs.append(f"control surface {i + 1}: {exc}")
            continue
        names.append(s.name)
        msgs += [f"control surface '{s.name}': {p}" for p in s.problems()]
    if len(set(names)) != len(names):
        msgs.append("control surface names must be unique")
    return msgs
