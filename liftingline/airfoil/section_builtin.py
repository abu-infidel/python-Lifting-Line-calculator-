"""Built-in section engine: fast, offline, deterministic.

* **Inviscid core from the panel method** (:mod:`panel2d`): lift slope, zero-lift angle and the c/4
  moment come from the actual geometry, so any airfoil (NACA code or ``.dat``) is handled.
* **Viscous corrections fitted to NeuralFoil** (:mod:`builtin_calibration`, regenerate with
  ``tools/calibrate_builtin.py``): lift-slope reduction with thickness/Reynolds number, ``cl_max`` and
  minimum drag. Hold-out accuracy against NeuralFoil is recorded in that module: lift slope ~3 %,
  ``cl_max`` ~5 %, minimum drag ~13 %, zero-lift angle ~0.1 deg.
* **Smooth saturating stall** ``cl = u / (1 + (|u|/cl_max)^6)^(1/6)``, ``u = a0 (alpha - alpha_L0)``:
  monotone, so the nonlinear lifting-line iteration stays well behaved; there is no post-stall lift
  drop (use the NeuralFoil engine for that) but drag rises steeply beyond ``cl_max``.

Not modelled: laminar-flow drag buckets (6-series sections), low-Reynolds laminar-bubble effects
(the fits start at Re = 5e5) and flaps. Use NeuralFoil when those matter.
"""

from __future__ import annotations

import math

import numpy as np

from ..core.section import LinearParams
from . import builtin_calibration as cal
from .geometry import Airfoil
from .panel2d import PanelMethodError, PanelSolver
from .section_base import AirfoilAero
from .section_builtin_features import FEATURES
from .thin_airfoil import thin_airfoil_from_airfoil

_KNEE_POWER = 6.0  # sharpness of the stall knee
_STALL_DRAG = 0.25  # drag rise per (excess lift / cl_max)^2 beyond the knee


def saturate(u, s_pos, s_neg, p: float = _KNEE_POWER):
    """Smooth saturation of the linear lift ``u = a0 (alpha - alpha_L0)`` at ``s_pos`` / ``s_neg``.

    Returns ``(cl, dcl/du, sat, excess)`` where ``excess = max(|u| - sat, 0) / sat`` measures how far
    beyond the knee the section is (used for the post-stall drag rise).
    """
    u = np.asarray(u, dtype=float)
    sat = np.where(u >= 0.0, s_pos, s_neg)
    g = 1.0 + (np.abs(u) / sat) ** p
    return u * g ** (-1.0 / p), g ** (-(1.0 + 1.0 / p)), sat, np.maximum(np.abs(u) - sat, 0.0) / sat


def _fit(name: str, t, m, L):
    spec = cal.FITS[name]
    return sum(c * FEATURES[f](t, m, L) for f, c in zip(spec["features"], spec["coef"]))


class BuiltinAero:
    def __init__(self, airfoil: Airfoil) -> None:
        self.label = f"Built-in {airfoil.name}"
        self.approximate = airfoil.approximate
        # clamp the regression inputs to the range the fits were made on (no wild extrapolation)
        self.t = float(np.clip(airfoil.t_over_c, 0.04, 0.25))
        self.m = float(np.clip(abs(airfoil.max_camber), 0.0, 0.09))
        self.notes: list[str] = []
        try:
            ps = PanelSolver(airfoil, 160)
            self.a0_inviscid, self.alpha_L0, self.cm_c4 = ps.lift_slope, ps.alpha_L0, ps.cm_c4_zero_lift
        except PanelMethodError as exc:  # pragma: no cover - needs a degenerate airfoil
            tat = thin_airfoil_from_airfoil(airfoil)
            self.a0_inviscid = 2.0 * math.pi * (1.0 + 0.77 * self.t)
            self.alpha_L0, self.cm_c4 = tat.alpha_L0, tat.cm_c4
            self.notes.append(f"panel method unavailable ({exc}); thin-airfoil estimates used")
        self.neg_stall_ratio = float(np.clip(1.0 - 9.0 * self.m, 0.5, 1.0))  # |cl_min| / cl_max

    # ---------------------------------------------------------------------- factors
    @staticmethod
    def _L(re, floor: float):
        return np.log10(np.clip(np.asarray(re, dtype=float), floor, cal.RE_MAX) / 1e6)

    def _a0(self, re):
        return self.a0_inviscid * _fit("a_ratio", self.t, self.m, self._L(re, cal.LIFT_RE_FLOOR))

    def _cl_max(self, re):
        return np.maximum(_fit("cl_max", self.t, self.m, self._L(re, cal.LIFT_RE_FLOOR)), 0.3)

    def linear(self, re: np.ndarray) -> LinearParams:
        re = np.asarray(re, dtype=float)
        a0, clm = self._a0(re), self._cl_max(re)
        return LinearParams(
            a0=a0,
            alpha_L0=np.full_like(re, self.alpha_L0),
            cm_c4=np.full_like(re, self.cm_c4),
            cl_max=clm,
            alpha_stall=self.alpha_L0 + clm / a0 + math.radians(cal.STALL_DELTA_DEG),
        )

    # ------------------------------------------------------------------ coefficients
    def coefficients(self, alpha: np.ndarray, re: np.ndarray):
        alpha = np.asarray(alpha, dtype=float)
        re = np.asarray(re, dtype=float) * np.ones_like(alpha)
        a0, clm = self._a0(re), self._cl_max(re)
        u = a0 * (alpha - self.alpha_L0)
        cl, dcl_du, sat, excess = saturate(u, clm, clm * self.neg_stall_ratio)
        dcl = a0 * dcl_du
        L_drag = self._L(re, cal.RE_MIN_DRAG)
        cd_min = np.exp(_fit("log_cd_min", self.t, self.m, L_drag))
        cl0 = _fit("cl_at_cd_min", self.t, self.m, self._L(re, cal.LIFT_RE_FLOOR))
        cd = cd_min + cal.K_CD * (cl - cl0) ** 2 + _STALL_DRAG * excess**2
        return cl, cd, np.full_like(alpha, self.cm_c4), dcl

    def confidence(self, alpha: np.ndarray, re: np.ndarray):
        return None


class BuiltinEngine:
    """Factory for :class:`BuiltinAero` objects (cached per airfoil geometry)."""

    name = "Built-in (fast)"

    def __init__(self, settings=None) -> None:
        self.settings = settings
        self._cache: dict[str, BuiltinAero] = {}

    def aero(self, airfoil: Airfoil) -> AirfoilAero:
        aero = self._cache.get(airfoil.digest)
        if aero is None:
            aero = self._cache[airfoil.digest] = BuiltinAero(airfoil)
        return aero
