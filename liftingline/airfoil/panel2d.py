"""Inviscid 2-D panel method: linear-strength vorticity, Neumann condition, Kutta condition.

Panels run counter-clockwise (Selig order); on panel ``j`` from node ``j`` to ``j+1`` the
vortex strength varies linearly between the nodal values ``gamma_j`` and ``gamma_{j+1}``.
In panel-local coordinates (xi along the panel, eta along the left normal, which points
*into* the body for counter-clockwise traversal) a unit-strength linear sheet induces

    u = -(1/2pi) [ g_a dth + (g_b - g_a)/l * (xi dth - eta ln(r1/r2)) ]
    v =  (1/2pi) [ g_a ln(r1/r2) + (g_b - g_a)/l * (xi ln(r1/r2) - l + eta dth) ]

with ``dth = theta2 - theta1`` (the angle subtended by the panel) and ``r1, r2`` the distances
to the panel ends. The flow tangency condition at the panel mid-points, plus the Kutta
condition ``gamma_0 + gamma_N = 0`` at the trailing edge, closes the (N+1) x (N+1) system.

Because the matrix does not depend on the angle of attack, two basis solutions (freestream
along x and along y) are factorised once and any angle follows by superposition. Lift comes
from the pressure integral (and is cross-checked against the circulation); in potential flow
``cl = R sin(alpha - alpha_L0)`` exactly, so ``alpha_L0`` and the lift slope ``R`` are obtained
directly from the two circulations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property

import numpy as np
from scipy.linalg import lu_factor, lu_solve

from .geometry import Airfoil

_TWO_PI = 2.0 * math.pi

#: condition number above which the panel system is treated as singular
_MAX_CONDITION = 1e11


class PanelMethodError(ValueError):
    """The panel system cannot be solved (e.g. zero-thickness or self-overlapping geometry)."""


@dataclass(frozen=True)
class PanelSolution:
    alpha: float  # rad
    cl: float  # pressure integration
    cl_circulation: float  # Kutta-Joukowski from the circulation (consistency check)
    cd: float  # pressure drag, ~0 by d'Alembert's paradox (a discretisation-error gauge)
    cm_c4: float  # about the quarter-chord point (0.25, 0), nose-up positive
    cp: np.ndarray  # at the panel mid-points
    x_mid: np.ndarray
    y_mid: np.ndarray
    gamma: np.ndarray  # nodal vortex strengths


class PanelSolver:
    """Linear-vortex panel solution for one airfoil.

    ``n_panels`` is the total panel count (about 160 gives lift-slope errors of a few 1e-3 for
    ordinary sections). Trailing-edge gaps below ``close_te_below`` (fraction of chord) are
    closed smoothly; larger (blunt) trailing edges are left open and the gap is not modelled.
    """

    def __init__(self, airfoil: Airfoil, n_panels: int = 160, close_te_below: float = 0.02) -> None:
        n_half = max(int(n_panels) // 2, 8)
        gap = airfoil.te_gap
        close = 1e-9 < gap < close_te_below
        self.airfoil = airfoil.resample(n_half, close_te=close, spacing="panel")
        self.te_closed = close or gap <= 1e-9
        self.nodes = self.airfoil.coords  # (N + 1, 2)
        self.n = len(self.nodes) - 1
        self._build()

    # --------------------------------------------------------------------- assembly
    def _build(self) -> None:
        P = self.nodes
        N = self.n
        d = P[1:] - P[:-1]
        length = np.hypot(d[:, 0], d[:, 1])
        if np.any(length < 1e-12):
            raise ValueError("degenerate (zero-length) panel; check the airfoil coordinates")
        t = d / length[:, None]
        nrm = np.column_stack([-t[:, 1], t[:, 0]])  # left normal (into the body)
        mid = 0.5 * (P[1:] + P[:-1])

        # vector from the start of panel j to control point i, in panel-j coordinates
        rel = mid[:, None, :] - P[None, :-1, :]  # (i, j, 2)
        xi = rel[..., 0] * t[None, :, 0] + rel[..., 1] * t[None, :, 1]
        eta = rel[..., 0] * nrm[None, :, 0] + rel[..., 1] * nrm[None, :, 1]
        l = length[None, :]
        r1 = np.hypot(xi, eta)
        r2 = np.hypot(xi - l, eta)
        dth = np.arctan2(eta, xi - l) - np.arctan2(eta, xi)
        lnr = np.log(r1 / r2)
        diag = np.arange(N)
        dth[diag, diag] = 0.0  # principal value on the panel itself
        lnr[diag, diag] = 0.0

        ua = -(dth * (1.0 - xi / l) + eta / l * lnr) / _TWO_PI
        ub = -(xi * dth - eta * lnr) / (l * _TWO_PI)
        va = ((1.0 - xi / l) * lnr + 1.0 - eta / l * dth) / _TWO_PI
        vb = ((xi / l) * lnr - 1.0 + eta / l * dth) / _TWO_PI

        tt = t @ t.T  # t_i . t_j
        tn = nrm @ t.T  # n_i . t_j
        nt = t @ nrm.T  # t_i . n_j
        nn = nrm @ nrm.T  # n_i . n_j
        # components along n_i (normal) and t_i (tangential) at control point i from panel j
        wa_n, wb_n = ua * tn + va * nn, ub * tn + vb * nn
        wa_t, wb_t = ua * tt + va * nt, ub * tt + vb * nt

        A = np.zeros((N + 1, N + 1))
        A[:N, :N] += wa_n
        A[:N, 1:] += wb_n
        A[N, 0] = A[N, N] = 1.0  # Kutta
        cond = float(np.linalg.cond(A, 1))
        if not np.isfinite(cond) or cond > _MAX_CONDITION:
            raise PanelMethodError(
                f"the panel system is singular (condition number {cond:.1e}): the airfoil is probably "
                "infinitely thin or has overlapping surfaces; give it a finite thickness"
            )
        self.condition = cond
        self._lu = lu_factor(A)

        Ct = np.zeros((N, N + 1))
        Ct[:, :N] += wa_t
        Ct[:, 1:] += wb_t
        Ct[diag, diag] += 0.25  # outside-surface jump: + gamma_mid / 2 along the travel direction
        Ct[diag, diag + 1] += 0.25

        self._t, self._n, self._len, self._mid = t, nrm, length, mid
        self._Ct = Ct
        rhs = np.zeros((N + 1, 2))
        rhs[:N, 0], rhs[:N, 1] = -nrm[:, 0], -nrm[:, 1]  # freestream along +x and +y
        gam = lu_solve(self._lu, rhs)  # (N + 1, 2)
        self._gam = gam
        self._vt = np.column_stack([t[:, 0] + Ct @ gam[:, 0], t[:, 1] + Ct @ gam[:, 1]])  # (N, 2)
        gm = 0.5 * (gam[:-1] + gam[1:])
        self._circ = (length[:, None] * gm).sum(axis=0)  # counter-clockwise circulation per basis

    # ------------------------------------------------------------------ solutions
    def solve(self, alpha: float) -> PanelSolution:
        """Solution at angle of attack ``alpha`` (rad) relative to the x axis."""
        ca, sa = math.cos(alpha), math.sin(alpha)
        vt = ca * self._vt[:, 0] + sa * self._vt[:, 1]
        cp = 1.0 - vt**2
        df = cp[:, None] * self._n * self._len[:, None]  # force on the body, per unit dynamic pressure
        fx, fy = float(df[:, 0].sum()), float(df[:, 1].sum())
        cl = -fx * sa + fy * ca
        cd = fx * ca + fy * sa
        x, y = self._mid[:, 0] - 0.25, self._mid[:, 1]
        # r x F is counter-clockwise-positive; with the leading edge on the left that lowers the
        # nose, so the nose-up-positive pitching moment is its negative
        cm = -float(np.sum(x * df[:, 1] - y * df[:, 0]))
        gamma = ca * self._gam[:, 0] + sa * self._gam[:, 1]
        cl_circ = -2.0 * (ca * self._circ[0] + sa * self._circ[1])
        return PanelSolution(
            alpha=alpha,
            cl=cl,
            cl_circulation=cl_circ,
            cd=cd,
            cm_c4=cm,
            cp=cp,
            x_mid=self._mid[:, 0],
            y_mid=self._mid[:, 1],
            gamma=gamma,
        )

    # ------------------------------------------------------------- linear parameters
    @cached_property
    def lift_slope(self) -> float:
        """Inviscid lift-curve slope dcl/dalpha at zero lift, per radian (circulation based)."""
        return 2.0 * math.hypot(self._circ[0], self._circ[1])

    @cached_property
    def alpha_L0(self) -> float:
        """Zero-lift angle of attack, rad (relative to the x axis of the supplied coordinates)."""
        return math.atan2(self._circ[0], -self._circ[1])

    @cached_property
    def cm_c4_zero_lift(self) -> float:
        """Pitching moment about c/4 at zero lift."""
        return self.solve(self.alpha_L0).cm_c4

    @cached_property
    def x_aerodynamic_centre(self) -> float:
        """Aerodynamic-centre position (fraction of chord) from the slope of cm(c/4) against cl."""
        a = self.alpha_L0
        s0, s1 = self.solve(a), self.solve(a + 0.02)
        dcl = s1.cl - s0.cl
        return 0.25 - (s1.cm_c4 - s0.cm_c4) / dcl if abs(dcl) > 1e-12 else 0.25
