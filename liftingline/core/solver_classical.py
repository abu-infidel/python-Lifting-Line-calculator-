"""Solver A - classical Prandtl lifting-line theory (Glauert's Fourier solution).

With ``y = -(b/2) cos(theta)`` the circulation is expanded as

    Gamma(theta) = 2 b V  sum_n  A_n sin(n theta)

and the lifting-line equation is enforced at ``N`` collocation angles
``theta_i = i pi / (N + 1)``:

    sum_n A_n sin(n theta) [ n mu + sin(theta) ] = mu (alpha_geo - alpha_L0) sin(theta),
    mu = c a0 / (4 b).

All harmonics are kept so asymmetric loading (control surfaces, roll) is handled. Tip
loss is built in: every sin(n theta) vanishes at the tips, i.e. Gamma -> 0 there, which
produces the downwash and the induced drag.

With real (nonlinear) section polars the same expansion is solved by Newton's method on
``A``:  R_i = 4 b sum_n A_n sin(n theta_i) - c_i cl(alpha_eff,i) = 0, with the closed-form
Jacobian ``4 b sin(n theta_i) + c_i cl'(alpha_eff,i) n sin(n theta_i) / sin(theta_i)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .flow import FlowCondition
from .forces import LoadDistribution, wind_axes
from .section import SectionModel
from .wing import Wing


class ControlEffects(Protocol):
    """Spanwise effect of deflected control surfaces (flaps, ailerons).

    ``d_alpha`` is the equivalent change of the section angle of attack (rad, trailing-edge
    down positive); ``d_cm`` the change of the c/4 pitching-moment coefficient. Both are
    functions of the *signed* lateral position ``y``.
    """

    def d_alpha(self, y: np.ndarray) -> np.ndarray: ...

    def d_cm(self, y: np.ndarray) -> np.ndarray: ...


@dataclass
class ClassicalSolution:
    A: np.ndarray  # Fourier coefficients A_1..A_N
    n_terms: int
    b: float  # projected span, m
    S: float  # planform area, m^2
    V: float  # airspeed, m/s
    alpha_root: float  # root-chord angle of attack, rad
    theta_nodes: np.ndarray
    converged: bool
    iterations: int
    residual: float  # max |R| of the lifting-line equations (0 for the linear solve = round-off)
    nonlinear: bool

    @property
    def AR(self) -> float:
        return self.b**2 / self.S

    # --- closed-form Fourier results (valid for the lifting-line representation) --------
    @property
    def CL_fourier(self) -> float:
        return math.pi * self.AR * float(self.A[0])

    @property
    def CDi_fourier(self) -> float:
        n = np.arange(1, self.n_terms + 1)
        return math.pi * self.AR * float(np.sum(n * self.A**2))

    @property
    def e_fourier(self) -> float:
        d = self.CDi_fourier
        return self.CL_fourier**2 / (math.pi * self.AR * d) if d > 0 else math.nan

    # --- evaluation of the series at arbitrary theta ---------------------------------
    def gamma(self, theta: np.ndarray) -> np.ndarray:
        n = np.arange(1, self.n_terms + 1)
        return self.V * 2.0 * self.b * (np.sin(np.outer(theta, n)) @ self.A)

    def alpha_induced(self, theta: np.ndarray) -> np.ndarray:
        """Induced angle sum n A_n sin(n theta)/sin(theta), with the exact limits at the tips."""
        theta = np.asarray(theta, dtype=float)
        n = np.arange(1, self.n_terms + 1)
        s = np.sin(theta)
        interior = s > 1e-12
        out = np.empty_like(theta)
        ratio = np.sin(np.outer(theta[interior], n)) / s[interior, None]
        out[interior] = ratio @ (n * self.A)
        if np.any(~interior):
            at_zero = theta[~interior] < 0.5 * math.pi
            tip0 = float(np.sum(n**2 * self.A))
            tip_pi = float(np.sum(n**2 * (-1.0) ** (n + 1) * self.A))
            out[~interior] = np.where(at_zero, tip0, tip_pi)
        return out


# ---------------------------------------------------------------------------------------
def _collocation(n_terms: int) -> np.ndarray:
    return math.pi * np.arange(1, n_terms + 1) / (n_terms + 1)


def solve_classical(
    wing: Wing,
    flow: FlowCondition,
    section: SectionModel,
    *,
    n_terms: int = 80,
    nonlinear: bool = False,
    controls: ControlEffects | None = None,
    tol: float = 1e-10,
    max_iter: int = 60,
) -> ClassicalSolution:
    """Solve the lifting-line equations of ``wing`` in ``flow`` with section data ``section``."""
    if n_terms < 4:
        raise ValueError("n_terms must be at least 4")
    N = n_terms
    b = wing.span
    theta = _collocation(N)
    sin_t = np.sin(theta)
    y = -0.5 * b * np.cos(theta)
    ay = np.abs(y)
    c = wing.chord(y)
    re = flow.reynolds(c)
    lin = section.linear(ay, re)

    alpha_root = math.radians(flow.alpha_deg + wing.incidence_deg)
    alpha_geo = alpha_root + np.radians(wing.twist_deg(y))
    if controls is not None:
        alpha_geo = alpha_geo + controls.d_alpha(y)

    n = np.arange(1, N + 1)
    S_mat = np.sin(np.outer(theta, n))  # (N, N)
    mu = c * lin.a0 / (4.0 * b)

    # ---- linear solve --------------------------------------------------------------
    lhs = S_mat * (n[None, :] * mu[:, None] + sin_t[:, None])
    rhs = mu * (alpha_geo - lin.alpha_L0) * sin_t
    A = np.linalg.solve(lhs, rhs)
    iterations, converged = 1, True
    residual = float(np.max(np.abs(lhs @ A - rhs)))

    if nonlinear:
        A, iterations, converged, residual = _newton(
            A, S_mat, n, sin_t, c, ay, re, alpha_geo, b, section, tol=tol, max_iter=max_iter
        )

    return ClassicalSolution(
        A=A,
        n_terms=N,
        b=b,
        S=wing.area,
        V=flow.V,
        alpha_root=alpha_root,
        theta_nodes=theta,
        converged=converged,
        iterations=iterations,
        residual=residual,
        nonlinear=nonlinear,
    )


def _residual(A, S_mat, n, sin_t, c, ay, re, alpha_geo, b, section):
    alpha_ind = (S_mat / sin_t[:, None]) @ (n * A)
    alpha_eff = alpha_geo - alpha_ind
    coeffs = section.evaluate(ay, alpha_eff, re)
    return 4.0 * b * (S_mat @ A) - c * coeffs.cl, coeffs.cl_alpha


def _newton(A0, S_mat, n, sin_t, c, ay, re, alpha_geo, b, section, *, tol, max_iter):
    """Damped Newton iteration on the Fourier coefficients (see module docstring)."""
    A = A0.copy()
    R, cla = _residual(A, S_mat, n, sin_t, c, ay, re, alpha_geo, b, section)
    scale = 4.0 * b
    for it in range(1, max_iter + 1):
        if float(np.max(np.abs(R))) <= tol * scale:
            return A, it, True, float(np.max(np.abs(R)))
        J = scale * S_mat + (c * cla)[:, None] * (S_mat * (n[None, :] / sin_t[:, None]))
        try:
            dA = -np.linalg.solve(J, R)
        except np.linalg.LinAlgError:
            dA = -np.linalg.lstsq(J, R, rcond=None)[0]
        norm0 = float(np.linalg.norm(R))
        step = 1.0
        for _ in range(12):  # backtracking line search on the residual norm
            trial = A + step * dA
            R_try, cla_try = _residual(trial, S_mat, n, sin_t, c, ay, re, alpha_geo, b, section)
            if float(np.linalg.norm(R_try)) < norm0 or step < 1e-3:
                break
            step *= 0.5
        A, R, cla = trial, R_try, cla_try
    return A, max_iter, False, float(np.max(np.abs(R)))


# ---------------------------------------------------------------------------------------
def build_load_classical(
    sol: ClassicalSolution,
    wing: Wing,
    flow: FlowCondition,
    section: SectionModel,
    *,
    controls: ControlEffects | None = None,
    n_grid: int | None = None,
) -> LoadDistribution:
    """Spanwise loads on a fine theta grid (Fourier series evaluated exactly).

    The grid uses equally spaced theta including both tips; integrating with
    ``dy = (b/2) sin(theta) dtheta`` makes every integral of a trigonometric polynomial
    (lift, induced drag) exact, and the profile-drag/moment integrals accurate to O(1e-6).
    """
    b = sol.b
    grid = n_grid if n_grid is not None else max(8 * (sol.n_terms + 1), 400)
    grid += grid % 2
    theta = np.linspace(0.0, math.pi, grid + 1)
    y = -0.5 * b * np.cos(theta)
    ay = np.abs(y)
    sin_t = np.sin(theta)
    h = math.pi / grid
    width = 0.5 * b * sin_t * h
    # Euler-Maclaurin end correction: for F = f(theta) sin(theta) the trapezoid rule misses
    # (h^2/12)(f(0) + f(pi)). It only matters for integrands that do not vanish at the tips
    # (profile drag, pitching moment); lift and induced drag (Gamma = 0 there) stay exact.
    width[0] += 0.5 * b * h * h / 12.0
    width[-1] += 0.5 * b * h * h / 12.0

    gamma = sol.gamma(theta)
    alpha_i = sol.alpha_induced(theta)
    c = wing.chord(y)
    alpha_geo = sol.alpha_root + np.radians(wing.twist_deg(y))
    d_cm = 0.0
    if controls is not None:
        alpha_geo = alpha_geo + controls.d_alpha(y)
        d_cm = controls.d_cm(y)
    alpha_eff = alpha_geo - alpha_i
    re = flow.reynolds(np.maximum(c, 1e-6))
    coeffs = section.evaluate(ay, alpha_eff, re)
    lin = section.linear(ay, re)
    cm = coeffs.cm + d_cm

    e_d, _, e_l = wind_axes(sol.alpha_root, 0.0)
    lift_pm = flow.rho * flow.V * gamma  # N/m
    induced_pm = lift_pm * alpha_i
    profile_pm = flow.q * c * coeffs.cd
    force_kj = (lift_pm[:, None] * e_l[None, :] + induced_pm[:, None] * e_d[None, :]) * width[:, None]
    force_profile = profile_pm[:, None] * e_d[None, :] * width[:, None]
    moment = np.zeros((theta.size, 3))
    moment[:, 1] = flow.q * c**2 * cm * width

    with np.errstate(divide="ignore", invalid="ignore"):
        cl_circ = np.where(c > 1e-9, 2.0 * gamma / (flow.V * np.maximum(c, 1e-12)), 0.0)
    pos = np.stack([wing.x_c4(y), y, wing.z_c4(y)], axis=1)
    return LoadDistribution(
        y=y,
        pos=pos,
        width=width,
        chord=c,
        force_kj=force_kj,
        force_profile=force_profile,
        moment_c4=moment,
        gamma=gamma,
        cl=cl_circ,
        cd=coeffs.cd,
        cm=cm,
        alpha_eff=alpha_eff,
        alpha_induced=alpha_i,
        re=re,
        cl_max=lin.cl_max,
        extras={"theta": theta, "alpha_geo": alpha_geo, "cl_section": coeffs.cl},
    )
