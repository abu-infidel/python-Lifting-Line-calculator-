"""Solver B - numerical lifting-line theory (horseshoe vortices; sweep, dihedral, sideslip, ground effect).

Generalises Prandtl's lifting line to wings whose quarter-chord line is swept and/or bent, in the
spirit of Phillips & Snyder (J. Aircraft 37, 2000). The wing is cut into ``N`` elements (cosine spacing
in the lateral coordinate). Element ``i`` carries a horseshoe vortex of circulation ``Gamma_i``: a bound
segment ``dl_i`` along the quarter-chord line plus two semi-infinite trailing legs along the freestream
direction. With the local velocity ``V_i = V_inf + sum_k Gamma_k v_ik`` at the element mid-point, the
Kutta-Joukowski force ``rho Gamma_i V_i x dl_i`` must equal the lift the 2-D section data predict:

    R_i = 2 Gamma_i |V_i x dl_i| - |V_perp,i|^2 A_i cl(alpha_i; Re_i) = 0.

**Sections are taken normal to the bound vortex.** At each element the section frame is
``(a, n, s)``: ``s`` along the bound vortex, ``a`` the chord direction in the plane normal to ``s``,
``n`` the section normal; ``V_perp`` is the velocity in that (a, n) plane and
``alpha_i = atan2(V.n, V.a)`` (+ control deflection), ``A_i = c_n |dl_i|`` with ``c_n`` the chord
measured in that plane. For an unswept planar wing this is exactly Prandtl's equation; for an infinite
swept wing it gives the classical result that the lift slope is reduced by ``cos(Lambda)``.

Profile drag is the exception: skin friction follows the freestream, so it is evaluated with the freestream
dynamic pressure, the element's surface area and the streamwise-chord Reynolds number (no sweep credit).

**Chord-scaled vortex cores (swept wings).** A bare lifting line with sweep has no converged solution: the
downwash of the trailing sheet at a point of the swept line contains the term ``2 sin(L) (dGamma/dy) ln(1/eps)``
(``L`` the sweep, ``eps`` the element size), because the sheet starts ahead of the control point on the
inboard side and behind it on the outboard side, and the kink of the bound line at the root induces a
``Gamma/rho`` velocity that notches the loading there. Refining the element count therefore makes the lift
fall without limit (about 4 % per doubling at 30 deg). The real wing has no such singularity because its
vorticity is spread over the chord, so the near field is cut off at a chord-sized core:

* the sweep/sideslip-induced (asymmetric) part of every trailing leg and the whole field of every bound
  segment are multiplied by a Vatistas (n = 2) factor ``h^2 / sqrt(h^4 + r_c^4)`` (``h`` = distance to the
  filament), with ``r_c = LEG_CORE * c`` for the legs and ``r_c = BOUND_CORE * c`` for the bound segments;
* the symmetric part of the leg field is untouched, and collinear bound segments induce nothing on one another,
  so **unswept planar wings reproduce the classical (Prandtl) result exactly** - the solver is continuous with
  the classical one as the sweep goes to zero.

The two constants were calibrated once against an independent vortex-lattice method (``tests/ref_vlm.py``) on
the *sweep effect* (swept relative to the same wing unswept), see ``tests/test_numerical_llt.py``.

The residual equations are solved by damped Newton iteration with the analytic Jacobian.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .forces import LoadDistribution
from .section import LinearParams, SectionCoeffs, SectionModel
from .wing import Wing

_FOUR_PI = 4.0 * math.pi

#: Vortex-core radii as fractions of the local chord (see the module docstring); 0 disables the regularisation.
LEG_CORE = 0.5
BOUND_CORE = 0.2


def freestream_direction(alpha: float, beta: float) -> np.ndarray:
    """Unit vector of the air velocity relative to the wing (internal frame: x aft, y starboard, z up)."""
    return np.array([math.cos(alpha) * math.cos(beta), -math.sin(beta), math.sin(alpha) * math.cos(beta)])


# ------------------------------------------------------------------------------ influence functions
def bound_segment_velocity(P: np.ndarray, A: np.ndarray, B: np.ndarray, core: float = 1e-9, rc=None) -> np.ndarray:
    """Velocity at points ``P`` (..., 3) from straight vortex segments ``A -> B`` of unit circulation.

    Biot-Savart for a finite segment; zero on the line through the segment (it induces nothing on itself).
    ``rc`` (optional, broadcastable to the segments) is a Vatistas n = 2 core radius: the singular
    ``1/h`` field is multiplied by ``h^2 / sqrt(h^4 + rc^4)``.
    """
    r1, r2, r0 = P - A, P - B, B - A
    cross = np.cross(r1, r2)
    c2 = np.einsum("...k,...k->...", cross, cross)  # |r0|^2 h^2
    n1 = np.linalg.norm(r1, axis=-1)
    n2 = np.linalg.norm(r2, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.einsum("...k,...k->...", r0, r1 / n1[..., None] - r2 / n2[..., None])
        if rc is None:
            den = c2
        else:
            den = np.sqrt(c2**2 + (np.asarray(rc) ** 2 * np.einsum("...k,...k->...", r0, r0)) ** 2)
        coef = np.where(c2 > core**2, term / den, 0.0)
    return cross * coef[..., None] / _FOUR_PI


def trailing_leg_velocity(P: np.ndarray, start: np.ndarray, u: np.ndarray, core: float = 1e-9, rc=None) -> np.ndarray:
    """Velocity at ``P`` from a semi-infinite vortex leg of unit circulation that starts at ``start`` and
    runs to infinity along ``+u`` (the circulation sense is along ``+u``).

    The field is ``(1 + cos(theta)) / h`` times the usual factor, where the ``1`` is the symmetric part and
    ``cos(theta)`` the part due to the leg starting ahead of / behind the point (sweep, sideslip). With ``rc``
    (Vatistas n = 2 core radius) only the latter is cut off in the near field, ``cos -> cos h^2/sqrt(h^4+rc^4)``.
    """
    r = P - start
    ur = np.cross(u, r)
    d2 = np.einsum("...k,...k->...", ur, ur)
    rn = np.linalg.norm(r, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        cos_theta = (r @ u) / rn
        if rc is not None:
            cos_theta = cos_theta * d2 / np.sqrt(d2**2 + np.asarray(rc) ** 4)
        fac = np.where(d2 > core**2, (1.0 + cos_theta) / d2, 0.0)
    return ur * fac[..., None] / _FOUR_PI


def horseshoe_velocity(
    P: np.ndarray, A: np.ndarray, B: np.ndarray, u: np.ndarray, core: float = 1e-9, rc_legs=None, rc_bound=None
) -> np.ndarray:
    """Velocity at ``P`` from horseshoes of unit circulation: legs along ``u`` from ``A`` (incoming) and
    ``B`` (outgoing) joined by the bound segment ``A -> B``. ``rc_legs = (rc_A, rc_B)`` and ``rc_bound`` are
    optional core radii (see the two component functions)."""
    rc_a, rc_b = rc_legs if rc_legs is not None else (None, None)
    return (
        bound_segment_velocity(P, A, B, core, rc_bound)
        - trailing_leg_velocity(P, A, u, core, rc_a)
        + trailing_leg_velocity(P, B, u, core, rc_b)
    )


# ------------------------------------------------------------------------------------- the solution
@dataclass
class NumericalSolution:
    gamma: np.ndarray  # (N,) circulation of each horseshoe, m^2/s
    nodes: np.ndarray  # (N + 1, 3) quarter-chord points
    control: np.ndarray  # (N, 3) element mid-points
    dl: np.ndarray  # (N, 3) bound-vortex vectors
    velocity: np.ndarray  # (N, 3) local velocity at the control points, m/s
    frame_a: np.ndarray  # (N, 3) section chord direction (aft)
    frame_n: np.ndarray  # (N, 3) section normal
    chord_n: np.ndarray  # (N,) chord measured in the section plane
    area: np.ndarray  # (N,) element area c_n |dl|
    u_inf: np.ndarray
    V_inf: float
    converged: bool
    iterations: int
    residual: float  # max normalised |R_i|
    n_elements: int
    ground_height: float | None = None


class _LinearizedSection:
    """Linear lift ``cl = a0 (alpha - alpha_L0)`` from the section's linear factors, with its real drag and
    moment: the numerical solver's counterpart of the classical solver's *linear* mode."""

    def __init__(self, inner: SectionModel) -> None:
        self.inner = inner
        self.label = inner.label

    def linear(self, y, re) -> LinearParams:
        return self.inner.linear(y, re)

    def evaluate(self, y, alpha, re) -> SectionCoeffs:
        lin = self.inner.linear(y, re)
        full = self.inner.evaluate(y, alpha, re)
        return SectionCoeffs(
            cl=lin.a0 * (np.asarray(alpha, dtype=float) - lin.alpha_L0), cd=full.cd, cm=full.cm, cl_alpha=lin.a0
        )


def _station_reynolds(wing: Wing, flow, y: np.ndarray) -> np.ndarray:
    """Section Reynolds number from the freestream speed and the *streamwise* chord (as in the classical
    solver). Skin friction follows the freestream, so sweep does not shorten the Reynolds length."""
    return flow.reynolds(np.maximum(wing.chord(y), 1e-6))


# ------------------------------------------------------------------------------------------ geometry
def _section_frames(wing: Wing, control: np.ndarray, dl: np.ndarray):
    """Section frame (a, n, s), normal-plane chord and element area for each element."""
    length = np.linalg.norm(dl, axis=1)
    s = dl / length[:, None]
    x_hat = np.array([1.0, 0.0, 0.0])
    a0 = x_hat[None, :] - (s @ x_hat)[:, None] * s
    cos_sweep = np.linalg.norm(a0, axis=1)  # = a0 . x_hat after normalising
    a0 = a0 / cos_sweep[:, None]
    n0 = np.cross(a0, s)
    theta = np.radians(wing.twist_deg(control[:, 1]))
    c, sn = np.cos(theta)[:, None], np.sin(theta)[:, None]
    a = a0 * c + np.cross(s, a0) * sn
    n = n0 * c + np.cross(s, n0) * sn
    chord_n = wing.chord(control[:, 1]) * cos_sweep
    return a, n, s, chord_n, chord_n * length, length


def _mirror_about_ground(points: np.ndarray, plane_point: np.ndarray, normal: np.ndarray) -> np.ndarray:
    d = (points - plane_point) @ normal
    return points - 2.0 * d[..., None] * normal


def _influence(
    control: np.ndarray,
    nodes: np.ndarray,
    u: np.ndarray,
    ground: tuple | None,
    core: float,
    rc_nodes: np.ndarray | None = None,
    rc_bound: np.ndarray | None = None,
) -> np.ndarray:
    """Tensor ``v[i, k, :]``: velocity at control point ``i`` from the horseshoe of element ``k`` (unit
    circulation), including the ground image system when a ground plane is given. ``rc_nodes`` (N + 1) and
    ``rc_bound`` (N) are the leg and bound-segment core radii; the images keep the radii of their originals."""
    P = control[:, None, :]
    A = nodes[None, :-1, :]
    B = nodes[None, 1:, :]
    legs = None if rc_nodes is None else (rc_nodes[None, :-1], rc_nodes[None, 1:])
    bound = None if rc_bound is None else rc_bound[None, :]
    v = horseshoe_velocity(P, A, B, u, core, legs, bound)
    if ground is not None:
        plane_point, normal = ground
        Ai = _mirror_about_ground(nodes[:-1], plane_point, normal)[None, :, :]
        Bi = _mirror_about_ground(nodes[1:], plane_point, normal)[None, :, :]
        v = v - horseshoe_velocity(P, Ai, Bi, u, core, legs, bound)  # the image has opposite circulation
    return v


def _initial_circulation(wing, flow, section, controls, control, V_inf, chord_n, lin, alpha_geo) -> np.ndarray:
    """Starting circulation for Newton: the classical (Fourier) solution of the same wing, which already
    contains the induced-angle reduction and is exact for unswept planar wings. Falls back to the
    two-dimensional estimate if the classical solve cannot be made."""
    guess = 0.5 * V_inf * chord_n * lin.a0 * (alpha_geo - lin.alpha_L0)
    try:
        from .solver_classical import solve_classical

        sol = solve_classical(wing, flow, section, n_terms=40, nonlinear=False, controls=controls)
        theta = np.arccos(np.clip(-2.0 * control[:, 1] / wing.span, -1.0, 1.0))
        g = sol.gamma(theta)
        return g if np.all(np.isfinite(g)) else guess
    except Exception:  # pragma: no cover - defensive
        return guess


# --------------------------------------------------------------------------------------------- solver
class NumericalProblem:
    """The discretised system ``R(Gamma) = 0`` with its analytic Jacobian (kept separate so it can be tested)."""

    def __init__(
        self,
        wing: Wing,
        flow,
        section: SectionModel,
        *,
        n_panels: int,
        controls=None,
        ground_height=None,
        leg_core: float = LEG_CORE,
        bound_core: float = BOUND_CORE,
    ):
        if n_panels < 4:
            raise ValueError("n_panels must be at least 4")
        self.wing, self.flow, self.section, self.controls = wing, flow, section, controls
        N = self.N = 2 * n_panels
        b = wing.span
        y_nodes = -0.5 * b * np.cos(np.pi * np.arange(N + 1) / N)
        self.nodes = np.stack([wing.x_c4(y_nodes), y_nodes, wing.z_c4(y_nodes)], axis=1)
        self.control = 0.5 * (self.nodes[1:] + self.nodes[:-1])
        self.dl = self.nodes[1:] - self.nodes[:-1]
        self.a, self.n, _, self.chord_n, self.area, _ = _section_frames(wing, self.control, self.dl)

        self.alpha_root = math.radians(flow.alpha_deg + wing.incidence_deg)
        self.u = freestream_direction(self.alpha_root, math.radians(flow.beta_deg))
        self.Vinf = flow.V * self.u
        self.ay = np.abs(self.control[:, 1])
        self.d_alpha = controls.d_alpha(self.control[:, 1]) if controls is not None else 0.0

        ground = None
        if ground_height is not None:
            e_l = np.array([-math.sin(self.alpha_root), 0.0, math.cos(self.alpha_root)])
            plane_point = np.array([wing.x_c4(0.0), 0.0, wing.z_c4(0.0)]) - ground_height * e_l
            ground = (plane_point, e_l)
        rc_nodes = leg_core * wing.chord(y_nodes) if leg_core > 0 else None
        rc_bound = bound_core * wing.chord(self.control[:, 1]) if bound_core > 0 else None
        self.vinf = _influence(self.control, self.nodes, self.u, ground, 1e-9 * b, rc_nodes, rc_bound)  # (N, N, 3)
        self.re = _station_reynolds(wing, flow, self.control[:, 1])  # from the freestream (Gamma dependence ignored)
        self.scale = flow.V**2 * self.area

    def residual(self, G: np.ndarray):
        V = self.Vinf[None, :] + np.einsum("ikc,k->ic", self.vinf, G)
        Va, Vn = np.einsum("ic,ic->i", V, self.a), np.einsum("ic,ic->i", V, self.n)
        Vp2 = Va**2 + Vn**2
        co = self.section.evaluate(self.ay, np.arctan2(Vn, Va) + self.d_alpha, self.re)
        cross = np.cross(V, self.dl)
        cn = np.linalg.norm(cross, axis=1)
        return 2.0 * G * cn - Vp2 * self.area * co.cl, (V, Va, Vn, Vp2, cross, cn, co)

    def jacobian(self, G: np.ndarray, aux) -> np.ndarray:
        _, Va, Vn, Vp2, cross, cn, co = aux
        dVa = np.einsum("ikc,ic->ik", self.vinf, self.a)
        dVn = np.einsum("ikc,ic->ik", self.vinf, self.n)
        dcn = np.einsum("ic,ikc->ik", cross, np.cross(self.vinf, self.dl[:, None, :])) / cn[:, None]
        dVp2 = 2.0 * (Va[:, None] * dVa + Vn[:, None] * dVn)
        dalpha = (Va[:, None] * dVn - Vn[:, None] * dVa) / Vp2[:, None]
        J = 2.0 * np.diag(cn) + 2.0 * G[:, None] * dcn
        return J - self.area[:, None] * (dVp2 * co.cl[:, None] + Vp2[:, None] * co.cl_alpha[:, None] * dalpha)


def solve_numerical(
    wing: Wing,
    flow,
    section: SectionModel,
    *,
    n_panels: int = 80,
    nonlinear: bool = True,
    controls=None,
    ground_height: float | None = None,
    tol: float = 1e-10,
    max_iter: int = 80,
    leg_core: float = LEG_CORE,
    bound_core: float = BOUND_CORE,
) -> NumericalSolution:
    """Solve the numerical lifting-line equations; ``n_panels`` elements per half-wing. ``leg_core`` and
    ``bound_core`` are the chord-scaled vortex-core radii (0 gives the bare, unregularised lifting line)."""
    if not nonlinear:
        section = _LinearizedSection(section)
    prob = NumericalProblem(
        wing,
        flow,
        section,
        n_panels=n_panels,
        controls=controls,
        ground_height=ground_height,
        leg_core=leg_core,
        bound_core=bound_core,
    )
    lin = section.linear(prob.ay, prob.re)
    alpha_geo = np.arctan2(prob.n @ prob.Vinf, prob.a @ prob.Vinf) + prob.d_alpha
    gamma = _initial_circulation(wing, flow, section, controls, prob.control, flow.V, prob.chord_n, lin, alpha_geo)

    R, aux = prob.residual(gamma)
    converged, it = False, 0
    for it in range(1, max_iter + 1):
        if float(np.max(np.abs(R) / prob.scale)) < tol:
            converged = True
            break
        J = prob.jacobian(gamma, aux)
        try:
            step = -np.linalg.solve(J, R)
        except np.linalg.LinAlgError:
            step = -np.linalg.lstsq(J, R, rcond=None)[0]
        norm0 = float(np.linalg.norm(R / prob.scale))
        damp = 1.0
        for _ in range(14):  # backtracking on the normalised residual
            trial = gamma + damp * step
            R_try, aux_try = prob.residual(trial)
            if float(np.linalg.norm(R_try / prob.scale)) < norm0 or damp < 1e-3:
                break
            damp *= 0.5
        gamma, R, aux = trial, R_try, aux_try
    else:
        converged = float(np.max(np.abs(R) / prob.scale)) < tol

    return NumericalSolution(
        gamma=gamma,
        nodes=prob.nodes,
        control=prob.control,
        dl=prob.dl,
        velocity=aux[0],
        frame_a=prob.a,
        frame_n=prob.n,
        chord_n=prob.chord_n,
        area=prob.area,
        u_inf=prob.u,
        V_inf=flow.V,
        converged=converged,
        iterations=it,
        residual=float(np.max(np.abs(R) / prob.scale)),
        n_elements=prob.N,
        ground_height=ground_height,
    )


# ---------------------------------------------------------------------------------------------- loads
def build_load_numerical(
    sol: NumericalSolution, wing: Wing, flow, section: SectionModel, *, controls=None, nonlinear: bool = True
) -> LoadDistribution:
    """Per-element loads from a converged solution (Kutta-Joukowski force, profile drag, section moment)."""
    if not nonlinear:
        section = _LinearizedSection(section)
    V, a, n = sol.velocity, sol.frame_a, sol.frame_n
    control, dl = sol.control, sol.dl
    ay = np.abs(control[:, 1])
    d_alpha = controls.d_alpha(control[:, 1]) if controls is not None else 0.0
    d_cm = controls.d_cm(control[:, 1]) if controls is not None else 0.0
    Va, Vn = np.einsum("ic,ic->i", V, a), np.einsum("ic,ic->i", V, n)
    Vp2 = Va**2 + Vn**2
    alpha_eff = np.arctan2(Vn, Va) + d_alpha
    Vinf = sol.V_inf * sol.u_inf
    alpha_geo = np.arctan2(n @ Vinf, a @ Vinf) + d_alpha
    re = _station_reynolds(wing, flow, control[:, 1])
    co = section.evaluate(ay, alpha_eff, re)
    lin = section.linear(ay, re)

    rho = flow.rho
    force_kj = rho * sol.gamma[:, None] * np.cross(V, dl)
    speed = np.linalg.norm(V, axis=1)
    # Profile drag: lift and moment follow the velocity normal to the bound vortex (independence principle),
    # but skin friction follows the freestream, so the drag uses q_inf and the element's surface area - a swept
    # wing gets no cos^2(sweep) credit and the unswept limit coincides with the classical solver.
    drag_pe = 0.5 * rho * sol.V_inf**2 * sol.area * co.cd  # profile drag of each element, N
    force_profile = drag_pe[:, None] * V / speed[:, None]
    s = dl / np.linalg.norm(dl, axis=1)[:, None]
    moment = (0.5 * rho * Vp2 * sol.area * sol.chord_n * (co.cm + d_cm))[:, None] * s

    cn = np.linalg.norm(np.cross(V, dl), axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        cl_circ = np.where(Vp2 * sol.area > 0, 2.0 * sol.gamma * cn / (Vp2 * sol.area), 0.0)
    return LoadDistribution(
        y=control[:, 1],
        pos=control,
        width=np.abs(dl[:, 1]),
        chord=wing.chord(control[:, 1]),
        force_kj=force_kj,
        force_profile=force_profile,
        moment_c4=moment,
        gamma=sol.gamma,
        cl=cl_circ,
        cd=co.cd,
        cm=co.cm + d_cm,
        alpha_eff=alpha_eff,
        alpha_induced=alpha_geo - alpha_eff,
        re=re,
        cl_max=lin.cl_max,
        extras={"cl_section": co.cl, "alpha_geo": alpha_geo, "frame_n": n, "frame_a": a},
    )


def run_numerical(req, section, nonlinear: bool, warnings: list[str], info: dict):
    """Entry point used by :func:`liftingline.core.analysis.run`."""
    sol = solve_numerical(
        req.wing,
        req.flow,
        section,
        n_panels=req.n_panels,
        nonlinear=nonlinear,
        controls=req.controls,
        ground_height=req.ground_height,
    )
    if not sol.converged:
        warnings.append(
            f"The numerical lifting-line iteration did not fully converge ({sol.iterations} iterations, normalised "
            f"residual {sol.residual:.1e}); the wing is probably at or beyond stall."
        )
    load = build_load_numerical(sol, req.wing, req.flow, section, controls=req.controls, nonlinear=nonlinear)
    info.update(n_elements=sol.n_elements, iterations=sol.iterations, residual=sol.residual)
    return sol, load
