"""Loads and totals.

Both solvers deliver a :class:`LoadDistribution` (force and moment vectors per
spanwise cell in the wing's *internal frame*: x aft, y starboard, z up). This module
turns it into totals in wind axes (L, D, Y) and *stability* axes (moments), using
the aerospace sign conventions:

* pitch nose-up positive, roll right-wing-down positive, yaw nose-right positive;
* lift perpendicular to the freestream (up positive), drag along it (aft positive),
  side force positive to starboard.

Moments are taken about the reference point ``(x_ref, 0, z_ref)``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np

from .flow import FlowCondition
from .wing import Wing


def wind_axes(alpha: float, beta: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unit vectors (drag, side, lift) of the wind axes in the internal frame (angles in rad)."""
    ca, sa, cb, sb = math.cos(alpha), math.sin(alpha), math.cos(beta), math.sin(beta)
    e_d = np.array([ca * cb, -sb, sa * cb])  # direction the air moves relative to the wing
    e_l = np.array([-sa, 0.0, ca])
    e_y = np.cross(e_l, e_d)
    return e_d, e_y, e_l


def stability_axes(alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unit vectors (aft x_s, starboard y_s, up-ish z_s) of the stability axes in the internal frame."""
    ca, sa = math.cos(alpha), math.sin(alpha)
    return np.array([ca, 0.0, sa]), np.array([0.0, 1.0, 0.0]), np.array([-sa, 0.0, ca])


@dataclass
class LoadDistribution:
    """Per-cell loads and diagnostics, internal frame, SI units."""

    y: np.ndarray  # lateral position of each cell, m
    pos: np.ndarray  # (n, 3) application point (quarter chord) of each cell, m
    width: np.ndarray  # projected width of each cell, m
    chord: np.ndarray  # local chord, m
    force_kj: np.ndarray  # (n, 3) Kutta-Joukowski force on each cell, N
    force_profile: np.ndarray  # (n, 3) profile-drag force on each cell, N
    moment_c4: np.ndarray  # (n, 3) section pitching-moment vector about its c/4 point, N m
    gamma: np.ndarray  # circulation, m^2/s
    cl: np.ndarray
    cd: np.ndarray
    cm: np.ndarray
    alpha_eff: np.ndarray  # effective angle of attack, rad
    alpha_induced: np.ndarray  # induced angle, rad
    re: np.ndarray
    cl_max: np.ndarray  # inf where unknown
    extras: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def cl_margin(self) -> np.ndarray:
        """cl / cl_max: 1 means the station is at its maximum lift coefficient."""
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(np.isfinite(self.cl_max) & (self.cl_max > 0), self.cl / self.cl_max, 0.0)


@dataclass
class Totals:
    # forces (N) and moments (N m) ------------------------------------------------
    L: float
    D: float  # total drag including extra parasite drag
    D_induced: float
    D_profile: float
    D_extra: float
    Y: float  # side force, + to starboard
    M_pitch: float  # nose-up +
    M_roll: float  # right-wing-down +
    M_yaw: float  # nose-right +
    # coefficients ------------------------------------------------------------------
    CL: float
    CD: float
    CD_induced: float
    CD_profile: float
    CD_extra: float
    CY: float
    Cm: float  # about the reference point, MAC-normalised
    C_roll: float  # b-normalised
    C_yaw: float  # b-normalised
    # reference & derived -------------------------------------------------------------
    S: float
    b: float
    AR: float
    MAC: float
    q: float
    e: float  # span efficiency CL^2/(pi AR CDi), nan when CDi = 0
    L_over_D: float
    root_bending_right: float  # N m, lift-induced, right semi-span about the root
    root_bending_left: float
    root_shear: float  # N, lift on one semi-span
    x_ref: float
    z_ref: float
    alpha_deg: float  # root-chord angle of attack (flow + incidence)
    beta_deg: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def compute_totals(
    load: LoadDistribution,
    wing: Wing,
    flow: FlowCondition,
    *,
    x_ref: float,
    z_ref: float = 0.0,
    extra_cd0: float = 0.0,
) -> Totals:
    alpha = math.radians(flow.alpha_deg + wing.incidence_deg)
    beta = math.radians(flow.beta_deg)
    e_d, e_y, e_l = wind_axes(alpha, beta)
    e_xs, e_ys, e_zs = stability_axes(alpha)

    f_kj = load.force_kj.sum(axis=0)
    f_pr = load.force_profile.sum(axis=0)
    S, b, q = wing.area, wing.span, flow.q

    d_extra = q * S * extra_cd0
    f_total = f_kj + f_pr + d_extra * e_d
    L = float(f_total @ e_l)
    D = float(f_total @ e_d)
    Y = float(f_total @ e_y)
    D_induced = float(f_kj @ e_d)
    D_profile = float(f_pr @ e_d)

    ref = np.array([x_ref, 0.0, z_ref])
    m_vec = (np.cross(load.pos - ref, load.force_kj + load.force_profile) + load.moment_c4).sum(axis=0)
    M_pitch = float(m_vec @ e_ys)
    M_roll = float(-(m_vec @ e_xs))
    M_yaw = float(-(m_vec @ e_zs))

    cell_lift = (load.force_kj + load.force_profile) @ e_l
    # a cell on the plane of symmetry (|y| ~ round-off) is shared equally by both semi-spans
    centre = np.abs(load.y) <= 1e-9 * b
    right = np.where(centre, 0.5, (load.y > 0.0).astype(float))
    left = np.where(centre, 0.5, (load.y < 0.0).astype(float))
    bending_r = float(np.sum(right * np.abs(load.y) * cell_lift))
    bending_l = float(np.sum(left * np.abs(load.y) * cell_lift))
    shear = float(np.sum(right * cell_lift))

    qS = q * S
    CL, CD, CDi, CDp = L / qS, D / qS, D_induced / qS, D_profile / qS
    e_span = CL**2 / (math.pi * wing.aspect_ratio * CDi) if CDi > 1e-14 else math.nan
    return Totals(
        L=L,
        D=D,
        D_induced=D_induced,
        D_profile=D_profile,
        D_extra=d_extra,
        Y=Y,
        M_pitch=M_pitch,
        M_roll=M_roll,
        M_yaw=M_yaw,
        CL=CL,
        CD=CD,
        CD_induced=CDi,
        CD_profile=CDp,
        CD_extra=extra_cd0,
        CY=Y / qS,
        Cm=M_pitch / (qS * wing.mac),
        C_roll=M_roll / (qS * b),
        C_yaw=M_yaw / (qS * b),
        S=S,
        b=b,
        AR=wing.aspect_ratio,
        MAC=wing.mac,
        q=q,
        e=e_span,
        L_over_D=L / D if abs(D) > 0.0 else math.inf,
        root_bending_right=bending_r,
        root_bending_left=bending_l,
        root_shear=shear,
        x_ref=x_ref,
        z_ref=z_ref,
        alpha_deg=flow.alpha_deg + wing.incidence_deg,
        beta_deg=flow.beta_deg,
    )
