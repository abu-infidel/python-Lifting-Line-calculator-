"""A small, independent vortex-lattice method used only as a *reference* for the lifting-line solvers.

Deliberately written differently from ``liftingline.core.solver_numerical`` (angle-based Biot-Savart,
control points at 3/4 chord, Trefftz-plane drag) so that agreement is meaningful. Planar wings with sweep
and twist; flat plate sections. Lengths in metres, velocity ``V`` along +x (air moving aft, angle of attack
``alpha`` about the y axis), z up.
"""

from __future__ import annotations

import math

import numpy as np

FOUR_PI = 4.0 * math.pi


def segment_velocity(P, A, B):
    """Velocity at points P (n, 3) from a finite vortex segment A->B (unit circulation), angle formula."""
    t = B - A
    L = np.linalg.norm(t)
    t = t / L
    w = P - A
    s = w @ t
    d_vec = w - s[:, None] * t[None, :]
    h = np.linalg.norm(d_vec, axis=1)
    cos1 = s / np.linalg.norm(w, axis=1)
    cos2 = (s - L) / np.linalg.norm(P - B, axis=1)
    out = np.zeros_like(P)
    ok = h > 1e-12
    out[ok] = (np.cross(t, d_vec[ok] / h[ok, None]) * ((cos1 - cos2)[ok] / (FOUR_PI * h[ok]))[:, None])
    return out


def leg_velocity(P, A, u):
    """Velocity at P from a semi-infinite vortex leg starting at A along +u (circulation along +u)."""
    w = P - A
    s = w @ u
    d_vec = w - s[:, None] * u[None, :]
    h = np.linalg.norm(d_vec, axis=1)
    cos1 = s / np.linalg.norm(w, axis=1)
    out = np.zeros_like(P)
    ok = h > 1e-12
    out[ok] = np.cross(u, d_vec[ok] / h[ok, None]) * ((1.0 + cos1)[ok] / (FOUR_PI * h[ok]))[:, None]
    return out


def vlm(wing, alpha_deg: float, n_span: int = 24, n_chord: int = 6, beta_deg: float = 0.0, ground_height=None, wake="freestream"):
    """Vortex-lattice results for a planar wing with flat-plate sections.

    Returns a dict with ``CL``, ``CDi`` (Trefftz plane), ``C_roll`` (right-wing-down +, b-normalised, stability
    axes), ``CY`` (+ to starboard). ``ground_height`` is the height of the wing's c/4 line above a ground plane
    parallel to the flow (image vortices of opposite sign). ``wake`` is the direction of the trailing legs:
    ``"freestream"`` (default; right for symmetric flow) or ``"body"`` (along the wing's x axis, the classical
    vortex-lattice choice - in sideslip the freestream legs of the inner chordwise panels pass arbitrarily close
    to the control points of the outer ones and the result is erratic, so sideslip must use ``"body"``).
    """
    alpha, beta = math.radians(alpha_deg), math.radians(beta_deg)
    u = np.array([math.cos(alpha) * math.cos(beta), -math.sin(beta), math.sin(alpha) * math.cos(beta)])
    e_l = np.array([-math.sin(alpha), 0.0, math.cos(alpha)])
    b = wing.span
    yj = -0.5 * b * np.cos(np.pi * np.arange(n_span + 1) / n_span)  # strip edges, whole span
    P_ctrl, normals, bound = [], [], []
    for s in range(n_span):
        y0, y1, ym = yj[s], yj[s + 1], 0.5 * (yj[s] + yj[s + 1])
        for p in range(n_chord):
            def xb(y):  # bound vortex at the panel's 1/4 chord
                return float(wing.x_le(y) + wing.chord(y) * (p + 0.25) / n_chord)

            bound.append((np.array([xb(y0), y0, 0.0]), np.array([xb(y1), y1, 0.0])))
            P_ctrl.append([float(wing.x_le(ym) + wing.chord(ym) * (p + 0.75) / n_chord), ym, 0.0])
            th = math.radians(float(wing.twist_deg(ym)))
            normals.append([math.sin(th), 0.0, math.cos(th)])
    P_ctrl, normals = np.array(P_ctrl), np.array(normals)
    n = len(bound)
    plane_pt = np.array([0.0, 0.0, 0.0]) - (ground_height or 0.0) * e_l
    w_dir = u if wake == "freestream" else np.array([1.0, 0.0, 0.0])

    def mirror(x):
        return x - 2.0 * ((x - plane_pt) @ e_l) * e_l

    M = np.zeros((n, n))
    for k, (A, B) in enumerate(bound):
        v = segment_velocity(P_ctrl, A, B) - leg_velocity(P_ctrl, A, w_dir) + leg_velocity(P_ctrl, B, w_dir)
        if ground_height is not None:
            Ai, Bi = mirror(A), mirror(B)
            v = v - (segment_velocity(P_ctrl, Ai, Bi) - leg_velocity(P_ctrl, Ai, w_dir) + leg_velocity(P_ctrl, Bi, w_dir))
        M[:, k] = np.einsum("ij,ij->i", v, normals)
    gamma = np.linalg.solve(M, -np.einsum("j,ij->i", u, normals))  # V = 1

    S = wing.area
    dl = np.array([B - A for A, B in bound])
    mid = np.array([0.5 * (A + B) for A, B in bound])
    force = gamma[:, None] * np.cross(u[None, :], dl)  # rho = V = 1, freestream Kutta-Joukowski
    e_d = u
    e_y = np.cross(e_l, e_d)
    q = 0.5
    CL = float(force.sum(axis=0) @ e_l) / (q * S)
    CY = float(force.sum(axis=0) @ e_y) / (q * S)
    moment = np.cross(mid, force).sum(axis=0)
    e_xs = np.array([math.cos(alpha), 0.0, math.sin(alpha)])
    C_roll = float(-(moment @ e_xs)) / (q * S * b)

    # Trefftz-plane induced drag from the strip circulations (image wake included)
    strip = gamma.reshape(n_span, n_chord).sum(axis=1)
    edges = np.concatenate([[0.0], strip, [0.0]])
    wake = edges[:-1] - edges[1:]
    yc = 0.5 * (yj[:-1] + yj[1:])
    w_T = np.zeros(n_span)
    for i, y in enumerate(yc):
        dy = y - yj
        w_T[i] = np.sum(wake / (2.0 * math.pi * dy))
        if ground_height is not None:
            w_T[i] -= np.sum(wake * dy / (2.0 * math.pi * (dy**2 + (2.0 * ground_height) ** 2)))
    CDi = (-0.5 * float(np.sum(strip * w_T * np.diff(yj)))) / (q * S) * 1.0
    return dict(CL=CL, CDi=CDi, C_roll=C_roll, CY=CY, y=yc, gamma=strip)
