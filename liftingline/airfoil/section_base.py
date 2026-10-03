"""Common machinery for section engines.

An *engine* turns an :class:`~liftingline.airfoil.geometry.Airfoil` into an :class:`AirfoilAero`:
coefficient functions ``cl, cd, cm (alpha, Re)`` plus the derived "factors" used by the linear
lifting-line solve (lift slope, zero-lift angle, moment, ``cl_max``).

Engines that are expensive to call (NeuralFoil, imported polars) work through *polar tables*: for
each airfoil an angle-of-attack table is built lazily at nodes of a **global Reynolds-number grid**
(factor ``10**0.25`` apart) and interpolated linearly in ``log10(Re)`` between neighbouring nodes.
Because the grid is global the tables are reused across sweeps and wings.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from scipy.interpolate import CubicSpline

from ..core.section import LinearParams
from .geometry import Airfoil

#: tabulated angle-of-attack window (degrees) and spacing
ALPHA_MIN_DEG, ALPHA_MAX_DEG, ALPHA_STEP_DEG = -25.0, 30.0, 0.5
#: global Reynolds-number node grid: Re_k = 10 ** (RE_LOG0 + RE_DLOG * k)
RE_LOG0, RE_DLOG, RE_NODES = 4.5, 0.25, 15  # 3.2e4 ... 1.0e8
RE_MIN = 10.0**RE_LOG0
RE_MAX = 10.0 ** (RE_LOG0 + RE_DLOG * (RE_NODES - 1))


class EngineUnavailable(RuntimeError):
    """A requested section engine cannot be used (missing optional package, ...)."""


@dataclass(frozen=True)
class EngineSettings:
    """Boundary-layer settings used by engines that model transition (NeuralFoil)."""

    n_crit: float = 9.0  # e^N amplification factor: 9 average wind tunnel, 12-14 sailplane, 4-8 dirty
    xtr_upper: float = 1.0  # forced transition location (x/c); 1.0 = free transition
    xtr_lower: float = 1.0
    model_size: str = "xlarge"  # NeuralFoil network size: xxsmall ... xxxlarge


@runtime_checkable
class AirfoilAero(Protocol):
    """Aerodynamic data of one airfoil, incompressible, as functions of alpha and Re."""

    label: str
    approximate: bool

    def linear(self, re: np.ndarray) -> LinearParams: ...

    def coefficients(self, alpha: np.ndarray, re: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(cl, cd, cm_c4, dcl/dalpha) at angles ``alpha`` (rad) and Reynolds numbers ``re``."""
        ...

    def confidence(self, alpha: np.ndarray, re: np.ndarray) -> np.ndarray | None:
        """Engine confidence in [0, 1] at each point, or None if the engine does not report one."""
        ...


@runtime_checkable
class SectionEngine(Protocol):
    name: str

    def aero(self, airfoil: Airfoil) -> AirfoilAero: ...


# --------------------------------------------------------------------------------------
def alpha_grid_deg() -> np.ndarray:
    return np.arange(ALPHA_MIN_DEG, ALPHA_MAX_DEG + 0.5 * ALPHA_STEP_DEG, ALPHA_STEP_DEG)


def re_node_value(k: int) -> float:
    return 10.0 ** (RE_LOG0 + RE_DLOG * k)


@dataclass(frozen=True)
class PolarTable:
    """Coefficients on the angle-of-attack grid at one Reynolds number."""

    alpha_deg: np.ndarray
    cl: np.ndarray
    cd: np.ndarray
    cm: np.ndarray
    conf: np.ndarray | None = None


def linear_parameters(alpha_deg: np.ndarray, cl: np.ndarray, cm: np.ndarray) -> dict[str, float]:
    """Linear-range factors from a polar: lift slope (per rad), zero-lift angle (rad), c/4 moment at
    zero lift, ``cl_max`` and the angle at which it occurs (rad).

    The zero-lift angle is the upward zero crossing of ``cl`` nearest ``alpha = 0`` (so a symmetric
    section gets exactly 0). The slope is the *secant through that point*: a least-squares line forced
    through ``(alpha_L0, 0)`` over ``alpha_L0 + [0.5, 6]`` deg. Real polars are not straight (laminar
    bubbles steepen them around 4-7 deg at moderate Re); a free-intercept fit would shift the zero-lift
    angle, which matters more than the exact slope. For accurate work use the polar itself (nonlinear
    solve); these factors are for the linear solve, for display and as the Newton starting point.
    """
    a = np.asarray(alpha_deg, dtype=float)
    cl = np.asarray(cl, dtype=float)
    sel = np.where((a >= -15.0) & (a <= 8.0))[0]
    crossings = [i for i in sel[:-1] if cl[i] <= 0.0 < cl[i + 1]]
    if crossings:
        i = min(crossings, key=lambda j: abs(a[j]))
        aL0_deg = a[i] - cl[i] * (a[i + 1] - a[i]) / (cl[i + 1] - cl[i])
    else:
        aL0_deg = float(a[sel][np.argmin(np.abs(cl[sel]))])
    win = (a >= aL0_deg + 0.5) & (a <= aL0_deg + 6.0)
    if win.sum() < 4:
        win = (a >= aL0_deg + 0.25) & (a <= aL0_deg + 8.0)
    x = np.radians(a[win] - aL0_deg)
    slope = float(np.sum(x * cl[win]) / np.sum(x * x))
    alpha_L0 = math.radians(aL0_deg)
    cm0 = float(np.interp(aL0_deg, a, cm))
    # first stall peak above the linear range; fall back to the global maximum of the window
    hi = np.where(a >= aL0_deg + 3.0)[0]
    peak = None
    for j in hi[1:-1]:
        if cl[j] >= cl[j - 1] and cl[j] > cl[j + 1] and cl[j] > 0.3:
            peak = j
            break
    if peak is None:
        peak = hi[int(np.argmax(cl[hi]))]
    return dict(
        a0=slope,
        alpha_L0=float(alpha_L0),
        cm_c4=cm0,
        cl_max=float(cl[peak]),
        alpha_stall=float(math.radians(a[peak])),
    )


class _Node:
    """Interpolants of one Reynolds-number node (``interp`` is a spline class taking ``(x, y)``)."""

    def __init__(self, table: PolarTable, interp=CubicSpline) -> None:
        a = table.alpha_deg
        self.lo, self.hi = float(a[0]), float(a[-1])
        self.cl = interp(a, table.cl)
        self.cd = interp(a, table.cd)
        self.cm = interp(a, table.cm)
        self.conf = None if table.conf is None else interp(a, table.conf)
        self.lin = linear_parameters(a, table.cl, table.cm)


class TabulatedAirfoilAero:
    """Base class: builds :class:`PolarTable` s lazily per Reynolds node and interpolates them."""

    label = "tabulated"
    approximate = False
    _interp = staticmethod(CubicSpline)  # dense tables: cubic spline; sparse imported polars use PCHIP

    def __init__(self) -> None:
        self._nodes: dict[int, _Node] = {}

    # to be provided by subclasses -------------------------------------------------
    def _build_table(self, re_node: float) -> PolarTable:  # pragma: no cover - abstract
        raise NotImplementedError

    # node handling ----------------------------------------------------------------
    def _node(self, k: int) -> _Node:
        node = self._nodes.get(k)
        if node is None:
            node = self._nodes[k] = _Node(self._build_table(re_node_value(k)), self._interp)
        return node

    @staticmethod
    def _locate(re: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Lower node index and interpolation weight (towards the upper node) for each Re."""
        x = (np.log10(np.clip(re, RE_MIN, RE_MAX)) - RE_LOG0) / RE_DLOG
        k0 = np.clip(np.floor(x).astype(int), 0, RE_NODES - 2)
        return k0, np.clip(x - k0, 0.0, 1.0)

    def _blend(self, re: np.ndarray, per_node):
        """Linear-in-log(Re) blend of ``per_node(node, mask)`` results (arrays shaped like ``re``)."""
        re = np.asarray(re, dtype=float)
        k0, w = self._locate(re)
        out = None
        for k in np.unique(np.concatenate([k0, k0 + 1])):
            weight = np.where(k0 == k, 1.0 - w, 0.0) + np.where(k0 + 1 == k, w, 0.0)
            sel = weight > 0.0
            if not np.any(sel):
                continue
            vals = per_node(self._node(int(k)), sel)
            if out is None:
                out = [np.zeros(re.shape) for _ in vals]
            for o, v in zip(out, vals):
                o[sel] += weight[sel] * v
        return out

    # AirfoilAero ------------------------------------------------------------------
    def linear(self, re: np.ndarray) -> LinearParams:
        re = np.asarray(re, dtype=float)

        def per_node(node: _Node, sel):
            n = int(sel.sum())
            L = node.lin
            return [np.full(n, L[key]) for key in ("a0", "alpha_L0", "cm_c4", "cl_max", "alpha_stall")]

        a0, aL0, cm, clm, ast = self._blend(re, per_node)
        return LinearParams(a0=a0, alpha_L0=aL0, cm_c4=cm, cl_max=clm, alpha_stall=ast)

    def coefficients(self, alpha: np.ndarray, re: np.ndarray):
        alpha = np.asarray(alpha, dtype=float)
        re = np.asarray(re, dtype=float) * np.ones_like(alpha)
        deg = np.degrees(alpha)

        def per_node(node: _Node, sel):
            d = np.clip(deg[sel], node.lo, node.hi)
            inside = (deg[sel] > node.lo) & (deg[sel] < node.hi)
            slope = np.where(inside, node.cl(d, 1) * (180.0 / math.pi), 0.0)  # per rad
            return [node.cl(d), node.cd(d), node.cm(d), slope]

        cl, cd, cm, slope = self._blend(re, per_node)
        return cl, np.maximum(cd, 1e-5), cm, slope

    def confidence(self, alpha: np.ndarray, re: np.ndarray):
        alpha = np.asarray(alpha, dtype=float)
        re = np.asarray(re, dtype=float) * np.ones_like(alpha)
        deg = np.degrees(alpha)
        probe = self._node(int(self._locate(np.array([float(np.median(re))]))[0][0]))
        if probe.conf is None:
            return None

        def per_node(node: _Node, sel):
            d = np.clip(deg[sel], node.lo, node.hi)
            c = np.clip(node.conf(d), 0.0, 1.0)
            return [np.where((deg[sel] < node.lo) | (deg[sel] > node.hi), 0.0, c)]

        return self._blend(re, per_node)[0]
