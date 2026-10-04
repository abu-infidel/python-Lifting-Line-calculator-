"""High-lift devices for takeoff and landing: deployable trailing-edge flaps and leading-edge slats.

What the devices do to a wing section, and the published (test-based) numbers used
----------------------------------------------------------------------------------
All increments are the handbook values of D. P. Raymer, *Aircraft Design: A Conceptual Approach* (AIAA), chapter 12
("Maximum lift", "Flaps"), which condense NACA/industry wind-tunnel results:

* **Maximum lift of the section**, ``d(cl_max)`` at the *landing* setting (Raymer, Table 12.2):

  ===========================  ================  ==========================  ==============
  trailing-edge flap           d(cl_max)         leading-edge device         d(cl_max)
  ===========================  ================  ==========================  ==============
  plain, split                 0.9               fixed slot                  0.2
  single slotted               1.3               leading-edge (droop) flap   0.3
  Fowler                       1.3 c'/c          Krueger flap                0.3
  double slotted               1.6 c'/c          slat                        0.4 c'/c
  triple slotted               1.9 c'/c
  ===========================  ================  ==========================  ==============

  ``c'/c`` is the chord with the device extended over the clean chord. At the *takeoff* setting the flaps are
  deflected less and give about 60-80 % of the landing increment; the middle of that range, 70 %, is used.
  Leading-edge devices are taken as fully effective in both phases.
* **Zero-lift angle of a flapped section**: -15 deg at the landing setting and -10 deg at takeoff (Raymer,
  "Flaps"). Leading-edge devices do not change it: they extend the lift curve to a higher stall angle.
* **Lift-curve slope**: a chord-extending flap (Fowler-type, ``c'/c > 1``) raises the slope, referred to the clean
  chord, in proportion to the chord extension (Raymer).
* **Profile drag of a flap** (Raymer's flap-drag correlation): ``d(cd) = F (cf/c) (delta - 10 deg)`` with
  ``F = 0.0144`` per degree for plain and split flaps and ``0.0074`` for slotted flaps, at the typical deflection of
  each flap type (takeoff 15-20 deg, landing 40-60 deg). Integrated over the span this reproduces Raymer's
  ``d(CD0) = F (cf/c) (S_flapped/S_ref) (delta - 10)`` exactly. Leading-edge devices add no drag in this model.
* **Pitching moment of a flapped section**: thin-airfoil flap theory relates the moment to the zero-lift-angle
  shift for the flap chord ``cf`` (the same relation the Controls tab uses), so a flap pitches the nose down.

How the solver uses them
------------------------
The clean section polar ``cl(alpha)`` (NeuralFoil, built-in engine, imported polar ...) is mapped, station by station,
onto the high-lift polar

    cl_HL(alpha) = s * cl(alpha_c),     alpha_c = alpha_L0 + k (alpha - alpha_L0 - d_alpha_L0) / s,

with ``s = (cl_max + d(cl_max)) / cl_max`` and ``k`` the slope factor. This keeps the shape of the section's own
stall, gives exactly the lift-curve slope ``k a0``, the zero-lift angle ``alpha_L0 + d_alpha_L0`` and the maximum lift
``cl_max + d(cl_max)``, and moves the stall angle accordingly (down for flaps, up for slats). Stations without a
device are evaluated unchanged (bit for bit), so the lifting-line solution itself decides how much of the section
gain the wing realises - the classical wing-level estimate ``d(CL_max) = 0.9 d(cl_max) (S_flapped / S_ref) cos
(sweep of the hinge line)`` (Raymer) is computed alongside for comparison. The ends of each device are blended over
4 % of the semi-span (the spanwise flow at a flap edge does the same), which keeps the lifting-line solution free of
the singular trailing vortex an abrupt jump would shed.

Devices act alike on both wings. They are meant for low-speed takeoff and landing; deploying them above about
Mach 0.35 produces a warning (flap placard speeds are far lower).
"""

from __future__ import annotations

import math
import textwrap
from dataclasses import dataclass, field, replace
from typing import Callable

import numpy as np

from .controls import flap_effectiveness, flap_moment_slope
from .section import LinearParams, SectionCoeffs, SectionModel

TAKEOFF, LANDING = "takeoff", "landing"
PHASES = (TAKEOFF, LANDING)
PHASE_LABELS = {TAKEOFF: "Takeoff", LANDING: "Landing"}

TRAILING, LEADING = "trailing", "leading"

#: change of the section zero-lift angle by a deflected flap, deg (Raymer)
DALPHA_L0_DEG = {TAKEOFF: -10.0, LANDING: -15.0}
#: takeoff increment of maximum lift as a fraction of the landing one (Raymer: 60-80 %; the middle of the range)
TAKEOFF_FRACTION = 0.7
#: wing-level estimate factor of Raymer's d(CL_max) = 0.9 d(cl_max) (S_flapped/S_ref) cos(sweep_hinge)
WING_FACTOR = 0.9
#: the ends of a device are blended over this fraction of the semi-span (centred on the nominal end)
EDGE_BLEND = 0.04
#: above this Mach number deployed high-lift devices are flagged (they are retracted long before)
MACH_WARNING = 0.35


@dataclass(frozen=True)
class DeviceType:
    """One kind of high-lift device with its published section increments."""

    key: str
    label: str
    edge: str  # TRAILING (flap) or LEADING (slat-type device)
    dcl_max: float  # section increment of cl_max at the landing setting (Raymer, Table 12.2)
    times_extension: bool  # the increment is quoted per unit chord extension (multiply by c'/c)
    default_extension: float  # a typical c'/c of the deployed device (1 = no chord extension)
    deflection_deg: tuple[float, float] = (0.0, 0.0)  # typical (takeoff, landing) deflection, for the drag estimate
    drag_factor: float = 0.0  # Raymer's F_flap per degree (trailing-edge flaps only)
    description: str = ""

    def table_text(self) -> str:
        return f"{self.dcl_max:g}" + (" x c'/c" if self.times_extension else "")


FLAP_TYPES: dict[str, DeviceType] = {
    t.key: t
    for t in (
        DeviceType("plain", "Plain flap", TRAILING, 0.9, False, 1.0, (20.0, 60.0), 0.0144,
                   "a hinged rear part of the airfoil"),
        DeviceType("split", "Split flap", TRAILING, 0.9, False, 1.0, (20.0, 60.0), 0.0144,
                   "a lower-surface plate hinged down"),
        DeviceType("slotted", "Single-slotted flap", TRAILING, 1.3, False, 1.0, (20.0, 40.0), 0.0074,
                   "a flap with a slot that re-energises the boundary layer"),
        DeviceType("fowler", "Fowler flap", TRAILING, 1.3, True, 1.25, (15.0, 40.0), 0.0074,
                   "a slotted flap that also moves aft and extends the chord"),
        DeviceType("double_slotted", "Double-slotted flap", TRAILING, 1.6, True, 1.2, (20.0, 50.0), 0.0074,
                   "two flap elements with two slots"),
        DeviceType("triple_slotted", "Triple-slotted flap", TRAILING, 1.9, True, 1.3, (20.0, 55.0), 0.0074,
                   "three flap elements (large transports)"),
    )
}

SLAT_TYPES: dict[str, DeviceType] = {
    t.key: t
    for t in (
        DeviceType("fixed_slot", "Fixed slot", LEADING, 0.2, False, 1.0, description="a permanent slot behind the nose"),
        DeviceType("le_flap", "Leading-edge flap (droop)", LEADING, 0.3, False, 1.0, description="the nose hinged down"),
        DeviceType("krueger", "Krueger flap", LEADING, 0.3, False, 1.0, description="a panel swung forward from the lower surface"),
        DeviceType("slat", "Slat", LEADING, 0.4, True, 1.1, description="a nose element moved forward, opening a slot"),
    )
}

DATA_SOURCE = (
    "D. P. Raymer, Aircraft Design: A Conceptual Approach (AIAA), ch. 12: Table 12.2 (section increments of maximum "
    "lift), zero-lift-angle shift of flapped sections, wing-level estimate and flap-drag correlation"
)


class HighLiftError(ValueError):
    """A high-lift device description that cannot be used."""


@dataclass(frozen=True)
class Device:
    """The geometry of one deployed device (it acts alike on both wings)."""

    type: str
    y_in: float = 0.1  # inboard end, fraction of the semi-span
    y_out: float = 0.6  # outboard end, fraction of the semi-span
    chord_fraction: float = 0.3  # flap chord / local chord (trailing-edge devices: drag and moment)
    chord_extension: float = 1.0  # c'/c with the device deployed

    def device_type(self) -> DeviceType:
        if self.type in FLAP_TYPES:
            return FLAP_TYPES[self.type]
        if self.type in SLAT_TYPES:
            return SLAT_TYPES[self.type]
        raise HighLiftError(f"unknown high-lift device type {self.type!r}")

    def problems(self, edge: str) -> list[str]:
        out = []
        table = FLAP_TYPES if edge == TRAILING else SLAT_TYPES
        what = "flap" if edge == TRAILING else "leading-edge device"
        if self.type not in table:
            out.append(f"unknown {what} type {self.type!r}")
        if not (0.0 <= self.y_in < self.y_out <= 1.0) or not (self.y_out - self.y_in >= 0.05):
            out.append(f"the {what} span must satisfy 0 <= inboard < outboard <= 100 % of the semi-span and cover at least 5 %")
        if edge == TRAILING and not (0.1 <= self.chord_fraction <= 0.45):
            out.append("the flap chord must be between 10 % and 45 % of the local chord")
        if not (1.0 <= self.chord_extension <= 1.5):
            out.append(f"the chord extension c'/c of the {what} must be between 1.0 and 1.5")
        return out


@dataclass(frozen=True)
class Increments:
    """Section changes at a set of stations (arrays)."""

    dalpha_L0: np.ndarray  # rad
    dcl_max: np.ndarray
    slope: np.ndarray  # factor on the lift-curve slope
    dcd: np.ndarray
    dcm: np.ndarray


@dataclass(frozen=True)
class WingEstimate:
    """Raymer's wing-level estimate (for comparison with the lifting-line result)."""

    dCL_max: float
    dCL_max_flaps: float
    dCL_max_slats: float
    dalpha_L0_deg: float  # wing zero-lift angle change
    dCD0_flaps: float
    flapped_area_ratio: float
    slatted_area_ratio: float
    hinge_sweep_deg: float
    le_sweep_deg: float


@dataclass(frozen=True)
class HighLiftSystem:
    """Which devices are deployed, for which phase. ``None`` for a device means it is retracted."""

    phase: str = LANDING
    flaps: Device | None = None
    slats: Device | None = None

    def __post_init__(self) -> None:
        msgs = self.problems()
        if msgs:
            raise HighLiftError("; ".join(msgs))

    def problems(self) -> list[str]:
        out = []
        if self.phase not in PHASES:
            out.append(f"unknown flight phase {self.phase!r} (takeoff or landing)")
        if self.flaps is not None:
            out += self.flaps.problems(TRAILING)
        if self.slats is not None:
            out += self.slats.problems(LEADING)
        return out

    @property
    def active(self) -> bool:
        return self.flaps is not None or self.slats is not None

    # ------------------------------------------------------------------ section values
    @property
    def flap_type(self) -> DeviceType | None:
        return None if self.flaps is None else FLAP_TYPES[self.flaps.type]

    @property
    def slat_type(self) -> DeviceType | None:
        return None if self.slats is None else SLAT_TYPES[self.slats.type]

    def flap_dcl_max(self) -> float:
        """Section increment of cl_max by the flaps in this phase."""
        t = self.flap_type
        if t is None:
            return 0.0
        value = t.dcl_max * (self.flaps.chord_extension if t.times_extension else 1.0)
        return value * (TAKEOFF_FRACTION if self.phase == TAKEOFF else 1.0)

    def slat_dcl_max(self) -> float:
        """Section increment of cl_max by the leading-edge devices (same in both phases)."""
        t = self.slat_type
        if t is None:
            return 0.0
        return t.dcl_max * (self.slats.chord_extension if t.times_extension else 1.0)

    def flap_dalpha_L0_deg(self) -> float:
        return DALPHA_L0_DEG[self.phase] if self.flaps is not None else 0.0

    def flap_slope_factor(self) -> float:
        """Lift-curve slope factor of a flapped section (chord extension of Fowler-type flaps)."""
        t = self.flap_type
        return self.flaps.chord_extension if (t is not None and t.times_extension) else 1.0

    def flap_deflection_deg(self) -> float:
        t = self.flap_type
        if t is None:
            return 0.0
        return t.deflection_deg[0] if self.phase == TAKEOFF else t.deflection_deg[1]

    def flap_dcd(self) -> float:
        """Section profile-drag increment of the flaps (Raymer's flap-drag correlation)."""
        t = self.flap_type
        if t is None:
            return 0.0
        return max(t.drag_factor * self.flaps.chord_fraction * (self.flap_deflection_deg() - 10.0), 0.0)

    def flap_dcm(self) -> float:
        """Section c/4 moment increment of the flaps: thin-airfoil ratio of moment to zero-lift-angle shift."""
        if self.flaps is None:
            return 0.0
        cf = self.flaps.chord_fraction
        delta_equivalent = -math.radians(self.flap_dalpha_L0_deg()) / flap_effectiveness(cf)
        return flap_moment_slope(cf) * delta_equivalent

    # ------------------------------------------------------------------ spanwise layout
    @staticmethod
    def _weight(eta: np.ndarray, dev: Device | None) -> np.ndarray:
        """1 inside the device, 0 outside, linear ramps of width EDGE_BLEND centred on the ends.

        An end at the root (0) or at the tip (1) is not blended: there is no neighbouring clean section."""
        if dev is None:
            return np.zeros_like(eta)
        h = 0.5 * EDGE_BLEND
        w = np.ones_like(eta)
        if dev.y_in > 0.0:
            w = np.minimum(w, np.clip((eta - (dev.y_in - h)) / EDGE_BLEND, 0.0, 1.0))
        if dev.y_out < 1.0:
            w = np.minimum(w, np.clip(((dev.y_out + h) - eta) / EDGE_BLEND, 0.0, 1.0))
        return w

    def increments(self, y: np.ndarray, half_span: float) -> Increments:
        eta = np.abs(np.asarray(y, dtype=float)) / half_span
        wf, ws = self._weight(eta, self.flaps), self._weight(eta, self.slats)
        return Increments(
            dalpha_L0=wf * math.radians(self.flap_dalpha_L0_deg()),
            dcl_max=wf * self.flap_dcl_max() + ws * self.slat_dcl_max(),
            slope=1.0 + wf * (self.flap_slope_factor() - 1.0),
            dcd=wf * self.flap_dcd(),
            dcm=wf * self.flap_dcm(),
        )

    # -------------------------------------------------------------------- descriptions
    def short_label(self) -> str:
        parts = [t.label.lower() for t in (self.flap_type, self.slat_type) if t is not None]
        return f"{PHASE_LABELS[self.phase].lower()} setting: " + " + ".join(parts) if parts else "clean"

    def note(self) -> str:
        """The assumptions of the high-lift model, for the validity notes of a result."""
        bits = []
        if self.flaps is not None:
            f = self.flaps
            bits.append(
                f"{self.flap_type.label.lower()}s over {100 * f.y_in:.0f}-{100 * f.y_out:.0f} % of the semi-span "
                f"(section d(cl_max) {self.flap_dcl_max():+.2f}, zero-lift angle {self.flap_dalpha_L0_deg():+.0f} deg, "
                f"typical deflection {self.flap_deflection_deg():.0f} deg, d(cd) {self.flap_dcd():+.4f})"
            )
        if self.slats is not None:
            s = self.slats
            bits.append(
                f"{self.slat_type.label.lower()} over {100 * s.y_in:.0f}-{100 * s.y_out:.0f} % "
                f"(section d(cl_max) {self.slat_dcl_max():+.2f})"
            )
        return (
            f"High-lift devices deployed, {PHASE_LABELS[self.phase].lower()} setting: " + "; ".join(bits)
            + ". Section increments are published handbook values (Raymer); stall and maximum-lift results are "
            "estimates of the same quality."
        )

    # --------------------------------------------------------------- wing-level estimate
    def wing_estimate(self, wing) -> WingEstimate:
        """Raymer's wing-level estimate of the maximum-lift and zero-lift-angle changes and the flap drag."""
        hs, area = wing.half_span, wing.area

        def strip(dev: Device | None, x_frac: float) -> tuple[float, float]:
            if dev is None:
                return 0.0, 0.0
            y = np.linspace(dev.y_in * hs, dev.y_out * hs, 401)
            c = wing.chord(y)
            s_dev = float(np.sum((c[1:] + c[:-1]) * np.diff(y)))  # both wings: 2 x trapezoidal rule
            x_line = wing.x_le(y) + x_frac * c
            sweep = math.degrees(math.atan2(float(x_line[-1] - x_line[0]), float(y[-1] - y[0])))
            return s_dev / area, sweep

        rf, hinge = strip(self.flaps, 1.0 - (self.flaps.chord_fraction if self.flaps else 0.0))
        rs, le = strip(self.slats, 0.0)
        cf, cs = math.cos(math.radians(hinge)), math.cos(math.radians(le))
        d_f = WING_FACTOR * self.flap_dcl_max() * rf * cf
        d_s = WING_FACTOR * self.slat_dcl_max() * rs * cs
        dcd0 = (self.flap_type.drag_factor * self.flaps.chord_fraction * rf * max(self.flap_deflection_deg() - 10.0, 0.0)
                if self.flaps is not None else 0.0)
        return WingEstimate(
            dCL_max=d_f + d_s, dCL_max_flaps=d_f, dCL_max_slats=d_s,
            dalpha_L0_deg=self.flap_dalpha_L0_deg() * rf * cf, dCD0_flaps=dcd0,
            flapped_area_ratio=rf, slatted_area_ratio=rs, hinge_sweep_deg=hinge, le_sweep_deg=le,
        )

    # ------------------------------------------------------------------ serialisation
    @classmethod
    def from_spec(cls, phase: str, flaps: dict | None, slats: dict | None) -> "HighLiftSystem":
        def dev(d: dict | None) -> Device | None:
            if d is None:
                return None
            try:
                return Device(
                    type=str(d["type"]), y_in=float(d["y_in"]), y_out=float(d["y_out"]),
                    chord_fraction=float(d.get("chord_fraction", 0.3)), chord_extension=float(d.get("chord_extension", 1.0)),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise HighLiftError(f"invalid high-lift device: {exc}") from exc

        return cls(phase=phase, flaps=dev(flaps), slats=dev(slats))


class HighLiftSection:
    """A :class:`~liftingline.core.section.SectionModel` with the high-lift devices applied to a clean one."""

    def __init__(self, base: SectionModel, system: HighLiftSystem, half_span: float) -> None:
        if not half_span > 0.0:
            raise HighLiftError("the semi-span must be positive")
        self.base, self.high_lift, self.half_span = base, system, float(half_span)
        self.label = f"{getattr(base, 'label', 'section data')} + high-lift devices"
        self.has_polars = bool(getattr(base, "has_polars", False))
        self.approximate = bool(getattr(base, "approximate", False))
        self.notes = list(getattr(base, "notes", ())) + [system.note()]

    def _map(self, y, re):
        inc = self.high_lift.increments(y, self.half_span)
        lin = self.base.linear(y, re)
        finite = np.isfinite(lin.cl_max) & (lin.cl_max > 0.0)
        s = np.where(finite, (lin.cl_max + inc.dcl_max) / np.where(finite, lin.cl_max, 1.0), 1.0)
        return lin, inc, s

    def linear(self, y, re) -> LinearParams:
        y = np.asarray(y, dtype=float)
        re = np.broadcast_to(np.asarray(re, dtype=float), y.shape).astype(float)
        lin, inc, s = self._map(y, re)
        k = inc.slope
        aL0 = lin.alpha_L0 + inc.dalpha_L0
        with np.errstate(invalid="ignore"):
            a_stall = np.where(np.isfinite(lin.alpha_stall), aL0 + s * (lin.alpha_stall - lin.alpha_L0) / k, lin.alpha_stall)
        return LinearParams(a0=k * lin.a0, alpha_L0=aL0, cm_c4=lin.cm_c4 + inc.dcm, cl_max=s * lin.cl_max, alpha_stall=a_stall)

    def mapped_alpha(self, y, alpha, re) -> np.ndarray:
        """Angle at which the clean polar is read for the high-lift section (``alpha`` itself where no device is)."""
        y = np.asarray(y, dtype=float)
        alpha = np.broadcast_to(np.asarray(alpha, dtype=float), y.shape).astype(float)
        re = np.broadcast_to(np.asarray(re, dtype=float), y.shape).astype(float)
        lin, inc, s = self._map(y, re)
        r = inc.slope / s
        # written so that r = 1 and d_alpha_L0 = 0 return alpha exactly (stations without a device)
        return alpha + (r - 1.0) * (alpha - lin.alpha_L0) - r * inc.dalpha_L0

    def evaluate(self, y, alpha, re) -> SectionCoeffs:
        y = np.asarray(y, dtype=float)
        alpha = np.broadcast_to(np.asarray(alpha, dtype=float), y.shape).astype(float)
        re = np.broadcast_to(np.asarray(re, dtype=float), y.shape).astype(float)
        lin, inc, s = self._map(y, re)
        r = inc.slope / s
        a_c = alpha + (r - 1.0) * (alpha - lin.alpha_L0) - r * inc.dalpha_L0
        c = self.base.evaluate(y, a_c, re)
        return SectionCoeffs(cl=s * c.cl, cd=c.cd + inc.dcd, cm=c.cm + inc.dcm, cl_alpha=inc.slope * c.cl_alpha)

    def confidence(self, y, alpha, re):
        conf = getattr(self.base, "confidence", None)
        if conf is None:
            return None
        return conf(y, self.mapped_alpha(y, alpha, re), re)


# ===================================================================================== stall report
#: conventional speed margins over the 1-g stall speed (Raymer, ch. 17; the certification rules use similar ones)
SPEED_FACTORS = {
    TAKEOFF: (("lift-off speed", 1.1), ("climb-out speed V2", 1.2)),
    LANDING: (("approach speed", 1.3), ("touchdown speed", 1.15)),
}


@dataclass
class ConfigStall:
    """Maximum lift of one configuration, from an angle-of-attack sweep.

    ``CL_max`` is the peak of the lift curve when the section data produce one (NeuralFoil, imported polars). Section
    data that saturate at cl_max (the built-in engine) give no peak; ``CL_max`` is then the lift coefficient at which
    the first section reaches its maximum (the critical-section method, slightly conservative)."""

    label: str
    CL_max: float
    alpha_CL_max: float
    definition: str  # "peak of the lift curve" or "first section stall"
    CL_first_stall: float
    alpha_first_stall: float
    y_first_stall: float
    CL_peak: float = math.nan
    alpha_peak: float = math.nan
    note: str = ""


@dataclass
class HighLiftReport:
    system: HighLiftSystem
    clean: ConfigStall
    deployed: ConfigStall
    estimate: WingEstimate
    clean_estimate_CL_max: float  # clean CL_max + Raymer's increment (comparison only)
    weight: float | None = None
    rho: float = math.nan
    area: float = math.nan
    speeds: dict[str, float] = field(default_factory=dict)  # name -> m/s
    notes: list[str] = field(default_factory=list)

    @property
    def dCL_max(self) -> float:
        return self.deployed.CL_max - self.clean.CL_max

    def stall_speed(self, which: str = "deployed") -> float:
        cfg = self.deployed if which == "deployed" else self.clean
        if not self.weight or not (cfg.CL_max > 0):
            return math.nan
        return math.sqrt(2.0 * self.weight / (self.rho * self.area * cfg.CL_max))


#: saturating section data: CL_max is taken where the wing's lift-curve slope has fallen to this fraction of its
#: attached-flow value (the knee of the lift curve)
KNEE_FRACTION = 0.2
DEF_PEAK, DEF_KNEE, DEF_FIRST = "peak of the lift curve", "knee of the lift curve", "first section stall"
#: section data with a post-stall lift drop: the wing has stalled when CL has fallen this much below its maximum
PEAK_DROP = 0.02
#: highest angle of attack searched for a lift peak (NeuralFoil's data are reliable up to about 30 degrees)
PEAK_SEARCH_LIMIT = 32.0


def section_saturates(req) -> bool:
    """True when the section data hold cl_max past the stall instead of losing lift (the built-in engine)."""
    hs = req.wing.half_span
    y = np.array([0.05, 0.5, 0.85]) * hs
    re = np.array([float(req.flow.reynolds(max(float(req.wing.chord(v)), 1e-3))) for v in y])
    lin = req.section.linear(y, re)
    if not np.all(np.isfinite(lin.alpha_stall)):
        return True
    for k in range(y.size):
        alpha = lin.alpha_stall[k] + np.radians(np.linspace(0.0, 15.0, 16))
        cl = req.section.evaluate(np.full(alpha.size, y[k]), alpha, np.full(alpha.size, re[k])).cl
        top = float(np.max(cl))
        if float(np.min(cl[int(np.argmax(cl)):])) < 0.95 * top:
            return False
    return True


def _knee(al: np.ndarray, cl: np.ndarray, mg: np.ndarray) -> tuple[float, float] | None:
    """(alpha, CL) where the lift-curve slope first drops below KNEE_FRACTION of its attached-flow value."""
    if al.size < 3:
        return None
    slopes = np.diff(cl) / np.diff(al)
    attached = slopes[(mg[1:] < 0.7) & np.isfinite(slopes)]
    if attached.size == 0:
        return None
    ref = float(np.max(attached))
    below = np.where((slopes < KNEE_FRACTION * ref) & (mg[1:] >= 0.7))[0]
    if below.size == 0:
        return None
    j = int(below[0])  # the interval (al[j], al[j+1]) is the first one past the knee: take its start
    return float(al[j]), float(cl[j])


def find_max_lift(req, label: str, *, alpha_start: float = 0.0, alpha_stop: float = 40.0, step: float = 2.0,
                  progress: Callable[[], None] | None = None, cancelled: Callable[[], bool] | None = None) -> ConfigStall:
    """Maximum lift of a configuration: a coarse angle-of-attack sweep until the lift has peaked (or, for section data
    that saturate at cl_max, has flattened out), then half-degree steps around the first stall and the maximum."""
    from .sweeps import ALPHA, SweepResult, run_sweep, stall_info

    pts = []

    def point(a: float):
        if cancelled is not None and cancelled():
            raise InterruptedError("cancelled")
        p = run_sweep(req, ALPHA, [a]).points[0]
        pts.append(p)
        if progress is not None:
            progress()
        return p

    def arrays():
        q = sorted((p for p in pts if p.ok), key=lambda p: p.value)
        return (np.array([p.value for p in q]), np.array([p.totals.CL for p in q]),
                np.array([p.max_margin if math.isfinite(p.max_margin) else 0.0 for p in q]))

    saturating = section_saturates(req)
    best, after_drop = -math.inf, 0
    a = alpha_start
    while a <= (alpha_stop if saturating else min(alpha_stop, PEAK_SEARCH_LIMIT)) + 1e-9:
        p = point(a)
        cl = p.totals.CL if p.ok else math.nan
        if math.isfinite(cl) and cl > best:
            best, after_drop = cl, 0
        elif math.isfinite(cl) and cl < (1.0 - PEAK_DROP) * best:
            after_drop += 1
        else:
            after_drop = 0
        if after_drop >= 2:  # the lift has clearly fallen past its maximum: stalled
            break
        if saturating:
            al, cls, mg = arrays()
            if al.size >= 4 and _knee(al[:-1], cls[:-1], mg[:-1]) is not None:
                break  # flat for two intervals in a row: past the knee
        a += step

    sweep0 = SweepResult(ALPHA, sorted(pts, key=lambda q: q.value))
    info0 = stall_info(sweep0)
    al, cls, mg = arrays()
    centres = []
    if cls.size:
        centres.append(float(al[int(np.argmax(cls))]))
        kn = _knee(al, cls, mg)
        if kn is not None:
            centres.append(kn[0])
    if info0.reached:
        centres.append(info0.alpha_first_stall)
    fine = set()
    for c in centres:
        fine.update(round(float(x), 6) for x in np.arange(c - 1.5, c + 1.51, 0.5))
    for x in sorted(fine):
        if alpha_start - 1e-9 <= x and not any(abs(q.value - x) < 1e-6 for q in pts):
            point(x)

    sweep = SweepResult(ALPHA, sorted(pts, key=lambda q: q.value))
    info = stall_info(sweep)
    al, cls, mg = arrays()
    cs = ConfigStall(
        label=label, CL_max=math.nan, alpha_CL_max=math.nan, definition="",
        CL_first_stall=info.CL_first_stall, alpha_first_stall=info.alpha_first_stall, y_first_stall=info.y_first_stall,
        note=info.note,
    )
    if not cls.size:
        cs.note = cs.note or "no converged points"
        return cs
    k = int(np.argmax(cls))
    if not saturating and k < cls.size - 1 and np.min(cls[k + 1:]) < (1.0 - PEAK_DROP) * cls[k]:
        cs.CL_peak, cs.alpha_peak = float(cls[k]), float(al[k])
        cs.CL_max, cs.alpha_CL_max, cs.definition = cs.CL_peak, cs.alpha_peak, DEF_PEAK
        return cs
    kn = _knee(al, cls, mg) if saturating else None
    if kn is not None:
        cs.alpha_CL_max, cs.CL_max = kn
        cs.definition = DEF_KNEE
    elif info.reached:
        cs.CL_max, cs.alpha_CL_max, cs.definition = info.CL_first_stall, info.alpha_first_stall, DEF_FIRST
    else:
        cs.note = cs.note or "no stall found in the swept range"
    return cs


def high_lift_report(clean_req, deployed_req, system: HighLiftSystem, *, weight: float | None = None,
                     progress: Callable[[int, int], None] | None = None,
                     cancelled: Callable[[], bool] | None = None) -> HighLiftReport:
    """Takeoff / landing stall analysis: maximum lift and stall of the clean and the configured wing, stall and
    reference speeds for a weight, and the comparison with the handbook wing-level estimate."""
    counter = [0]
    expected = 50  # rough number of solutions, for the progress bar only

    def tick() -> None:
        counter[0] += 1
        if progress is not None:
            progress(min(counter[0], expected - 1), expected)

    clean = find_max_lift(clean_req, "clean wing", progress=tick, cancelled=cancelled)
    deployed = find_max_lift(deployed_req, system.short_label(), progress=tick, cancelled=cancelled)
    est = system.wing_estimate(deployed_req.wing)
    rep = HighLiftReport(
        system=system, clean=clean, deployed=deployed, estimate=est,
        clean_estimate_CL_max=clean.CL_max + est.dCL_max, weight=weight if weight and weight > 0 else None,
        rho=deployed_req.flow.rho, area=deployed_req.wing.area,
    )
    if rep.weight:
        vs = rep.stall_speed()
        rep.speeds["stall speed (deployed)"] = vs
        rep.speeds["stall speed (clean)"] = rep.stall_speed("clean")
        for name, factor in SPEED_FACTORS[system.phase]:
            rep.speeds[f"{name} ({factor:g} Vs)"] = factor * vs
    for cfg in (clean, deployed):
        if not math.isfinite(cfg.CL_max):
            rep.notes.append(f"{cfg.label}: no maximum lift found ({cfg.note}).")
        elif cfg.definition == DEF_KNEE:
            rep.notes.append(
                f"{cfg.label}: the section data saturate at cl_max (no lift peak, built-in engine), so CL_max is read at "
                f"the knee of the lift curve, where its slope has fallen to {KNEE_FRACTION:.0%} of the attached-flow slope."
            )
        elif cfg.definition == DEF_FIRST:
            rep.notes.append(f"{cfg.label}: CL_max is the lift coefficient at the first section stall (no peak or knee found).")
    if progress is not None:
        progress(expected, expected)
    return rep


def format_report(rep: HighLiftReport, speed_unit: str = "m/s", speed_factor: float = 1.0, weight_unit: str = "N",
                  weight_factor: float = 1.0) -> str:
    """Plain-text summary of a :class:`HighLiftReport` (speeds and weight converted with the factors to the units)."""

    s, e = rep.system, rep.estimate
    lines = [f"High-lift devices - {PHASE_LABELS[s.phase].lower()} setting", ""]
    if s.flaps is not None:
        f = s.flaps
        lines.append(
            f"Flaps : {s.flap_type.label}, {100 * f.y_in:.0f}-{100 * f.y_out:.0f} % semi-span, chord {100 * f.chord_fraction:.0f} %"
            + (f", c'/c {f.chord_extension:.2f}" if s.flap_type.times_extension else "")
            + f"; typical deflection {s.flap_deflection_deg():.0f} deg"
        )
        lines.append(
            f"        section: d(cl_max) {s.flap_dcl_max():+.3f}, zero-lift angle {s.flap_dalpha_L0_deg():+.0f} deg, "
            f"d(cd) {s.flap_dcd():+.4f}, d(cm) {s.flap_dcm():+.3f}"
        )
    if s.slats is not None:
        sl = s.slats
        lines.append(
            f"Slats : {s.slat_type.label}, {100 * sl.y_in:.0f}-{100 * sl.y_out:.0f} % semi-span"
            + (f", c'/c {sl.chord_extension:.2f}" if s.slat_type.times_extension else "")
        )
        lines.append(f"        section: d(cl_max) {s.slat_dcl_max():+.3f}")
    lines.append("")
    lines.append(f"{'':16s}{'CL_max':>8s}{'at alpha':>12s}   {'first section stall':<34s}")
    for name, cfg in (("Clean wing", rep.clean), ("Devices out", rep.deployed)):
        lines.append(
            f"{name:16s}{cfg.CL_max:8.3f}{cfg.alpha_CL_max:8.2f} deg   alpha {cfg.alpha_first_stall:6.2f} deg, CL "
            f"{cfg.CL_first_stall:.3f}, near |y| = {cfg.y_first_stall:.2f} m"
        )
    lines.append("")
    lines.append(f"Increase of CL_max (lifting line)        {rep.dCL_max:+.3f}")
    lines.append(f"Handbook estimate (Raymer)               {e.dCL_max:+.3f}   = 0.9 d(cl_max) (S'/S) cos(sweep of the hinge line)")
    lines.append(
        f"   flapped area {100 * e.flapped_area_ratio:.0f} % of S (hinge sweep {e.hinge_sweep_deg:.1f} deg), slatted area "
        f"{100 * e.slatted_area_ratio:.0f} % of S (leading-edge sweep {e.le_sweep_deg:.1f} deg)"
    )
    if e.dCL_max > 0:
        lines.append(f"   lifting line / handbook               {rep.dCL_max / e.dCL_max:.2f}")
    if s.flaps is not None:
        lines.append(f"Flap drag d(CD0) (handbook)              {e.dCD0_flaps:+.4f}")
    if rep.weight:
        lines.append("")
        lines.append(f"Weight {rep.weight * weight_factor:.0f} {weight_unit}, density {rep.rho:.4f} kg/m^3, wing area {rep.area:.3f} m^2")
        for name, v in rep.speeds.items():
            lines.append(f"   {name:34s}{v * speed_factor:9.1f} {speed_unit}")
    lines.append("")
    for para in rep.notes + ["Data: " + DATA_SOURCE + ".",
                             "Speed margins: conventional factors over the 1-g stall speed (Raymer, ch. 17)."]:
        lines += textwrap.wrap(para, 100, subsequent_indent="   ")
    return "\n".join(lines)


def without_devices(system: HighLiftSystem) -> HighLiftSystem:
    return replace(system, flaps=None, slats=None)
